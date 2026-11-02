#!/usr/bin/env python3
"""Export TinyStories-8M (roneneldan, GPT-Neo) weights for the Rul runtime
and produce an f64 numpy reference generation to verify against.

Outputs (in ./export/):
  weights.bin    - every tensor the Rul graphs need, f64 LE, concatenated
  manifest.txt   - "name offset len" per tensor (offsets in f64 elements)
  reference.json - prompt ids, numpy greedy continuation ids, and the
                   step-0 (next-token after prompt) logits are in:
  step0_logits.bin - f64 logits (vocab,) after the prompt

GPT-Neo facts replicated here (from the checkpoint's own config):
  hidden 256, 8 layers, 16 heads (head_dim 16), ffn 1024, vocab 50257,
  gelu_new activation, LayerNorm eps 1e-5, q/k/v projections have NO bias,
  lm_head is tied to wte. Local-attention layers use window 256, which
  covers every context this demo generates (<= 256 tokens), so local ==
  global here and no window logic is needed.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_load import load_state_dict  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "export")
os.makedirs(OUT, exist_ok=True)

D, L, H, DH, FFN, V = 256, 8, 16, 16, 1024, 50257
EPS = 1e-5

sd = load_state_dict(os.path.join(HERE, "pytorch_model.bin"))
W = {k: np.asarray(v, dtype=np.float64) for k, v in sd.items()}

# ---------------------------------------------------------------- export ----
tensors = []  # (name, array)


def put(name, arr):
    tensors.append((name, np.ascontiguousarray(arr, dtype="<f8")))


put("wte", W["transformer.wte.weight"])
put("wpe", W["transformer.wpe.weight"])
put("lnf_g", W["transformer.ln_f.weight"])
put("lnf_b", W["transformer.ln_f.bias"])
for l in range(L):
    p = f"transformer.h.{l}"
    put(f"l{l}_ln1_g", W[f"{p}.ln_1.weight"]); put(f"l{l}_ln1_b", W[f"{p}.ln_1.bias"])
    put(f"l{l}_ln2_g", W[f"{p}.ln_2.weight"]); put(f"l{l}_ln2_b", W[f"{p}.ln_2.bias"])
    for proj in ("q", "k", "v"):
        M = W[f"{p}.attn.attention.{proj}_proj.weight"]  # (256, 256)
        for h in range(H):
            put(f"l{l}_{proj}{h}", M[h * DH:(h + 1) * DH, :])
    Wo = W[f"{p}.attn.attention.out_proj.weight"]  # (256, 256) rows=out
    for h in range(H):
        put(f"l{l}_wo{h}", Wo[:, h * DH:(h + 1) * DH])
    put(f"l{l}_bo", W[f"{p}.attn.attention.out_proj.bias"])
    put(f"l{l}_wfc", W[f"{p}.mlp.c_fc.weight"]); put(f"l{l}_bfc", W[f"{p}.mlp.c_fc.bias"])
    put(f"l{l}_wcp", W[f"{p}.mlp.c_proj.weight"]); put(f"l{l}_bcp", W[f"{p}.mlp.c_proj.bias"])

blob = np.concatenate([a.ravel() for _, a in tensors])
blob.tofile(os.path.join(OUT, "weights.bin"))
off = 0
with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    for name, a in tensors:
        f.write(f"{name} {off} {a.size}\n")
        off += a.size
print(f"weights.bin: {blob.size:,} f64 = {blob.nbytes/1e6:.1f} MB, {len(tensors)} tensors")

# ------------------------------------------------------------- reference ----
def ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    return xc / np.sqrt(var + EPS) * g + b


def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def forward(ids):
    t = len(ids)
    x = W["transformer.wte.weight"][ids] + W["transformer.wpe.weight"][:t]
    for l in range(L):
        p = f"transformer.h.{l}"
        y = ln(x, W[f"{p}.ln_1.weight"], W[f"{p}.ln_1.bias"])
        q = y @ W[f"{p}.attn.attention.q_proj.weight"].T
        k = y @ W[f"{p}.attn.attention.k_proj.weight"].T
        v = y @ W[f"{p}.attn.attention.v_proj.weight"].T
        o = np.zeros_like(x)
        mask = np.triu(np.full((t, t), -1e30), k=1)
        for h in range(H):
            s = slice(h * DH, (h + 1) * DH)
            scores = q[:, s] @ k[:, s].T / np.sqrt(DH) + mask
            scores = scores - scores.max(-1, keepdims=True)
            e = np.exp(scores)
            att = e / e.sum(-1, keepdims=True)
            o[:, s] = att @ v[:, s]
        x = x + o @ W[f"{p}.attn.attention.out_proj.weight"].T + W[f"{p}.attn.attention.out_proj.bias"]
        z = ln(x, W[f"{p}.ln_2.weight"], W[f"{p}.ln_2.bias"])
        f = z @ W[f"{p}.mlp.c_fc.weight"].T + W[f"{p}.mlp.c_fc.bias"]
        x = x + gelu_new(f) @ W[f"{p}.mlp.c_proj.weight"].T + W[f"{p}.mlp.c_proj.bias"]
    y = ln(x, W["transformer.ln_f.weight"], W["transformer.ln_f.bias"])
    return y @ W["transformer.wte.weight"].T


tok = Tokenizer.from_file(os.path.join(HERE, "tokenizer.json"))
prompt = "Once upon a time"
ids = tok.encode(prompt).ids
step0 = forward(ids)[-1]
step0.astype("<f8").tofile(os.path.join(OUT, "step0_logits.bin"))

gen = list(ids)
for _ in range(40):
    logits = forward(gen)
    gen.append(int(np.argmax(logits[-1])))
with open(os.path.join(OUT, "reference.json"), "w") as f:
    json.dump({"prompt": prompt, "prompt_ids": ids,
               "greedy_ids_40": gen[len(ids):]}, f, indent=1)
print("prompt ids:", ids)
print("numpy reference story:")
print(tok.decode(gen))
