#!/usr/bin/env python3
"""Parse stories15M.bin (llama2.c format) + tokenizer.bin, export f64
weights for the Rul runtime, and produce an f64 numpy reference
(Llama forward with RoPE) whose greedy ids the Rul runner must match.

Checkpoint layout (verified by exact float count):
  header: 7 x int32 (dim, hidden, n_layers, n_heads, n_kv_heads,
          vocab_size [negative => unshared wcls], seq_len)
  fp32: wte, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3, rms_final,
        freq_cis_real, freq_cis_imag, [wcls if unshared]
"""
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "stories15M.bin")
TOKB = os.path.join(HERE, "llama2.c", "tokenizer.bin")
OUT = os.path.join(HERE, "stories15m_export")
os.makedirs(OUT, exist_ok=True)

raw = open(CKPT, "rb").read()
dim, hidden, L, H, KV, vocab_raw, seq = struct.unpack_from("<7i", raw, 0)
shared = vocab_raw > 0
V = abs(vocab_raw)
HD = dim // H
print(f"dim={dim} hidden={hidden} L={L} H={H} KV={KV} V={V} seq={seq} "
      f"head_dim={HD} shared_wcls={shared}")

f = np.frombuffer(raw, dtype="<f4", offset=28).astype(np.float64)
pos = 0


def take(*shape):
    global pos
    n = int(np.prod(shape))
    a = f[pos:pos + n].reshape(shape)
    pos += n
    return a


wte = take(V, dim)
rms_att = take(L, dim)
wq = take(L, dim, dim)
wk = take(L, dim, KV * HD)
wv = take(L, dim, KV * HD)
wo = take(L, dim, dim)
rms_ffn = take(L, dim)
w1 = take(L, hidden, dim)
w2 = take(L, dim, hidden)
w3 = take(L, hidden, dim)
rms_final = take(dim)
freq_re = take(seq, HD // 2)
freq_im = take(seq, HD // 2)
wcls = wte if shared else take(V, dim)
assert pos == len(f), f"float count mismatch: consumed {pos} of {len(f)}"
print(f"checkpoint parsed exactly: {pos:,} floats")

# ---------------------------------------------------------------- export ----
tensors = []


def put(name, arr):
    tensors.append((name, np.ascontiguousarray(arr, dtype="<f8")))


put("wte", wte)
put("wcls", wcls)
put("rms_final", rms_final)
put("freq_re", freq_re)
put("freq_im", freq_im)
for l in range(L):
    put(f"l{l}_rms_att", rms_att[l])
    put(f"l{l}_rms_ffn", rms_ffn[l])
    for h in range(H):
        put(f"l{l}_wq{h}", wq[l][h * HD:(h + 1) * HD, :])
        put(f"l{l}_wo{h}", wo[l][:, h * HD:(h + 1) * HD])
    kvh = KV // H if KV >= H else 1  # kv head per query head (MHA here: KV==H)
    for h in range(H):
        kh = h * kvh if KV == H else h // (H // KV)
        put(f"l{l}_wk{h}", wk[l][kh * HD:(kh + 1) * HD, :])
        put(f"l{l}_wv{h}", wv[l][kh * HD:(kh + 1) * HD, :])
    put(f"l{l}_w1", w1[l])
    put(f"l{l}_w2", w2[l])
    put(f"l{l}_w3", w3[l])

blob = np.concatenate([a.ravel() for _, a in tensors])
blob.tofile(os.path.join(OUT, "weights.bin"))
# F32 storage-lane artifact: the checkpoint is natively FP32, so this
# cast is exact (halves the file; the engine can consume it directly
# as MatVec/MatVecT weights via InputData::F32).
blob.astype("<f4").tofile(os.path.join(OUT, "weights_f32.bin"))
off = 0
with open(os.path.join(OUT, "manifest.txt"), "w") as fh:
    for name, a in tensors:
        fh.write(f"{name} {off} {a.size}\n")
        off += a.size
print(f"weights.bin: {blob.size:,} f64 = {blob.nbytes/1e6:.1f} MB, {len(tensors)} tensors")

# ------------------------------------------------------------- tokenizer ----
tb = open(TOKB, "rb").read()
max_len, = struct.unpack_from("<i", tb, 0)
p = 4
vocab = []  # (bytes, score)
for _ in range(V):
    score, ln = struct.unpack_from("<fi", tb, p)
    p += 8
    vocab.append((tb[p:p + ln], score))
    p += ln
vocab_map = {b: i for i, (b, _s) in enumerate(vocab)}


def encode(text: str, bos=True):
    """Replicates run.c's tokenizer exactly: BOS, then the dummy-prefix
    token (the literal " " piece), per-codepoint lookup with byte
    fallback (byte + 3), then best-score merges. Note this
    tokenizer.bin stores pieces with literal spaces, not U+2581."""
    toks = [1] if bos else []
    if bos and text:
        toks.append(vocab_map[b" "])
    ids = []
    for ch in text:
        b = ch.encode("utf-8")
        if b in vocab_map:
            ids.append(vocab_map[b])
        else:
            for byte in b:
                ids.append(byte + 3)
    toks += ids
    while True:
        best_score, best_idx, best_id = -1e30, -1, -1
        for i in range(len(toks) - 1):
            merged = vocab[toks[i]][0] + vocab[toks[i + 1]][0]
            mid = vocab_map.get(merged)
            if mid is not None and vocab[mid][1] > best_score:
                best_score, best_idx, best_id = vocab[mid][1], i, mid
        if best_idx < 0:
            break
        toks[best_idx:best_idx + 2] = [best_id]
    return toks


def decode(ids):
    out = b""
    for i in ids:
        if i in (1, 2):
            continue
        out += vocab[i][0]
    text = out.decode("utf-8", errors="replace")
    # run.c skips one leading space of the piece right after BOS
    return text[1:] if text.startswith(" ") else text


# ------------------------------------------------------------- reference ----
def rmsnorm(x, g):
    return x * (1.0 / np.sqrt((x * x).mean() + 1e-5)) * g


def rope(vec, pos_i):
    v = vec.reshape(H, HD).copy()
    fcr, fci = freq_re[pos_i], freq_im[pos_i]
    ev, od = v[:, 0::2], v[:, 1::2]
    out = np.empty_like(v)
    out[:, 0::2] = ev * fcr - od * fci
    out[:, 1::2] = ev * fci + od * fcr
    return out.ravel()


def forward(ids):
    t = len(ids)
    x = wte[ids].copy()
    kc = np.zeros((L, t, dim)); vc = np.zeros((L, t, dim))
    logits = None
    for pos_i in range(t):
        h = x[pos_i]
        for l in range(L):
            y = rmsnorm(h, rms_att[l])
            q = rope(wq[l] @ y, pos_i)
            k = rope(wk[l] @ y, pos_i)
            v = wv[l] @ y
            kc[l, pos_i] = k; vc[l, pos_i] = v
            o = np.zeros(dim)
            for hd_ in range(H):
                s = slice(hd_ * HD, (hd_ + 1) * HD)
                scores = kc[l, :pos_i + 1, s] @ q[s] / np.sqrt(HD)
                scores = scores - scores.max()
                e = np.exp(scores)
                att = e / e.sum()
                o[s] = att @ vc[l, :pos_i + 1, s]
            h = h + wo[l] @ o
            z = rmsnorm(h, rms_ffn[l])
            f1 = w1[l] @ z; f3 = w3[l] @ z
            silu = f1 * (1.0 / (1.0 + np.exp(-f1)))
            h = h + w2[l] @ (silu * f3)
        x[pos_i] = h
        logits = wcls @ rmsnorm(h, rms_final)
    return logits


prompt = "Once upon a time"
pids = encode(prompt)
print("prompt ids (mine):", pids)
step0 = forward(pids)
step0.astype("<f8").tofile(os.path.join(OUT, "step0_logits.bin"))
gen = list(pids)
while len(gen) < 256:  # run.c caps steps at seq_len (256)
    nxt = int(np.argmax(forward(gen)))
    gen.append(nxt)
    if nxt == 2:
        break
ref = {"prompt": prompt, "prompt_ids": pids, "ids_to_260": gen}
with open(os.path.join(OUT, "reference.json"), "w") as fh:
    json.dump(ref, fh, indent=1)
print("reference text:")
print(decode(gen))
