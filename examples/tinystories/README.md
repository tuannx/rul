# TinyStories-8M on the Rul runtime (fork experiment)

The real **TinyStories-8M** model (Eldan & Li, Microsoft Research;
`roneneldan/TinyStories-8M` on Hugging Face — GPT-Neo: hidden 256,
8 layers, 16 heads, ffn 1024, vocab 50257, ~19.7M params) generating text
through verified Rul graphs on this fork. This is the transformer
counterpart of the MLP decision-model spike in `examples/decision/`.

## Run it

```bash
# 1. Download the checkpoint + tokenizer into this directory
#    (huggingface.co/roneneldan/TinyStories-8M: pytorch_model.bin, tokenizer.json)
# 2. Export f64 arenas + a numpy reference (restricted unpickler, no torch):
python3 examples/tinystories/export_weights.py
# 3. Generate with the Rul runtime:
cargo run --release --example tinystories examples/tinystories/export 40
```

`weights.bin` (~158 MB f64) is a build artifact and is **not** committed.

## Measured on the dev VM (prompt "Once upon a time", greedy, 40 tokens)

- Step-0 logits vs the f64 numpy reference: **max |diff| = 3.6e-14**
  over all 50,257 logits
- Greedy continuation: **40/40 tokens identical** to the reference —
  *"Once upon a time, there was a little girl named Lily. She was very
  excited to play with her friends. …"*
- Speed: **~20 ms/token (~51 tok/s)** end-to-end including graph
  assembly per step; prompt processing ~20 ms/token

## How it maps onto Rul

- Added for this: opcode **`Rsqrt` (0x1B)** (LayerNorm). With `Exp`
  (0x1A) and `MatVec` (0x24) from the decision spike, a full GPT-Neo
  block is expressible: LayerNorm from ReduceSum/Div/Rsqrt,
  attention scores as `MatVec(K_cache, q)`, softmax composed from
  Exp/ReduceSum/Div, per-head output projections summed as column
  blocks of Wo (no concat op needed), and `gelu_new` composed via
  `tanh(z) = 1 - 2/(exp(2z)+1)`.
- The host owns the token loop and the KV cache — the same split as
  llama2.c. Per token and layer the host runs two Rul graphs
  (LayerNorm, then the block) plus a final ln_f + classifier graph.
  The host itself computes only the k/v projections of the *new*
  token (those values must be extracted into the cache) and the
  embedding-row lookup.
- GPT-Neo's local-attention layers (window 256) need no window logic:
  the whole context stays below 256 tokens, so local == global.
- Checkpoint loading is done by `safe_load.py`, a restricted unpickler
  whitelisting exactly the four globals the file references
  (verified via pickletools) — no checkpoint code is executed.

## Honest limits

- Weights are f64: 19.7M params = ~158 MB, not the ~40 MB the fp32
  original occupies and far from any int8 budget. A narrow dtype lane
  is still the missing piece for size-competitive deployment.
- AOT still rejects these graphs (`UnsupportedGraph`); execution is
  Tier 0/1. The per-step block graph is reassembled as the cache grows.
- Generation quality is the model's own: greedy decoding of an 8M
  model loops phrases and eventually emits separator tokens after the
  story ends — the reference implementation does exactly the same.
