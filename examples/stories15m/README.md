# stories15M on the Rul runtime — head-to-head vs llama2.c

The same checkpoint llama2.c itself benchmarks with
(`karpathy/tinyllamas` **stories15M**: Llama arch, dim 288, 6 layers,
6 heads, head_dim 48, ffn 768, vocab 32000, seq 256, shared classifier,
15,204,000 params) running on this fork, raced against `run.c` under
identical conditions: same checkpoint, prompt "Once upon a time",
greedy (temperature 0), 256 total tokens (run.c caps steps at
seq_len), tok/s in llama2.c's own convention `(steps-1)/time`,
median of repeated runs on the same machine
(AMD EPYC 9D25, 4 vCPU, gcc 13.3, single-threaded unless noted).

| runtime | tok/s (median) | runs |
|---|---|---|
| llama2.c `run.c`, stock `-O3` | **71.7** | 71.9, 72.9, 71.7, 71.6, 70.7 |
| llama2.c `run.c`, `-O3 -march=native` | 69.4 | 68.5, 69.4, 69.4 |
| llama2.c `runomp` config (`-Ofast -fopenmp -march=native`, 4 threads) | 55.8 | 59.9, 41.5, 55.8 |
| **Rul (this fork, Tier 1)** | **64.0** | 63.5, 64.0, 62.4, 66.2, 66.0 |

So the Rul graph runtime lands at **~89% of hand-written C** on the
C program's own benchmark — despite two handicaps stated plainly:

- **Precision**: llama2.c computes in fp32; Rul computes in f64,
  moving twice the bytes per weight through memory.
- **Assembly**: the Rul timing includes re-assembling the block graph
  at every step (its structure grows with the KV cache); run.c has no
  equivalent overhead. Interleaved A/B runs confirmed the gap is
  stable, not VM drift.

Correctness, three ways: Rul's 256 greedy token ids are **identical**
to an independent f64 numpy reference (step-0 logits max diff
4.6e-14 over 32,000 logits), and the numpy reference's decoded text
matches the story `run.c` itself prints.

## Run it

```bash
# checkpoint: huggingface.co/karpathy/tinyllamas -> stories15M.bin
# tokenizer:  llama2.c repo -> tokenizer.bin
python3 examples/stories15m/stories15m_export.py   # -> stories15m_export/
cargo run --release --example stories15m stories15m_export 256
```

The exporter verifies the checkpoint by exact float count
(15,204,000) and replicates run.c's tokenizer semantics token-for-token
(dummy-prefix space piece, byte fallback `byte+3`, best-score merges —
this tokenizer.bin stores literal-space pieces, not U+2581).

## Mapping notes (Llama arch on Rul)

- RMSNorm is LayerNorm without mean-centering: `x * Rsqrt(mean(x^2)+eps) * g`
  — no new opcode beyond the fork's `Rsqrt`.
- SwiGLU's sigmoid is composed from `Exp`: `silu(x) = x / (1 + exp(-x))`.
- RoPE is applied host-side to q and to each new k before it enters
  the cache (the host already projects q/k/v for the current token;
  cached keys are stored pre-rotated, so attention inside the graph
  is unchanged). Everything else — attention, output projection,
  both norms, FFN, classifier — runs inside Rul graphs.
