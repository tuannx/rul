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
| Rul + `MatVecT` no-copy cache read (below) | **66.9** | 67.8, 65.8, 66.9, 66.9, 67.4, 63.4 |

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

## Where the time goes (profiled, 256 tokens, `RUL_PROFILE=1`)

| phase | ms/token | share |
|---|---|---|
| Rul graph execution | 12.9 | 83% |
| host q/k/v projections + RoPE | 1.4 | 9% |
| graph assembly (text `.rls` per step) | 0.69 | 4.4% |
| V^T transpose copies | 0.46 | 2.9% |

A bare-loop micro-benchmark of the same matvecs streams weights at
~9.2 GB/s on this VM; the pure-matvec floor is ~13.3 ms/token
(~75 tok/s). In other words **~85% of the runtime is irreducible
f64 weight streaming** (122 MB read per token) — no-copy work can
only reclaim the remaining ~15%.

First slice of that work, measured: opcode **`MatVecT` (0x25)**
computes `W^T . x` over the row-major cache directly, so the V
cache is read in its append layout with **zero transpose copies**:
66.9 vs 64.0 tok/s median (**+4.5%**), ids unchanged (256/256).
Remaining identified overhead: operand clones inside the evaluator
(~0.9 ms/token) and text-program assembly (~0.65 ms/token) — together
worth roughly another +8-10%, up to the ~75 tok/s streaming floor.
Beyond that floor, only fewer bytes per weight (an fp32/int8 lane)
or more memory bandwidth can help; llama2.c's own OpenMP build shows
threads do not add bandwidth on this VM.

## F32 storage lane — built, lossless, and (here) not faster

The fork's ISA now has an F32 storage lane (`ElemType::F32 = 0x4`,
`lane=f32x8`): an F32 arena may feed only `MatVec`/`MatVecT` as
weights, elements are widened inside the product loop, accumulation
stays f64. For this checkpoint the lane is **lossless** — every
weight is natively FP32, and the runner's step-0 logits diff
(4.6e-14) and 256/256 greedy ids are unchanged when consuming
`weights_f32.bin` (98 MB instead of 195 MB).

It is also, measured on this VM, **slower**, and no loop shape fixed
it. Classifier matvec (32,000x288), ms per call:

| variant | ms |
|---|---|
| F64 plain (incumbent) | 8.15–8.30 |
| F32 storage, widen in loop | 9.57 |
| F32 storage, widen row/block first | 11.57–11.69 |
| F32 accumulate | 9.97 |
| pure F32 compute (x pre-cast) | 7.96 |

End-to-end the F32-lane runner scored 58 tok/s vs 67 for F64.
Interpretation: at this VM's ~9 GB/s effective streaming the F64
loop is bound by the load+FMA pipeline, not by byte count, so
halving bytes buys nothing while conversions cost µops; even pure
F32 compute gains only ~2%. The lane is kept for its footprint
halving and because the calculus changes on wider-bandwidth
hardware — but the benchmark runner deliberately stays on F64.
