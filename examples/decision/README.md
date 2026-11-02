# Decision-model runtime spike (fork experiment)

A tiny MLP decision model — 6 features -> 16 hidden (ReLU) -> 3 classes
(approve / review / reject for card transactions) — trained with numpy in
`generate.py`, exported as a Rul graph (`model.rls`) and run by the new
`MatVec` (0x24) and `Exp` (0x1A) opcodes added on this branch.

This is a **fork-only spike** to size up a neural runtime for Rul; it is not
an upstream proposal yet.

## Run it

```bash
python3 examples/decision/generate.py   # retrain + re-export (seed 42)
cargo run --release --example decision_model
```

## Measured on the dev VM (seed 42, 2,000 held-out samples)

- numpy MLP test accuracy vs teacher labels: **0.9375**
  (teacher labels carry 5% noise, so the ceiling is ~= 0.95)
- Rul accuracy: **0.9375** (1875/2000) — identical decisions
- max |p_rul - p_numpy| over all samples/classes: **3.3e-16**
- Tier 0 vs Tier 2 confidence agreement (first 200 samples): **200/200**
- Tier 2 score-only latency: **~1.9 us/decision**; with full white-box
  trace: **~3.7 us/decision**

## Scope and honest limits

- Training stays outside Rul (numpy here; PyTorch would do). Rul is the
  verified runtime only — the graph is a static DAG, there is no backprop,
  no training loop, no optimizer state.
- `MatVec` shape rule is proven by the verifier: the weight column length
  must be a positive multiple of the input vector length
  (`VerifyErrorKind::InvalidMatVecShape` otherwise). No new proof entries
  or format changes: both opcodes fit the existing 16-byte `NodeFrame`.
- Softmax is *composed*, not a new opcode: `Exp` -> `ReduceSum` -> `Div`.
  ReLU is `Max(x, 0)`. Argmax stays a host-side read of the probability
  column via the white-box trace.
- **AOT does not support `Exp`/`MatVec` yet**: `rul build` rejects such
  graphs with an explicit `UnsupportedGraph` error instead of
  miscompiling. Execution is Tier 0/1/2 (Tier 2 falls back to Tier 1 for
  these graphs today).
- Everything is f64. A ~50 MB weight budget therefore holds ~6M params,
  not the ~25-45M an int8/f16 lane would allow; adding a narrow dtype for
  weight arenas is the obvious next step after this spike.
- CLI note (pre-existing upstream behavior): `rul run|trace` resolves
  `data/arena_*.bin` relative to the **current working directory**, not
  the `.rls` file's project directory; the synthetic-data warning added
  in #9 makes a miss loud instead of silent.
