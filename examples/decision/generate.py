#!/usr/bin/env python3
"""Train a tiny MLP decision model (approve / review / reject for card
transactions) with numpy, then export it as a Rul program:

  model.rls          - 2-layer MLP + softmax expressed with MatVec/Exp nodes
  data/arena_1.bin   - W1 (16x6, row-major f64)
  data/arena_2.bin   - b1 (16)
  data/arena_3.bin   - W2 (3x16, row-major f64)
  data/arena_4.bin   - b2 (3)
  data/samples.bin   - held-out test features (N x 6 f64)
  data/labels.bin    - held-out labels (N f64: 0=approve, 1=review, 2=reject)
  data/probs.bin     - numpy reference probabilities for the test set (N x 3)

The Rul graph's arena 0 is the per-transaction feature vector, supplied by
the host at run time (see examples/decision_model.rs).

Labels come from a hand-written nonlinear risk rule (the "teacher") with 5%
label noise, so the MLP's ceiling is the teacher's own consistency.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

rng = np.random.default_rng(42)
D_IN, D_HID, D_OUT = 6, 16, 3


def gen(n):
    amount = np.exp(rng.uniform(np.log(5), np.log(5000), n))
    x0 = (np.log(amount) - np.log(5)) / (np.log(5000) - np.log(5))
    hour = rng.integers(0, 24, n)
    x1 = np.where((hour >= 0) & (hour <= 5), 1.0, 0.15)
    x2 = np.minimum(rng.poisson(2.0, n) / 10.0, 1.0)  # velocity
    x3 = rng.beta(2, 5, n)                            # merchant risk
    x4 = rng.beta(1.5, 6, n)                          # device risk
    x5 = rng.beta(1, 9, n)                            # chargeback history
    X = np.stack([x0, x1, x2, x3, x4, x5], axis=1)
    risk = (0.30 * x0 + 0.35 * x4 + 0.20 * x3 + 0.25 * x2 + 0.15 * x5
            + 0.10 * x1 + 0.25 * x0 * x4)
    y = np.where(risk < 0.38, 0, np.where(risk < 0.62, 1, 2))
    flip = rng.random(n) < 0.05
    y = np.where(flip, rng.integers(0, 3, n), y)
    return X, y


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


Xtr, ytr = gen(20000)
Xte, yte = gen(2000)

# --- tiny MLP, full-batch Adam, manual backprop ---------------------------
W1 = rng.normal(0, np.sqrt(2 / D_IN), (D_HID, D_IN)); b1 = np.zeros(D_HID)
W2 = rng.normal(0, np.sqrt(2 / D_HID), (D_OUT, D_HID)); b2 = np.zeros(D_OUT)
params = [W1, b1, W2, b2]
m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
Y = np.eye(D_OUT)[ytr]; N = len(Xtr)
for step in range(1, 401):
    H_pre = Xtr @ W1.T + b1; H = np.maximum(H_pre, 0)
    P = softmax(H @ W2.T + b2)
    dZ = (P - Y) / N
    grads = [dZ.T @ H, dZ.sum(0), (dZ @ W2) * (H_pre > 0), None]
    grads[0] = (dZ @ W2 * (H_pre > 0)).T @ Xtr if False else grads[0]
    dH = (dZ @ W2) * (H_pre > 0)
    grads = [dH.T @ Xtr, dH.sum(0), dZ.T @ H, dZ.sum(0)]
    for i in range(4):
        m[i] = 0.9 * m[i] + 0.1 * grads[i]; v[i] = 0.999 * v[i] + 0.001 * grads[i] ** 2
        mh = m[i] / (1 - 0.9 ** step); vh = v[i] / (1 - 0.999 ** step)
        params[i] -= 0.01 * mh / (np.sqrt(vh) + 1e-8)
    W1, b1, W2, b2 = params

Pte = softmax(np.maximum(Xte @ W1.T + b1, 0) @ W2.T + b2)
acc = float((Pte.argmax(1) == yte).mean())
print(f"numpy MLP test accuracy vs teacher labels: {acc:.4f}")
print("test class distribution:", np.bincount(yte, minlength=3).tolist())

# --- export ------------------------------------------------------------------
def wbin(name, arr):
    np.asarray(arr, dtype="<f8").tofile(os.path.join(DATA, name))

wbin("arena_1.bin", W1.ravel()); wbin("arena_2.bin", b1)
wbin("arena_3.bin", W2.ravel()); wbin("arena_4.bin", b2)
wbin("samples.bin", Xte); wbin("labels.bin", yte.astype(np.float64))
wbin("probs.bin", Pte)

RLS = """cap 0x0
arena 0 len=6 lane=f64x8 mut=false align=64 ver=1
arena 1 len=96 lane=f64x8 mut=false align=64 ver=1
arena 2 len=16 lane=f64x8 mut=false align=64 ver=1
arena 3 len=48 lane=f64x8 mut=false align=64 ver=1
arena 4 len=3 lane=f64x8 mut=false align=64 ver=1
node $x = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(0,1) a=0 b=0
node $w1 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(1,1) a=1 b=0
node $b1 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(2,1) a=2 b=0
node $w2 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(3,1) a=3 b=0
node $b2 = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(4,1) a=4 b=0
node $h1 = MatVec f64x8 fg=1 pol=simd proof=0 tok=pure a=$w1 b=$x
node $h1b = Add f64x8 fg=1 pol=simd proof=0 tok=pure a=$h1 b=$b1
node $zero = ConstF64 f64x1 fg=1 pol=simd proof=0 tok=pure const=0.0
node $h = Max f64x8 fg=1 pol=simd proof=0 tok=pure a=$h1b b=$zero
node $s = MatVec f64x8 fg=1 pol=simd proof=0 tok=pure a=$w2 b=$h
node $sb = Add f64x8 fg=1 pol=simd proof=0 tok=pure a=$s b=$b2
node $e = Exp f64x8 fg=1 pol=simd proof=0 tok=pure a=$sb b=0
node $es = ReduceSum f64x1 fg=1 pol=simd proof=0 tok=pure a=$e b=0
node $p = Div f64x8 fg=1 pol=simd proof=0 tok=pure a=$e b=$es
node $conf = ReduceMax f64x1 fg=1 pol=simd proof=0 tok=pure a=$p b=0
node $out = AssertContract f64x1 fg=1 pol=simd proof=0 tok=pure a=$conf b=0
output $out
"""
with open(os.path.join(HERE, "model.rls"), "w") as f:
    f.write(RLS)
with open(os.path.join(DATA, "expected.json"), "w") as f:
    json.dump({"numpy_test_accuracy": acc, "n_test": len(yte),
               "classes": ["approve", "review", "reject"]}, f, indent=1)
print("exported model.rls + data/ to", HERE)
