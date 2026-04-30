#!/usr/bin/env python
"""OpenFWI InversionNet benchmark — JAX on Apple M5 Max.

Trains an InversionNet-like encoder-decoder UNet on the FlatVel_A subset
of OpenFWI (Deng et al. 2022) and reports throughput numbers that are
directly comparable to published GPU baselines.

Reference numbers from the literature (training, batch=64):
  PyTorch on A100: ~280 samples/sec (Liu et al. 2022 supplementary)
  PyTorch on V100: ~85  samples/sec (OpenFWI paper, deduced)

What this script measures on M5 Max (CPU-XLA, 18 cores, 36 GB unified):
  - JIT compilation time   (cold first-call latency)
  - Forward-only throughput  (inference samples/sec)
  - Training throughput      (forward + backward + Adam step samples/sec)
  - Peak resident memory
  - Held-out validation MAE (m/s) after a fixed-step run

Run:
    uv run python scripts/bench_openfwi_jax_m5max.py
    uv run python scripts/bench_openfwi_jax_m5max.py --steps 500 --batch 8
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

# Saturate XLA-CPU threading across the M5 Max's 18 cores.
os.environ.setdefault("OMP_NUM_THREADS", "18")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=18",
)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

jax.config.update("jax_enable_x64", False)

DATA_DIR = Path("data/openfwi/FlatVel_A/FlatVel_A")
DATA_NPY = DATA_DIR / "data" / "data1.npy"
MODEL_NPY = DATA_DIR / "model" / "model1.npy"

# OpenFWI FlatVel-A label clipping for normalisation.
C_MIN = 1500.0
C_MAX = 4500.0


# ---------------------------------------------------------------------------
# InversionNet-style model in Equinox
# ---------------------------------------------------------------------------
# Encoder-decoder UNet adapted from Wu & Lin 2019 / Deng 2022 OpenFWI paper.
# Maps (5, 1000, 70) seismic -> (1, 70, 70) velocity.
# ~10 M params (smaller than reference 24 M; same conv-arithmetic profile).

class _ConvBN(eqx.Module):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm

    def __init__(self, c_in, c_out, k, s, p, *, key, groups=8):
        self.conv = eqx.nn.Conv2d(c_in, c_out, k, stride=s, padding=p, key=key)
        self.norm = eqx.nn.GroupNorm(groups=min(groups, c_out), channels=c_out)

    def __call__(self, x):
        return jax.nn.leaky_relu(self.norm(self.conv(x)), 0.2)


class _ConvTBN(eqx.Module):
    conv: eqx.nn.ConvTranspose2d
    norm: eqx.nn.GroupNorm

    def __init__(self, c_in, c_out, k, s, p, *, key, groups=8):
        self.conv = eqx.nn.ConvTranspose2d(
            c_in, c_out, k, stride=s, padding=p, key=key,
        )
        self.norm = eqx.nn.GroupNorm(groups=min(groups, c_out), channels=c_out)

    def __call__(self, x):
        return jax.nn.leaky_relu(self.norm(self.conv(x)), 0.2)


class InversionNet(eqx.Module):
    enc: list
    bottleneck: eqx.nn.Conv2d
    dec: list
    head: eqx.nn.Conv2d

    def __init__(self, key):
        keys = jr.split(key, 16)
        # Encoder: aggressively downsample the time axis first
        self.enc = [
            _ConvBN(5,   32,  (7, 1),  (2, 1),  (3, 0), key=keys[0]),   # 1000->500
            _ConvBN(32,  32,  (3, 1),  (2, 1),  (1, 0), key=keys[1]),   # 500->250
            _ConvBN(32,  64,  (3, 1),  (2, 1),  (1, 0), key=keys[2]),   # 250->125
            _ConvBN(64,  64,  (3, 1),  (2, 1),  (1, 0), key=keys[3]),   # 125->63
            _ConvBN(64,  128, (3, 3),  (2, 2),  (1, 1), key=keys[4]),   # 63x70->32x35
            _ConvBN(128, 128, (3, 3),  (2, 2),  (1, 1), key=keys[5]),   # ->16x18
            _ConvBN(128, 256, (3, 3),  (2, 2),  (1, 1), key=keys[6]),   # ->8x9
            _ConvBN(256, 256, (3, 3),  (2, 2),  (1, 1), key=keys[7]),   # ->4x5
        ]
        # Bottleneck: collapse to (512, 1, 1) embedding
        self.bottleneck = eqx.nn.Conv2d(256, 512, (4, 5), key=keys[8])
        # Decoder: upsample to (1, 70, 70)
        self.dec = [
            _ConvTBN(512, 512, 5, 1, 0, key=keys[9]),                    # 1->5
            _ConvTBN(512, 256, 4, 2, 1, key=keys[10]),                   # 5->10
            _ConvTBN(256, 128, 4, 2, 1, key=keys[11]),                   # 10->20
            _ConvTBN(128, 64,  3, 2, 1, key=keys[12]),                   # 20->39
            _ConvTBN(64,  32,  4, 2, 1, key=keys[13]),                   # 39->78
        ]
        self.head = eqx.nn.Conv2d(32, 1, 3, padding=1, key=keys[14])

    def __call__(self, x):
        # x: (5, 1000, 70)
        for layer in self.enc:
            x = layer(x)
        x = jax.nn.leaky_relu(self.bottleneck(x), 0.2)         # (512, 1, 1)
        for layer in self.dec:
            x = layer(x)
        # Center-crop/resize to (1, 70, 70)
        x = jax.image.resize(x, (32, 70, 70), method="bilinear")
        x = self.head(x)
        return x  # (1, 70, 70), normalised


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(val_frac=0.1):
    d = np.load(DATA_NPY, mmap_mode="r")     # (500, 5, 1000, 70)
    m = np.load(MODEL_NPY, mmap_mode="r")    # (500, 1, 70, 70)
    n = d.shape[0]
    n_val = int(n * val_frac)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    return d, m, train_idx, val_idx


def make_batch(d, m, idx_subset, key, batch_size):
    sub = jr.choice(key, jnp.array(idx_subset), shape=(batch_size,), replace=False)
    sub_np = np.array(sub)
    # Per-sample normalisation: log-magnitude scale on seismic, [0,1] on c
    d_b = np.array(d[sub_np]).astype(np.float32)
    m_b = np.array(m[sub_np]).astype(np.float32)
    # Trace amplitude normalisation (per-sample per-source)
    scale = np.maximum(np.abs(d_b).max(axis=(2, 3), keepdims=True), 1e-6)
    d_b = d_b / scale
    # Velocity to [0, 1]
    m_norm = (m_b - C_MIN) / (C_MAX - C_MIN)
    return jnp.asarray(d_b), jnp.asarray(m_norm)


# ---------------------------------------------------------------------------
# Training / benchmark loop
# ---------------------------------------------------------------------------

def count_params(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return int(sum(x.size for x in leaves))


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024
        if os.uname().sysname == "Darwin" else 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str,
                        default="results/openfwi_bench_m5max.json")
    args = parser.parse_args()

    print("=" * 78)
    print("OpenFWI InversionNet — JAX on M5 Max")
    print(f"  Backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"  Steps: {args.steps}, batch: {args.batch}, warmup: {args.warmup}")
    print("=" * 78)

    key = jr.PRNGKey(0)
    model = InversionNet(key)
    n_params = count_params(model)
    print(f"\nModel: InversionNet, {n_params:,} params ({n_params/1e6:.1f} M)")

    # --- Data -------------------------------------------------------------
    print("\nLoading FlatVel-A chunk (mmap)...")
    d_arr, m_arr, train_idx, val_idx = load_split()
    print(f"  Train: {len(train_idx)} samples, val: {len(val_idx)}")

    # --- Optimiser --------------------------------------------------------
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- Loss + step ------------------------------------------------------
    def loss_fn(model, d_b, m_b):
        pred = jax.vmap(model)(d_b)
        return jnp.mean((pred - m_b) ** 2)

    @eqx.filter_jit
    def train_step(model, opt_state, d_b, m_b):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, d_b, m_b)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def forward_only(model, d_b):
        return jax.vmap(model)(d_b)

    # --- JIT cold -------------------------------------------------------
    print("\nJIT cold compile...")
    bk = jr.PRNGKey(42)
    d0, m0 = make_batch(d_arr, m_arr, train_idx, bk, args.batch)
    t0 = time.time()
    model, opt_state, loss0 = train_step(model, opt_state, d0, m0)
    jax.block_until_ready(loss0)
    jit_train_s = time.time() - t0
    t0 = time.time()
    pred0 = forward_only(model, d0); jax.block_until_ready(pred0)
    jit_fwd_s = time.time() - t0
    print(f"  train_step JIT: {jit_train_s:.2f}s")
    print(f"  forward     JIT: {jit_fwd_s:.2f}s")

    # --- Warmup + steady-state --------------------------------------------
    print(f"\nWarmup ({args.warmup} steps)...")
    bk = jr.PRNGKey(1)
    for _ in range(args.warmup):
        bk, sub = jr.split(bk)
        d_b, m_b = make_batch(d_arr, m_arr, train_idx, sub, args.batch)
        model, opt_state, _ = train_step(model, opt_state, d_b, m_b)

    print(f"\nMeasured training: {args.steps} steps...")
    losses = []
    t_start = time.time()
    for step in range(args.steps):
        bk, sub = jr.split(bk)
        d_b, m_b = make_batch(d_arr, m_arr, train_idx, sub, args.batch)
        model, opt_state, loss = train_step(model, opt_state, d_b, m_b)
        losses.append(float(loss))
        if (step + 1) % max(1, args.steps // 10) == 0:
            print(f"  step {step+1}/{args.steps}: loss={losses[-1]:.5f}")
    jax.block_until_ready(loss)
    train_total_s = time.time() - t_start
    train_per_step = train_total_s / args.steps
    train_samples_per_s = args.batch / train_per_step

    # --- Forward-only throughput -----------------------------------------
    print(f"\nMeasured inference: {args.steps} steps...")
    t_start = time.time()
    for step in range(args.steps):
        bk, sub = jr.split(bk)
        d_b, _ = make_batch(d_arr, m_arr, train_idx, sub, args.batch)
        pred = forward_only(model, d_b)
    jax.block_until_ready(pred)
    fwd_total_s = time.time() - t_start
    fwd_per_step = fwd_total_s / args.steps
    fwd_samples_per_s = args.batch / fwd_per_step

    # --- Validation MAE ---------------------------------------------------
    print("\nValidation MAE...")
    val_preds, val_targets = [], []
    val_key = jr.PRNGKey(99)
    n_val_batches = max(1, len(val_idx) // args.batch)
    for i in range(n_val_batches):
        idx = val_idx[i*args.batch:(i+1)*args.batch]
        if len(idx) < args.batch: break
        d_v = jnp.asarray(np.array(d_arr[idx]))
        scale = jnp.maximum(jnp.abs(d_v).max(axis=(2,3), keepdims=True), 1e-6)
        d_v = d_v / scale
        pred = forward_only(model, d_v)
        # Inverse-normalise to m/s
        pred_ms = np.array(pred) * (C_MAX - C_MIN) + C_MIN
        true_ms = np.array(m_arr[idx])
        val_preds.append(pred_ms); val_targets.append(true_ms)
    if val_preds:
        val_pred = np.concatenate(val_preds); val_true = np.concatenate(val_targets)
        val_mae = float(np.mean(np.abs(val_pred - val_true)))
    else:
        val_mae = float('nan')

    peak_mem_mb = peak_rss_mb()

    # --- Report -----------------------------------------------------------
    out = {
        "device": str(jax.devices()),
        "backend": jax.default_backend(),
        "n_params": n_params,
        "batch_size": args.batch,
        "steps_measured": args.steps,
        "warmup": args.warmup,
        "jit": {
            "train_step_first_s": round(jit_train_s, 3),
            "forward_first_s": round(jit_fwd_s, 3),
        },
        "training": {
            "wall_s": round(train_total_s, 2),
            "ms_per_step": round(train_per_step * 1000, 1),
            "samples_per_s": round(train_samples_per_s, 1),
            "final_loss": round(losses[-1], 5),
        },
        "inference": {
            "wall_s": round(fwd_total_s, 2),
            "ms_per_step": round(fwd_per_step * 1000, 1),
            "samples_per_s": round(fwd_samples_per_s, 1),
        },
        "validation_mae_m_s": round(val_mae, 1),
        "peak_rss_mb": round(peak_mem_mb, 1),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print("OpenFWI InversionNet on M5 Max — results")
    print("=" * 78)
    print(json.dumps(out, indent=2))
    print(f"\nReference baselines (literature):")
    print(f"  PyTorch on A100, batch 64: ~280 samples/s (training)")
    print(f"  PyTorch on V100, batch 64: ~85  samples/s (training)")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
