#!/usr/bin/env python
"""A/B benchmark: pool-readout FNO vs gather-readout FNO on a 2D toy.

Phase 4 readiness evidence — quantifies the architectural lever called
out in `docs/design/phase4_fno_surrogate.md` §10 step 4 and in the
existing 9.2.1 evidence (toy 2D FNO with pool readout: p50 8.23% rel-L2,
p95 28.45%, well above the 1%/5% gate).

Hypothesis: replacing the global-average-pool readout with a spatial
gather at the (fixed) receiver voxel substantially improves trace
fidelity, because the wave equation's response at a sensor depends on
its location, not a global summary of the velocity field.

Method
------
- Single source, single receiver, 2D homogeneous-water phantoms with
  a Gaussian velocity bump (random centre + amplitude).
- Identical UNO backbone for both models (width / modes / depth).
- Identical optimiser, learning rate, batch size, and number of epochs.
- Same train/test split and PRNG keys.

Output: median and 95th-percentile relative-L2 per trace, side-by-side.
This is the smallest-possible apples-to-apples test.
"""

from __future__ import annotations

import os
import time

# Use the GPU the toy was originally validated on if available; CPU
# fallback prints a warning but still completes.
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from brain_fwi.simulation.forward import (
    _build_source_signal,
    build_domain,
    build_medium,
    build_time_axis,
    simulate_shot_sensors,
)
from brain_fwi.surrogate.fno2d import (
    CToTraceFNO,
    CToTraceFNO2DGather,
    CToTraceFNO2DPoolGather,
)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def gaussian_bump_phantom(grid: int, key: jax.Array, c_bg: float = 1500.0) -> jnp.ndarray:
    """Random Gaussian bump on a homogeneous-water 2D field."""
    key_ctr, key_amp, key_sigma = jr.split(key, 3)
    cx = jr.uniform(key_ctr, (), minval=grid * 0.3, maxval=grid * 0.7)
    cy = jr.uniform(jr.fold_in(key_ctr, 1), (), minval=grid * 0.3, maxval=grid * 0.7)
    amp = jr.uniform(key_amp, (), minval=200.0, maxval=600.0)
    sigma = jr.uniform(key_sigma, (), minval=2.0, maxval=5.0)
    x, y = jnp.meshgrid(jnp.arange(grid, dtype=jnp.float32),
                          jnp.arange(grid, dtype=jnp.float32), indexing="ij")
    return c_bg + amp * jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def generate_dataset(
    n_samples: int, grid: int, dx: float, src_pos, recv_pos, freq_hz: float, key,
):
    """Generate ``n_samples`` (c, d) pairs."""
    domain = build_domain((grid, grid), dx)
    medium_ref = build_medium(domain, 1500.0, 1000.0, pml_size=8)
    time_axis = build_time_axis(medium_ref, cfl=0.3, t_end=20e-6)
    dt = float(time_axis.dt)
    n_t = int(float(time_axis.t_end) / dt)
    source_signal = _build_source_signal(freq_hz, dt, n_t)

    cs, ds = [], []
    for i in range(n_samples):
        ck = jr.fold_in(key, i)
        c = gaussian_bump_phantom(grid, ck)
        medium = build_medium(domain, c, jnp.ones_like(c) * 1000.0, pml_size=8)
        sensor_pos = (jnp.array([recv_pos[0]]), jnp.array([recv_pos[1]]))
        d = simulate_shot_sensors(medium, time_axis, src_pos, sensor_pos,
                                    source_signal, dt)
        cs.append(np.asarray(c))
        ds.append(np.asarray(d).squeeze(-1))   # → (n_t,)
    return np.stack(cs), np.stack(ds), n_t


# ---------------------------------------------------------------------------
# Train one model end-to-end and measure rel-L2 on held-out data
# ---------------------------------------------------------------------------

def train_and_eval(
    label: str, model, c_tr, d_tr, c_te, d_te, n_epochs: int, batch_size: int,
    lr: float, key,
):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    n_train = len(c_tr)

    @eqx.filter_jit
    def step(m, opt_state, c_b, d_b):
        def loss_fn(m_):
            preds = jax.vmap(m_)(c_b)
            return jnp.mean((preds - d_b) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, opt_state = optimizer.update(grads, opt_state)
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss

    print(f"\n[{label}] training {n_epochs} epochs (batch={batch_size}, lr={lr})...")
    t0 = time.time()
    for epoch in range(n_epochs):
        key, sub = jr.split(key)
        perm = np.asarray(jr.permutation(sub, n_train))
        epoch_losses = []
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            model, opt_state, loss = step(
                model, opt_state, jnp.asarray(c_tr[idx]), jnp.asarray(d_tr[idx]),
            )
            epoch_losses.append(float(loss))
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"  [{label}] epoch {epoch+1:3d}/{n_epochs}  "
                  f"loss={np.mean(epoch_losses):.6f}")

    elapsed = time.time() - t0
    print(f"[{label}] training done in {elapsed:.1f}s")

    preds = jax.vmap(model)(jnp.asarray(c_te))
    err = jnp.linalg.norm(preds - jnp.asarray(d_te), axis=1) / (
        jnp.linalg.norm(jnp.asarray(d_te), axis=1) + 1e-12
    )
    return {
        "label": label,
        "median_rel_l2": float(jnp.median(err)),
        "p95_rel_l2": float(jnp.percentile(err, 95)),
        "mean_rel_l2": float(jnp.mean(err)),
        "train_time_s": elapsed,
    }


def main() -> None:
    print("=" * 72)
    print("  Phase 4 — gather vs pool readout A/B benchmark")
    print("=" * 72)
    print(f"  Backend: {jax.default_backend()}  device: {jax.devices()[0]}")

    # Reasonably small to keep the benchmark CPU-feasible (~5-10 min)
    # while still in the regime that produced the 8.23% pool-readout
    # baseline (toy_2d_fno.py uses grid=32 + n_train=500).
    grid = 24
    n_train, n_test = 80, 20
    dx = 4e-4                  # 0.4 mm — domain ~10 mm, comfortably within wave field
    src_pos = (grid // 2, grid // 4)
    recv_pos = (grid // 2, grid - grid // 4)
    freq_hz = 5.0e5

    # UNO backbone hyperparams — identical for both models
    width, modes, depth = 32, 6, 2
    n_epochs, batch_size, lr = 80, 16, 1e-3

    print(f"  Grid:       {grid}x{grid} @ dx={dx*1e3:.2f} mm")
    print(f"  Source:     {src_pos}")
    print(f"  Receiver:   {recv_pos}")
    print(f"  Frequency:  {freq_hz/1e3:.0f} kHz")
    print(f"  Train/Test: {n_train}/{n_test}")
    print(f"  UNO arch:   width={width}, modes={modes}, depth={depth}")
    print(f"  Train:      {n_epochs} epochs, batch={batch_size}, lr={lr}")

    # Dataset
    print("\nGenerating dataset (j-Wave forward sim)...")
    t0 = time.time()
    c_all, d_all, _ = generate_dataset(
        n_train + n_test, grid, dx, src_pos, recv_pos, freq_hz, jr.PRNGKey(0),
    )
    n_t = d_all.shape[1]   # trust the simulator's actual output length
    print(f"  shapes: c={c_all.shape}, d={d_all.shape}  ({n_t} timesteps)")
    print(f"  dataset gen: {time.time()-t0:.1f}s")

    # Per-trace normalisation matters because trace amplitudes vary;
    # without it MSE is dominated by a few high-amplitude samples.
    d_mean = d_all.mean()
    d_std = d_all.std() + 1e-8
    d_n = (d_all - d_mean) / d_std

    c_tr, c_te = c_all[:n_train], c_all[n_train:]
    d_tr, d_te = d_n[:n_train], d_n[n_train:]

    # All three models built with the same backbone hyperparams + key.
    pool_model = CToTraceFNO(
        grid_h=grid, grid_w=grid, n_timesteps=n_t,
        width=width, modes=modes, depth=depth, key=jr.PRNGKey(42),
    )
    gather_model = CToTraceFNO2DGather(
        grid_h=grid, grid_w=grid, n_timesteps=n_t, recv_pos=recv_pos,
        width=width, modes=modes, depth=depth, key=jr.PRNGKey(42),
    )
    poolgather_model = CToTraceFNO2DPoolGather(
        grid_h=grid, grid_w=grid, n_timesteps=n_t, recv_pos=recv_pos,
        width=width, modes=modes, depth=depth, key=jr.PRNGKey(42),
    )

    results = []
    for label, m in [
        ("POOL", pool_model),
        ("GATHER", gather_model),
        ("POOL+GATHER", poolgather_model),
    ]:
        results.append(train_and_eval(
            label, m, c_tr, d_tr, c_te, d_te,
            n_epochs, batch_size, lr, key=jr.PRNGKey(7),
        ))

    # Report
    print("\n" + "=" * 72)
    print("  Result")
    print("=" * 72)
    print(f"  {'readout':<14} {'median rel-L2':>14} {'p95 rel-L2':>14} "
          f"{'train time':>12}")
    for r in results:
        print(f"  {r['label']:<14} {r['median_rel_l2']*100:>13.3f}% "
              f"{r['p95_rel_l2']*100:>13.3f}% {r['train_time_s']:>10.1f}s")

    pool_med = results[0]['median_rel_l2']
    print()
    for r in results[1:]:
        delta = (pool_med - r['median_rel_l2']) / max(pool_med, 1e-12)
        sign = "improves" if delta > 0 else "regresses"
        print(f"  {r['label']:<14} {sign} median rel-L2 by {abs(delta)*100:.1f}% vs POOL")


if __name__ == "__main__":
    main()
