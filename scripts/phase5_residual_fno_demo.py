"""Phase 5 — empirical syn-vs-exp residual FNO (baseline for CANN).

Trains a 1D Fourier Neural Operator to map ICL **synthetic** traces to
ICL **experimental** traces on the same src/rcv pairs. The learned
operator is the data-driven analogue of "what's missing from Robins
et al.'s forward model"; mostly attenuation + dispersion + calibration.

Why this exists: the diagnostic in
``scripts/icl_syn_vs_exp_diagnostic.py`` showed median correlation
0.871 between bandpassed paired traces. That number is the *baseline*
gap a Phase 5 modelling effort has to beat. This FNO is the *empirical
floor*: it has no inductive bias other than spectral mixing, so any
physics-based model (CANN α(ω) + K-K c(ω) running through a real
forward sim) needs to either match or exceed it.

If a ten-thousand-pair-trained FNO closes the gap from 0.871 → 0.99,
that tells us the missing physics is *learnable* from these traces;
Phase 5's value is then in *generalising* across geometries / pulses
the FNO can't, not in fitting capacity.

If a fully-trained FNO can't go past 0.92, the residual is data
limited (noise, inconsistencies) and the upper bound on any model —
CANN included — is closer to 0.92 than 1.0.

Procedure:
  1. Load 8 192 random paired traces (synthetic + experimental at the
     same trace indices). Bandpass each to 200 kHz – 1 MHz, normalise
     to unit L2.
  2. 80/20 train/val split.
  3. Train a 1D ClassicFNO (1 in, 1 out, 32 hidden, 16 modes, 4 blocks)
     with Adam for ~200 epochs, batch 32.
  4. Report on the val split:
       - Median |⟨ŷ, y⟩| (cosine correlation, FNO output vs true exp)
       - Median |⟨x, y⟩| (cosine correlation, identity baseline = the
         0.87 figure we already have)
       - rel-L2 of FNO output vs experimental
       - rel-L2 of identity-baseline (synthetic as predicted experimental)

Run:
    ICL_DUAL_PROBE_PATH=/path/to/icl-dual-probe-2023 \
        uv run python scripts/phase5_residual_fno_demo.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from pdequinox.arch import ClassicFNO
from scipy.signal import butter, sosfiltfilt

from brain_fwi.data.icl_dual_probe import load_icl_dual_probe


def _bandpass(x: np.ndarray, dt: float, lo: float, hi: float) -> np.ndarray:
    fnyq = 0.5 / dt
    sos = butter(4, [lo / fnyq, hi / fnyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _cosine_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("nt,nt->n", _norm(a), _norm(b))


def _rel_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(pred - target, axis=-1) /
            (np.linalg.norm(target, axis=-1) + 1e-12))


def main(
    n_pairs: int = 8192,
    n_epochs: int = 200,
    batch_size: int = 32,
    seed: int = 0,
):
    data_path = Path(
        os.environ.get(
            "ICL_DUAL_PROBE_PATH",
            "/Users/mhough/Workspace/brain-fwi/data/icl-dual-probe-2023",
        )
    )
    print(f"Loading {n_pairs} paired traces from {data_path}…")
    syn = load_icl_dual_probe(data_path, mode="synthetic")
    exp = load_icl_dual_probe(data_path, mode="experimental")
    try:
        dt = syn["dt"]
        rng = np.random.default_rng(seed)
        # Sort indices so h5py is happy. Both files share the trace
        # index space (same src_recv_pairs).
        idx = np.sort(rng.choice(syn["traces"].shape[0], size=n_pairs, replace=False))
        t0 = time.time()
        syn_chunk = np.asarray(syn["traces"][idx, :]).astype(np.float32)
        exp_chunk = np.asarray(exp["traces"][idx, :]).astype(np.float32)
        print(f"  loaded in {time.time()-t0:.1f}s; shape {syn_chunk.shape}")
    finally:
        syn["_h5_file"].close()
        exp["_h5_file"].close()

    print("Preprocessing: bandpass 200 kHz – 1 MHz, unit-L2 normalise per-trace…")
    syn_b = _bandpass(syn_chunk, dt, 2e5, 1e6).astype(np.float32)
    exp_b = _bandpass(exp_chunk, dt, 2e5, 1e6).astype(np.float32)
    syn_n = _norm(syn_b).astype(np.float32)
    exp_n = _norm(exp_b).astype(np.float32)

    n_train = int(0.8 * n_pairs)
    perm = rng.permutation(n_pairs)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    x_train, y_train = syn_n[train_idx], exp_n[train_idx]
    x_val,   y_val   = syn_n[val_idx],   exp_n[val_idx]

    # Identity-baseline (use synthetic as predicted experimental)
    base_corr = _cosine_corr(x_val, y_val)
    base_rl2  = _rel_l2(x_val, y_val)
    print(f"\nBaseline (synthetic ≈ experimental, identity map):")
    print(f"  median cosine corr: {np.median(base_corr):+.4f}")
    print(f"  median rel-L2     : {np.median(base_rl2):.4f}")

    # FNO setup
    print("\nBuilding ClassicFNO (1D, in=1, out=1, hidden=32, modes=16, blocks=4)…")
    model = ClassicFNO(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        num_modes=16,
        num_blocks=4,
        boundary_mode="periodic",
        key=jr.PRNGKey(seed),
    )

    @eqx.filter_jit
    def model_apply(m, x):
        # x: (B, T). Add channel dim, vmap over batch, drop channel.
        return jax.vmap(m)(x[:, None, :])[:, 0, :]

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(m, st, xb, yb):
        def loss_fn(mm):
            pred = model_apply(mm, xb)
            return jnp.mean((pred - yb) ** 2)
        loss, g = eqx.filter_value_and_grad(loss_fn)(m)
        upd, st = optimizer.update(g, st, m)
        m = eqx.apply_updates(m, upd)
        return m, st, loss

    print(f"Training for {n_epochs} epochs, batch {batch_size}, "
          f"{n_train} train traces…")
    n_batches = n_train // batch_size
    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_perm = rng.permutation(n_train)
        epoch_loss = 0.0
        for b in range(n_batches):
            bi = epoch_perm[b * batch_size:(b + 1) * batch_size]
            xb = jnp.asarray(x_train[bi])
            yb = jnp.asarray(y_train[bi])
            model, opt_state, loss = train_step(model, opt_state, xb, yb)
            epoch_loss += float(loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Val pass
            pred_v = np.asarray(model_apply(model, jnp.asarray(x_val)))
            v_corr = float(np.median(_cosine_corr(pred_v, y_val)))
            v_rl2 = float(np.median(_rel_l2(pred_v, y_val)))
            print(f"  ep {epoch+1:>4d}  train MSE {epoch_loss/n_batches:.4e}  "
                  f"val corr {v_corr:+.4f}  val rel-L2 {v_rl2:.4f}  "
                  f"[{time.time()-t0:.0f}s]")

    pred_val = np.asarray(model_apply(model, jnp.asarray(x_val)))
    fno_corr = _cosine_corr(pred_val, y_val)
    fno_rl2 = _rel_l2(pred_val, y_val)

    print("\n" + "=" * 70)
    print(f"{'metric':<28s}{'identity':>14s}{'FNO':>14s}{'gain':>14s}")
    print("=" * 70)
    print(f"{'median cosine corr':<28s}"
          f"{np.median(base_corr):>+14.4f}{np.median(fno_corr):>+14.4f}"
          f"{np.median(fno_corr) - np.median(base_corr):>+14.4f}")
    print(f"{'median rel-L2':<28s}"
          f"{np.median(base_rl2):>14.4f}{np.median(fno_rl2):>14.4f}"
          f"{np.median(fno_rl2) - np.median(base_rl2):>+14.4f}")
    print(f"{'frac corr > 0.95':<28s}"
          f"{(base_corr > 0.95).mean()*100:>13.1f}%"
          f"{(fno_corr > 0.95).mean()*100:>13.1f}%")
    print(f"{'frac corr < 0.50':<28s}"
          f"{(base_corr < 0.50).mean()*100:>13.1f}%"
          f"{(fno_corr < 0.50).mean()*100:>13.1f}%")
    print("=" * 70)

    print("\nINTERPRETATION")
    print("-" * 70)
    if np.median(fno_corr) > np.median(base_corr) + 0.05:
        print("FNO meaningfully reduces the syn↔exp residual. The missing")
        print("physics is learnable from these traces; this is the empirical")
        print("floor a Phase-5 CANN+K-K model needs to beat or match. Phase 5's")
        print("value will be in *generalisation* (other geometries / pulses)")
        print("rather than *fitting capacity* on this dataset.")
    else:
        print("FNO did NOT meaningfully reduce the residual. The 0.87 gap")
        print("is likely data-limited (noise, calibration, near-field) and")
        print("Phase 5 cannot expect to push correlation past ~0.9 either.")


if __name__ == "__main__":
    main()
