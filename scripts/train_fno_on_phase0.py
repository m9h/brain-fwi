#!/usr/bin/env python
"""Train the UNO/FNO surrogate on a Phase-0 dataset.

Used by ``scripts/modal_train_fno_phase0.py`` to scale training to H100s.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from brain_fwi.data import ShardedReader
from brain_fwi.surrogate.fno3d import CToTraceFNO3D
from brain_fwi.surrogate.train import train_fno_surrogate
from brain_fwi.surrogate.validation import (
    trace_fidelity,
    format_gate_report,
    gradient_accuracy,
    _predict_all_shots,
)
from brain_fwi.simulation.forward import (
    build_domain,
    build_medium,
    build_time_axis,
    simulate_shot_sensors,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True,
                    help="Phase-0 dataset root")
    ap.add_argument("--out", type=Path, required=True,
                    help="Prefix for model and metrics output")
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--lambda-spec", type=float, default=0.3)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--num-modes", type=int, default=12)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--held-out-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # --- Setup ----------------------------------------------------------
    reader = ShardedReader(args.data)
    n_samples = len(reader)
    if n_samples < 5:
        print(f"Dataset too small: {n_samples} samples")
        return 1

    print("=" * 70)
    print(f"  Phase 4 Step 3: Train UNO Surrogate on {args.data}")
    print("=" * 70)
    print(f"  samples:      {n_samples}")
    print(f"  jax devices:  {jax.devices()}")

    # --- Split ----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(reader.sample_ids)
    n_test = max(1, int(round(n_samples * args.held_out_fraction)))
    held_out_ids = perm[:n_test].tolist()
    train_ids = perm[n_test:].tolist()

    print(f"  split:        train={len(train_ids)}  held_out={len(held_out_ids)}")

    # --- Init model -----------------------------------------------------
    # Read first sample to get shapes
    first = reader[train_ids[0]]
    c_voxel = first["sound_speed_voxel"]
    d_true = first["observed_data"]
    # d_true shape is (n_src, n_t, n_recv)
    grid_shape = c_voxel.shape
    n_t, n_recv = d_true.shape[1], d_true.shape[2]

    # Compute output scale (target.std() across dataset)
    # Sample up to 10 random training items to estimate std
    print("  estimating output scale...")
    n_est = min(len(train_ids), 10)
    est_stds = []
    for i in range(n_est):
        d_est = reader[train_ids[i]]["observed_data"]
        est_stds.append(float(np.std(d_est)))
    output_scale = float(np.mean(est_stds))

    print(f"  grid_shape:   {grid_shape}")
    print(f"  n_t:          {n_t}")
    print(f"  n_receivers:  {n_recv}")
    print(f"  output_scale: {output_scale:.4e}")
    print(f"  architecture: hidden={args.hidden_channels}, modes={args.num_modes}, depth={args.depth}")

    key = jr.PRNGKey(args.seed)
    model_key, train_key = jr.split(key)

    model = CToTraceFNO3D(
        grid_shape=grid_shape,
        n_timesteps=n_t,
        n_receivers=n_recv,
        hidden_channels=args.hidden_channels,
        num_modes=args.num_modes,
        depth=args.depth,
        output_scale=output_scale,
        key=model_key,
    )

    # --- Train ----------------------------------------------------------
    t0 = time.time()
    trained, losses = train_fno_surrogate(
        model,
        reader,
        n_steps=args.n_steps,
        key=train_key,
        learning_rate=args.learning_rate,
        lambda_spec=args.lambda_spec,
        held_out_ids=held_out_ids,
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"\n  training time: {train_time/60:.1f} min")

    # --- Validate -------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Validation (§7.2 Trace-fidelity Gate)")
    print("=" * 70)
    
    held_out_samples = [reader[sid] for sid in held_out_ids]
    from brain_fwi.surrogate.train import _extract_source_positions
    src_pos = _extract_source_positions(first)

    trace_metrics = trace_fidelity(
        trained,
        held_out_samples,
        source_positions=src_pos,
    )

    # --- Gradient Accuracy (§7.3) ---------------------------------------
    print("\n" + "=" * 70)
    print("  Validation (§7.3 Gradient-accuracy Gate)")
    print("=" * 70)
    
    n_grad = min(len(held_out_ids), 20)
    grad_ids = held_out_ids[:n_grad]
    grad_samples = [reader[sid] for sid in grad_ids]
    
    # Pre-build j-Wave components from the first sample
    dx = float(first["dx"])
    freq = float(first.get("frequency", 500000.0))
    domain = build_domain(grid_shape, dx)
    
    # Assume fixed density for gradient comparison
    rho_const = jnp.asarray(first["density_voxel"], dtype=jnp.float32)

    def surrogate_fwd(c_v):
        c_n = (c_v - 1400.0) / (3200.0 - 1400.0)
        return _predict_all_shots(trained, c_n, src_pos)

    def jwave_fwd(c_v):
        medium = build_medium(domain, c_v, rho_const)
        # We need a time_axis and signal that match the model's n_timesteps
        dt_val = float(first.get("dt", 0.3 * dx / 3200.0))
        sig = jnp.asarray(first["source_signal"], dtype=jnp.float32)
        if len(sig) > trained.n_timesteps:
            sig = sig[:trained.n_timesteps]
        
        from jwave.geometry import TimeAxis
        t_axis = TimeAxis(t_end=float(trained.n_timesteps * dt_val), dt=dt_val)

        # Convert src_pos list to sensor grid tuple
        sensor_grid = (
            jnp.array([s[0] for s in src_pos]),
            jnp.array([s[1] for s in src_pos]),
            jnp.array([s[2] for s in src_pos]),
        )

        return jnp.stack([
            simulate_shot_sensors(medium, t_axis, src, sensor_grid, sig, dt_val)
            for src in src_pos
        ], axis=0)

    # Use a subset of velocity fields for speed
    c_grad_samples = [jnp.asarray(s["sound_speed_voxel"], dtype=jnp.float32) for s in grad_samples]
    
    grad_metrics = gradient_accuracy(
        surrogate_forward=surrogate_fwd,
        jwave_forward=jwave_fwd,
        c_samples=c_grad_samples,
    )
    
    print(format_gate_report(trace_metrics, grad_metrics))

    # --- Persist --------------------------------------------------------
    out_dir = args.out if args.out.is_dir() else args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = args.out.with_suffix(".eqx") if not args.out.is_dir() else args.out / "model.eqx"
    json_path = args.out.with_suffix(".json") if not args.out.is_dir() else args.out / "metrics.json"

    eqx.tree_serialise_leaves(model_path, trained)
    
    report = {
        "config": vars(args),
        "metrics": trace_metrics,
        "grad_metrics": grad_metrics,
        "loss_history": [float(l) for l in losses],
        "train_time_s": train_time,
    }
    json_path.write_text(json.dumps(report, indent=2, default=str))
    
    print(f"\n  model written:   {model_path}")
    print(f"  metrics written: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
