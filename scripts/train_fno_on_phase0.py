#!/usr/bin/env python
"""Train the 3D FNO surrogate on the Phase-0 dataset.

Phase 4 §10 step 5 — first measurement of the §7.2 trace-fidelity gate
against real MIDA traces (vs the 8-13% rel-L2 the toy 2D experiment
got with synthetic phantoms).

Pipeline::

  1. ShardedReader on the dataset root.
  2. Hold out a held_out_fraction split (default 20%) for validation.
  3. Build a CToTraceFNO3D with the configured backbone size.
  4. train_fno_surrogate over the training split.
  5. trace_fidelity gate on the held-out split, report median + p95
     rel-L2 + the §7.2 gate verdict.

Usage::

    python scripts/train_fno_on_phase0.py \\
        --data /data/datasets/brain-fwi/phase0_v1_mida_96 \\
        --out  /data/datasets/brain-fwi/fno_surrogate_v1 \\
        --n-steps 1000
"""

from __future__ import annotations

import argparse
import json
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
    format_gate_report,
    trace_fidelity,
)


def _split_ids(reader, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    ids = list(reader.sample_ids)
    rng.shuffle(ids)
    n_held = max(1, int(len(ids) * fraction))
    return ids[n_held:], ids[:n_held]


def _resolve_source_positions(sample) -> list[tuple[int, int, int]]:
    if "transducer_positions_grid" in sample:
        arr = np.asarray(sample["transducer_positions_grid"])
    else:
        positions_m = np.asarray(sample["transducer_positions"])
        dx = float(sample["dx"])
        arr = np.round(positions_m / dx).astype(np.int32)
    return [tuple(int(x) for x in row) for row in arr]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True,
                    help="Phase-0 dataset root (contains manifest.json).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output directory for the trained surrogate.")
    ap.add_argument("--held-out-fraction", type=float, default=0.2)
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--lambda-spec", type=float, default=0.3)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--num-modes", type=int, default=12)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print("  Phase 4 FNO surrogate training on real Phase-0 data")
    print("=" * 78)
    print(f"  JAX backend: {jax.default_backend()}  device: {jax.devices()[0]}")
    print(f"  Dataset:     {args.data}")

    reader = ShardedReader(args.data, fields=[
        "sound_speed_voxel", "observed_data",
        "transducer_positions", "transducer_positions_grid", "dx",
    ])
    print(f"  Total samples: {len(reader)}")

    train_ids, held_out_ids = _split_ids(reader, args.held_out_fraction, args.seed)
    print(f"  Train: {len(train_ids)} / Held out: {len(held_out_ids)}")

    # Get model dimensions from the first sample (all samples are same
    # geometry per gen_phase0.py).
    first = reader[train_ids[0]]
    grid_shape = tuple(int(x) for x in np.asarray(first["sound_speed_voxel"]).shape)
    n_t, n_recv = np.asarray(first["observed_data"]).shape[1:]
    src_positions = _resolve_source_positions(first)
    print(f"  Grid:        {grid_shape}")
    print(f"  Trace shape: ({n_t} timesteps, {n_recv} receivers)")
    print(f"  Sources:     {len(src_positions)}")
    print(f"  FNO backbone: hidden={args.hidden_channels}, "
          f"modes={args.num_modes}, depth={args.depth}")

    key = jr.PRNGKey(args.seed)
    model_key, train_key = jr.split(key)
    model = CToTraceFNO3D(
        grid_shape=grid_shape,
        n_timesteps=int(n_t),
        n_receivers=int(n_recv),
        hidden_channels=args.hidden_channels,
        num_modes=args.num_modes,
        depth=args.depth,
        key=model_key,
    )

    print(f"\nTraining for {args.n_steps} steps...")
    t0 = time.time()
    trained, losses = train_fno_surrogate(
        model, reader,
        n_steps=args.n_steps,
        key=train_key,
        learning_rate=args.learning_rate,
        lambda_spec=args.lambda_spec,
        source_positions=src_positions,
        held_out_ids=held_out_ids,
        log_every=50,
        verbose=True,
    )
    train_wall = time.time() - t0
    print(f"\nTraining wall: {train_wall/60:.1f} min")

    # §7.2 gate on held-out
    print("\nEvaluating §7.2 trace-fidelity gate on held-out samples...")
    held_out_samples = [reader[sid] for sid in held_out_ids]
    metrics = trace_fidelity(trained, held_out_samples, src_positions)
    print(format_gate_report(metrics))

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(str(args.out / "fno_surrogate.eqx"), trained)
        (args.out / "training_summary.json").write_text(json.dumps({
            "dataset": str(args.data),
            "n_train": len(train_ids),
            "n_held_out": len(held_out_ids),
            "n_steps": args.n_steps,
            "learning_rate": args.learning_rate,
            "lambda_spec": args.lambda_spec,
            "hidden_channels": args.hidden_channels,
            "num_modes": args.num_modes,
            "depth": args.depth,
            "train_wall_min": train_wall / 60.0,
            "loss_history": [float(x) for x in losses],
            "trace_fidelity_metrics": {k: (v if isinstance(v, (int, float, bool, str))
                                            else float(v))
                                        for k, v in metrics.items()},
            "held_out_ids": held_out_ids,
        }, indent=2))
        print(f"\nSaved to {args.out}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
