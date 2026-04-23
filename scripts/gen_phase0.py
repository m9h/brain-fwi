#!/usr/bin/env python
"""Phase-0 dataset generator for learning-based brain FWI.

Iterates over (subject x augmentation) combinations, runs a j-Wave
forward simulation, pretrains a SIREN field on the ground-truth
sound-speed volume, and streams each sample through
:class:`brain_fwi.data.ShardedWriter` with manifest-backed resume.

Supports two phantom sources:

  * ``--phantom synthetic`` — parametric three-layer ellipsoidal head
    (``phantoms.synthetic.make_three_layer_head``). Fast, deterministic,
    no external data required.
  * ``--phantom mida`` — real MIDA v1.0 anatomical head
    (``phantoms.mida.make_mida_phantom``). Richer anatomy, but single
    subject — variability comes from augmentation. Requires the MIDA
    NIfTI on disk (default path honours the layout at
    ``/data/datasets/MIDAv1-0/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii``).

Each sample captures:
  - ``tissue_labels``, ``sound_speed_voxel``, ``density_voxel``
  - ``transducer_positions``, ``sensor_positions``, ``source_signal``
  - ``observed_data`` (helmet-recorded traces from the j-Wave forward)
  - ``siren_weights_bytes`` — compact theta for downstream SBI / neural
    operator work

Design reference: ``docs/design/data_pipeline.md``. Augmentation
(per-tissue Gaussian property jitter + smooth deformation warp) comes
from ``phantoms/augment.py`` unchanged for both phantom paths.

Resume is automatic: reruns skip samples already listed in
``<out>/manifest.json``.

Example (CPU smoke test, tiny grid)::

    uv run python scripts/gen_phase0.py \\
        --out data/phase0_smoke --phantom synthetic --grid-size 32 \\
        --dx 0.004 --n-subjects 1 --n-augments 2 --n-elements 32 \\
        --siren-pretrain-steps 100 --iters-per-ramp 1 --freq 5e4

MIDA at production scale on DGX Spark::

    uv run python scripts/gen_phase0.py \\
        --out /data/datasets/brain-fwi/phase0_mida \\
        --phantom mida --grid-size 128 --dx 0.001 \\
        --n-subjects 1 --n-augments 100 --n-elements 256
"""

from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path
from typing import Tuple

# Request GPU before importing JAX (mirrors run_full_usct.py).
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", False)

from brain_fwi.data import ShardedWriter
from brain_fwi.inversion.param_field import init_siren_from_velocity
from brain_fwi.phantoms.augment import (
    jittered_properties,
    random_deformation_warp,
)
from brain_fwi.phantoms.mida import make_mida_phantom
from brain_fwi.phantoms.properties import map_labels_to_all
from brain_fwi.phantoms.synthetic import make_three_layer_head
from brain_fwi.simulation.forward import (
    _build_source_signal,
    build_domain,
    build_medium,
    build_time_axis,
    generate_observed_data,
)
from brain_fwi.transducers.helmet import (
    helmet_array_3d,
    transducer_positions_to_grid,
)


# ---------------------------------------------------------------------------
# Phantom dispatch
# ---------------------------------------------------------------------------

def _build_phantom_labels(
    phantom: str,
    grid_shape: Tuple[int, int, int],
    dx: float,
    mida_path: Path | None,
) -> np.ndarray:
    """Return the canonical label volume for the chosen phantom source."""
    if phantom == "synthetic":
        return make_three_layer_head(grid_shape, dx)
    if phantom == "mida":
        if mida_path is None or not Path(mida_path).exists():
            raise FileNotFoundError(
                f"MIDA NIfTI not found at {mida_path!r}. Pass --mida-path."
            )
        # Use make_mida_phantom to get the resampled + lesion-injected label
        # map; we discard the returned (c, rho, alpha) because the
        # augmentation pipeline redraws them from (jitter x labels).
        labels, _c, _rho, _alpha = make_mida_phantom(
            mida_path, grid_shape, dx, add_lesion=False, crop_cube=True,
        )
        return np.asarray(labels).astype(np.int32)
    raise ValueError(f"Unknown phantom {phantom!r}; expected synthetic or mida")


def _build_helmet(
    n_elements: int, grid_shape: Tuple[int, int, int], dx: float,
) -> Tuple[np.ndarray, tuple, list]:
    cx_m = grid_shape[0] * dx / 2
    cy_m = grid_shape[1] * dx / 2
    cz_m = grid_shape[2] * dx / 2
    r_ap = min(0.095, (grid_shape[0] * dx / 2) - 5 * dx)
    r_lr = min(0.075, (grid_shape[1] * dx / 2) - 5 * dx)
    r_si = min(0.090, (grid_shape[2] * dx / 2) - 5 * dx)
    positions = helmet_array_3d(
        n_elements=n_elements,
        center=(cx_m, cy_m, cz_m),
        radius_ap=r_ap, radius_lr=r_lr, radius_si=r_si,
        standoff=0.005, coverage_angle=2.8, exclude_face=True,
    )
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    n = len(pos_grid[0])
    src_list = [
        (int(pos_grid[0][i]), int(pos_grid[1][i]), int(pos_grid[2][i]))
        for i in range(n)
    ]
    return np.asarray(positions), pos_grid, src_list


def _serialize_siren(siren) -> np.ndarray:
    """Flatten an Equinox SIREN module into a uint8 byte blob."""
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, siren)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_sample(
    sample_id: str,
    subject_idx: int,
    aug_idx: int,
    grid_shape: Tuple[int, int, int],
    dx: float,
    freq_hz: float,
    n_elements: int,
    deformation_voxels: float,
    jitter_intensity: float,
    siren_hidden: int,
    siren_layers: int,
    siren_omega: float,
    siren_pretrain_steps: int,
    base_seed: int,
    phantom: str,
    mida_path: Path | None,
    verbose: bool = True,
) -> dict:
    subject_seed = base_seed * 10_000 + subject_idx * 100 + aug_idx
    np_rng = np.random.default_rng(subject_seed)
    jax_key = jr.PRNGKey(subject_seed)

    base_labels = _build_phantom_labels(phantom, grid_shape, dx, mida_path)
    warped_labels = random_deformation_warp(
        base_labels, np_rng,
        max_displacement_voxels=deformation_voxels,
        smoothness_voxels=max(8.0, min(grid_shape) / 6.0),
    )

    jax_key, jkey_prop = jr.split(jax_key)
    props = jittered_properties(
        jnp.asarray(warped_labels), jkey_prop, intensity=jitter_intensity,
    )
    # Replace any air labels (label 0 in the synthetic phantom; MIDA labels
    # 26-31 / 85 / 97 slip through this mask because they map via the "air"
    # group into c=343 only if not caught here) with water coupling for
    # USCT acquisition.
    coupling_mask = jnp.asarray(warped_labels) == 0
    sound_speed = jnp.where(coupling_mask, 1500.0, props["sound_speed"])
    density = jnp.where(coupling_mask, 1000.0, props["density"])

    positions, pos_grid, src_list = _build_helmet(n_elements, grid_shape, dx)
    sensor_grid = pos_grid

    domain = build_domain(grid_shape, dx)
    medium = build_medium(domain, sound_speed, density, pml_size=20)
    time_axis = build_time_axis(medium, cfl=0.3, t_end=None)
    dt = float(time_axis.dt)
    n_samples = int(float(time_axis.t_end) / dt)
    source_signal = _build_source_signal(freq_hz, dt, n_samples)

    t0 = time.time()
    observed = generate_observed_data(
        sound_speed=sound_speed,
        density=density,
        dx=dx,
        src_positions_grid=src_list,
        sensor_positions_grid=sensor_grid,
        freq=freq_hz,
        time_axis=time_axis,
        source_signal=source_signal,
        dt=dt,
        verbose=False,
    )
    sim_time = time.time() - t0
    if verbose:
        print(f"    forward sim: {sim_time:.1f}s for {len(src_list)} shots")

    jax_key, jkey_siren = jr.split(jax_key)
    siren_field = init_siren_from_velocity(
        sound_speed,
        c_min=1400.0, c_max=3200.0,
        hidden_dim=siren_hidden, n_hidden=siren_layers, omega_0=siren_omega,
        pretrain_steps=siren_pretrain_steps, learning_rate=1e-3,
        key=jkey_siren, verbose=False,
    )
    siren_v = siren_field.to_velocity(1400.0, 3200.0)
    siren_rel_err = float(
        jnp.mean(jnp.abs(siren_v - sound_speed)) / jnp.mean(sound_speed)
    )
    if verbose:
        print(f"    SIREN pretrain rel-err: {siren_rel_err:.3%}")

    return {
        "sample_id": sample_id,
        "subject_id": f"{phantom}_{subject_idx:03d}",
        "subject_idx": subject_idx,
        "aug_idx": aug_idx,
        "seed": subject_seed,
        "dx": float(dx),
        "freq_hz": float(freq_hz),
        "dt": float(dt),
        "grid_shape": np.asarray(grid_shape, dtype=np.int32),
        "tissue_labels": np.asarray(warped_labels).astype(np.uint8),
        "sound_speed_voxel": np.asarray(sound_speed).astype(np.float16),
        "density_voxel": np.asarray(density).astype(np.float16),
        "transducer_positions": positions.astype(np.float32),
        "sensor_positions": positions.astype(np.float32),
        "source_signal": np.asarray(source_signal).astype(np.float32),
        # observed_data stays float32: pressure magnitudes at kHz
        # frequencies overflow float16's +-65504 range. Per-sample
        # amplitude normalisation + float16 is the planned migration
        # (see data_pipeline.md §5).
        "observed_data": np.asarray(observed).astype(np.float32),
        "siren_weights_bytes": _serialize_siren(siren_field.siren),
        "siren_arch": {
            "hidden_dim": siren_hidden,
            "n_hidden": siren_layers,
            "omega_0": siren_omega,
            "in_dim": 3,
            "out_dim": 1,
        },
        "siren_rel_err": siren_rel_err,
        "phantom_version": f"{phantom}@v1",
        "acoustic_version": "aubry2022_itrusst",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-0 dataset generator")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset dir")
    parser.add_argument("--phantom", choices=("synthetic", "mida"),
                        default="synthetic")
    parser.add_argument(
        "--mida-path", type=Path,
        default=Path("/data/datasets/MIDAv1-0/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii"),
    )
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--dx", type=float, default=0.002)
    parser.add_argument("--freq", type=float, default=5.0e5,
                        help="Source centre frequency (Hz)")
    parser.add_argument("--n-subjects", type=int, default=1)
    parser.add_argument("--n-augments", type=int, default=4)
    parser.add_argument("--n-elements", type=int, default=64)
    parser.add_argument("--deformation-voxels", type=float, default=2.0)
    parser.add_argument("--jitter-intensity", type=float, default=1.0)
    parser.add_argument("--siren-hidden", type=int, default=128)
    parser.add_argument("--siren-layers", type=int, default=3)
    parser.add_argument("--siren-omega", type=float, default=30.0)
    parser.add_argument("--siren-pretrain-steps", type=int, default=600)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--version", type=str, default="phase0_v1")
    args = parser.parse_args()

    n = args.grid_size
    grid_shape = (n, n, n)

    print("=" * 70)
    print("  Phase-0 dataset generator")
    print("=" * 70)
    print(f"  Backend:        {jax.default_backend()}")
    print(f"  Phantom:        {args.phantom}"
          + (f" ({args.mida_path})" if args.phantom == "mida" else ""))
    print(f"  Grid:           {n}^3, dx={args.dx * 1e3:.2f} mm")
    print(f"  Frequency:      {args.freq / 1e3:.0f} kHz")
    print(f"  Transducers:    {args.n_elements}")
    print(f"  Subjects x aug: {args.n_subjects} x {args.n_augments} "
          f"= {args.n_subjects * args.n_augments} samples")
    print(f"  Output:         {args.out}")

    writer = ShardedWriter(
        args.out,
        shard_size=args.shard_size,
        version=args.version,
        metadata={
            "phantom": args.phantom,
            "grid_size": n,
            "dx": args.dx,
            "freq_hz": args.freq,
            "n_elements": args.n_elements,
            "deformation_voxels": args.deformation_voxels,
            "jitter_intensity": args.jitter_intensity,
        },
    )
    print(f"  Resuming from:  {writer.n_completed} sample(s) already written")

    total_t0 = time.time()
    generated = 0
    skipped = 0

    for subj in range(args.n_subjects):
        for aug in range(args.n_augments):
            sample_id = f"{args.phantom}_{subj:03d}_{aug:03d}"
            if writer.is_complete(sample_id):
                skipped += 1
                continue

            print(f"\n[{subj+1}/{args.n_subjects}  aug {aug+1}/{args.n_augments}] "
                  f"{sample_id}")
            sample = generate_sample(
                sample_id=sample_id,
                subject_idx=subj,
                aug_idx=aug,
                grid_shape=grid_shape,
                dx=args.dx,
                freq_hz=args.freq,
                n_elements=args.n_elements,
                deformation_voxels=args.deformation_voxels,
                jitter_intensity=args.jitter_intensity,
                siren_hidden=args.siren_hidden,
                siren_layers=args.siren_layers,
                siren_omega=args.siren_omega,
                siren_pretrain_steps=args.siren_pretrain_steps,
                base_seed=args.base_seed,
                phantom=args.phantom,
                mida_path=args.mida_path,
            )
            writer.write(sample)
            generated += 1

    writer.close()
    total_t = time.time() - total_t0
    print("\n" + "=" * 70)
    print(f"  Done. Generated {generated}, skipped {skipped}, "
          f"total {writer.n_completed} in dataset.")
    print(f"  Wall time: {total_t / 60:.1f} min "
          f"({total_t / max(1, generated):.1f}s per new sample)")
    print(f"  Manifest:  {writer.manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
