#!/usr/bin/env python
"""3D Brain Recovery Simulation with Ultrasound Probe Helmet.

Complete working simulation of transcranial ultrasound Full Waveform
Inversion for brain sound-speed imaging. Includes:

  1. Anatomically-informed synthetic 3D head phantom (scalp, skull, CSF,
     grey matter, white matter, ventricles) with ITRUSST acoustic properties
  2. Kernel Flow-inspired probe helmet (256 elements, whole-head coverage,
     Fibonacci sphere sampling with face exclusion)
  3. Forward data generation via j-Wave pseudospectral solver
  4. Multi-frequency FWI with JAX autodiff gradients
  5. Reconstruction quality metrics and HDF5 output

Designed to run at two scales:
  --fast   : 48^3 grid, 32 elements, ~5 min on CPU (proof of concept)
  default  : 96^3 grid, 128 elements, ~1 hr on GPU (research quality)
  --full   : 192^3 grid, 256 elements, needs >=24GB VRAM (publication)

Usage:
    uv run python examples/03_3d_brain_recovery.py --fast
    uv run python examples/03_3d_brain_recovery.py
    uv run python examples/03_3d_brain_recovery.py --full --device gpu
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from brain_fwi.phantoms.properties import map_labels_to_all
from brain_fwi.transducers.helmet import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    simulate_shot_sensors, _build_source_signal,
)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi
from brain_fwi.inversion.losses import l2_loss


# ============================================================================
# 3D Head Phantom
# ============================================================================

def create_head_phantom_3d(
    grid_shape: tuple,
    dx: float,
    include_lesion: bool = True,
) -> dict:
    """Create an anatomically-informed 3D head phantom.

    Concentric ellipsoidal layers with realistic proportions:
    - Scalp: 3mm thick
    - Skull: 7mm thick (cortical bone)
    - CSF: 2mm subarachnoid space
    - Grey matter: cortical ribbon ~4mm
    - White matter: core
    - Ventricles: lateral ventricles
    - Optional lesion: 1cm diameter haemorrhage-like inclusion

    Returns dict with 'labels', 'sound_speed', 'density', 'attenuation'.
    """
    nx, ny, nz = grid_shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    # Head semi-axes in grid units
    # Adult head: ~19cm AP, ~15cm LR, ~18cm SI
    head_a = min(0.095 / dx, cx - 3)  # AP, clamped to fit grid
    head_b = min(0.075 / dx, cy - 3)  # LR
    head_c = min(0.090 / dx, cz - 3)  # SI

    x, y, z = jnp.meshgrid(
        jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij"
    )

    # Normalized ellipsoidal radius
    r = jnp.sqrt(
        ((x - cx) / head_a) ** 2 +
        ((y - cy) / head_b) ** 2 +
        ((z - cz) / head_c) ** 2
    )

    # Layer boundaries (outside → inside)
    scalp_t = 0.003 / (head_a * dx)
    skull_t = 0.007 / (head_a * dx)
    csf_t = 0.002 / (head_a * dx)
    cortex_t = 0.004 / (head_a * dx)

    r_scalp = 1.0
    r_skull_o = r_scalp - scalp_t
    r_skull_i = r_skull_o - skull_t
    r_csf = r_skull_i
    r_csf_i = r_csf - csf_t
    r_cortex_i = r_csf_i - cortex_t

    # BrainWeb label convention: 0=bg, 1=CSF, 2=GM, 3=WM, 6=scalp, 7=skull
    labels = jnp.zeros(grid_shape, dtype=jnp.int32)
    labels = jnp.where(r <= r_scalp, 6, labels)      # scalp
    labels = jnp.where(r <= r_skull_o, 7, labels)     # skull
    labels = jnp.where(r <= r_skull_i, 1, labels)     # CSF
    labels = jnp.where(r <= r_csf_i, 2, labels)       # grey matter
    labels = jnp.where(r <= r_cortex_i, 3, labels)    # white matter

    # Lateral ventricles (two ellipsoids)
    for y_offset in [-0.015 / dx, 0.015 / dx]:
        vent_r = jnp.sqrt(
            ((x - cx) / (0.010 / dx)) ** 2 +
            ((y - cy - y_offset) / (0.008 / dx)) ** 2 +
            ((z - cz + 0.005 / dx) / (0.025 / dx)) ** 2
        )
        labels = jnp.where((vent_r <= 1.0) & (r <= r_cortex_i), 1, labels)

    # Optional lesion (simulated haemorrhage — higher speed of sound)
    if include_lesion:
        lesion_r = jnp.sqrt(
            ((x - cx + 0.02 / dx) / (0.005 / dx)) ** 2 +
            ((y - cy + 0.01 / dx) / (0.005 / dx)) ** 2 +
            ((z - cz) / (0.005 / dx)) ** 2
        )
        # Label 8 = blood vessels (1584 m/s) — simulates haemorrhage
        labels = jnp.where((lesion_r <= 1.0) & (r <= r_cortex_i), 8, labels)

    props = map_labels_to_all(labels)

    # Replace air (label 0, outside head) with water for coupling
    c = jnp.where(labels == 0, 1500.0, props["sound_speed"])
    rho = jnp.where(labels == 0, 1000.0, props["density"])

    return {
        "labels": labels,
        "sound_speed": c,
        "density": rho,
        "attenuation": props["attenuation"],
    }


# ============================================================================
# Probe Helmet Design
# ============================================================================

def create_probe_helmet(
    n_elements: int,
    grid_shape: tuple,
    dx: float,
) -> dict:
    """Create a probe helmet array matched to the head phantom.

    Positions elements on an ellipsoidal shell with:
    - Fibonacci sphere sampling for near-uniform coverage
    - Face exclusion (no elements in front of face, below equator)
    - 5mm water coupling standoff from scalp
    - Farthest-point subsampling for optimal spatial distribution

    Returns dict with 'positions_m', 'positions_grid', 'src_list', 'sensor_grid'.
    """
    cx_m = grid_shape[0] * dx / 2
    cy_m = grid_shape[1] * dx / 2
    cz_m = grid_shape[2] * dx / 2

    # Match helmet to head size (slightly larger than scalp)
    # Head semi-axes + standoff
    r_ap = min(0.095, (grid_shape[0] * dx / 2) - 5 * dx)
    r_lr = min(0.075, (grid_shape[1] * dx / 2) - 5 * dx)
    r_si = min(0.090, (grid_shape[2] * dx / 2) - 5 * dx)

    positions = helmet_array_3d(
        n_elements=n_elements,
        center=(cx_m, cy_m, cz_m),
        radius_ap=r_ap,
        radius_lr=r_lr,
        radius_si=r_si,
        standoff=0.005,  # 5mm water coupling
        coverage_angle=2.8,  # ~160 deg
        exclude_face=True,
    )

    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    n_actual = len(pos_grid[0])

    src_list = [
        (int(pos_grid[0][i]), int(pos_grid[1][i]), int(pos_grid[2][i]))
        for i in range(n_actual)
    ]

    return {
        "positions_m": positions,
        "positions_grid": pos_grid,
        "src_list": src_list,
        "sensor_grid": pos_grid,
        "n_elements": n_actual,
    }


# ============================================================================
# Main simulation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D Brain Recovery Simulation with US Probe Helmet"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 48^3, 32 elements (~5 min CPU)")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: 192^3, 256 elements (needs GPU)")
    parser.add_argument("--n-elements", type=int, default=None,
                        help="Override number of helmet elements")
    parser.add_argument("--iters", type=int, default=None,
                        help="Override iterations per frequency band")
    parser.add_argument("--shots", type=int, default=None,
                        help="Override shots per iteration")
    parser.add_argument("--no-lesion", action="store_true",
                        help="Omit simulated haemorrhage lesion")
    parser.add_argument("--output", type=str, default="/data/datasets/brain-fwi/brain_recovery_3d.h5",
                        help="Output HDF5 path")
    args = parser.parse_args()

    # Configuration presets
    if args.fast:
        N, n_elem, iters, shots, dx = 48, 32, 8, 4, 0.003
        freq_bands = [(30e3, 60e3)]
    elif args.full:
        N, n_elem, iters, shots, dx = 192, 256, 20, 8, 0.001
        freq_bands = [(50e3, 100e3), (100e3, 200e3), (200e3, 300e3)]
    else:
        N, n_elem, iters, shots, dx = 96, 128, 15, 4, 0.002
        freq_bands = [(40e3, 80e3), (80e3, 150e3)]

    # Override from CLI
    if args.n_elements is not None:
        n_elem = args.n_elements
    if args.iters is not None:
        iters = args.iters
    if args.shots is not None:
        shots = args.shots

    grid_shape = (N, N, N)
    domain_cm = N * dx * 100

    print("=" * 70)
    print("3D Brain Recovery — Transcranial Ultrasound FWI")
    print("=" * 70)
    print(f"  Device: {jax.devices()[0]}")
    print(f"  Grid: {N}^3 = {N**3:,} voxels, dx={dx*1e3:.1f} mm")
    print(f"  Domain: {domain_cm:.1f}^3 cm")
    print(f"  Helmet: {n_elem} elements")
    print(f"  Freq bands: {len(freq_bands)}, {iters} iters/band, {shots} shots/iter")

    # ================================================================
    # Step 1: Head Phantom
    # ================================================================
    print(f"\n[1/4] Creating 3D head phantom...")
    t0 = time.time()

    phantom = create_head_phantom_3d(
        grid_shape, dx, include_lesion=not args.no_lesion,
    )
    c_true = phantom["sound_speed"]
    rho = phantom["density"]
    labels = phantom["labels"]

    n_skull = int(jnp.sum(labels == 7))
    n_brain = int(jnp.sum((labels == 2) | (labels == 3)))
    n_csf = int(jnp.sum(labels == 1))
    n_lesion = int(jnp.sum(labels == 8))

    print(f"  Skull voxels: {n_skull:,}")
    print(f"  Brain voxels: {n_brain:,} (GM+WM)")
    print(f"  CSF voxels:   {n_csf:,}")
    if n_lesion > 0:
        print(f"  Lesion voxels: {n_lesion:,} (simulated haemorrhage)")
    print(f"  Speed range: [{float(jnp.min(c_true)):.0f}, {float(jnp.max(c_true)):.0f}] m/s")
    print(f"  Created in {time.time()-t0:.1f} s")

    # ================================================================
    # Step 2: Probe Helmet
    # ================================================================
    print(f"\n[2/4] Designing probe helmet...")
    t0 = time.time()

    helmet = create_probe_helmet(n_elem, grid_shape, dx)

    print(f"  {helmet['n_elements']} transducer elements")
    print(f"  Coverage: ~160 deg (face excluded)")
    print(f"  Standoff: 5mm water coupling")
    print(f"  Created in {time.time()-t0:.1f} s")

    # ================================================================
    # Step 3: Forward Data Generation
    # ================================================================
    print(f"\n[3/4] Generating observed data...")
    t0 = time.time()

    c_max_fwi = 3200.0  # FWI velocity upper bound
    max_freq = max(fmax for _, fmax in freq_bands)
    wavelength = 1500.0 / max_freq
    ppw = dx / wavelength
    print(f"  Max frequency: {max_freq/1e3:.0f} kHz")
    print(f"  Min wavelength: {wavelength*1e3:.1f} mm ({1/ppw:.0f} points/wavelength)")

    # CRITICAL: Compute ONE reference time axis from c_max — used for BOTH
    # data generation and FWI. This prevents dt mismatch that causes divergence.
    domain = build_domain(grid_shape, dx)
    ref_medium = build_medium(domain, c_max_fwi, 1000.0, pml_size=10)
    time_axis = build_time_axis(ref_medium, cfl=0.3)
    dt = float(time_axis.dt)
    t_end = float(time_axis.t_end)
    n_samples = int(t_end / dt)

    print(f"  dt={dt*1e6:.3f} us, t_end={t_end*1e3:.2f} ms, {n_samples} samples")

    source_signal = _build_source_signal(max_freq, dt, n_samples)

    # Build true medium for data generation (using same time_axis)
    medium_true = build_medium(domain, c_true, rho, pml_size=10)

    # Use a subset of sources for data generation (speed)
    n_data_src = min(helmet["n_elements"], shots * len(freq_bands) * 2)
    print(f"  Generating data for {n_data_src}/{helmet['n_elements']} sources...")

    all_data = []
    for i in range(n_data_src):
        if (i + 1) % max(1, n_data_src // 10) == 0 or i == 0:
            print(f"    Shot {i+1}/{n_data_src}...", end="\r")
        src_pos = helmet["src_list"][i]
        d = simulate_shot_sensors(
            medium_true, time_axis, src_pos,
            helmet["sensor_grid"], source_signal, dt,
        )
        all_data.append(d)

    observed = jnp.stack(all_data, axis=0)
    print(f"  Observed data: {observed.shape}                ")
    print(f"  Generated in {time.time()-t0:.1f} s")

    # ================================================================
    # Step 4: FWI Recovery
    # ================================================================
    print(f"\n[4/4] Running Full Waveform Inversion...")
    t0 = time.time()

    c_init = jnp.full(grid_shape, 1500.0, dtype=jnp.float32)

    config = FWIConfig(
        freq_bands=freq_bands,
        n_iters_per_band=iters,
        shots_per_iter=shots,
        learning_rate=0.5,
        c_min=1400.0,
        c_max=3200.0,
        pml_size=10,
        gradient_smooth_sigma=3.0,
        loss_fn="l2",
        skip_bandpass=True,
        verbose=True,
    )

    result = run_fwi(
        observed_data=observed,
        initial_velocity=c_init,
        density=rho,
        dx=dx,
        src_positions_grid=helmet["src_list"][:n_data_src],
        sensor_positions_grid=helmet["sensor_grid"],
        source_signal=source_signal,
        dt=dt,
        t_end=t_end,
        config=config,
    )

    fwi_time = time.time() - t0
    print(f"\n  FWI completed in {fwi_time:.1f} s")

    # ================================================================
    # Results & Metrics
    # ================================================================
    c_recon = result.velocity

    # Compute metrics inside the head (exclude water coupling)
    brain_mask = (labels == 2) | (labels == 3)  # GM + WM
    skull_mask = labels == 7
    head_mask = labels > 0  # everything non-water

    def rmse(a, b, mask):
        return float(jnp.sqrt(jnp.mean((a - b) ** 2 * mask) / jnp.maximum(jnp.mean(mask), 1e-10)))

    rmse_brain = rmse(c_recon, c_true, brain_mask)
    rmse_skull = rmse(c_recon, c_true, skull_mask)
    rmse_head = rmse(c_recon, c_true, head_mask)
    rmse_init = rmse(c_init, c_true, head_mask)

    print(f"\n  Results:")
    print(f"    RMSE (head, initial): {rmse_init:.1f} m/s")
    print(f"    RMSE (head, recon):   {rmse_head:.1f} m/s")
    print(f"    RMSE (brain only):    {rmse_brain:.1f} m/s")
    print(f"    RMSE (skull only):    {rmse_skull:.1f} m/s")
    print(f"    Improvement: {(1 - rmse_head/rmse_init)*100:.1f}%")
    print(f"    Final loss: {result.loss_history[-1]:.6f}")

    if n_lesion > 0:
        lesion_mask = labels == 8
        c_lesion_true = float(jnp.mean(c_true * lesion_mask) / jnp.maximum(jnp.mean(lesion_mask), 1e-10))
        c_lesion_recon = float(jnp.mean(c_recon * lesion_mask) / jnp.maximum(jnp.mean(lesion_mask), 1e-10))
        print(f"    Lesion speed (true):  {c_lesion_true:.0f} m/s")
        print(f"    Lesion speed (recon): {c_lesion_recon:.0f} m/s")

    # ================================================================
    # Save results
    # ================================================================
    import h5py
    output = Path(args.output)
    print(f"\n  Saving to {output}...")

    with h5py.File(str(output), "w") as f:
        # Volumes
        f.create_dataset("velocity_true", data=np.array(c_true), compression="gzip")
        f.create_dataset("velocity_recon", data=np.array(c_recon), compression="gzip")
        f.create_dataset("velocity_init", data=np.array(c_init), compression="gzip")
        f.create_dataset("density", data=np.array(rho), compression="gzip")
        f.create_dataset("labels", data=np.array(labels), compression="gzip")

        # Helmet geometry
        f.create_dataset("helmet_positions", data=np.array(helmet["positions_m"]))

        # FWI diagnostics
        f.create_dataset("loss_history", data=np.array(result.loss_history))
        for i, v in enumerate(result.velocity_history):
            f.create_dataset(f"velocity_band_{i}", data=np.array(v), compression="gzip")

        # Metadata
        f.attrs["grid_shape"] = grid_shape
        f.attrs["dx"] = dx
        f.attrs["n_elements"] = helmet["n_elements"]
        f.attrs["freq_bands"] = str(freq_bands)
        f.attrs["n_iters_per_band"] = iters
        f.attrs["shots_per_iter"] = shots
        f.attrs["fwi_time_seconds"] = fwi_time
        f.attrs["rmse_head"] = rmse_head
        f.attrs["rmse_brain"] = rmse_brain
        f.attrs["rmse_skull"] = rmse_skull
        f.attrs["improvement_pct"] = (1 - rmse_head / rmse_init) * 100

    print(f"  Saved ({output.stat().st_size / 1e6:.1f} MB)")
    print(f"\n{'='*70}")
    print(f"  Total time: {time.time() - t0 + fwi_time:.1f} s")
    print(f"  Visualize with: h5dump -H {output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
