#!/usr/bin/env python
"""Example 1: 2D Axial Brain FWI.

Full Waveform Inversion of a 2D axial cross-section of the human head.
Uses the BrainWeb phantom (or synthetic fallback) with a ring array of
128 ultrasound transducers, following:

  - Guasch et al. (2020). Full-waveform inversion imaging of the human
    brain. npj Digital Medicine.
  - Stride breast 2D FWI example (trustimaging/stride)
  - j-Wave FWI notebook (ucl-bug/jwave)

Pipeline:
  1. Load head phantom (BrainWeb or synthetic)
  2. Map tissue labels → acoustic properties (ITRUSST values)
  3. Create 128-element ring array (Kernel Flow-inspired coverage)
  4. Generate synthetic observed data (forward sim for each source)
  5. Run multi-frequency FWI (100→200→300 kHz)
  6. Visualize results

Run:
    uv run python examples/01_2d_axial_fwi.py [--synthetic] [--n-elements 64]
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", False)  # float32 for speed

from brain_fwi.phantoms import make_synthetic_head, map_labels_to_speed, map_labels_to_density
from brain_fwi.transducers import ring_array_2d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    generate_observed_data, _build_source_signal,
)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi


def main():
    parser = argparse.ArgumentParser(description="2D Axial Brain FWI")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic phantom (no download)")
    parser.add_argument("--n-elements", type=int, default=128,
                        help="Number of transducer elements")
    parser.add_argument("--grid-size", type=int, default=256,
                        help="Grid dimension (NxN)")
    parser.add_argument("--dx", type=float, default=0.5e-3,
                        help="Grid spacing in metres (default 0.5mm)")
    parser.add_argument("--shots-per-iter", type=int, default=4,
                        help="Sources per FWI iteration")
    parser.add_argument("--iters-per-band", type=int, default=30,
                        help="Iterations per frequency band")
    parser.add_argument("--output", type=str, default="fwi_result.png",
                        help="Output figure path")
    args = parser.parse_args()

    print("=" * 70)
    print("Brain FWI — 2D Axial Cross-Section")
    print("=" * 70)

    # ================================================================
    # 1. Head phantom
    # ================================================================
    print("\n[1] Loading head phantom...")
    t0 = time.time()

    grid_shape = (args.grid_size, args.grid_size)
    dx = args.dx

    if args.synthetic:
        labels, props = make_synthetic_head(grid_shape=grid_shape, dx=dx)
        c_true = props["sound_speed"]
        rho = props["density"]
        print(f"  Synthetic phantom: {grid_shape}, dx={dx*1e3:.1f} mm")
    else:
        try:
            from brain_fwi.phantoms import load_brainweb_slice
            bw_labels, bw_dx = load_brainweb_slice(axis="axial", slice_idx=90)

            # Resample to target grid if needed
            if bw_labels.shape != grid_shape:
                from scipy.ndimage import zoom
                scale = np.array(grid_shape) / np.array(bw_labels.shape)
                bw_labels = zoom(bw_labels, scale, order=0).astype(np.int32)

            labels = jnp.array(bw_labels)
            c_true = map_labels_to_speed(labels)
            rho = map_labels_to_density(labels)
            print(f"  BrainWeb phantom: {grid_shape}, dx={dx*1e3:.1f} mm")
        except Exception as e:
            print(f"  BrainWeb unavailable ({e}), using synthetic phantom")
            labels, props = make_synthetic_head(grid_shape=grid_shape, dx=dx)
            c_true = props["sound_speed"]
            rho = props["density"]

    # Replace air (outside head) with water for coupling medium
    c_true = jnp.where(c_true < 500.0, 1500.0, c_true)
    rho = jnp.where(rho < 10.0, 1000.0, rho)

    domain_size_cm = args.grid_size * dx * 100
    print(f"  Domain: {domain_size_cm:.1f} x {domain_size_cm:.1f} cm")
    print(f"  Speed range: [{float(jnp.min(c_true)):.0f}, {float(jnp.max(c_true)):.0f}] m/s")
    print(f"  Loaded in {time.time()-t0:.1f} s")

    # ================================================================
    # 2. Transducer ring array
    # ================================================================
    print(f"\n[2] Creating {args.n_elements}-element ring array...")

    # Centre of domain in physical coordinates
    cx = grid_shape[0] * dx / 2
    cy = grid_shape[1] * dx / 2

    # Ring array around head
    positions = ring_array_2d(
        n_elements=args.n_elements,
        center=(cx, cy),
        semi_major=0.10,   # ~10 cm radius (head + standoff)
        semi_minor=0.08,   # ~8 cm
        standoff=0.005,    # 5mm water coupling
    )

    # Convert to grid indices
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    n_sensors = len(pos_grid[0])

    # Build list of source positions (each element can be a source)
    src_positions_list = [
        (int(pos_grid[0][i]), int(pos_grid[1][i]))
        for i in range(args.n_elements)
    ]
    # Use all elements as sensors
    sensor_pos = pos_grid

    print(f"  Ring radius: ~{0.10*100:.0f} x {0.08*100:.0f} cm (elliptical)")
    print(f"  {n_sensors} sensors, {args.n_elements} sources")

    # ================================================================
    # 3. Generate synthetic observed data
    # ================================================================
    print("\n[3] Generating observed data (ground truth)...")
    t0 = time.time()

    # Use highest frequency for data generation
    freq_data = 300e3  # 300 kHz
    domain = build_domain(grid_shape, dx)
    medium_true = build_medium(domain, c_true, rho, pml_size=20)
    time_axis = build_time_axis(medium_true, cfl=0.3)
    dt = float(time_axis.dt)
    t_end = float(time_axis.t_end)
    n_samples = int(t_end / dt)

    print(f"  Frequency: {freq_data/1e3:.0f} kHz")
    print(f"  dt={dt*1e6:.3f} us, t_end={t_end*1e3:.2f} ms, {n_samples} samples")

    source_signal = _build_source_signal(freq_data, dt, n_samples)

    observed = generate_observed_data(
        c_true, rho, dx,
        src_positions_list, sensor_pos,
        freq_data,
        pml_size=20,
        cfl=0.3,
        t_end=t_end,
        verbose=True,
    )

    print(f"  Observed data shape: {observed.shape}")
    print(f"  Generated in {time.time()-t0:.1f} s")

    # ================================================================
    # 4. FWI
    # ================================================================
    print("\n[4] Running Full Waveform Inversion...")
    t0 = time.time()

    # Initial model: homogeneous water
    c_init = jnp.full(grid_shape, 1500.0, dtype=jnp.float32)

    # Inversion mask: only update inside the ring array
    y, x = jnp.meshgrid(jnp.arange(grid_shape[1]), jnp.arange(grid_shape[0]))
    mask_r = jnp.sqrt(((x - grid_shape[0]//2) / (0.095/dx))**2 +
                       ((y - grid_shape[1]//2) / (0.075/dx))**2)
    fwi_mask = jnp.where(mask_r <= 1.0, 1.0, 0.0)

    config = FWIConfig(
        freq_bands=[
            (50e3, 100e3),    # Low freq: broad features
            (100e3, 200e3),   # Mid freq: skull + brain
            (200e3, 300e3),   # High freq: fine detail
        ],
        n_iters_per_band=args.iters_per_band,
        shots_per_iter=args.shots_per_iter,
        learning_rate=5.0,
        c_min=1400.0,
        c_max=3200.0,
        pml_size=20,
        gradient_smooth_sigma=3.0,
        loss_fn="multiscale",
        envelope_weight=0.5,
        mask=fwi_mask,
        verbose=True,
    )

    result = run_fwi(
        observed_data=observed,
        initial_velocity=c_init,
        density=rho,
        dx=dx,
        src_positions_grid=src_positions_list,
        sensor_positions_grid=sensor_pos,
        source_signal=source_signal,
        dt=dt,
        t_end=t_end,
        config=config,
    )

    print(f"\n  FWI completed in {time.time()-t0:.1f} s")
    c_recon = result.velocity
    print(f"  Reconstructed speed range: [{float(jnp.min(c_recon)):.0f}, "
          f"{float(jnp.max(c_recon)):.0f}] m/s")

    # ================================================================
    # 5. Visualization
    # ================================================================
    print(f"\n[5] Saving results to {args.output}...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # True model
    im0 = axes[0, 0].imshow(np.array(c_true).T, cmap="seismic",
                             vmin=1400, vmax=3200, origin="lower",
                             extent=[0, domain_size_cm, 0, domain_size_cm])
    axes[0, 0].set_title("True Sound Speed")
    axes[0, 0].set_xlabel("cm")
    plt.colorbar(im0, ax=axes[0, 0], label="m/s")

    # Transducer positions
    pos_cm = np.array(positions) * 100
    axes[0, 0].plot(pos_cm[:, 0], pos_cm[:, 1], "kv", ms=3, label="Transducers")

    # Initial model
    im1 = axes[0, 1].imshow(np.array(c_init).T, cmap="seismic",
                             vmin=1400, vmax=3200, origin="lower",
                             extent=[0, domain_size_cm, 0, domain_size_cm])
    axes[0, 1].set_title("Initial Model (homogeneous water)")
    plt.colorbar(im1, ax=axes[0, 1], label="m/s")

    # Reconstructed model
    im2 = axes[0, 2].imshow(np.array(c_recon).T, cmap="seismic",
                             vmin=1400, vmax=3200, origin="lower",
                             extent=[0, domain_size_cm, 0, domain_size_cm])
    axes[0, 2].set_title("FWI Reconstruction")
    plt.colorbar(im2, ax=axes[0, 2], label="m/s")

    # Difference (error)
    diff = np.array(c_recon - c_true)
    im3 = axes[1, 0].imshow(diff.T, cmap="RdBu_r",
                             vmin=-200, vmax=200, origin="lower",
                             extent=[0, domain_size_cm, 0, domain_size_cm])
    axes[1, 0].set_title("Error (Recon - True)")
    plt.colorbar(im3, ax=axes[1, 0], label="m/s")

    # Band progression
    for i, v in enumerate(result.velocity_history):
        axes[1, 1].imshow(np.array(v).T, cmap="seismic",
                          vmin=1400, vmax=3200, origin="lower", alpha=0.5)
    axes[1, 1].set_title(f"Band Progression ({len(result.velocity_history)} bands)")

    # Loss curve
    axes[1, 2].semilogy(result.loss_history)
    n_per = config.n_iters_per_band
    for i in range(len(config.freq_bands)):
        axes[1, 2].axvline(x=i * n_per, color="gray", ls="--", alpha=0.5)
        fmin, fmax = config.freq_bands[i]
        axes[1, 2].text(i * n_per + 2, max(result.loss_history) * 0.8,
                        f"{fmin/1e3:.0f}-{fmax/1e3:.0f}kHz", fontsize=8)
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("Loss")
    axes[1, 2].set_title("Convergence")
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle("Brain FWI — 2D Axial Cross-Section\n"
                 f"{args.n_elements} transducers, {grid_shape[0]}x{grid_shape[1]} grid, "
                 f"dx={dx*1e3:.1f} mm",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"  Saved to {args.output}")

    # Print summary statistics
    rmse = float(jnp.sqrt(jnp.mean((c_recon - c_true) ** 2 * fwi_mask)))
    mae = float(jnp.mean(jnp.abs(c_recon - c_true) * fwi_mask))
    print(f"\n  RMSE (masked): {rmse:.1f} m/s")
    print(f"  MAE  (masked): {mae:.1f} m/s")


if __name__ == "__main__":
    main()
