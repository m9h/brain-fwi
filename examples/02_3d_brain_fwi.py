#!/usr/bin/env python
"""Example 2: 3D Brain FWI.

Full 3D Full Waveform Inversion using:
  - BrainWeb 3D tissue phantom (181x217x181, 1mm iso)
  - 256-element helmet array (Kernel Flow-inspired coverage)
  - Multi-frequency banding: 50→100→200 kHz
  - j-Wave PSTD solver with JAX autodiff

This example requires a GPU with >=16GB VRAM for the 3D simulations.
On DGX Spark (CUDA 13), a full run takes ~2-4 hours depending on
the number of iterations and frequency bands.

For a quick test, use --grid-size 64 --iters 5 --n-elements 32.

Run:
    uv run python examples/02_3d_brain_fwi.py --synthetic --grid-size 64
    uv run python examples/02_3d_brain_fwi.py  # full BrainWeb (needs download)
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from brain_fwi.phantoms import make_synthetic_head, map_labels_to_speed, map_labels_to_density
from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    generate_observed_data, _build_source_signal,
)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi


def make_synthetic_head_3d(grid_shape, dx):
    """Generate a 3D synthetic head phantom (concentric ellipsoids).

    Extends the 2D make_synthetic_head to 3D by creating ellipsoidal
    layers for scalp, skull, CSF, grey matter, and white matter.
    """
    from brain_fwi.phantoms.properties import map_labels_to_all

    nx, ny, nz = grid_shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    # Head semi-axes (metres → grid units)
    head_a = 0.095 / dx  # AP
    head_b = 0.075 / dx  # LR
    head_c = 0.090 / dx  # SI

    x, y, z = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), jnp.arange(nz),
                            indexing="ij")
    r = jnp.sqrt(((x - cx) / head_a) ** 2 +
                  ((y - cy) / head_b) ** 2 +
                  ((z - cz) / head_c) ** 2)

    # Layer thicknesses
    skull_t = 0.007 / (head_a * dx)
    scalp_t = 0.003 / (head_a * dx)
    csf_t = 0.002 / (head_a * dx)

    scalp_outer = 1.0
    skull_outer = 1.0 - scalp_t
    skull_inner = skull_outer - skull_t
    csf_inner = skull_inner - csf_t

    labels = jnp.zeros(grid_shape, dtype=jnp.int32)
    labels = jnp.where(r <= scalp_outer, 6, labels)   # scalp
    labels = jnp.where(r <= skull_outer, 7, labels)    # skull
    labels = jnp.where(r <= skull_inner, 1, labels)    # CSF
    labels = jnp.where(r <= csf_inner, 2, labels)      # grey matter

    # White matter core
    wm_r = r / csf_inner
    labels = jnp.where((wm_r <= 0.6) & (r <= csf_inner), 3, labels)

    properties = map_labels_to_all(labels)
    return labels, properties


def main():
    parser = argparse.ArgumentParser(description="3D Brain FWI")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic 3D phantom")
    parser.add_argument("--grid-size", type=int, default=96,
                        help="Grid dimension (NxNxN)")
    parser.add_argument("--dx", type=float, default=1.5e-3,
                        help="Grid spacing in metres (default 1.5mm)")
    parser.add_argument("--n-elements", type=int, default=128,
                        help="Number of helmet transducer elements")
    parser.add_argument("--iters", type=int, default=15,
                        help="Iterations per frequency band")
    parser.add_argument("--shots", type=int, default=4,
                        help="Sources per iteration")
    args = parser.parse_args()

    print("=" * 70)
    print("Brain FWI — 3D Full Volume")
    print("=" * 70)
    print(f"  Device: {jax.devices()[0]}")

    # ================================================================
    # 1. Head phantom
    # ================================================================
    print("\n[1] Creating 3D head phantom...")
    t0 = time.time()

    N = args.grid_size
    grid_shape = (N, N, N)
    dx = args.dx

    if args.synthetic or True:  # Always synthetic for 3D (BrainWeb needs resampling)
        labels, props = make_synthetic_head_3d(grid_shape, dx)
        c_true = props["sound_speed"]
        rho = props["density"]
    # Replace air with water
    c_true = jnp.where(c_true < 500.0, 1500.0, c_true)
    rho = jnp.where(rho < 10.0, 1000.0, rho)

    domain_cm = N * dx * 100
    print(f"  Grid: {grid_shape}, dx={dx*1e3:.1f} mm")
    print(f"  Domain: {domain_cm:.1f}^3 cm")
    print(f"  Speed range: [{float(jnp.min(c_true)):.0f}, {float(jnp.max(c_true)):.0f}] m/s")
    print(f"  Created in {time.time()-t0:.1f} s")

    # ================================================================
    # 2. Helmet array
    # ================================================================
    print(f"\n[2] Creating {args.n_elements}-element helmet array...")

    cx_m = N * dx / 2
    positions = helmet_array_3d(
        n_elements=args.n_elements,
        center=(cx_m, cx_m, cx_m),
        radius_ap=0.04 * (N * dx / 0.2),  # scale to grid
        radius_lr=0.035 * (N * dx / 0.2),
        radius_si=0.038 * (N * dx / 0.2),
        standoff=0.003,
    )

    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    n_elem = len(pos_grid[0])

    src_positions_list = [
        (int(pos_grid[0][i]), int(pos_grid[1][i]), int(pos_grid[2][i]))
        for i in range(n_elem)
    ]
    sensor_pos = pos_grid

    print(f"  {n_elem} transducer elements on ellipsoidal helmet")

    # ================================================================
    # 3. Generate observed data
    # ================================================================
    print("\n[3] Generating observed data...")
    t0 = time.time()

    freq_data = 200e3
    domain = build_domain(grid_shape, dx)
    medium_true = build_medium(domain, c_true, rho, pml_size=10)
    time_axis = build_time_axis(medium_true, cfl=0.3)
    dt = float(time_axis.dt)
    t_end = float(time_axis.t_end)
    n_samples = int(t_end / dt)

    source_signal = _build_source_signal(freq_data, dt, n_samples)

    # Only use a subset of sources for data generation (3D is expensive)
    n_data_sources = min(args.n_elements, 32)
    observed = generate_observed_data(
        c_true, rho, dx,
        src_positions_list[:n_data_sources], sensor_pos,
        freq_data, pml_size=10, cfl=0.3, t_end=t_end,
    )

    print(f"  Observed data: {observed.shape}")
    print(f"  Generated in {time.time()-t0:.1f} s")

    # ================================================================
    # 4. FWI
    # ================================================================
    print("\n[4] Running 3D FWI...")
    t0 = time.time()

    c_init = jnp.full(grid_shape, 1500.0, dtype=jnp.float32)

    config = FWIConfig(
        freq_bands=[
            (30e3, 80e3),
            (80e3, 150e3),
        ],
        n_iters_per_band=args.iters,
        shots_per_iter=args.shots,
        learning_rate=5.0,
        c_min=1400.0,
        c_max=3200.0,
        pml_size=10,
        gradient_smooth_sigma=2.0,
        loss_fn="multiscale",
        verbose=True,
    )

    result = run_fwi(
        observed_data=observed,
        initial_velocity=c_init,
        density=rho,
        dx=dx,
        src_positions_grid=src_positions_list[:n_data_sources],
        sensor_positions_grid=sensor_pos,
        source_signal=source_signal,
        dt=dt,
        t_end=t_end,
        config=config,
    )

    print(f"\n  3D FWI completed in {time.time()-t0:.1f} s")

    # Save results
    import h5py
    output_path = "fwi_3d_result.h5"
    with h5py.File(output_path, "w") as f:
        f.create_dataset("velocity_true", data=np.array(c_true))
        f.create_dataset("velocity_recon", data=np.array(result.velocity))
        f.create_dataset("density", data=np.array(rho))
        f.create_dataset("loss_history", data=np.array(result.loss_history))
        f.attrs["dx"] = dx
        f.attrs["grid_shape"] = grid_shape
        f.attrs["n_elements"] = args.n_elements
    print(f"  Saved to {output_path}")

    # Quick RMSE
    rmse = float(jnp.sqrt(jnp.mean((result.velocity - c_true) ** 2)))
    print(f"  RMSE: {rmse:.1f} m/s")


if __name__ == "__main__":
    main()
