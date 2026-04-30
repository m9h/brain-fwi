#!/usr/bin/env python
"""Example 4: TUS Forward Simulation (SimNIBS-compatible).

Performs a forward acoustic simulation of Transcranial Ultrasound 
Stimulation (TUS) using a focused bowl transducer and a SimNIBS-derived 
head model. Redoes simulations from the SimNIBS 4 / PRESTUS paper 
using j-Wave.

Key features:
  - Focused bowl transducer geometry
  - SimNIBS (CHARM) tissue property mapping
  - j-Wave PSTD solver on GPU
  - Focal spot visualization (peak pressure)
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", False)

from brain_fwi.phantoms.simnibs import load_simnibs_acoustic
from brain_fwi.phantoms.mida import resample_volume
from brain_fwi.transducers.focused import bowl_transducer_3d
from brain_fwi.transducers.helmet import transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    simulate_shot_sensors, _build_source_signal,
)

def check_convergence(c: jnp.ndarray, dx: float, freq: float, min_ppw: float = 6.0):
    """Verify that the grid resolution supports the requested frequency.

    Args:
        c: Sound speed volume (m/s).
        dx: Grid spacing (m).
        freq: Operating frequency (Hz).
        min_ppw: Minimum points per wavelength.
    """
    min_c = float(jnp.min(c))
    wavelength = min_c / freq
    ppw = wavelength / dx
    
    print(f"\n[Convergence Check]")
    print(f"  Min sound speed: {min_c:.1f} m/s")
    print(f"  Wavelength:      {wavelength*1e3:.2f} mm")
    print(f"  PPW:             {ppw:.2f} (target: >{min_ppw})")
    
    if ppw < min_ppw:
        print(f"  WARNING: Grid may be too coarse (PPW < {min_ppw}). "
              f"Consider reducing dx or frequency.")
    else:
        print(f"  Resolution OK.")

def main():
    parser = argparse.ArgumentParser(description="TUS Forward Simulation")
    parser.add_argument("--m2m", type=str, help="Path to SimNIBS m2m directory")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Focal target coordinates relative to head center (m)")
    parser.add_argument("--frequency", type=float, default=500000.0,
                        help="Transducer frequency (Hz)")
    parser.add_argument("--grid-size", type=int, default=128,
                        help="Grid size (NxNxN)")
    parser.add_argument("--dx", type=float, default=0.001,
                        help="Grid spacing (m)")
    args = parser.parse_args()

    print("=" * 70)
    print("Transcranial Ultrasound Simulation (j-Wave)")
    print("=" * 70)

    # 1. Load Head Model
    grid_shape = (args.grid_size, args.grid_size, args.grid_size)
    dx = args.dx

    if args.m2m and Path(args.m2m).exists():
        print(f"\n[1] Loading SimNIBS phantom from {args.m2m}...")
        props = load_simnibs_acoustic(args.m2m)
        c = resample_volume(np.asarray(props["sound_speed"]), grid_shape, order=1)
        rho = resample_volume(np.asarray(props["density"]), grid_shape, order=1)
        alpha = resample_volume(np.asarray(props["attenuation"]), grid_shape, order=1)
        labels = resample_volume(np.asarray(props["labels"]), grid_shape, order=0)
    else:
        print("\n[1] No m2m provided, using synthetic three-layer head...")
        from brain_fwi.phantoms.synthetic import make_three_layer_head
        from brain_fwi.phantoms.properties import map_labels_to_all
        labels_np = make_three_layer_head(grid_shape, dx)
        labels = jnp.asarray(labels_np)
        props = map_labels_to_all(labels)
        c = props["sound_speed"]
        rho = props["density"]
        alpha = props["attenuation"]
        # Water coupling
        c = jnp.where(labels == 0, 1500.0, c)
        rho = jnp.where(labels == 0, 1000.0, rho)

    check_convergence(c, dx, args.frequency)

    # 2. Define Transducer (Focused Bowl)
    print("\n[2] Setting up focused bowl transducer...")
    cx_m, cy_m, cz_m = [s * dx / 2 for s in grid_shape]
    focal_point = (cx_m + args.target[0], cy_m + args.target[1], cz_m + args.target[2])
    
    # Example transducer: H-115 style (64mm diam, 52mm focal length)
    positions = bowl_transducer_3d(
        focal_length=0.052,
        aperture_diameter=0.064,
        focal_point=focal_point,
        direction=(0.0, 0.0, -1.0), # Pointing down from top
        n_points=500
    )
    
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    
    print(f"  Focal Target: {focal_point}")
    print(f"  Aperture: 64 mm, Focal Length: 52 mm")
    print(f"  Elements: {len(pos_grid[0])}")

    # 3. Simulation
    print("\n[3] Running forward simulation...")
    domain = build_domain(grid_shape, dx)
    medium = build_medium(domain, c, rho, attenuation=alpha)
    time_axis = build_time_axis(medium, cfl=0.3)
    
    print(f"  Grid: {grid_shape}, dx={dx*1e3:.1f} mm")
    print(f"  Time: {time_axis.t_end*1e3:.2f} ms, dt={time_axis.dt*1e6:.3f} us")

    # In j-Wave, we can simulate multiple sources by sum or by shot.
    # For a bowl, we can treat it as a single source with many points.
    from jwave.geometry import Sources
    from jwave.acoustics.time_varying import (
        fourier_wave_prop_params,
        momentum_conservation_rhs,
        mass_conservation_rhs,
        pressure_from_density,
        TimeWavePropagationSettings,
    )

    dt_val = float(time_axis.dt)
    n_samples = int(time_axis.Nt)
    signal = _build_source_signal(args.frequency, dt_val, n_samples)
    
    # Build multi-point source
    src_pos_tuple = tuple(pos_grid)
    # All points share the same signal
    signals = jnp.tile(signal[jnp.newaxis, :], (len(pos_grid[0]), 1))
    
    sources = Sources(
        positions=src_pos_tuple,
        signals=signals,
        dt=dt_val,
        domain=domain
    )

    # For TUS, we want the maximum pressure at each voxel
    print("  Solving...")
    
    settings = TimeWavePropagationSettings(checkpoint=True)
    params = fourier_wave_prop_params(medium, time_axis, settings=settings)
    c_ref = params["c_ref"]
    pml_rho = params["pml_rho"]
    pml_u = params["pml_u"]

    # Initialize fields
    ndim = 3
    shape = tuple(list(domain.N) + [ndim])
    shape_one = tuple(list(domain.N) + [1])

    p0 = pml_rho.replace_params(jnp.zeros(shape_one))
    u0 = pml_u.replace_params(jnp.zeros(shape))
    rho0 = p0.replace_params(jnp.stack([p0.params[..., i] for i in range(ndim)], axis=-1)) / ndim
    rho0 = rho0 / (medium.sound_speed ** 2)

    # We add peak_pressure to the carry
    init_carry = (p0, u0, rho0, jnp.zeros(domain.N))
    
    @jax.jit
    def step(carry, n):
        p, u, rho_f, peak = carry
        mass_src_field = sources.on_grid(n)

        du = momentum_conservation_rhs(
            p, u, medium, c_ref=c_ref, dt=dt_val, params=params["fourier"],
        )
        u = pml_u * (pml_u * u + dt_val * du)

        drho = mass_conservation_rhs(
            p, u, mass_src_field, medium,
            c_ref=c_ref, dt=dt_val, params=params["fourier"],
        )
        rho_f = pml_rho * (pml_rho * rho_f + dt_val * drho)

        p = pressure_from_density(rho_f, medium)
        
        # Update peak pressure (absolute)
        p_abs = jnp.abs(p.on_grid[..., 0])
        peak = jnp.maximum(peak, p_abs)
        
        return (p, u, rho_f, peak), None

    # Run loop
    output_steps = jnp.arange(0, time_axis.Nt, 1)
    (p_final, u_final, rho_final, peak_pressure), _ = jax.lax.scan(
        step, init_carry, output_steps
    )
    
    print("  Simulation complete.")
    
    # 4. Visualization
    print("\n[4] Generating results...")
    peak_pressure_np = np.asarray(peak_pressure)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(np.asarray(c[:, :, grid_shape[2]//2]).T, origin="lower", cmap="viridis")
    plt.scatter(pos_grid[0], pos_grid[1], c="r", s=1, label="Transducer")
    plt.title("Sound Speed (Axial)")
    plt.colorbar(label="m/s")
    
    plt.subplot(132)
    plt.imshow(peak_pressure_np[:, :, grid_shape[2]//2].T, origin="lower", cmap="hot")
    plt.title("Peak Pressure (Axial)")
    plt.colorbar(label="Pa")
    
    plt.subplot(133)
    # Coronal slice through target
    target_idx = [int(f/dx) for f in focal_point]
    plt.imshow(peak_pressure_np[target_idx[0], :, :].T, origin="lower", cmap="hot")
    plt.title("Peak Pressure (Coronal)")
    plt.colorbar(label="Pa")
    
    plt.tight_layout()
    plt.savefig("tus_simulation.png")
    print("  Saved plot to tus_simulation.png")

if __name__ == "__main__":
    main()
