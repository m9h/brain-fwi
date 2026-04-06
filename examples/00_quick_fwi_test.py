#!/usr/bin/env python
"""Quick FWI convergence diagnostic.

Runs a minimal 2D FWI on a simple circular inclusion phantom
to verify the pipeline converges. Uses the same time_axis for
data generation and inversion (critical for consistency).

This is the fastest way to verify the FWI works (~30s on CPU).
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from brain_fwi.transducers.helmet import ring_array_2d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    simulate_shot_sensors, _build_source_signal,
)
from brain_fwi.inversion.fwi import (
    _params_to_velocity, _velocity_to_params, _smooth_gradient,
)
from brain_fwi.inversion.losses import l2_loss
import optax


def main():
    print("Quick FWI convergence diagnostic")
    print("=" * 50)
    t0 = time.time()

    # Setup: 48x48, 16 transducers, circular inclusion
    grid_shape = (48, 48)
    dx = 0.002
    c_min, c_max = 1400.0, 1800.0
    pml_size = 10

    cx, cy = 24, 24
    x, y = jnp.meshgrid(jnp.arange(48), jnp.arange(48), indexing="ij")
    r = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    c_true = jnp.where(r < 10, 1600.0, 1500.0)
    rho = jnp.ones(grid_shape) * 1000.0

    # Transducers
    center_m = (cx * dx, cy * dx)
    positions = ring_array_2d(n_elements=16, center=center_m,
                              semi_major=0.04, semi_minor=0.04)
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    src_list = [(int(pos_grid[0][i]), int(pos_grid[1][i])) for i in range(16)]

    # CRITICAL: one time_axis for everything
    domain = build_domain(grid_shape, dx)
    ref_medium = build_medium(domain, c_max, 1000.0, pml_size=pml_size)
    time_axis = build_time_axis(ref_medium, cfl=0.3, t_end=40e-6)
    dt = float(time_axis.dt)
    n_samples = int(float(time_axis.t_end) / dt)
    freq = 50e3
    source_signal = _build_source_signal(freq, dt, n_samples)

    print(f"  dt={dt*1e6:.2f} us, {n_samples} samples")

    # Generate observed data
    print("  Generating observed data (16 shots)...")
    medium_true = build_medium(domain, c_true, rho, pml_size=pml_size)
    observed = []
    for i, src in enumerate(src_list):
        d = simulate_shot_sensors(medium_true, time_axis, src, pos_grid, source_signal, dt)
        observed.append(d)
    observed = jnp.stack(observed, axis=0)
    print(f"  Observed shape: {observed.shape}")

    # Inversion mask: only update inside the transducer ring
    mask_r = jnp.sqrt(((x - cx) / 18.0) ** 2 + ((y - cy) / 18.0) ** 2)
    inv_mask = jnp.where(mask_r <= 1.0, 1.0, 0.0)

    # Manual FWI loop (no bandpass, direct L2, all sources)
    params = _velocity_to_params(jnp.full(grid_shape, 1500.0), c_min, c_max)
    optimizer = optax.adam(0.1)  # conservative LR
    opt_state = optimizer.init(params)

    print("\n  Iter | Loss       | c_min  | c_max  | MSE vs true")
    print("  " + "-" * 55)

    for it in range(25):
        def loss_fn(p):
            vel = _params_to_velocity(p, c_min, c_max)
            med = build_medium(domain, vel, rho, pml_size=pml_size)
            total = 0.0
            for si in range(16):
                pred = simulate_shot_sensors(med, time_axis, src_list[si], pos_grid, source_signal, dt)
                obs_i = observed[si]
                min_t = min(pred.shape[0], obs_i.shape[0])
                total = total + l2_loss(pred[:min_t], obs_i[:min_t])
            return total / 16.0

        loss_val, grad = jax.value_and_grad(loss_fn)(params)
        grad = _smooth_gradient(grad, sigma=5.0)
        grad = grad * inv_mask  # mask to interior

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        vel = _params_to_velocity(params, c_min, c_max)
        mse = float(jnp.mean((vel - c_true) ** 2))

        if (it + 1) % 5 == 0 or it == 0:
            mse_masked = float(jnp.sum((vel - c_true) ** 2 * inv_mask) / jnp.sum(inv_mask))
            print(f"  {it+1:4d} | {float(loss_val):.6f} | {float(jnp.min(vel)):6.0f} | {float(jnp.max(vel)):6.0f} | {mse:.1f} ({mse_masked:.1f} masked)")

    # Final assessment
    vel_final = _params_to_velocity(params, c_min, c_max)
    mse_init = float(jnp.mean((jnp.full(grid_shape, 1500.0) - c_true) ** 2))
    mse_final = float(jnp.mean((vel_final - c_true) ** 2))

    print(f"\n  MSE (init):  {mse_init:.1f}")
    print(f"  MSE (final): {mse_final:.1f}")
    print(f"  Improved: {'YES' if mse_final < mse_init else 'NO'}")
    print(f"  Total time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
