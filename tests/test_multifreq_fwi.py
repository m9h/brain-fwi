"""Test multi-frequency FWI convergence (RED → GREEN).

Validates that the frequency-banding strategy works:
bandpass the source signal, simulate with bandpassed source,
compare with bandpassed observed data. Two bands should converge
better than one by reducing cycle-skipping at low frequencies.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.transducers.helmet import ring_array_2d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    simulate_shot_sensors, _build_source_signal,
)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi


class TestMultiFreqConvergence:
    """Multi-frequency FWI should converge with bandpass enabled."""

    def test_two_band_fwi_loss_decreases(self):
        """FWI with 2 frequency bands should reduce the loss."""
        grid_shape = (48, 48)
        dx = 0.002
        c_min, c_max = 1400.0, 1800.0
        pml_size = 10

        cx, cy = 24, 24
        x, y = jnp.meshgrid(jnp.arange(48), jnp.arange(48), indexing="ij")
        r = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        c_true = jnp.where(r < 10, 1600.0, 1500.0)
        rho = jnp.ones(grid_shape) * 1000.0

        positions = ring_array_2d(
            n_elements=16, center=(cx * dx, cy * dx),
            semi_major=0.04, semi_minor=0.04,
        )
        pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
        src_list = [(int(pos_grid[0][i]), int(pos_grid[1][i])) for i in range(16)]

        # Reference time axis from c_max
        domain = build_domain(grid_shape, dx)
        ref_medium = build_medium(domain, c_max, 1000.0, pml_size=pml_size)
        time_axis = build_time_axis(ref_medium, cfl=0.3, t_end=40e-6)
        dt = float(time_axis.dt)
        n_samples = int(float(time_axis.t_end) / dt)
        freq = 50e3
        source_signal = _build_source_signal(freq, dt, n_samples)

        # Generate observed data with consistent time axis
        from brain_fwi.simulation.forward import generate_observed_data
        observed = generate_observed_data(
            c_true, rho, dx, src_list, pos_grid, freq,
            pml_size=pml_size, time_axis=time_axis,
            source_signal=source_signal, dt=dt, verbose=False,
        )

        # Mask
        mask_r = jnp.sqrt(((x - cx) / 18.0) ** 2 + ((y - cy) / 18.0) ** 2)
        inv_mask = jnp.where(mask_r <= 1.0, 1.0, 0.0)

        config = FWIConfig(
            freq_bands=[(10e3, 40e3), (40e3, 80e3)],
            n_iters_per_band=5,
            shots_per_iter=16,
            learning_rate=0.1,
            c_min=c_min,
            c_max=c_max,
            pml_size=pml_size,
            gradient_smooth_sigma=5.0,
            loss_fn="l2",
            mask=inv_mask,
            skip_bandpass=False,  # actually USE bandpass
            verbose=False,
        )

        c_init = jnp.full(grid_shape, 1500.0)
        result = run_fwi(
            observed_data=observed,
            initial_velocity=c_init,
            density=rho,
            dx=dx,
            src_positions_grid=src_list,
            sensor_positions_grid=pos_grid,
            source_signal=source_signal,
            dt=dt,
            t_end=float(time_axis.t_end),
            config=config,
        )

        # Loss should decrease across both bands
        assert result.loss_history[-1] < result.loss_history[0], \
            f"Multi-freq loss did not decrease: {result.loss_history[0]:.6f} → {result.loss_history[-1]:.6f}"
