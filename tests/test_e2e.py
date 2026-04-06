"""End-to-end FWI test on a tiny grid.

Verifies the full pipeline: phantom → transducers → forward data → FWI → reconstruction.
Uses a 48x48 grid with 8 transducers and 1 frequency band for speed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.phantoms.brainweb import make_synthetic_head
from brain_fwi.phantoms.properties import map_labels_to_speed, map_labels_to_density
from brain_fwi.transducers.helmet import ring_array_2d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    generate_observed_data, _build_source_signal,
)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi


class TestEndToEndFWI:
    """Full pipeline test on a tiny problem."""

    @pytest.fixture
    def tiny_fwi_problem(self):
        """Set up a minimal FWI problem: 48x48 grid, 16 transducers.

        CRITICAL: Uses a single consistent time_axis computed from c_max
        for BOTH data generation and FWI. This prevents the dt mismatch
        that causes FWI divergence.
        """
        grid_shape = (48, 48)
        dx = 0.002  # 2mm
        c_min, c_max = 1400.0, 1800.0
        pml_size = 10

        # Simple two-layer phantom: water + circular inclusion
        cx, cy = 24, 24
        x, y = jnp.meshgrid(jnp.arange(48), jnp.arange(48), indexing="ij")
        r = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        c_true = jnp.where(r < 10, 1600.0, 1500.0)  # inclusion at 1600 m/s
        rho = jnp.ones(grid_shape) * 1000.0

        # 16 transducers in a ring (more coverage = better inversion)
        center_m = (24 * dx, 24 * dx)
        positions = ring_array_2d(
            n_elements=16, center=center_m,
            semi_major=0.04, semi_minor=0.04, standoff=0.0,
        )
        pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
        src_list = [(int(pos_grid[0][i]), int(pos_grid[1][i])) for i in range(16)]

        # Compute ONE reference time axis from c_max — used everywhere
        freq = 50e3
        domain = build_domain(grid_shape, dx)
        ref_medium = build_medium(domain, c_max, 1000.0, pml_size=pml_size)
        time_axis = build_time_axis(ref_medium, cfl=0.3, t_end=40e-6)
        dt = float(time_axis.dt)
        t_end = float(time_axis.t_end)
        n_samples = int(t_end / dt)
        source_signal = _build_source_signal(freq, dt, n_samples)

        return {
            "grid_shape": grid_shape,
            "dx": dx,
            "c_true": c_true,
            "rho": rho,
            "src_list": src_list,
            "sensor_pos": pos_grid,
            "freq": freq,
            "dt": dt,
            "t_end": t_end,
            "source_signal": source_signal,
            "time_axis": time_axis,
            "pml_size": pml_size,
            "c_min": c_min,
            "c_max": c_max,
        }

    def test_full_pipeline_reduces_error(self, tiny_fwi_problem):
        """FWI should produce a model closer to truth than the initial guess."""
        s = tiny_fwi_problem

        # Generate observed data using the SAME time_axis as FWI
        observed = generate_observed_data(
            s["c_true"], s["rho"], s["dx"],
            s["src_list"], s["sensor_pos"],
            s["freq"], pml_size=s["pml_size"], cfl=0.3,
            time_axis=s["time_axis"],
            source_signal=s["source_signal"],
            dt=s["dt"],
            verbose=False,
        )

        # Initial model: homogeneous water
        c_init = jnp.full(s["grid_shape"], 1500.0)

        config = FWIConfig(
            freq_bands=[(20e3, 60e3)],
            n_iters_per_band=15,
            shots_per_iter=8,
            learning_rate=2.0,
            c_min=s["c_min"],
            c_max=s["c_max"],
            pml_size=s["pml_size"],
            gradient_smooth_sigma=2.0,
            loss_fn="l2",
            verbose=False,
        )

        result = run_fwi(
            observed_data=observed,
            initial_velocity=c_init,
            density=s["rho"],
            dx=s["dx"],
            src_positions_grid=s["src_list"],
            sensor_positions_grid=s["sensor_pos"],
            source_signal=s["source_signal"],
            dt=s["dt"],
            t_end=s["t_end"],
            config=config,
        )

        # Loss should have decreased
        assert result.loss_history[-1] < result.loss_history[0], \
            f"Loss did not decrease: {result.loss_history[0]:.6f} → {result.loss_history[-1]:.6f}"

        # Error should decrease
        error_init = float(jnp.mean((c_init - s["c_true"]) ** 2))
        error_recon = float(jnp.mean((result.velocity - s["c_true"]) ** 2))
        assert error_recon < error_init, \
            f"FWI did not improve: init_err={error_init:.1f}, recon_err={error_recon:.1f}"
