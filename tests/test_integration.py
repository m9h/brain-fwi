"""Integration tests: forward simulation + FWI gradient (RED → GREEN).

Tests the full pipeline on a tiny grid (32x32) to verify:
1. j-Wave produces non-zero pressure at receivers
2. Gradients flow through the forward operator
3. A single FWI step reduces the loss
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.simulation.forward import (
    build_domain,
    build_medium,
    build_time_axis,
    simulate_shot_sensors,
    _build_source_signal,
    _to_array,
)
from brain_fwi.inversion.fwi import FWIConfig
from brain_fwi.inversion.losses import l2_loss
from brain_fwi.transducers.helmet import ring_array_2d, transducer_positions_to_grid


@pytest.fixture
def tiny_setup():
    """Tiny 48x48 homogeneous domain for fast integration tests."""
    grid_shape = (48, 48)
    dx = 0.002  # 2mm spacing
    freq = 50e3  # 50 kHz (wavelength = 30mm, ~15 PPW)

    c_true = jnp.ones(grid_shape) * 1500.0
    rho = jnp.ones(grid_shape) * 1000.0

    domain = build_domain(grid_shape, dx)
    medium = build_medium(domain, c_true, rho, pml_size=10)
    time_axis = build_time_axis(medium, cfl=0.3, t_end=30e-6)
    dt = float(time_axis.dt)
    n_samples = int(float(time_axis.t_end) / dt)
    source_signal = _build_source_signal(freq, dt, n_samples)

    # Source at grid centre, sensors at 4 corners (inside PML-safe zone)
    src_pos = (24, 24)
    sensor_grid = (
        jnp.array([15, 15, 33, 33]),
        jnp.array([15, 33, 15, 33]),
    )

    return {
        "grid_shape": grid_shape,
        "dx": dx,
        "freq": freq,
        "c_true": c_true,
        "rho": rho,
        "domain": domain,
        "medium": medium,
        "time_axis": time_axis,
        "dt": dt,
        "source_signal": source_signal,
        "src_pos": src_pos,
        "sensor_grid": sensor_grid,
    }


class TestForwardSimulation:
    """Validate that j-Wave forward simulation works end-to-end."""

    def test_sensor_data_nonzero(self, tiny_setup):
        """Forward simulation should produce non-zero pressure at sensors."""
        s = tiny_setup
        data = simulate_shot_sensors(
            s["medium"], s["time_axis"], s["src_pos"],
            s["sensor_grid"], s["source_signal"], s["dt"],
        )
        assert data.ndim >= 1
        assert float(jnp.max(jnp.abs(data))) > 0, "Sensor data is all zeros"

    def test_sensor_data_finite(self, tiny_setup):
        """No NaN or Inf in sensor data."""
        s = tiny_setup
        data = simulate_shot_sensors(
            s["medium"], s["time_axis"], s["src_pos"],
            s["sensor_grid"], s["source_signal"], s["dt"],
        )
        assert jnp.all(jnp.isfinite(data)), "Sensor data contains NaN/Inf"

    def test_sensor_count_matches(self, tiny_setup):
        """Output should have one trace per sensor."""
        s = tiny_setup
        data = simulate_shot_sensors(
            s["medium"], s["time_axis"], s["src_pos"],
            s["sensor_grid"], s["source_signal"], s["dt"],
        )
        n_sensors = len(s["sensor_grid"][0])
        # Should have shape (n_timesteps, n_sensors) or at least n_sensors columns
        assert data.shape[-1] == n_sensors


class TestGradientFlow:
    """Validate that gradients flow through the full FWI pipeline."""

    def test_loss_gradient_nonzero(self, tiny_setup):
        """Gradient of L2 loss w.r.t. velocity should be non-zero."""
        s = tiny_setup

        # Generate "observed" data with true model
        observed = simulate_shot_sensors(
            s["medium"], s["time_axis"], s["src_pos"],
            s["sensor_grid"], s["source_signal"], s["dt"],
        )

        c_min, c_max = 1400.0, 3200.0
        # Perturbed initial velocity (slightly different from true).
        # Direct-velocity parameterisation: params are stored in m/s and
        # bounds are enforced by clip inside the loss function.
        c_init = jnp.ones(s["grid_shape"]) * 1480.0
        params = c_init

        # Pre-compute time_axis OUTSIDE the traced function.
        # TimeAxis.from_medium calls float() which breaks JAX tracing.
        fixed_time_axis = s["time_axis"]

        def loss_fn(p):
            velocity = jnp.clip(p, c_min, c_max)
            domain = build_domain(s["grid_shape"], s["dx"])
            medium = build_medium(domain, velocity, s["rho"], pml_size=10)

            pred = simulate_shot_sensors(
                medium, fixed_time_axis, s["src_pos"],
                s["sensor_grid"], s["source_signal"], s["dt"],
            )
            min_t = min(pred.shape[0], observed.shape[0])
            return l2_loss(pred[:min_t], observed[:min_t])

        loss_val, grad = jax.value_and_grad(loss_fn)(params)

        assert jnp.all(jnp.isfinite(grad)), "Gradient contains NaN/Inf"
        assert float(jnp.max(jnp.abs(grad))) > 0, "Gradient is all zeros"

    def test_gradient_step_reduces_loss(self, tiny_setup):
        """A single gradient step should reduce the loss (descent direction)."""
        s = tiny_setup

        observed = simulate_shot_sensors(
            s["medium"], s["time_axis"], s["src_pos"],
            s["sensor_grid"], s["source_signal"], s["dt"],
        )

        c_min, c_max = 1400.0, 3200.0
        c_init = jnp.ones(s["grid_shape"]) * 1480.0
        params = c_init

        # Pre-compute time_axis outside traced scope
        fixed_time_axis = s["time_axis"]

        def loss_fn(p):
            velocity = jnp.clip(p, c_min, c_max)
            domain = build_domain(s["grid_shape"], s["dx"])
            medium = build_medium(domain, velocity, s["rho"], pml_size=10)

            pred = simulate_shot_sensors(
                medium, fixed_time_axis, s["src_pos"],
                s["sensor_grid"], s["source_signal"], s["dt"],
            )
            min_t = min(pred.shape[0], observed.shape[0])
            return l2_loss(pred[:min_t], observed[:min_t])

        loss_before, grad = jax.value_and_grad(loss_fn)(params)

        # Gradient should be non-zero (descent direction exists)
        assert float(jnp.max(jnp.abs(grad))) > 0, "Gradient is zero"

        # Take a gradient step — if loss > threshold, it should decrease
        lr = 10.0
        params_new = params - lr * grad
        loss_after = loss_fn(params_new)

        # Either loss decreased, or it was already near zero
        assert float(loss_after) < float(loss_before) + 1e-8, \
            f"Loss increased: {float(loss_before):.10f} → {float(loss_after):.10f}"


class TestRingArrayIntegration:
    """Test ring array + forward simulation together."""

    def test_ring_array_produces_valid_grid_indices(self):
        """Ring array positions should map to valid grid indices."""
        grid_shape = (128, 128)
        dx = 0.001
        cx, cy = 0.064, 0.064  # centre of 12.8cm grid

        pos = ring_array_2d(n_elements=32, center=(cx, cy),
                            semi_major=0.05, semi_minor=0.05)
        idx = transducer_positions_to_grid(pos, dx, grid_shape)

        # All indices should be within grid bounds
        assert jnp.all(idx[0] >= 0) and jnp.all(idx[0] < grid_shape[0])
        assert jnp.all(idx[1] >= 0) and jnp.all(idx[1] < grid_shape[1])
