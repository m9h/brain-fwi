"""Tests for FWI engine (RED phase).

Tests the reparameterization, gradient computation, and basic convergence.
Uses tiny grids (32x32) for speed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.inversion.fwi import (
    _params_to_velocity,
    _velocity_to_params,
    _smooth_gradient,
    FWIConfig,
)


class TestReparameterization:
    """Validate sigmoid reparameterization for velocity bounds."""

    def test_roundtrip(self):
        """velocity → params → velocity should be identity."""
        c_min, c_max = 1400.0, 3200.0
        v_orig = jnp.array([1500.0, 1800.0, 2800.0])
        params = _velocity_to_params(v_orig, c_min, c_max)
        v_recovered = _params_to_velocity(params, c_min, c_max)
        np.testing.assert_allclose(np.array(v_recovered), np.array(v_orig),
                                   atol=5.0)

    def test_bounds_enforced(self):
        """Output velocity should always be within [c_min, c_max]."""
        c_min, c_max = 1400.0, 3200.0
        # Extreme parameter values
        params = jnp.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        v = _params_to_velocity(params, c_min, c_max)
        assert float(jnp.min(v)) >= c_min - 0.1
        assert float(jnp.max(v)) <= c_max + 0.1

    def test_zero_params_gives_midpoint(self):
        """params=0 → sigmoid(0)=0.5 → midpoint velocity."""
        c_min, c_max = 1400.0, 3200.0
        v = _params_to_velocity(jnp.array([0.0]), c_min, c_max)
        expected = (c_min + c_max) / 2
        assert float(v[0]) == pytest.approx(expected, abs=1.0)

    def test_differentiable(self):
        """Reparameterization must be differentiable."""
        c_min, c_max = 1400.0, 3200.0
        params = jnp.array([0.0, 1.0, -1.0])
        grad = jax.grad(lambda p: jnp.sum(_params_to_velocity(p, c_min, c_max)))(params)
        assert grad.shape == (3,)
        assert not jnp.any(jnp.isnan(grad))
        assert jnp.all(grad > 0)  # sigmoid is monotonically increasing

    def test_2d_array(self):
        """Should work on 2D velocity fields."""
        c_min, c_max = 1400.0, 3200.0
        v = jnp.ones((32, 32)) * 1500.0
        params = _velocity_to_params(v, c_min, c_max)
        assert params.shape == (32, 32)
        v_back = _params_to_velocity(params, c_min, c_max)
        np.testing.assert_allclose(np.array(v_back), 1500.0, atol=5.0)


class TestGradientSmoothing:
    """Validate Gaussian gradient smoothing."""

    def test_no_smoothing(self):
        """sigma=0 should return input unchanged."""
        grad = jnp.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0]])
        result = _smooth_gradient(grad, sigma=0.0)
        np.testing.assert_allclose(np.array(result), np.array(grad))

    def test_smoothing_reduces_peak(self):
        """Smoothing should reduce the peak value of a delta function."""
        n = 32
        grad = jnp.zeros((n, n))
        grad = grad.at[n//2, n//2].set(1.0)
        smoothed = _smooth_gradient(grad, sigma=3.0)
        assert float(jnp.max(smoothed)) < float(jnp.max(grad))

    def test_smoothing_preserves_total(self):
        """Smoothing should approximately preserve total gradient mass."""
        n = 32
        grad = jnp.zeros((n, n))
        grad = grad.at[n//2, n//2].set(1.0)
        smoothed = _smooth_gradient(grad, sigma=2.0)
        np.testing.assert_allclose(
            float(jnp.sum(smoothed)), float(jnp.sum(grad)),
            rtol=0.1  # within 10% (edge effects)
        )

    def test_shape_preserved(self):
        grad = jnp.ones((64, 64))
        smoothed = _smooth_gradient(grad, sigma=2.0)
        assert smoothed.shape == (64, 64)


class TestFWIConfig:
    """Validate FWI configuration defaults."""

    def test_default_freq_bands(self):
        config = FWIConfig()
        assert len(config.freq_bands) == 3
        # Bands should be ascending
        for i in range(len(config.freq_bands) - 1):
            assert config.freq_bands[i][1] <= config.freq_bands[i+1][1]

    def test_default_velocity_bounds(self):
        config = FWIConfig()
        assert config.c_min < config.c_max
        assert config.c_min > 0

    def test_custom_config(self):
        config = FWIConfig(
            freq_bands=[(100e3, 200e3)],
            n_iters_per_band=10,
            shots_per_iter=2,
        )
        assert len(config.freq_bands) == 1
        assert config.n_iters_per_band == 10
