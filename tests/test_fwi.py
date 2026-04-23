"""Tests for FWI engine.

Tests gradient smoothing, loss selection, and basic convergence.
Parameterisation-specific behaviour (VoxelField, SIRENField) lives in
tests/test_param_field.py. Uses tiny grids (32x32) for speed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.inversion.fwi import (
    _smooth_gradient,
    FWIConfig,
)


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
