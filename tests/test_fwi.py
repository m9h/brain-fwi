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


class TestCheckpointGridShapeGuard:
    """Regression test for the 917/919 stale-checkpoint bug.

    On 2026-04-23 overnight, a 192^3 checkpoint written by one FWI run
    was transparently loaded by a later MIDA 96^3 run because the
    checkpoint_dir was shared. This caused:
      - Job 917 to crash with a broadcast error inside j-Wave when
        the (192,...) params were multiplied against the (96,...)
        medium.
      - Job 919 to silently skip Band 1 (loading foreign params) and
        NaN through bands 2-3 at 192^3 because the loaded params were
        from an unrelated configuration.
    The fix stamps grid_shape on save and validates on load.
    """

    def _write_and_read(self, saved_shape, expected_shape, tmp_path):
        import jax.numpy as _jnp
        from brain_fwi.inversion.fwi import _load_checkpoint, _save_checkpoint

        ckpt = tmp_path / "fwi_checkpoint.h5"
        params = _jnp.zeros(saved_shape, dtype=_jnp.float32)
        _save_checkpoint(
            ckpt, band_idx=0, params=params,
            loss_history=[0.1], velocity_history=[params],
            grid_shape=saved_shape,
        )
        return _load_checkpoint(ckpt, expected_grid_shape=expected_shape)

    def test_matching_shape_loads(self, tmp_path):
        loaded = self._write_and_read((8, 8, 8), (8, 8, 8), tmp_path)
        assert loaded is not None
        assert tuple(loaded["params"].shape) == (8, 8, 8)

    def test_mismatched_shape_raises(self, tmp_path):
        with pytest.raises(ValueError, match="grid"):
            self._write_and_read((16, 16, 16), (8, 8, 8), tmp_path)

    def test_no_checkpoint_returns_none(self, tmp_path):
        from brain_fwi.inversion.fwi import _load_checkpoint
        assert _load_checkpoint(tmp_path / "nope.h5",
                                expected_grid_shape=(8, 8, 8)) is None

    def test_legacy_checkpoint_without_grid_stamp_still_works(self, tmp_path):
        """A checkpoint written before the guard landed has no grid_shape
        attribute. If the params happen to match the expected shape, loading
        should still succeed — we only refuse on an actual mismatch."""
        import h5py
        import numpy as _np
        ckpt = tmp_path / "legacy.h5"
        with h5py.File(str(ckpt), "w") as f:
            f.attrs["completed_bands"] = 1
            f.create_dataset("params", data=_np.zeros((8, 8, 8), dtype=_np.float32))
            f.create_dataset("loss_history", data=_np.array([0.1]))
            f.create_dataset("velocity_band_0", data=_np.zeros((8, 8, 8), dtype=_np.float32))
        from brain_fwi.inversion.fwi import _load_checkpoint
        loaded = _load_checkpoint(ckpt, expected_grid_shape=(8, 8, 8))
        assert loaded is not None
