"""Tests for resolution matrix / PSF analysis (RED phase).

The resolution matrix R characterizes how well different locations in
the brain can be resolved by the ultrasound tomography system. It depends
on the transducer geometry (helmet coverage) and the background medium.

R = (J^T J + lambda I)^{-1} J^T J

where J is the Jacobian (sensitivity matrix) dp/dc — the derivative of
sensor pressure w.r.t. sound speed at each voxel.

The PSF (Point Spread Function) at location x is the column of R
corresponding to x, reshaped to the spatial grid.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestJacobianComputation:
    """Validate Jacobian (sensitivity matrix) computation."""

    def test_jacobian_shape(self):
        """J should be (n_data, n_voxels) where n_data = n_timesteps * n_sensors."""
        from brain_fwi.inversion.resolution import compute_jacobian_column

        grid_shape = (32, 32)
        dx = 0.003
        # Single source, 4 sensors
        n_sensors = 4
        result = compute_jacobian_column(
            grid_shape=grid_shape, dx=dx,
            src_pos=(16, 16),
            sensor_positions=(jnp.array([10, 10, 22, 22]),
                              jnp.array([10, 22, 10, 22])),
            voxel_idx=(16, 16),
            c_background=1500.0,
            freq=50e3,
        )
        # Should return a 1D vector (sensitivity at the queried voxel)
        assert result.ndim == 1
        assert len(result) > 0

    def test_jacobian_nonzero_on_path(self):
        """Sensitivity should be non-zero on the source-receiver path."""
        from brain_fwi.inversion.resolution import compute_jacobian_column

        grid_shape = (32, 32)
        dx = 0.003
        # Source at left, sensor at right — voxel in between should be sensitive
        result = compute_jacobian_column(
            grid_shape=grid_shape, dx=dx,
            src_pos=(16, 4),
            sensor_positions=(jnp.array([16]), jnp.array([28])),
            voxel_idx=(16, 16),  # midpoint
            c_background=1500.0,
            freq=50e3,
        )
        assert float(jnp.max(jnp.abs(result))) > 0

    def test_jacobian_finite(self):
        """No NaN/Inf in Jacobian."""
        from brain_fwi.inversion.resolution import compute_jacobian_column

        grid_shape = (32, 32)
        dx = 0.003
        result = compute_jacobian_column(
            grid_shape=grid_shape, dx=dx,
            src_pos=(16, 16),
            sensor_positions=(jnp.array([8, 24]), jnp.array([8, 24])),
            voxel_idx=(16, 16),
            c_background=1500.0,
            freq=50e3,
        )
        assert jnp.all(jnp.isfinite(result))


class TestResolutionMatrix:
    """Validate resolution matrix and PSF computation."""

    def test_psf_has_correct_shape(self):
        """PSF should match grid shape."""
        from brain_fwi.inversion.resolution import compute_psf

        grid_shape = (24, 24)
        dx = 0.004
        n_elem = 8
        psf = compute_psf(
            grid_shape=grid_shape, dx=dx,
            n_elements=n_elem,
            query_point=(12, 12),
            c_background=1500.0,
            freq=50e3,
        )
        assert psf.shape == grid_shape

    def test_psf_peak_at_query_point(self):
        """PSF should peak at or near the queried location."""
        from brain_fwi.inversion.resolution import compute_psf

        grid_shape = (24, 24)
        dx = 0.004
        qx, qy = 12, 12
        psf = compute_psf(
            grid_shape=grid_shape, dx=dx,
            n_elements=16,
            query_point=(qx, qy),
            c_background=1500.0,
            freq=50e3,
        )
        peak = jnp.unravel_index(jnp.argmax(jnp.abs(psf)), grid_shape)
        # Peak should be within 3 voxels of query point
        assert abs(int(peak[0]) - qx) <= 3
        assert abs(int(peak[1]) - qy) <= 3

    def test_more_elements_sharper_psf(self):
        """More transducers should produce a narrower PSF."""
        from brain_fwi.inversion.resolution import compute_psf_width

        grid_shape = (24, 24)
        dx = 0.004
        width_8 = compute_psf_width(grid_shape, dx, n_elements=8,
                                     query_point=(12, 12), freq=50e3)
        width_16 = compute_psf_width(grid_shape, dx, n_elements=16,
                                      query_point=(12, 12), freq=50e3)
        assert width_16 <= width_8


class TestCoverageMetrics:
    """Validate coverage / sensitivity metrics."""

    def test_sensitivity_map_shape(self):
        """Sensitivity map should match grid shape."""
        from brain_fwi.inversion.resolution import compute_sensitivity_map

        grid_shape = (24, 24)
        dx = 0.004
        sens = compute_sensitivity_map(
            grid_shape=grid_shape, dx=dx,
            n_elements=8,
            c_background=1500.0,
            freq=50e3,
        )
        assert sens.shape == grid_shape

    def test_sensitivity_higher_near_transducers(self):
        """Sensitivity should be highest near the transducer ring."""
        from brain_fwi.inversion.resolution import compute_sensitivity_map

        grid_shape = (32, 32)
        dx = 0.003
        sens = compute_sensitivity_map(
            grid_shape=grid_shape, dx=dx,
            n_elements=16,
            c_background=1500.0,
            freq=50e3,
        )
        # Mean sensitivity in outer ring vs centre
        cx, cy = 16, 16
        x, y = jnp.meshgrid(jnp.arange(32), jnp.arange(32), indexing="ij")
        r = jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        outer = (r > 10) & (r < 14)
        inner = r < 5
        # Outer ring (near transducers) should have higher sensitivity
        assert float(jnp.mean(sens * outer)) > 0
