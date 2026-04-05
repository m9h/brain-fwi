"""Tests for transducer array geometry (RED phase)."""

import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.transducers.helmet import (
    ring_array_2d,
    helmet_array_3d,
    transducer_positions_to_grid,
    compute_normals_2d,
)


class TestRingArray2D:
    """Validate 2D ring array generation."""

    def test_output_shape(self):
        pos = ring_array_2d(n_elements=64)
        assert pos.shape == (64, 2)

    def test_128_elements(self):
        pos = ring_array_2d(n_elements=128)
        assert pos.shape == (128, 2)

    def test_positions_on_ellipse(self):
        """All positions should lie on the specified ellipse."""
        a, b = 0.10, 0.08
        pos = ring_array_2d(n_elements=64, center=(0.0, 0.0),
                            semi_major=a, semi_minor=b, standoff=0.0)
        # Check: (x/a)^2 + (y/b)^2 ≈ 1 for all points
        r = (pos[:, 0] / a) ** 2 + (pos[:, 1] / b) ** 2
        np.testing.assert_allclose(np.array(r), 1.0, atol=1e-5)

    def test_standoff_increases_radius(self):
        """Standoff should place elements further from centre."""
        pos_no_standoff = ring_array_2d(n_elements=32, standoff=0.0)
        pos_with_standoff = ring_array_2d(n_elements=32, standoff=0.01)
        r0 = jnp.linalg.norm(pos_no_standoff, axis=1)
        r1 = jnp.linalg.norm(pos_with_standoff, axis=1)
        assert float(jnp.mean(r1)) > float(jnp.mean(r0))

    def test_centre_offset(self):
        """Custom centre should shift all positions."""
        pos = ring_array_2d(n_elements=32, center=(0.1, 0.05))
        centroid = jnp.mean(pos, axis=0)
        assert float(centroid[0]) == pytest.approx(0.1, abs=0.01)
        assert float(centroid[1]) == pytest.approx(0.05, abs=0.01)

    def test_uniform_angular_spacing(self):
        """Elements should be approximately uniformly spaced."""
        pos = ring_array_2d(n_elements=64, semi_major=0.1, semi_minor=0.1,
                            standoff=0.0, center=(0.0, 0.0))
        # For a circle, consecutive distance should be nearly constant
        diffs = jnp.roll(pos, -1, axis=0) - pos
        distances = jnp.linalg.norm(diffs, axis=1)
        cv = float(jnp.std(distances) / jnp.mean(distances))
        assert cv < 0.05  # < 5% variation for a circle

    def test_exclude_arc(self):
        """Excluding an arc should reduce coverage range."""
        pos_full = ring_array_2d(n_elements=64)
        pos_gap = ring_array_2d(n_elements=64,
                                exclude_arc=(-0.5, 0.5))
        # With exclusion, no elements should be in the excluded region
        angles = jnp.arctan2(pos_gap[:, 1], pos_gap[:, 0])
        in_gap = jnp.sum((angles > -0.5) & (angles < 0.5))
        assert int(in_gap) < 5  # few or no elements in gap


class TestHelmetArray3D:
    """Validate 3D helmet array generation."""

    def test_output_shape(self):
        pos = helmet_array_3d(n_elements=128)
        assert pos.shape == (128, 3)

    def test_256_elements(self):
        pos = helmet_array_3d(n_elements=256)
        assert pos.shape == (256, 3)

    def test_positions_on_ellipsoid(self):
        """All positions should be approximately on the target ellipsoid."""
        a, b, c = 0.10, 0.08, 0.09
        pos = helmet_array_3d(n_elements=64, radius_ap=a, radius_lr=b,
                              radius_si=c, standoff=0.0, exclude_face=False)
        r = (pos[:, 0] / a) ** 2 + (pos[:, 1] / b) ** 2 + (pos[:, 2] / c) ** 2
        np.testing.assert_allclose(np.array(r), 1.0, atol=0.1)

    def test_exclude_face_removes_anterior_inferior(self):
        """Face exclusion should remove elements in front-lower region."""
        pos = helmet_array_3d(n_elements=128, exclude_face=True)
        # Should have fewer elements with large positive x and negative z
        front_low = (pos[:, 0] > 0.06) & (pos[:, 2] < -0.02)
        assert int(jnp.sum(front_low)) < 5


class TestGridConversion:
    """Validate physical → grid index conversion."""

    def test_origin_at_zero(self):
        pos = jnp.array([[0.005, 0.005]])
        idx = transducer_positions_to_grid(pos, dx=0.001, grid_shape=(256, 256))
        assert int(idx[0][0]) == 5
        assert int(idx[1][0]) == 5

    def test_clipping(self):
        """Out-of-bounds positions should be clipped to grid edges."""
        pos = jnp.array([[-0.01, -0.01], [1.0, 1.0]])
        idx = transducer_positions_to_grid(pos, dx=0.001, grid_shape=(128, 128))
        assert int(idx[0][0]) == 0
        assert int(idx[0][1]) == 127

    def test_round_trip_accuracy(self):
        """Grid index * dx should be close to original position."""
        pos = jnp.array([[0.0456, 0.0789]])
        dx = 0.001
        idx = transducer_positions_to_grid(pos, dx, grid_shape=(256, 256))
        recovered = jnp.array([idx[0][0] * dx, idx[1][0] * dx])
        np.testing.assert_allclose(np.array(recovered), np.array(pos[0]),
                                   atol=dx)


class TestNormals2D:
    """Validate inward-pointing normal computation."""

    def test_normals_point_inward(self):
        """Normals should point toward the specified centre."""
        pos = ring_array_2d(n_elements=32, center=(0.05, 0.05))
        centre = jnp.array([0.05, 0.05])
        normals = compute_normals_2d(pos, centre)

        # Dot product of normal with (centre - pos) should be positive
        to_centre = centre - pos
        dots = jnp.sum(normals * to_centre, axis=1)
        assert jnp.all(dots > 0)

    def test_normals_are_unit_vectors(self):
        pos = ring_array_2d(n_elements=32)
        centre = jnp.array([0.0, 0.0])
        normals = compute_normals_2d(pos, centre)
        norms = jnp.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(np.array(norms), 1.0, atol=1e-5)
