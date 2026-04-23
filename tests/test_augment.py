"""Tests for phantom augmentation (jittered_properties + deformation warp)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from brain_fwi.phantoms.augment import (
    jittered_properties,
    random_deformation_warp,
)


def _tiny_label_volume() -> np.ndarray:
    """Small toy volume with a skull-like shell + brain interior."""
    vol = np.zeros((16, 16, 16), dtype=np.int32)
    vol[2:14, 2:14, 2:14] = 7  # skull
    vol[4:12, 4:12, 4:12] = 2  # grey matter
    vol[6:10, 6:10, 6:10] = 1  # CSF
    return vol


class TestJitteredProperties:
    def test_shapes_match_labels(self):
        labels = jnp.asarray(_tiny_label_volume())
        out = jittered_properties(labels, jr.PRNGKey(0))
        for k in ("sound_speed", "density", "attenuation"):
            assert out[k].shape == labels.shape

    def test_zero_intensity_returns_nominal(self):
        labels = jnp.asarray(_tiny_label_volume())
        out = jittered_properties(labels, jr.PRNGKey(0), intensity=0.0)
        # skull cortical nominal c = 2800
        skull_mask = np.asarray(labels) == 7
        np.testing.assert_allclose(
            np.asarray(out["sound_speed"])[skull_mask].mean(), 2800.0, atol=1e-3
        )

    def test_same_key_reproducible(self):
        labels = jnp.asarray(_tiny_label_volume())
        a = jittered_properties(labels, jr.PRNGKey(42))
        b = jittered_properties(labels, jr.PRNGKey(42))
        np.testing.assert_array_equal(np.asarray(a["sound_speed"]), np.asarray(b["sound_speed"]))

    def test_different_keys_differ(self):
        labels = jnp.asarray(_tiny_label_volume())
        a = jittered_properties(labels, jr.PRNGKey(1))
        b = jittered_properties(labels, jr.PRNGKey(2))
        assert not np.allclose(np.asarray(a["sound_speed"]), np.asarray(b["sound_speed"]))

    def test_air_label_has_no_jitter(self):
        """Label 0 sigma is zero by default; c, rho, alpha must stay nominal."""
        labels = jnp.zeros((4, 4, 4), dtype=jnp.int32)
        out = jittered_properties(labels, jr.PRNGKey(9), intensity=5.0)
        np.testing.assert_allclose(np.asarray(out["sound_speed"]), 343.0, atol=1e-3)
        np.testing.assert_allclose(np.asarray(out["density"]), 1.225, atol=1e-4)

    def test_within_clip_bounds(self):
        """Multipliers are clipped to [0.5, 1.5] so outputs stay physical."""
        labels = jnp.asarray(_tiny_label_volume())
        out = jittered_properties(labels, jr.PRNGKey(123), intensity=10.0)
        c = np.asarray(out["sound_speed"])
        # skull nominal 2800; clipped multiplier range [0.5, 1.5] → [1400, 4200]
        skull_mask = np.asarray(labels) == 7
        assert c[skull_mask].max() <= 2800.0 * 1.5 + 1e-3
        assert c[skull_mask].min() >= 2800.0 * 0.5 - 1e-3

    def test_multipliers_logged(self):
        labels = jnp.asarray(_tiny_label_volume())
        out = jittered_properties(labels, jr.PRNGKey(0))
        assert out["multipliers"].shape[-1] == 3


class TestDeformationWarp:
    def test_shape_and_dtype_preserved(self):
        labels = _tiny_label_volume()
        rng = np.random.default_rng(0)
        warped = random_deformation_warp(labels, rng, max_displacement_voxels=2.0)
        assert warped.shape == labels.shape
        assert warped.dtype == labels.dtype

    def test_zero_displacement_is_identity(self):
        labels = _tiny_label_volume()
        rng = np.random.default_rng(0)
        warped = random_deformation_warp(labels, rng, max_displacement_voxels=0.0)
        np.testing.assert_array_equal(warped, labels)

    def test_preserves_label_set(self):
        """Nearest-neighbor warp must not introduce new label values."""
        labels = _tiny_label_volume()
        rng = np.random.default_rng(7)
        warped = random_deformation_warp(labels, rng, max_displacement_voxels=4.0)
        assert set(np.unique(warped)).issubset(set(np.unique(labels)))

    def test_rejects_non_3d(self):
        labels = np.zeros((16, 16), dtype=np.int32)
        with pytest.raises(ValueError, match="3D"):
            random_deformation_warp(labels, np.random.default_rng(0))

    def test_reproducible_with_same_rng(self):
        labels = _tiny_label_volume()
        a = random_deformation_warp(labels, np.random.default_rng(42))
        b = random_deformation_warp(labels, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)
