"""Tests for MIDA head model loading and acoustic mapping (RED phase).

The MIDA model has 153 anatomical structures at 500um resolution.
These tests validate the tissue-to-acoustic mapping and resampling
without requiring the actual MIDA data files (uses synthetic stubs).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.phantoms.mida import (
    MIDA_TISSUE_GROUPS,
    MIDA_ACOUSTIC_PROPERTIES,
    map_mida_labels_to_acoustic,
    resample_volume,
)


class TestMIDAPropertyTable:
    """Validate the MIDA tissue group acoustic property mapping."""

    def test_all_groups_have_properties(self):
        """Every tissue group should have speed, density, attenuation."""
        for group, props in MIDA_ACOUSTIC_PROPERTIES.items():
            assert "sound_speed" in props, f"Group {group!r} missing sound_speed"
            assert "density" in props, f"Group {group!r} missing density"
            assert "attenuation" in props, f"Group {group!r} missing attenuation"

    def test_cortical_bone_speed(self):
        """Cortical bone: ~2800 m/s per ITRUSST benchmark."""
        props = MIDA_ACOUSTIC_PROPERTIES["cortical_bone"]
        assert 2500 <= props["sound_speed"] <= 3200

    def test_trabecular_bone_speed(self):
        """Trabecular/diploe: slower than cortical."""
        cortical = MIDA_ACOUSTIC_PROPERTIES["cortical_bone"]["sound_speed"]
        trabecular = MIDA_ACOUSTIC_PROPERTIES["trabecular_bone"]["sound_speed"]
        assert trabecular < cortical

    def test_brain_tissue_speed(self):
        """Brain tissue: ~1560 m/s."""
        for group in ["grey_matter", "white_matter"]:
            c = MIDA_ACOUSTIC_PROPERTIES[group]["sound_speed"]
            assert 1500 <= c <= 1600

    def test_csf_near_water(self):
        c = MIDA_ACOUSTIC_PROPERTIES["csf"]["sound_speed"]
        assert 1480 <= c <= 1520

    def test_air_speed(self):
        c = MIDA_ACOUSTIC_PROPERTIES["air"]["sound_speed"]
        assert c == pytest.approx(343.0, abs=10)

    def test_skull_layers_distinguished(self):
        """MIDA has separate cortical and trabecular bone layers."""
        assert "cortical_bone" in MIDA_ACOUSTIC_PROPERTIES
        assert "trabecular_bone" in MIDA_ACOUSTIC_PROPERTIES


class TestMIDALabelMapping:
    """Validate MIDA tissue label → acoustic property mapping."""

    def test_simple_label_array(self):
        """Map a small array of MIDA labels to acoustic properties."""
        # Use labels that map to known groups
        labels = np.array([0, 1, 2, 10, 50], dtype=np.int32)
        props = map_mida_labels_to_acoustic(labels)
        assert "sound_speed" in props
        assert props["sound_speed"].shape == (5,)

    def test_all_speeds_positive(self):
        labels = np.arange(154, dtype=np.int32)
        props = map_mida_labels_to_acoustic(labels)
        assert np.all(np.array(props["sound_speed"]) > 0)

    def test_all_densities_positive(self):
        labels = np.arange(154, dtype=np.int32)
        props = map_mida_labels_to_acoustic(labels)
        assert np.all(np.array(props["density"]) > 0)

    def test_3d_label_volume(self):
        """Should work on 3D volumes (the MIDA use case)."""
        labels = np.random.randint(0, 154, size=(10, 10, 10), dtype=np.int32)
        props = map_mida_labels_to_acoustic(labels)
        assert props["sound_speed"].shape == (10, 10, 10)

    def test_background_maps_to_water(self):
        """Label 0 (background) should map to water for coupling medium."""
        labels = np.array([0], dtype=np.int32)
        props = map_mida_labels_to_acoustic(labels)
        c = float(props["sound_speed"][0])
        # Background → water (1500 m/s) for acoustic coupling
        assert c == pytest.approx(1500.0, abs=50)


class TestResampling:
    """Validate volume resampling for grid adjustment."""

    def test_identity_resample(self):
        """Resampling to same shape should preserve values."""
        vol = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        result = resample_volume(vol, target_shape=(2, 2, 2))
        np.testing.assert_allclose(result, vol, atol=0.1)

    def test_downsample(self):
        vol = np.ones((10, 10, 10), dtype=np.float32) * 1500.0
        result = resample_volume(vol, target_shape=(5, 5, 5))
        assert result.shape == (5, 5, 5)
        np.testing.assert_allclose(result, 1500.0, atol=1.0)

    def test_upsample_shape(self):
        vol = np.ones((5, 5, 5), dtype=np.float32)
        result = resample_volume(vol, target_shape=(10, 10, 10))
        assert result.shape == (10, 10, 10)

    def test_label_resample_nearest(self):
        """Label volumes should use nearest-neighbor interpolation."""
        labels = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
        result = resample_volume(labels, target_shape=(4, 4, 4), order=0)
        assert result.dtype == np.int32
        # All values should be from the original set
        assert set(np.unique(result)).issubset(set(range(8)))
