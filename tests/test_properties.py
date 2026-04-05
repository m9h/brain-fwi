"""Tests for acoustic tissue property mapping (RED phase).

Validates the ITRUSST benchmark values and label-to-property mapping.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.phantoms.properties import (
    TISSUE_PROPERTIES,
    map_labels_to_speed,
    map_labels_to_density,
    map_labels_to_attenuation,
    map_labels_to_all,
    remap_sci_labels,
)


class TestTissuePropertyTable:
    """Validate the acoustic property lookup table."""

    def test_all_12_brainweb_labels_present(self):
        """BrainWeb has 12 tissue classes (0-11)."""
        for label in range(12):
            assert label in TISSUE_PROPERTIES, f"Label {label} missing"

    def test_skull_speed_itrusst_benchmark(self):
        """Skull cortical bone: 2800 m/s per ITRUSST BM3."""
        c, rho, alpha = TISSUE_PROPERTIES[7]
        assert c == pytest.approx(2800.0, abs=100.0)

    def test_skull_density_itrusst_benchmark(self):
        """Skull density: ~1850 kg/m^3 per ITRUSST BM3."""
        c, rho, alpha = TISSUE_PROPERTIES[7]
        assert rho == pytest.approx(1850.0, abs=100.0)

    def test_brain_speed_reasonable(self):
        """Grey and white matter: ~1560 m/s."""
        for label in [2, 3]:  # GM, WM
            c, _, _ = TISSUE_PROPERTIES[label]
            assert 1500.0 <= c <= 1600.0

    def test_csf_speed_near_water(self):
        """CSF is close to water (1500 m/s)."""
        c, _, _ = TISSUE_PROPERTIES[1]
        assert c == pytest.approx(1500.0, abs=20.0)

    def test_trabecular_bone_properties(self):
        """Trabecular bone (label 11): slower and less dense than cortical."""
        c_cortical, rho_cortical, _ = TISSUE_PROPERTIES[7]
        c_trab, rho_trab, _ = TISSUE_PROPERTIES[11]
        assert c_trab < c_cortical
        assert rho_trab < rho_cortical

    def test_all_speeds_positive(self):
        for label, (c, rho, alpha) in TISSUE_PROPERTIES.items():
            assert c > 0, f"Label {label} has non-positive speed"
            assert rho > 0, f"Label {label} has non-positive density"
            assert alpha >= 0, f"Label {label} has negative attenuation"


class TestLabelMapping:
    """Validate the vectorized label → property mapping."""

    def test_single_label(self):
        labels = jnp.array([7])  # skull
        c = map_labels_to_speed(labels)
        assert c[0] == pytest.approx(2800.0)

    def test_2d_label_array(self):
        """Map a 2D label grid and check shape preservation."""
        labels = jnp.array([[0, 1, 2], [3, 7, 11]], dtype=jnp.int32)
        c = map_labels_to_speed(labels)
        assert c.shape == (2, 3)

    def test_speed_matches_table(self):
        """Every label maps to the correct table value."""
        for label in range(12):
            c = map_labels_to_speed(jnp.array([label]))
            expected = TISSUE_PROPERTIES[label][0]
            assert float(c[0]) == pytest.approx(expected, rel=1e-4)

    def test_density_matches_table(self):
        for label in range(12):
            rho = map_labels_to_density(jnp.array([label]))
            expected = TISSUE_PROPERTIES[label][1]
            assert float(rho[0]) == pytest.approx(expected, rel=1e-4)

    def test_out_of_range_labels_clipped(self):
        """Labels > 11 should be clipped to valid range, not crash."""
        labels = jnp.array([99, -1])
        c = map_labels_to_speed(labels)
        assert c.shape == (2,)
        # Should map to max label (11) and 0 respectively
        assert float(c[0]) == pytest.approx(TISSUE_PROPERTIES[11][0], rel=1e-3)
        assert float(c[1]) == pytest.approx(TISSUE_PROPERTIES[0][0], rel=1e-3)

    def test_map_all_returns_three_keys(self):
        labels = jnp.array([2, 7])
        result = map_labels_to_all(labels)
        assert "sound_speed" in result
        assert "density" in result
        assert "attenuation" in result
        assert result["sound_speed"].shape == (2,)


class TestSCIRemapping:
    """Validate SCI Institute label remapping."""

    def test_sci_skull_maps_to_brainweb_skull(self):
        """SCI label 2 (skull) → BrainWeb label 7."""
        sci = jnp.array([2])
        bw = remap_sci_labels(sci)
        assert int(bw[0]) == 7

    def test_sci_gm_maps_to_brainweb_gm(self):
        """SCI label 4 (GM) → BrainWeb label 2."""
        sci = jnp.array([4])
        bw = remap_sci_labels(sci)
        assert int(bw[0]) == 2

    def test_sci_scalp_maps_to_brainweb_skin(self):
        """SCI label 1 (scalp) → BrainWeb label 6 (muscle/skin)."""
        sci = jnp.array([1])
        bw = remap_sci_labels(sci)
        assert int(bw[0]) == 6
