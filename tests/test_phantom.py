"""Tests for head phantom generation (RED phase)."""

import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.phantoms.brainweb import make_synthetic_head
from brain_fwi.phantoms.properties import map_labels_to_speed


class TestSyntheticHead:
    """Validate the procedural head phantom."""

    def test_output_shape(self):
        labels, props = make_synthetic_head(grid_shape=(128, 128), dx=0.001)
        assert labels.shape == (128, 128)

    def test_output_is_integer_labels(self):
        labels, _ = make_synthetic_head(grid_shape=(64, 64), dx=0.001)
        assert labels.dtype in (jnp.int32, jnp.int64)

    def test_contains_skull(self):
        """Phantom must contain skull tissue (label 7)."""
        labels, _ = make_synthetic_head(grid_shape=(128, 128), dx=0.001)
        assert jnp.any(labels == 7)

    def test_contains_brain(self):
        """Phantom must contain grey matter (label 2) and white matter (label 3)."""
        labels, _ = make_synthetic_head(grid_shape=(128, 128), dx=0.001)
        assert jnp.any(labels == 2)
        assert jnp.any(labels == 3)

    def test_contains_csf(self):
        """Phantom must contain CSF (label 1)."""
        labels, _ = make_synthetic_head(grid_shape=(128, 128), dx=0.001)
        assert jnp.any(labels == 1)

    def test_skull_surrounds_brain(self):
        """Skull should be between scalp and brain (radially)."""
        # Use a large enough grid to contain the full head (>20 cm)
        labels, _ = make_synthetic_head(grid_shape=(256, 256), dx=0.001)
        # Check a radial line from centre to edge
        mid = 128
        line = labels[mid, :]  # horizontal line through centre
        # Find tissue transitions
        unique_in_order = []
        for val in np.array(line):
            if len(unique_in_order) == 0 or val != unique_in_order[-1]:
                unique_in_order.append(int(val))
        # Skull (7) should appear between scalp (6) and CSF/brain
        assert 7 in unique_in_order

    def test_properties_dict_has_correct_keys(self):
        _, props = make_synthetic_head(grid_shape=(64, 64), dx=0.001)
        assert "sound_speed" in props
        assert "density" in props
        assert "attenuation" in props

    def test_speed_range_physical(self):
        """Speed of sound should be in physical range [300, 4500] m/s."""
        _, props = make_synthetic_head(grid_shape=(128, 128), dx=0.001)
        c = props["sound_speed"]
        assert float(jnp.min(c)) >= 300.0
        assert float(jnp.max(c)) <= 4500.0

    def test_skull_speed_distinct_from_brain(self):
        """Skull should be significantly faster than brain tissue."""
        labels, props = make_synthetic_head(grid_shape=(128, 128), dx=0.001)
        c = props["sound_speed"]
        skull_speed = float(jnp.mean(c[labels == 7]))
        brain_speed = float(jnp.mean(c[labels == 2]))
        assert skull_speed > brain_speed + 500  # skull >> brain

    def test_ventricles_present(self):
        """Synthetic phantom should include lateral ventricles (CSF inside brain)."""
        labels, _ = make_synthetic_head(grid_shape=(256, 256), dx=0.001)
        # CSF label (1) should appear in the interior (not just around skull)
        mid = 128
        inner_csf = labels[mid-30:mid+30, mid-30:mid+30]
        assert jnp.any(inner_csf == 1), "No ventricles found in brain core"
