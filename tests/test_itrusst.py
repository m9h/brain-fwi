"""Tests for the ITRUSST benchmark phantom generators."""

import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.phantoms.itrusst import (
    make_bm1_water_box,
    make_bm2_single_layer_plate,
    make_bm3_three_layer_plate,
    make_bm4_curved_three_layer_plate,
)


class TestBM1:
    def test_shape_and_keys(self):
        p = make_bm1_water_box((16, 16, 16))
        for k in ("sound_speed", "density", "attenuation", "labels"):
            assert k in p
            assert p[k].shape == (16, 16, 16)

    def test_homogeneous_water(self):
        p = make_bm1_water_box((8, 8, 8))
        np.testing.assert_allclose(np.array(p["sound_speed"]), 1500.0)
        np.testing.assert_allclose(np.array(p["density"]), 1000.0)
        np.testing.assert_allclose(np.array(p["attenuation"]), 0.0)


class TestBM2:
    def test_plate_thickness_matches_spec(self):
        p = make_bm2_single_layer_plate((32, 32, 32), dx=0.001,
                                         plate_thickness_m=0.007)
        # Bone label = 1, exactly 7 mm at dx=1mm = 7 voxels through axis 2.
        bone = np.array(p["labels"]) == 1
        col = bone[16, 16, :]
        assert int(col.sum()) == 7
        # Plate is centred: 7 voxels straddle the middle slice
        assert col[16] or col[15]

    def test_bone_speed_is_cortical(self):
        p = make_bm2_single_layer_plate((32, 32, 32), dx=0.001)
        bone_mask = np.array(p["labels"]) == 1
        c_bone = np.array(p["sound_speed"])[bone_mask]
        np.testing.assert_allclose(c_bone, 2800.0)
        # Water everywhere else
        c_water = np.array(p["sound_speed"])[~bone_mask]
        np.testing.assert_allclose(c_water, 1500.0)


class TestBM3:
    def test_three_layers_present(self):
        p = make_bm3_three_layer_plate((32, 32, 64), dx=0.001)
        labels = np.array(p["labels"])
        assert set(np.unique(labels).tolist()) == {0, 1, 2}

    def test_layer_ordering_outer_diploe_inner(self):
        """Going through the plate along its normal should go
        water -> cortical -> diploe -> cortical -> water."""
        p = make_bm3_three_layer_plate((32, 32, 64), dx=0.001)
        col = np.array(p["labels"])[16, 16, :]
        # Find transitions
        transitions = col[:-1] != col[1:]
        segments = []
        cur = int(col[0])
        for i, t in enumerate(transitions):
            if t:
                segments.append(cur)
                cur = int(col[i + 1])
        segments.append(cur)
        assert segments == [0, 1, 2, 1, 0]

    def test_speeds_match_bm3_spec(self):
        p = make_bm3_three_layer_plate((32, 32, 32), dx=0.001)
        c = np.array(p["sound_speed"])
        labels = np.array(p["labels"])
        np.testing.assert_allclose(c[labels == 0], 1500.0)
        np.testing.assert_allclose(c[labels == 1], 2800.0)
        np.testing.assert_allclose(c[labels == 2], 2300.0)


class TestBM4:
    def test_curved_plate_has_all_three_layers(self):
        p = make_bm4_curved_three_layer_plate(
            (64, 64, 64), dx=0.002, skull_radius_m=0.085,
            centre_offset_m=(0.0, 0.0, 0.0),
        )
        labels = np.array(p["labels"])
        assert set(np.unique(labels).tolist()).issuperset({0, 1, 2}), (
            f"expected water + cortical + trabecular, got "
            f"{sorted(np.unique(labels).tolist())}"
        )

    def test_curvature_present(self):
        """Bone voxels should not all live on a single axis-aligned slab."""
        p = make_bm4_curved_three_layer_plate(
            (64, 64, 64), dx=0.002, centre_offset_m=(0.0, 0.0, 0.0),
        )
        labels = np.array(p["labels"])
        bone = (labels == 1) | (labels == 2)
        # Slice z-column positions of bone voxels — should span more than
        # the flat-plate thickness (~7 mm / 2 mm = 3-4 voxels) because the
        # sphere intersects the grid at varying z.
        bone_idx = np.argwhere(bone)
        z_span = bone_idx[:, 2].max() - bone_idx[:, 2].min()
        assert z_span >= 5, f"plate looks flat, not curved (z span {z_span})"
