"""Tests for SCI Institute head-model loader."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.phantoms.sci_head import (
    SCI_ACOUSTIC_PROPERTIES,
    SCI_LABEL_NAMES,
    load_sci_head_acoustic,
    load_sci_head_segmentation,
    map_sci_labels_to_acoustic,
)


class TestSCIAcousticTable:
    def test_all_eight_tissues_present(self):
        assert set(SCI_LABEL_NAMES.keys()) == set(range(1, 9))
        for lab in range(1, 9):
            assert lab in SCI_ACOUSTIC_PROPERTIES

    def test_cortical_bone_matches_itrusst(self):
        skull = SCI_ACOUSTIC_PROPERTIES[6]   # label 6 = Skull
        assert skull["sound_speed"] == 2800.0
        assert skull["density"] == 1850.0

    def test_background_is_water_coupling(self):
        bg = SCI_ACOUSTIC_PROPERTIES[8]
        assert bg["sound_speed"] == 1500.0
        assert bg["density"] == 1000.0

    def test_label_names_match_verified_ordering(self):
        """Verified against HeadSegmentation.nrrd by slice inspection
        (scripts/inspect_sci_head.py). See sci_head.py docstring."""
        assert SCI_LABEL_NAMES[1] == "Eyes"
        assert SCI_LABEL_NAMES[2] == "Gray Matter"
        assert SCI_LABEL_NAMES[3] == "White Matter"
        assert SCI_LABEL_NAMES[6] == "Skull"
        assert SCI_LABEL_NAMES[7] == "Scalp"


class TestMapLabels:
    def test_shape_preserved(self):
        labels = np.ones((4, 4, 4), dtype=np.int32) * 2  # GM (verified ordering)
        props = map_sci_labels_to_acoustic(labels)
        assert props["sound_speed"].shape == (4, 4, 4)
        np.testing.assert_allclose(np.array(props["sound_speed"]), 1560.0)

    def test_mixed_labels(self):
        # Using the verified SCI ordering: 2=GM, 3=WM, 6=skull, 8=background
        labels = np.array([[2, 3], [6, 8]], dtype=np.int32)
        props = map_sci_labels_to_acoustic(labels)
        c = np.array(props["sound_speed"])
        assert c[0, 0] == 1560.0  # GM
        assert c[0, 1] == 1560.0  # WM
        assert c[1, 0] == 2800.0  # skull
        assert c[1, 1] == 1500.0  # background -> water

    def test_override_properties(self):
        labels = np.array([6], dtype=np.int32)  # skull
        custom = {6: {"sound_speed": 2500.0, "density": 1700.0, "attenuation": 3.0}}
        props = map_sci_labels_to_acoustic(labels, properties=custom)
        assert float(props["sound_speed"][0]) == 2500.0


class TestLoaderSyntheticNRRD:
    def test_roundtrip(self, tmp_path):
        pytest.importorskip("nrrd")
        import nrrd
        labels = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 1]]], dtype=np.uint8)
        path = tmp_path / "HeadSegmentation.nrrd"
        nrrd.write(str(path), labels)

        loaded = load_sci_head_segmentation(path)
        np.testing.assert_array_equal(loaded, labels.astype(np.int32))


class TestRealSCIHead:
    """Uses the real SCI head model at /data/datasets/sci_head_model/ if present."""

    _path = Path("/data/datasets/sci_head_model/segmentation/HeadSegmentation.nrrd")

    @pytest.mark.skipif(not _path.exists(),
                        reason="SCI head NRRD not available on this host")
    def test_real_segmentation_loads(self):
        labels = load_sci_head_segmentation(self._path)
        # Shape from the README: 208 x 256 x 256
        assert labels.shape == (208, 256, 256)
        # 8 tissue labels plus possibly a 0 background
        u = set(np.unique(labels).tolist())
        assert u.issubset(set(range(0, 9)))
        assert u.intersection({1, 2, 3, 4, 5, 6, 7, 8}) == set(range(1, 9))

    @pytest.mark.skipif(not _path.exists(),
                        reason="SCI head NRRD not available on this host")
    def test_real_segmentation_acoustic(self):
        result = load_sci_head_acoustic(self._path)
        assert "labels" in result
        c = np.array(result["sound_speed"])
        # At least some voxels should be cortical bone
        assert float(c.max()) >= 2500.0
        # Coupling medium is water
        assert float(c.min()) >= 1500.0 - 1e-3
