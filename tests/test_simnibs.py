"""Tests for SimNIBS / CHARM head-model loader."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.phantoms.simnibs import (
    SIMNIBS_LABEL_NAMES,
    find_simnibs_tissue_labeling,
    load_simnibs_acoustic,
    load_simnibs_segmentation,
    map_simnibs_labels_to_acoustic,
    simnibs_acoustic_table,
)


class TestSimNIBSLabelTable:
    def test_charm_v1_labels_present(self):
        """The CHARM v1 11-tissue labels should all have an entry."""
        expected = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11}
        assert set(SIMNIBS_LABEL_NAMES.keys()) == expected

    def test_label_4_intentionally_unused(self):
        assert 4 not in SIMNIBS_LABEL_NAMES

    def test_compact_vs_spongy_bone_distinguished(self):
        t = simnibs_acoustic_table()
        assert t[7]["sound_speed"] == 2800.0   # compact = cortical
        assert t[8]["sound_speed"] == 2300.0   # spongy = trabecular
        assert t[8]["attenuation"] > t[7]["attenuation"]  # spongy loses more

    def test_air_pockets_water_by_default(self):
        t = simnibs_acoustic_table(use_air=False)
        assert t[11]["sound_speed"] == 1500.0

    def test_air_pockets_real_air_when_requested(self):
        t = simnibs_acoustic_table(use_air=True)
        assert t[11]["sound_speed"] == 343.0


class TestMapLabels:
    def test_mixed_labels(self):
        # 1 = WM, 7 = compact bone, 0 = background
        labels = np.array([[1, 7], [0, 2]], dtype=np.int32)
        props = map_simnibs_labels_to_acoustic(labels)
        c = np.array(props["sound_speed"])
        assert c[0, 0] == 1560.0    # WM
        assert c[0, 1] == 2800.0    # compact bone
        assert c[1, 0] == 1500.0    # background -> water
        assert c[1, 1] == 1560.0    # GM

    def test_override_beats_default(self):
        labels = np.array([1], dtype=np.int32)
        props = map_simnibs_labels_to_acoustic(
            labels,
            properties={1: {"sound_speed": 1700.0, "density": 1100.0, "attenuation": 1.5}},
        )
        assert float(props["sound_speed"][0]) == 1700.0


class TestFilesystemLayout:
    def test_finds_standard_filename(self, tmp_path):
        m2m = tmp_path / "m2m_subj01"
        m2m.mkdir()
        (m2m / "tissue_labeling.nii.gz").touch()
        found = find_simnibs_tissue_labeling(m2m)
        assert found.name == "tissue_labeling.nii.gz"

    def test_finds_upsampled_variant(self, tmp_path):
        m2m = tmp_path / "m2m_subj02"
        (m2m / "label_prep").mkdir(parents=True)
        (m2m / "label_prep" / "tissue_labeling_upsampled.nii.gz").touch()
        found = find_simnibs_tissue_labeling(m2m)
        assert found.name == "tissue_labeling_upsampled.nii.gz"

    def test_missing_raises(self, tmp_path):
        m2m = tmp_path / "m2m_empty"
        m2m.mkdir()
        with pytest.raises(FileNotFoundError, match="SimNIBS tissue-labeling"):
            find_simnibs_tissue_labeling(m2m)


class TestSegmentationLoader:
    def test_nifti_roundtrip(self, tmp_path):
        import nibabel as nib
        m2m = tmp_path / "m2m_fake"
        m2m.mkdir()
        labels = np.zeros((6, 6, 6), dtype=np.int16)
        labels[1:5, 1:5, 1:5] = 2  # gray matter interior
        labels[2:4, 2:4, 2:4] = 1  # white matter core
        img = nib.Nifti1Image(labels, affine=np.eye(4))
        nib.save(img, m2m / "tissue_labeling.nii.gz")

        out = load_simnibs_acoustic(m2m)
        assert out["labels"].shape == (6, 6, 6)
        assert out["sound_speed"].shape == (6, 6, 6)
        c = np.array(out["sound_speed"])
        # Core is WM (1) -> 1560
        assert c[3, 3, 3] == 1560.0
        # Outside the volume is background (0) -> water 1500
        assert c[0, 0, 0] == 1500.0

    def test_direct_file_path_also_works(self, tmp_path):
        """Passing a direct file path should skip m2m directory search."""
        import nibabel as nib
        labels = np.ones((4, 4, 4), dtype=np.int16) * 7  # compact bone
        path = tmp_path / "my_labels.nii.gz"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), path)

        out = load_simnibs_acoustic(path)
        np.testing.assert_allclose(np.array(out["sound_speed"]), 2800.0)
