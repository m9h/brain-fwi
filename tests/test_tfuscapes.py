"""Tests for TFUScapes loader and CT-to-acoustic mapping.

The actual TFUScapes sample at /data/datasets/tfuscapes/ is optional —
all tests here synthesise tiny CT volumes in memory so the suite runs
without the dataset on disk.
"""

import io
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.phantoms.tfuscapes import (
    ct_to_acoustic,
    discover_tfuscapes_samples,
    head_mask_from_ct,
    load_tfuscapes_sample,
)


class TestCTToAcoustic:
    def test_water_at_zero_hu(self):
        """Default background='water' → HU=0 voxels map to water."""
        ct = np.zeros((4, 4, 4), dtype=np.float32)
        props = ct_to_acoustic(ct, background="water")
        np.testing.assert_allclose(np.array(props["sound_speed"]), 1500.0)
        np.testing.assert_allclose(np.array(props["density"]), 1000.0)
        np.testing.assert_allclose(np.array(props["attenuation"]), 0.0)

    def test_cortical_bone_at_1500_hu(self):
        ct = np.full((4, 4, 4), 1500.0, dtype=np.float32)
        props = ct_to_acoustic(ct)
        np.testing.assert_allclose(np.array(props["sound_speed"]), 2800.0)
        np.testing.assert_allclose(np.array(props["density"]), 1850.0)
        np.testing.assert_allclose(np.array(props["attenuation"]), 4.0)

    def test_trabecular_bone_at_1000_hu(self):
        ct = np.full((4, 4, 4), 1000.0, dtype=np.float32)
        props = ct_to_acoustic(ct, background="interp")
        np.testing.assert_allclose(np.array(props["sound_speed"]), 2300.0)
        np.testing.assert_allclose(np.array(props["density"]), 1700.0)
        np.testing.assert_allclose(np.array(props["attenuation"]), 8.0)

    def test_monotonic_c_rho_on_ramp(self):
        """Sound speed and density should be monotonically non-decreasing
        along a HU ramp from 0 to 1500."""
        ct = np.linspace(0.0, 1500.0, 32, dtype=np.float32)
        props = ct_to_acoustic(ct.reshape(32, 1, 1), background="interp")
        c = np.array(props["sound_speed"]).ravel()
        rho = np.array(props["density"]).ravel()
        assert np.all(np.diff(c) >= 0), "c should be non-decreasing in HU"
        assert np.all(np.diff(rho) >= 0), "rho should be non-decreasing in HU"

    def test_bounds(self):
        """Output is bounded by the extreme anchors."""
        ct = np.array([-500.0, 0.0, 3000.0, 5000.0], dtype=np.float32)
        props = ct_to_acoustic(ct.reshape(-1, 1, 1), background="interp")
        c = np.array(props["sound_speed"]).ravel()
        assert c.min() >= 1500.0
        assert c.max() <= 2800.0

    def test_water_background_replacement(self):
        """background='water' replaces HU=0 voxels; HU>0 voxels unaffected."""
        ct = np.array([[0.0, 1500.0], [0.0, 1000.0]], dtype=np.float32)
        props_water = ct_to_acoustic(ct, background="water")
        props_interp = ct_to_acoustic(ct, background="interp")
        # HU=0 region differs
        assert float(props_water["sound_speed"][0, 0]) == 1500.0
        # HU=1500 region is the same (water vs interp coincide at cortical)
        np.testing.assert_allclose(
            np.array(props_water["sound_speed"])[ct > 0],
            np.array(props_interp["sound_speed"])[ct > 0],
        )

    def test_air_background_replacement(self):
        ct = np.zeros((2, 2, 2), dtype=np.float32)
        props = ct_to_acoustic(ct, background="air")
        np.testing.assert_allclose(np.array(props["sound_speed"]), 343.0)
        np.testing.assert_allclose(np.array(props["density"]), 1.225)

    def test_unknown_background_raises(self):
        with pytest.raises(ValueError, match="background="):
            ct_to_acoustic(np.zeros((2, 2, 2), dtype=np.float32), background="plasma")


class TestHeadMask:
    def test_mask_excludes_zero_background(self):
        ct = np.array([[0.0, 50.0], [1500.0, 0.0]], dtype=np.float32)
        mask = head_mask_from_ct(ct)
        expected = np.array([[False, True], [True, False]])
        np.testing.assert_array_equal(mask, expected)


class TestDiscoverSamples:
    def test_accepts_either_root_layout(self, tmp_path):
        """Accepts either dataset-root or .../data/ as input."""
        data_dir = tmp_path / "data" / "SUBJ_01"
        data_dir.mkdir(parents=True)
        (data_dir / "exp_0.npz").write_bytes(b"\x00")
        (data_dir / "exp_1.npz").write_bytes(b"\x00")

        samples_via_root = discover_tfuscapes_samples(tmp_path)
        samples_via_data = discover_tfuscapes_samples(tmp_path / "data")
        assert len(samples_via_root) == 2
        assert samples_via_root == samples_via_data


class TestLoadSample:
    def test_roundtrip_on_synthetic_npz(self, tmp_path):
        """Create a tiny synthetic TFUScapes-style npz and load it."""
        subj_dir = tmp_path / "SUBJ_TEST"
        subj_dir.mkdir()
        path = subj_dir / "exp_0.npz"

        rng = np.random.default_rng(42)
        ct = rng.uniform(0.0, 1500.0, size=(8, 8, 8)).astype(np.float32)
        pmap = rng.uniform(0.0, 1000.0, size=(8, 8, 8)).astype(np.float32)
        tr_coords = rng.integers(0, 8, size=(4, 3)).astype(np.int64)
        np.savez(path, ct=ct, pmap=pmap, tr_coords=tr_coords)

        sample = load_tfuscapes_sample(path)

        assert sample["subject_id"] == "SUBJ_TEST"
        assert sample["experiment_id"] == "exp_0"
        np.testing.assert_array_equal(sample["ct"], ct)
        np.testing.assert_array_equal(sample["transducer_positions_grid"], tr_coords.astype(np.int32))
        assert sample["sound_speed"].shape == (8, 8, 8)
        assert sample["density"].shape == (8, 8, 8)

    @pytest.mark.skipif(
        not Path("/data/datasets/tfuscapes/data/A00028185/exp_0.npz").exists(),
        reason="TFUScapes sample not available on this host",
    )
    def test_real_sample_on_disk(self):
        """Sanity-check against the real sample if it's present."""
        path = Path("/data/datasets/tfuscapes/data/A00028185/exp_0.npz")
        sample = load_tfuscapes_sample(path)
        assert sample["ct"].shape == (256, 256, 256)
        # Head region should have non-trivial bone signal
        c = np.array(sample["sound_speed"])
        assert float(c.max()) >= 2500.0
        # Coupling medium is water, not air
        rho = np.array(sample["density"])
        assert float(rho.min()) >= 999.0
