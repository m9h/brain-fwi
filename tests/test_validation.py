"""Tests for FWI reconstruction comparison harness.

Captures the contract for ``brain_fwi.validation.compare``:

  - ``regional_rmse`` returns a {region_name: rmse} dict, computes only
    over voxels matching the region's label(s), and handles empty
    regions gracefully.
  - ``compare_reconstructions`` loads two ``run_full_usct.py`` HDF5
    outputs and returns a comparison dict with per-region RMSE for
    both reconstructions plus global stats.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from brain_fwi.validation.compare import (
    compare_reconstructions,
    regional_rmse,
)


class TestRegionalRMSE:
    def test_single_label_single_region(self):
        true = np.full((4,), 1500.0)
        recon = np.full((4,), 1550.0)
        labels = np.full((4,), 7, dtype=np.int32)
        out = regional_rmse(recon, true, labels, regions={"skull": 7})
        assert out["skull"] == pytest.approx(50.0)

    def test_multiple_regions(self):
        labels = np.array([7, 7, 2, 2], dtype=np.int32)
        true = np.array([1500.0, 1500.0, 1560.0, 1560.0])
        recon = np.array([1520.0, 1520.0, 1570.0, 1570.0])
        out = regional_rmse(
            recon, true, labels, regions={"skull": 7, "grey": 2},
        )
        assert out["skull"] == pytest.approx(20.0)
        assert out["grey"] == pytest.approx(10.0)

    def test_region_as_label_list(self):
        """Region spec can be a list to union multiple label ids."""
        labels = np.array([2, 3, 7], dtype=np.int32)
        true = np.array([1560.0, 1560.0, 2800.0])
        recon = np.array([1570.0, 1570.0, 2800.0])
        out = regional_rmse(recon, true, labels, regions={"brain": [2, 3]})
        assert out["brain"] == pytest.approx(10.0)

    def test_empty_region_returns_nan(self):
        labels = np.full((4,), 7, dtype=np.int32)
        out = regional_rmse(
            np.zeros(4), np.zeros(4), labels, regions={"csf": 1},
        )
        assert np.isnan(out["csf"])

    def test_3d_volume(self):
        """RMSE is computed over masked voxels regardless of ndim."""
        labels = np.zeros((4, 4, 4), dtype=np.int32)
        labels[1:3, 1:3, 1:3] = 7  # 8 skull voxels
        true = np.full((4, 4, 4), 1500.0)
        recon = true.copy()
        recon[1:3, 1:3, 1:3] += 30.0
        out = regional_rmse(recon, true, labels, regions={"skull": 7})
        assert out["skull"] == pytest.approx(30.0)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            regional_rmse(
                np.zeros(4), np.zeros(5),
                np.zeros(4, dtype=np.int32), regions={"any": 0},
            )


class TestCompareReconstructions:
    @staticmethod
    def _write_result(
        path: Path,
        labels: np.ndarray,
        true: np.ndarray,
        recon: np.ndarray,
        final_loss: float,
    ) -> None:
        """Minimal HDF5 shaped like run_full_usct.py output."""
        with h5py.File(path, "w") as f:
            f.create_dataset("labels", data=labels)
            f.create_dataset("velocity_true", data=true)
            f.create_dataset("velocity_recon", data=recon)
            f.create_dataset("loss_history", data=np.array([1.0, 0.5, final_loss]))
            f.attrs["grid_shape"] = list(true.shape)

    def test_loads_two_files_and_compares(self, tmp_path: Path):
        labels = np.array([7, 7, 2, 2, 0, 0], dtype=np.int32)
        true = np.array([1500.0, 1500.0, 1560.0, 1560.0, 343.0, 343.0])

        vox_path = tmp_path / "voxel.h5"
        sir_path = tmp_path / "siren.h5"
        self._write_result(vox_path, labels, true,
                           recon=true + np.array([30, 30, 5, 5, 0, 0]),
                           final_loss=0.10)
        self._write_result(sir_path, labels, true,
                           recon=true + np.array([20, 20, 15, 15, 0, 0]),
                           final_loss=0.12)

        out = compare_reconstructions(
            voxel_path=vox_path, siren_path=sir_path,
            regions={"skull": 7, "brain": [2, 3]},
        )

        assert out["voxel"]["regional_rmse"]["skull"] == pytest.approx(30.0)
        assert out["voxel"]["regional_rmse"]["brain"] == pytest.approx(5.0)
        assert out["siren"]["regional_rmse"]["skull"] == pytest.approx(20.0)
        assert out["siren"]["regional_rmse"]["brain"] == pytest.approx(15.0)
        assert out["voxel"]["final_loss"] == pytest.approx(0.10)
        assert out["siren"]["final_loss"] == pytest.approx(0.12)

    def test_default_regions_cover_brain_and_skull(self, tmp_path: Path):
        """Default region set must at least include skull + brain."""
        labels = np.array([7, 2, 3], dtype=np.int32)
        true = np.array([1500.0, 1560.0, 1560.0])
        recon = true.copy()
        vox = tmp_path / "v.h5"
        sir = tmp_path / "s.h5"
        self._write_result(vox, labels, true, recon, 0.0)
        self._write_result(sir, labels, true, recon, 0.0)
        out = compare_reconstructions(voxel_path=vox, siren_path=sir)
        # at minimum the canonical labels must be reported
        regions = set(out["voxel"]["regional_rmse"].keys())
        assert {"skull", "brain"}.issubset(regions)
