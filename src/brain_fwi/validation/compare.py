"""Compare two FWI reconstructions (e.g., voxel vs SIREN) by region.

Consumes the HDF5 output produced by ``run_full_usct.py``:

    /labels            — tissue label volume
    /velocity_true     — ground-truth sound speed
    /velocity_recon    — reconstructed sound speed
    /loss_history      — per-iteration loss values
    attrs: grid_shape, dx_m, n_elements, freq_bands_hz, ...

Regions are given as ``{name: label or [labels]}``; per-region RMSE is
computed over voxels whose label matches. Empty regions return NaN
(explicit signal to the caller, not a silent zero).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

import h5py
import numpy as np


# BrainWeb label convention (see src/brain_fwi/phantoms/properties.py).
DEFAULT_REGIONS: Dict[str, Union[int, List[int]]] = {
    "skull": 7,
    "brain": [2, 3],
    "csf": 1,
    "scalp": 6,
}


def regional_rmse(
    recon: np.ndarray,
    true: np.ndarray,
    labels: np.ndarray,
    regions: Mapping[str, Union[int, List[int]]],
) -> Dict[str, float]:
    """Per-region RMSE between ``recon`` and ``true``.

    Args:
        recon: Reconstructed scalar field.
        true: Ground-truth scalar field. Must match ``recon.shape``.
        labels: Integer tissue labels. Must match ``recon.shape``.
        regions: {name: label or [labels]} mapping.

    Returns:
        ``{name: rmse}``. Empty regions return ``nan``.
    """
    if recon.shape != true.shape:
        raise ValueError(
            f"shape mismatch: recon {recon.shape} vs true {true.shape}"
        )
    if recon.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: recon {recon.shape} vs labels {labels.shape}"
        )

    diff_sq = (recon.astype(np.float64) - true.astype(np.float64)) ** 2

    out: Dict[str, float] = {}
    for name, spec in regions.items():
        label_set = [spec] if isinstance(spec, int) else list(spec)
        mask = np.isin(labels, label_set)
        n = int(mask.sum())
        if n == 0:
            out[name] = float("nan")
        else:
            out[name] = float(np.sqrt(diff_sq[mask].mean()))
    return out


def _load_result(path: Path) -> Dict[str, Any]:
    """Pull the fields needed for comparison out of an HDF5 result file."""
    with h5py.File(str(path), "r") as f:
        return {
            "labels": np.asarray(f["labels"]),
            "velocity_true": np.asarray(f["velocity_true"]),
            "velocity_recon": np.asarray(f["velocity_recon"]),
            "loss_history": np.asarray(f["loss_history"]),
            "attrs": dict(f.attrs),
        }


def compare_reconstructions(
    voxel_path: Path,
    siren_path: Path,
    regions: Union[Mapping[str, Union[int, List[int]]], None] = None,
) -> Dict[str, Any]:
    """Compare two ``run_full_usct.py`` HDF5 outputs.

    Args:
        voxel_path: Path to the voxel-path reconstruction file.
        siren_path: Path to the SIREN-path reconstruction file.
        regions: Region spec. ``None`` → ``DEFAULT_REGIONS``.

    Returns:
        Dict with structure::

            {
              "voxel":  {"regional_rmse": {...}, "final_loss": float,
                          "global_rmse": float},
              "siren":  {"regional_rmse": {...}, "final_loss": float,
                          "global_rmse": float},
              "regions": {...},    # the spec actually used
            }
    """
    regions = dict(DEFAULT_REGIONS if regions is None else regions)

    def _summarise(result: Dict[str, Any]) -> Dict[str, Any]:
        recon = result["velocity_recon"]
        true = result["velocity_true"]
        labels = result["labels"]
        return {
            "regional_rmse": regional_rmse(recon, true, labels, regions),
            "global_rmse": float(
                np.sqrt(np.mean((recon.astype(np.float64) - true.astype(np.float64)) ** 2))
            ),
            "final_loss": float(result["loss_history"][-1]),
        }

    return {
        "voxel": _summarise(_load_result(Path(voxel_path))),
        "siren": _summarise(_load_result(Path(siren_path))),
        "regions": regions,
    }


def format_comparison(comparison: Dict[str, Any]) -> str:
    """Pretty-print a comparison dict as an aligned text table."""
    regions = list(comparison["regions"].keys())
    header = f"{'region':<10}  {'voxel RMSE':>12}  {'siren RMSE':>12}  {'Δ (siren−voxel)':>20}"
    lines = [header, "-" * len(header)]
    for r in regions:
        v = comparison["voxel"]["regional_rmse"][r]
        s = comparison["siren"]["regional_rmse"][r]
        delta = (s - v) if (np.isfinite(v) and np.isfinite(s)) else float("nan")
        lines.append(f"{r:<10}  {v:>12.2f}  {s:>12.2f}  {delta:>+20.2f}")
    lines.append("-" * len(header))
    lines.append(
        f"{'global':<10}  "
        f"{comparison['voxel']['global_rmse']:>12.2f}  "
        f"{comparison['siren']['global_rmse']:>12.2f}"
    )
    lines.append(
        f"{'final loss':<10}  "
        f"{comparison['voxel']['final_loss']:>12.6f}  "
        f"{comparison['siren']['final_loss']:>12.6f}"
    )
    return "\n".join(lines)
