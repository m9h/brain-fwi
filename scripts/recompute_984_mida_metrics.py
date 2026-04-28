#!/usr/bin/env python
"""Re-compute regional reconstruction metrics for job 984 with MIDA-aware truth.

Job 984's stdout report card used `validation/compare.py`, which looks
up "True c" via BrainWeb canonical values per label. MIDA uses different
label IDs (Skull=40, Brain Gray Matter=10, …) so every "True c" in the
984 report is wrong. The H5 itself does the right thing — it stores the
actual phantom velocity at `velocity_true` and the integer label map
at `labels`. We just need to compute regional stats against those.

Output: prints a corrected table and writes a JSON sidecar at
``brain_usct_192_mida_voxel_984_metrics_mida_aware.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from brain_fwi.phantoms.mida import (
    MIDA_LABEL_NAMES,
    MIDA_LABEL_TO_GROUP,
)


# Acoustic groups grouped further into reporting buckets ordered to
# match the original 984 report card so the diff is obvious.
_REPORT_GROUPS: dict[str, list[str]] = {
    "Brain (GM + WM)":        ["grey_matter", "white_matter"],
    "Grey matter":            ["grey_matter"],
    "White matter":           ["white_matter"],
    "CSF":                    ["csf"],
    "Skull (cortical)":       ["cortical_bone"],
    "Skull (trabecular)":     ["trabecular_bone"],
    "Skull (all bone)":       ["cortical_bone", "trabecular_bone"],
    "Scalp / skin":           ["skin"],
    "Soft tissue (muscle/fat)": ["muscle", "fat", "connective"],
    "Eye":                    ["eye"],
    "Vessels":                ["blood_vessels"],
    "Coupling (water)":       ["water"],
    "Air cavities":           ["air"],
    "Lesion (BrainWeb 8)":    ["__lesion__"],
}

# Lesion isn't a MIDA group — it's a label-8 voxel injected by
# `make_mida_phantom` per `phantoms/mida.py`. Treat as its own bucket.
_LESION_LABEL = 8


def _label_set(group_names: list[str]) -> set[int]:
    if group_names == ["__lesion__"]:
        return {_LESION_LABEL}
    return {
        lab for lab, g in MIDA_LABEL_TO_GROUP.items()
        if g in group_names
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--h5", type=Path,
        default=Path(
            "/data/datasets/brain-fwi/brain_usct_192_mida_voxel_984.h5"
        ),
        help="FWI run output H5 (must contain velocity_true, velocity_recon, labels)",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.h5.exists():
        print(f"Not found: {args.h5}", file=sys.stderr)
        return 1

    with h5py.File(str(args.h5), "r") as f:
        v_true = np.asarray(f["velocity_true"], dtype=np.float32)
        v_init = np.asarray(f["velocity_init"], dtype=np.float32)
        v_recon = np.asarray(f["velocity_recon"], dtype=np.float32)
        labels = np.asarray(f["labels"], dtype=np.int32)
        n_iters = int(f.attrs.get("n_iters_per_band", 0))
        n_bands = len(eval(str(f.attrs.get("freq_bands_hz", "[]"))))
        wall_min = float(f.attrs.get("total_time_s", 0)) / 60.0

    print("=" * 78)
    print("  MIDA-aware regional metrics — job 984 (MIDA 192^3 voxel FWI)")
    print("=" * 78)
    print(f"  Source:      {args.h5}")
    print(f"  Grid:        {labels.shape}")
    print(f"  Bands:       {n_bands} × {n_iters} iters each")
    print(f"  Wall time:   {wall_min:.1f} min")
    print(f"  Velocity range:")
    print(f"    init:      [{v_init.min():.0f}, {v_init.max():.0f}] m/s")
    print(f"    true:      [{v_true.min():.0f}, {v_true.max():.0f}] m/s")
    print(f"    recon:     [{v_recon.min():.0f}, {v_recon.max():.0f}] m/s")

    print(f"\n  {'Region':<28} {'voxels':>9} {'true c':>8} {'recon c':>9} "
          f"{'init RMSE':>10} {'recon RMSE':>11} {'improve':>8}")
    print("  " + "-" * 76)

    rows: list[dict] = []
    for region_name, group_names in _REPORT_GROUPS.items():
        lab_set = _label_set(group_names)
        mask = np.isin(labels, list(lab_set))
        n = int(mask.sum())
        if n == 0:
            continue
        c_true = float(v_true[mask].mean())
        c_recon = float(v_recon[mask].mean())
        rmse_init = float(np.sqrt(np.mean((v_init[mask] - v_true[mask]) ** 2)))
        rmse_recon = float(np.sqrt(np.mean((v_recon[mask] - v_true[mask]) ** 2)))
        improve = (
            (rmse_init - rmse_recon) / rmse_init * 100.0
            if rmse_init > 1e-6 else 0.0
        )
        print(f"  {region_name:<28} {n:>9d} {c_true:>8.0f} {c_recon:>9.0f} "
              f"{rmse_init:>10.1f} {rmse_recon:>11.1f} {improve:>+7.1f}%")
        rows.append({
            "region": region_name,
            "n_voxels": n,
            "c_true_mean": c_true,
            "c_recon_mean": c_recon,
            "rmse_init": rmse_init,
            "rmse_recon": rmse_recon,
            "improve_pct": improve,
        })

    # Whole-head and inside-head summary
    print("  " + "-" * 76)
    head_mask = labels != 50  # MIDA "Background" → water coupling
    n_head = int(head_mask.sum())
    rmse_head = float(np.sqrt(np.mean(
        (v_recon[head_mask] - v_true[head_mask]) ** 2
    )))
    rmse_head_init = float(np.sqrt(np.mean(
        (v_init[head_mask] - v_true[head_mask]) ** 2
    )))
    print(f"  {'Head (all non-background)':<28} {n_head:>9d} "
          f"{float(v_true[head_mask].mean()):>8.0f} "
          f"{float(v_recon[head_mask].mean()):>9.0f} "
          f"{rmse_head_init:>10.1f} {rmse_head:>11.1f} "
          f"{(rmse_head_init - rmse_head) / max(rmse_head_init, 1e-6) * 100:>+7.1f}%")

    summary = {
        "h5": str(args.h5),
        "grid_shape": list(labels.shape),
        "wall_time_min": wall_min,
        "velocity_range_m_s": {
            "init": [float(v_init.min()), float(v_init.max())],
            "true": [float(v_true.min()), float(v_true.max())],
            "recon": [float(v_recon.min()), float(v_recon.max())],
        },
        "regions": rows,
        "head_overall_rmse_init": rmse_head_init,
        "head_overall_rmse_recon": rmse_head,
    }

    out = args.json_out or args.h5.with_name(
        args.h5.stem + "_metrics_mida_aware.json"
    )
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  JSON summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
