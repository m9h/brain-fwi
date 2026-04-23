#!/usr/bin/env python
"""Verify the SCI Institute head-model label ordering.

The distributed ``HeadSegmentation.nrrd`` has 8 integer labels (1-8)
but the original README does not document the mapping. This script
loads the segmentation, reports per-label voxel counts and
bounding-box geometry, and computes each label's fraction of voxels
adjacent to the exterior background — the outermost tissue has the
highest contact fraction.

Combined with a mid-slice visualisation (saved as PNG), this lets us
verify the label ordering used by ``phantoms/sci_head.py`` against
the actual data.

Usage::

    uv run python scripts/inspect_sci_head.py \\
        /data/datasets/sci_head_model/segmentation/HeadSegmentation.nrrd \\
        /data/datasets/brain-fwi/sci_head_inspection.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nrrd_path", type=Path,
                        help="SCI HeadSegmentation.nrrd path")
    parser.add_argument("output_png", type=Path,
                        help="Destination for the per-axis slice PNG")
    args = parser.parse_args()

    import numpy as np
    import nrrd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from scipy.ndimage import binary_dilation

    data, _ = nrrd.read(str(args.nrrd_path))
    sx, sy, sz = data.shape
    print(f"Shape: {data.shape}, dtype: {data.dtype}, total voxels: {data.size:,}")

    print(f"\n{'Label':>5}  {'Count':>10}  {'Frac':>6}  {'Centroid':<24}  {'Bbox':<14}  {'Bg-contact'}")
    print("-" * 90)
    for lab in sorted(int(x) for x in np.unique(data)):
        mask = data == lab
        n = int(mask.sum())
        frac = n / data.size
        idx = np.argwhere(mask)
        centroid = idx.mean(axis=0)
        bbox = idx.max(axis=0) - idx.min(axis=0)
        if lab == 0 or n == 0:
            contact = "-"
        else:
            touching = binary_dilation(mask) & (data == 8)
            contact = f"{int(touching.sum()):>7} ({100*int(touching.sum())/n:4.1f}%)"
        print(f"{lab:>5}  {n:>10,}  {frac:>6.2%}  "
              f"({centroid[0]:5.1f},{centroid[1]:5.1f},{centroid[2]:5.1f})  "
              f"{bbox[0]:>3}x{bbox[1]:>3}x{bbox[2]:>3}  {contact}")

    # Visualisation — 9-colour palette so labels are easy to distinguish.
    colors = [
        (0, 0, 0, 0),          # 0 transparent
        (1, 0, 0),              # 1 red    — eyes
        (0.7, 0.7, 1.0),        # 2 light purple — GM
        (0.5, 0.5, 0.85),       # 3 deeper purple — WM
        (0.0, 0.85, 1.0),       # 4 cyan   — CSF
        (0.25, 0.25, 0.25),     # 5 dark   — sinus/air
        (1.0, 1.0, 0.7),        # 6 yellow — skull
        (0.9, 0.65, 0.55),      # 7 skin   — scalp
        (0.0, 0.0, 0.15),       # 8 ~black — background
    ]
    cmap = ListedColormap(colors)

    slices = [
        (f"Sagittal (x={sx//2})", data[sx // 2, :, :]),
        (f"Coronal  (y={sy//2})", data[:, sy // 2, :]),
        (f"Axial    (z={sz//2})", data[:, :, sz // 2]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, img) in zip(axes, slices):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=8, origin="lower")
        ax.set_title(title)
        ax.axis("off")

    names = ["Eyes", "Gray Matter", "White Matter", "CSF",
             "Sinus", "Skull", "Scalp", "Background"]
    patches = [Patch(color=colors[i + 1], label=f"{i+1} {names[i]}")
               for i in range(8)]
    fig.legend(handles=patches, loc="center right",
               bbox_to_anchor=(1.02, 0.5), title="Label")
    plt.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(args.output_png), dpi=120, bbox_inches="tight")
    print(f"\nSaved slice figure -> {args.output_png}")


if __name__ == "__main__":
    main()
