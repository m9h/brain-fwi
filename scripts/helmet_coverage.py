#!/usr/bin/env python
"""Compare the current cap vs the clinical helmet: element layout + interior
transmission-path coverage. Cheap geometric demo (no FWI) of why fuller-azimuth,
denser coverage conditions the inversion better.

Coverage proxy: for a sample of source->receiver straight chords, count how many
pass through each brain voxel (transmission illumination). A good imaging array
covers the interior densely and uniformly.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain_fwi.transducers.helmet import helmet_array_3d, clinical_helmet_3d

CTR = (0.0, 0.0, 0.0)
RAP, RLR, RSI = 0.095, 0.080, 0.100      # skull semi-axes (m)
RB = 0.072                                # brain ellipsoid (scaled-down)


def chord_coverage(pos, n_src=64, grid=64, half=0.11):
    """Count src->recv chords through each voxel of a centred grid; return the
    central axial slice + the fraction of brain voxels covered + uniformity."""
    pos = np.asarray(pos)
    rng = np.random.default_rng(0)
    src = pos[rng.choice(len(pos), size=min(n_src, len(pos)), replace=False)]
    vol = np.zeros((grid, grid, grid), np.float32)
    edges = np.linspace(-half, half, grid + 1)
    t = np.linspace(0, 1, 60)
    for s in src:
        seg = s[None, None, :] * (1 - t)[None, :, None] + pos[:, None, :] * t[None, :, None]
        p = seg.reshape(-1, 3)
        ijk = np.clip(((p + half) / (2 * half) * grid).astype(int), 0, grid - 1)
        np.add.at(vol, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)
    gx, gy, gz = np.meshgrid(*[(edges[:-1] + edges[1:]) / 2] * 3, indexing="ij")
    brain = (gx / RB) ** 2 + (gy / (RB * RLR / RAP)) ** 2 + (gz / (RB * RSI / RAP)) ** 2 < 1
    cov = vol[brain]
    frac = float((cov > 0).mean())
    cov_uniformity = float(cov[cov > 0].std() / (cov[cov > 0].mean() + 1e-9))
    return vol[:, :, grid // 2].T, brain[:, :, grid // 2].T, frac, cov_uniformity


def main():
    cap = np.asarray(helmet_array_3d(n_elements=256, center=CTR, radius_ap=RAP,
                                     radius_lr=RLR, radius_si=RSI, exclude_face=True))
    clin = np.asarray(clinical_helmet_3d(center=CTR, radius_ap=RAP, radius_lr=RLR,
                                         radius_si=RSI, freq=300e3, standoff=0.007))
    arrays = [("Current cap (256 elem)", cap), (f"Clinical helmet ({len(clin)} elem)", clin)]

    fig, ax = plt.subplots(2, 3, figsize=(14, 8.8), facecolor="white")
    for row, (name, pos) in enumerate(arrays):
        # axial (x-y, top view) and sagittal (x-z, side view) element scatter
        ax[row, 0].scatter(pos[:, 0] * 1e3, pos[:, 1] * 1e3, s=8, c="#2a6fdb")
        ax[row, 0].add_patch(plt.matplotlib.patches.Ellipse((0, 0), 2 * RAP * 1e3,
                             2 * RLR * 1e3, fill=False, color="k", lw=1))
        ax[row, 0].set_title(f"{name}\naxial (top view)", fontsize=11)
        ax[row, 0].set_aspect("equal"); ax[row, 0].set_xlabel("anterior → (mm)")
        ax[row, 1].scatter(pos[:, 0] * 1e3, pos[:, 2] * 1e3, s=8, c="#2a6fdb")
        ax[row, 1].add_patch(plt.matplotlib.patches.Ellipse((0, 0), 2 * RAP * 1e3,
                             2 * RSI * 1e3, fill=False, color="k", lw=1))
        ax[row, 1].set_title("sagittal (side view)", fontsize=11)
        ax[row, 1].set_aspect("equal"); ax[row, 1].set_xlabel("anterior → (mm)")
        # interior coverage map
        cov, brain, frac, unif = chord_coverage(pos)
        im = ax[row, 2].imshow(np.where(brain, cov, np.nan), origin="lower", cmap="magma")
        ax[row, 2].set_title(f"interior transmission coverage\n"
                             f"{frac*100:.0f}% of brain covered, CoV {unif:.2f}", fontsize=11)
        ax[row, 2].set_xticks([]); ax[row, 2].set_yticks([])
        fig.colorbar(im, ax=ax[row, 2], fraction=0.046, pad=0.04, label="# chords")
        print(f"{name}: {len(pos)} elem, brain covered {frac*100:.0f}%, CoV {unif:.2f}")
    fig.suptitle("Imaging helmet coverage: current cap vs clinical (Imperial/Guasch-style) helmet",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "results/absorption_aware_fwi_3d/helmet_coverage.png"
    fig.savefig(out, dpi=140, facecolor="white"); print("saved", out)


if __name__ == "__main__":
    main()
