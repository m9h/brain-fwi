#!/usr/bin/env python
"""Render presentation-grade headline figures from saved 3D-FWI npz results.

Each npz (written by examples/06_absorption_aware_fwi_3d.py) holds c_true,
c_init, c_aware, c_lossless, labels, roi, rmse, misfit. This script reproduces
the single-phantom headline (truth | absorption-aware | lossless + an error
bar) and a combined multi-row title-slide figure.

Usage:
    python scripts/make_headline.py --single results/.../mida96.npz out.png \
        --title "..." --subtitle "..." --vmin 1490 --vmax 1600 --bar brain
    python scripts/make_headline.py --combined          # the title-slide figure
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("results/absorption_aware_fwi_3d")
FOOTER = ("JAX autodiff through j-Wave  ·  k-Wave-validated absorption  ·  "
          "96³ free-local (192³ on B200)")


def _slice(d, use_lesion):
    lab = d["labels"]
    les = (lab == 1)
    if use_lesion and les.any():
        zc = int(np.argmax(les.sum(axis=(0, 1))))
    else:
        roi = (lab == 2) | np.isin(lab, (3, 4))
        zc = int(np.argmax(roi.sum(axis=(0, 1))))
    return zc, les


def _draw(ax, img2d, skull2d, lesm2d, title, vmin, vmax):
    im = ax.imshow(img2d, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.contour(skull2d, [0.5], colors="white", linewidths=1.0)
    if lesm2d is not None and lesm2d.any():
        ax.contour(lesm2d, [0.5], colors="red", linewidths=1.5)
    ax.set_title(title, fontsize=13, pad=6); ax.set_xticks([]); ax.set_yticks([])
    return im


def _bar(ax, vals, ylab):
    bars = ax.bar(["start", "ignore\nabsorption", "absorption\n-aware"], vals,
                  color=["#9e9e9e", "#d64550", "#2a9d3f"], width=0.66)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f}",
                ha="center", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.2); ax.tick_params(labelsize=8.5)


def _bar_vals(d, bar_metric):
    if bar_metric == "lesion":
        ct = d["c_true"]; les = (d["labels"] == 1)
        lr = lambda c: float(np.sqrt(np.mean((c[les] - ct[les]) ** 2)))
        return [lr(d["c_init"]), lr(d["c_lossless"]), lr(d["c_aware"])], "lesion RMSE (m/s) ↓"
    r0, rl, ra = [float(x) for x in d["rmse"]]
    return [r0, rl, ra], "brain RMSE (m/s) ↓"


def render_single(npz, out, title, subtitle, vmin, vmax, use_lesion, bar_metric):
    d = np.load(npz); zc, les = _slice(d, use_lesion)
    sl = lambda a: a[:, :, zc].T
    skull = sl(d["labels"] == 5).astype(float); lesm = sl(les).astype(float)
    ra, rl = float(d["rmse"][2]), float(d["rmse"][1])
    fig = plt.figure(figsize=(16.5, 5.6), facecolor="white")
    sfs = fig.subfigures(1, 2, width_ratios=[3.25, 0.82], wspace=0.02)
    axs = sfs[0].subplots(1, 3)
    tt = ["Ground truth", f"Absorption-aware FWI\nbrain RMSE {ra:.1f} m/s",
          f"Ignoring absorption\nbrain RMSE {rl:.1f} m/s"]
    for a, img, t in zip(axs, [d["c_true"], d["c_aware"], d["c_lossless"]], tt):
        im = _draw(a, sl(img), skull, lesm, t, vmin, vmax)
    cb = sfs[0].colorbar(im, ax=axs, location="right", fraction=0.035, pad=0.02)
    cb.set_label("sound speed (m/s)", fontsize=10)
    vals, ylab = _bar_vals(d, bar_metric)
    _bar(sfs[1].subplots(1, 1), vals, ylab)
    fig.suptitle(title, fontsize=15, y=1.08)
    fig.text(0.5, 1.0, subtitle, ha="center", fontsize=12.5, style="italic")
    fig.text(0.5, -0.04, FOOTER, ha="center", fontsize=10.5, color="#444")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out)


# The three demos, with display windows tuned to each brain's velocity range.
SPECS = [
    dict(npz=RES / "clean96_headline.npz", name="Synthetic anatomical head",
         vmin=1500, vmax=1620, use_lesion=True, bar="brain"),
    dict(npz=RES / "mida96.npz", name="MIDA ITRUSST head (real anatomy)",
         vmin=1490, vmax=1600, use_lesion=False, bar="brain"),
    dict(npz=RES / "birnbaum96.npz", name="Birnbaum patient (real + stroke lesion)",
         vmin=1500, vmax=1645, use_lesion=True, bar="lesion"),
]


def render_combined(out):
    specs = [s for s in SPECS if Path(s["npz"]).exists()]
    n = len(specs)
    fig = plt.figure(figsize=(15, 4.7 * n), facecolor="white")
    sfs = fig.subfigures(n, 1, hspace=0.14)
    if n == 1:
        sfs = [sfs]
    for sf, s in zip(sfs, specs):
        d = np.load(s["npz"]); zc, les = _slice(d, s["use_lesion"])
        sl = lambda a: a[:, :, zc].T
        skull = sl(d["labels"] == 5).astype(float); lesm = sl(les).astype(float)
        ra, rl = float(d["rmse"][2]), float(d["rmse"][1])
        vals, _ = _bar_vals(d, s["bar"])
        gain = 100.0 * (vals[1] - vals[2]) / vals[1]
        axs = sf.subplots(1, 4, width_ratios=[1, 1, 1, 0.7])
        tt = ["Ground truth", f"Absorption-aware ({s['bar']} RMSE {vals[2]:.0f})",
              f"Ignoring absorption ({vals[1]:.0f})"]
        for a, img, t in zip(axs[:3], [d["c_true"], d["c_aware"], d["c_lossless"]], tt):
            im = _draw(a, sl(img), skull, lesm, t, s["vmin"], s["vmax"])
        _bar(axs[3], vals, f"{s['bar']} RMSE ↓")
        sf.suptitle(f"{s['name']}   —   absorption-aware +{gain:.0f}%",
                    fontsize=13, x=0.5, y=0.97, fontweight="bold")
    fig.suptitle("Differentiable 3D FWI through a known skull: modeling absorption is essential",
                 fontsize=16, y=1.035)
    fig.text(0.5, 0.005, FOOTER, ha="center", fontsize=10.5, color="#444")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", action="store_true")
    ap.add_argument("--single", nargs=2, metavar=("NPZ", "OUT"))
    ap.add_argument("--title", default=""); ap.add_argument("--subtitle", default="")
    ap.add_argument("--vmin", type=float, default=1500); ap.add_argument("--vmax", type=float, default=1620)
    ap.add_argument("--lesion", action="store_true"); ap.add_argument("--bar", default="brain")
    args = ap.parse_args()
    if args.single:
        render_single(args.single[0], args.single[1], args.title, args.subtitle,
                      args.vmin, args.vmax, args.lesion, args.bar)
    if args.combined or not args.single:
        render_combined(RES / "HEADLINE_combined.png")


if __name__ == "__main__":
    main()
