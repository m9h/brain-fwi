#!/usr/bin/env python
"""Funding-push demo: a real head imaged with the clinical helmet, all sensors
FLUSH to the scalp (conformal, ~1-voxel coupling). Produces (1) the device
geometry on the real head and (2) the full 3-plane reconstruction.

  python scripts/funding_helmet_demo.py --geometry-only     # fast geometry render
  python scripts/funding_helmet_demo.py                      # geometry + FWI recon
"""
from __future__ import annotations
import argparse, importlib.util, time
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

spec = importlib.util.spec_from_file_location("ex06", "examples/06_absorption_aware_fwi_3d.py")
ex06 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex06)
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data, _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi
from brain_fwi.robustness import brain_roi_rmse

ap = argparse.ArgumentParser()
ap.add_argument("--phantom", choices=["mida", "birnbaum", "synthetic"], default="mida")
ap.add_argument("--n", type=int, default=96)
ap.add_argument("--n-elem", type=int, default=600)
ap.add_argument("--geometry-only", action="store_true")
ap.add_argument("--pub", action="store_true",
                help="192^3 publication preset: 3 bands, 14 iters, 1024 flush elements")
ap.add_argument("--mida-path", default=None, help="override MIDA .nii path (for cloud)")
args = ap.parse_args()
N, n_elem = args.n, args.n_elem
bands = [(50e3, 100e3), (100e3, 160e3)]; n_iters, shots = 12, 10
if args.pub:                                  # publication-quality 192^3 preset
    if args.n == 96: N = 192
    if args.n_elem == 600: n_elem = 1024
    bands = [(50e3, 100e3), (100e3, 180e3), (180e3, 280e3)]; n_iters, shots = 14, 12
RES = "results/absorption_aware_fwi_3d"

if args.phantom == "mida":
    c_true, rho_true, alpha_true, labels, dx = (
        ex06.load_mida_head(N, args.mida_path) if args.mida_path else ex06.load_mida_head(N))
elif args.phantom == "birnbaum":
    c_true, rho_true, alpha_true, labels, dx = ex06.load_head(N, 0)
else:
    c_true, rho_true, alpha_true, labels, dx = ex06.synthetic_head(N)
roi = ex06.brain_roi(labels)
print(f"{args.phantom} head {N}^3, dx={dx*1e3:.2f}mm, brain ROI={roi.sum()}", flush=True)

# Clinical helmet, FLUSH to the scalp (1-voxel coupling standoff), full coverage.
src_positions, sensor_positions = ex06.make_helmet(
    labels, dx, n_elem, n_src=24, kind="clinical", freq=600e3, standoff=1.0 * dx)
rx, ry, rz = (np.asarray(a) for a in sensor_positions)
solid = (labels == 5) | np.isin(labels, ex06.B.BRAIN) | (labels == 1)
in_solid = int(sum(solid[x, y, z] for x, y, z in zip(rx, ry, rz)))
print(f"helmet: {len(rx)} sensors flush ({in_solid} in solid), {len(src_positions)} sources", flush=True)


def render_geometry(out):
    skull = labels == 5
    head = skull | np.isin(labels, ex06.B.BRAIN) | (labels == 1)
    sx, sy, sz = np.where(skull)
    sub = np.random.default_rng(0).choice(len(sx), size=min(4000, len(sx)), replace=False)
    cz = int(np.where(roi)[2].mean()); cy = int(np.where(roi)[1].mean()); cx = int(np.where(roi)[0].mean())
    fig = plt.figure(figsize=(18, 5.2), facecolor="white")
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.scatter(sx[sub], sy[sub], sz[sub], s=2, c="lightgray", alpha=0.12)
    ax.scatter(rx, ry, rz, s=14, c="#2a6fdb", depthshade=False)
    ax.scatter([p[0] for p in src_positions], [p[1] for p in src_positions],
               [p[2] for p in src_positions], s=40, c="red", depthshade=False)
    ax.set_title(f"Clinical helmet on the real head\n{len(rx)} sensors (blue) flush to scalp, "
                 f"{len(src_positions)} sources (red)", fontsize=10)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=-60); ax.set_axis_off()
    for i, (axis, mid, name) in enumerate([(2, cz, "axial"), (1, cy, "coronal"), (0, cx, "sagittal")]):
        a = fig.add_subplot(1, 4, i + 2)
        sl = np.take(head, mid, axis=axis).T.astype(float)
        a.imshow(sl, origin="lower", cmap="bone", vmin=0, vmax=1.4)
        near = np.abs([rx, ry, rz][axis] - mid) <= 1
        oth = [k for k in (0, 1, 2) if k != axis]
        pts = [rx, ry, rz]
        a.scatter(pts[oth[0]][near], pts[oth[1]][near], s=10, c="#2a6fdb")
        a.set_title(f"{name}: sensors flush on scalp", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Future imaging helmet on a real head ({args.phantom.upper()}, {N}³, dx {dx*1e3:.1f} mm) "
                 "— all sensors flush to the scalp", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", out, flush=True)


render_geometry(f"{RES}/funding_helmet_geometry.png")
if args.geometry_only:
    raise SystemExit(0)

# ---- reconstruction ----
ref = build_medium(build_domain((N, N, N), dx), ex06.C_MAX, 1000.0, pml_size=8)
t_end = 1.9 * (N * dx) / 1500.0
ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt = float(ta.dt); nt = int(ta.Nt)
f0 = max(f for _, f in bands)
sig = _build_source_signal(f0, dt, nt)
alpha_j = jnp.asarray(alpha_true)
print(f"generating observed (with absorption), Nt={nt}...", flush=True)
observed = generate_observed_data(
    jnp.asarray(c_true), jnp.asarray(rho_true), dx, src_positions, sensor_positions, f0,
    pml_size=8, time_axis=ta, source_signal=sig, dt=dt, attenuation=alpha_j,
    alpha_power=ex06.Y_POWER, verbose=False)
c_init = c_true.copy(); c_init[roi] = ex06.C_BRAIN0
cfg = FWIConfig(freq_bands=bands, n_iters_per_band=n_iters, shots_per_iter=shots, learning_rate=30.0,
                c_min=ex06.C_MIN, c_max=ex06.C_MAX, pml_size=8, cfl=0.3, gradient_smooth_sigma=1.5,
                mask=jnp.asarray(roi.astype(np.float32)), attenuation=alpha_j,
                alpha_power=ex06.Y_POWER, verbose=True)
print("running absorption-aware FWI...", flush=True)
t0 = time.time()
res = run_fwi(observed, jnp.asarray(c_init), jnp.asarray(rho_true), dx, src_positions,
              sensor_positions, sig, dt, t_end, config=cfg, key=jr.PRNGKey(0))
recon = np.asarray(res.velocity)
rmse = brain_roi_rmse(recon, c_true, roi)
print(f"brain RMSE {rmse:.2f} m/s ({time.time()-t0:.0f}s)", flush=True)

# full 3-plane image
cz = int(np.where(roi)[2].mean()); cy = int(np.where(roi)[1].mean()); cx = int(np.where(roi)[0].mean())
vmin, vmax = 1490, 1600
fig, ax = plt.subplots(2, 3, figsize=(13, 8.6), facecolor="white")
planes = [("axial", lambda a: a[:, :, cz].T), ("coronal", lambda a: a[:, cy, :].T),
          ("sagittal", lambda a: a[cx, :, :].T)]
for j, (name, slf) in enumerate(planes):
    for r, (img, lab) in enumerate([(c_true, "truth"), (recon, "reconstruction")]):
        im = ax[r, j].imshow(slf(img), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax[r, j].set_title(f"{lab} — {name}", fontsize=11); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="sound speed (m/s)")
fig.suptitle(f"Full 3D brain reconstruction through the flush clinical helmet "
             f"({args.phantom.upper()}, {N}³)  —  brain RMSE {rmse:.1f} m/s", fontsize=14, y=0.98)
out = f"{RES}/funding_helmet_recon.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); print("saved", out, flush=True)
np.savez(f"{RES}/funding_helmet_recon.npz", c_true=c_true, recon=recon, labels=labels,
         rmse=rmse, dx=dx, sensors=np.array([rx, ry, rz]))
