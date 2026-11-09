#!/usr/bin/env python
"""End-to-end multiparameter (c + alpha) FWI on anatomy (Phase 6 capstone).

Clinical setup: skull / scalp / CSF known from CT (frozen at truth); invert the
BRAIN's sound speed and attenuation jointly, with:
  - preconditioning (interior illumination),
  - the c-first hierarchical schedule (recover c, then release alpha),
  - the constitutive c->alpha coupling (velocity predicts alpha via tissue).

A co-varying stroke-like lesion (higher c AND higher alpha) is injected in the
white matter as the coupling's target. The honest counterpoint: grey vs white
matter differ ONLY in attenuation (same c), so the coupling is degenerate there
and cannot separate them — measured by gm_wm_separability on the recon.

  python scripts/multiparam_head_demo.py --smoke   # fast sanity (32^3)
  python scripts/multiparam_head_demo.py           # 64^3 demo
"""
from __future__ import annotations
import argparse, time
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain_fwi.phantoms.synthetic import make_gm_wm_contrast_head
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data,
    _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi
from brain_fwi.constitutive import speed_alpha_anchors
from brain_fwi.robustness import gm_wm_separability

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

N = 32 if args.smoke else 64
dx = 4e-3 if args.smoke else 2e-3
Y = 1.1
c, rho, alpha, labels, dx, gm, wm = make_gm_wm_contrast_head((N, N, N), dx)
c = c.copy(); alpha = alpha.copy()
GM, WM, C_MAX = 2, 3, 2900.0
brain = (labels == GM) | (labels == WM)

# Inject a co-varying lesion (stroke-like: high c AND high alpha) in WM.
zz, yy, xx = np.meshgrid(*[np.arange(N)] * 3, indexing="ij")
bx, by, bz = [int(np.where(brain)[i].mean()) for i in range(3)]
les_r = max(2.0, N * 0.09)
lesion = ((xx - bx) ** 2 + (yy - by) ** 2 + (zz - bz) ** 2 <= les_r ** 2) & (labels == WM)
c[lesion] = 1700.0
alpha[lesion] = 3.0
print(f"{N}^3 dx={dx*1e3:.1f}mm  brain={brain.sum()} GM={gm.sum()} WM={wm.sum()} "
      f"lesion={lesion.sum()}", flush=True)

# Transducers: Fibonacci sphere around the head (receivers), subset as sources.
n_recv, n_src = (40, 6) if args.smoke else (200, 12)
gi = (1 + np.sqrt(5)) / 2
k = np.arange(n_recv)
th = np.arccos(1 - 2 * (k + 0.5) / n_recv); ph = 2 * np.pi * k / gi
rad = N * 0.34
px = np.clip(np.round(bx + rad * np.sin(th) * np.cos(ph)).astype(int), 9, N - 10)
py = np.clip(np.round(by + rad * np.sin(th) * np.sin(ph)).astype(int), 9, N - 10)
pz = np.clip(np.round(bz + rad * np.cos(th)).astype(int), 9, N - 10)
recv = (px, py, pz)
si = np.linspace(0, n_recv - 1, n_src).astype(int)
srcs = [(int(px[i]), int(py[i]), int(pz[i])) for i in si]

pml = 8
dom = build_domain((N, N, N), dx)
ref = build_medium(dom, C_MAX, 1000.0, pml_size=pml)
t_end = 1.9 * (N * dx) / 1500.0
ta = build_time_axis(ref, cfl=0.2, t_end=t_end); dt, nt = float(ta.dt), int(ta.Nt)
bands = [(80e3, 150e3)] if args.smoke else [(80e3, 150e3), (150e3, 250e3)]
n_iters = 3 if args.smoke else 8
f0 = max(b for _, b in bands)
sig = _build_source_signal(f0, dt, nt)

print(f"generating observed (absorption-aware), Nt={nt}, {len(srcs)} shots...", flush=True)
observed = generate_observed_data(
    jnp.asarray(c), jnp.asarray(rho), dx, srcs, recv, f0, pml_size=pml,
    time_axis=ta, source_signal=sig, dt=dt, attenuation=jnp.asarray(alpha),
    alpha_power=Y, verbose=False)

# Clinical init: skull/scalp/CSF known; brain c homogeneous, brain alpha unknown (0).
c_init = c.copy(); c_init[brain] = 1560.0
a_init = alpha.copy(); a_init[brain] = 0.0
brain_mask = jnp.asarray(brain.astype(np.float32))
cfg = FWIConfig(
    freq_bands=bands, n_iters_per_band=n_iters, shots_per_iter=min(8, len(srcs)),
    learning_rate=25.0, c_min=1450.0, c_max=C_MAX, pml_size=pml, cfl=0.2,
    gradient_smooth_sigma=1.0, mask=brain_mask, precondition=True, precondition_floor=0.05,
    invert_attenuation=True, attenuation_init=jnp.asarray(a_init), attenuation_lr=1.5,
    attenuation_max=12.0, attenuation_mask=brain_mask,
    attenuation_release_frac=0.4,                       # c-first
    attenuation_speed_anchors=speed_alpha_anchors(), attenuation_speed_weight=0.4,
    verbose=True)
print("running multiparameter FWI (precondition + c-first + coupling)...", flush=True)
t0 = time.time()
res = run_fwi(observed, jnp.asarray(c_init), jnp.asarray(rho), dx, srcs, recv,
              sig, dt, t_end, config=cfg, key=jr.PRNGKey(0))
c_rec = np.asarray(res.velocity); a_rec = np.asarray(res.attenuation)
print(f"done ({time.time()-t0:.0f}s)", flush=True)

def rmse(a, b, m): return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))
les_c = rmse(c_rec, c, lesion); les_a = rmse(a_rec, alpha, lesion)
sep_true = gm_wm_separability(alpha, gm, wm)
sep_rec = gm_wm_separability(a_rec, gm, wm)
print(f"lesion c RMSE={les_c:.1f} m/s (in {c_rec[lesion].mean():.0f}, true 1700)", flush=True)
print(f"lesion a RMSE={les_a:.2f} (in {a_rec[lesion].mean():.2f}, true 3.0)", flush=True)
print(f"GM/WM separability: truth={sep_true:.1f}  recon={sep_rec:.3f} "
      f"(recon GM a={a_rec[gm].mean():.3f} WM a={a_rec[wm].mean():.3f})", flush=True)

# figure: c and alpha, truth vs recon, at the lesion plane
zc = int(bz)
fig, ax = plt.subplots(2, 3, figsize=(13, 8.4), facecolor="white")
sl = lambda A: A[:, :, zc].T
for j, (img, ttl, vlo, vhi, cm) in enumerate([
    (c, "c truth", 1500, 2000, "viridis"), (c_rec, "c recon", 1500, 2000, "viridis"),
    (c_rec - c, "c error", -200, 200, "coolwarm")]):
    im = ax[0, j].imshow(sl(img), origin="lower", cmap=cm, vmin=vlo, vmax=vhi)
    ax[0, j].set_title(ttl); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
    fig.colorbar(im, ax=ax[0, j], fraction=0.046)
for j, (img, ttl, vlo, vhi, cm) in enumerate([
    (alpha, "alpha truth", 0, 4, "magma"), (a_rec, "alpha recon", 0, 4, "magma"),
    (a_rec - alpha, "alpha error", -2, 2, "coolwarm")]):
    im = ax[1, j].imshow(sl(img), origin="lower", cmap=cm, vmin=vlo, vmax=vhi)
    ax[1, j].set_title(ttl); ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
    fig.colorbar(im, ax=ax[1, j], fraction=0.046)
fig.suptitle(
    f"Multiparameter FWI on anatomy ({N}³, clinical: skull known, invert brain c+α)\n"
    f"lesion α {a_rec[lesion].mean():.2f}/3.0 (partial, illumination-limited); "
    f"GM/WM α NOT separated (recon GM {a_rec[gm].mean():.2f} vs WM {a_rec[wm].mean():.2f}, "
    f"truth 0.6/0.9 — degenerate c)", y=0.99, fontsize=11)
import os; os.makedirs("results/absorption_aware_fwi_3d", exist_ok=True)
base = f"results/absorption_aware_fwi_3d/multiparam_head_demo{'_smoke' if args.smoke else ''}"
fig.savefig(base + ".png", dpi=140, bbox_inches="tight", facecolor="white")
np.savez(base + ".npz", c_true=c, c_rec=c_rec, alpha_true=alpha, a_rec=a_rec,
         labels=labels, gm=gm, wm=wm, lesion=lesion, dx=dx,
         les_c=les_c, les_a=les_a, sep_true=sep_true, sep_rec=sep_rec)
print("saved", base + ".png", flush=True)
