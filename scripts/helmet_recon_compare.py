#!/usr/bin/env python
"""Reconstruct the same head with the current cap vs the future clinical helmet.

Both runs are absorption-aware FWI on the synthetic head at 96^3; the only
difference is the array: current 160-element Kernel-Flow cap vs the
frequency-aware, scalp-conformal clinical helmet (more, better-placed elements).
Shows what the future device buys in reconstruction quality.
"""
from __future__ import annotations
import importlib.util, time
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("ex06", "examples/06_absorption_aware_fwi_3d.py")
ex06 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex06)
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data, _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi
from brain_fwi.robustness import brain_roi_rmse

N = 64
c_true, rho_true, alpha_true, labels, dx = ex06.synthetic_head(N)
roi = ex06.brain_roi(labels)
ref = build_medium(build_domain((N, N, N), dx), ex06.C_MAX, 1000.0, pml_size=8)
t_end = 1.9 * (N * dx) / 1500.0
ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt = float(ta.dt); nt = int(ta.Nt)
bands = [(50e3, 100e3), (100e3, 160e3)]; f0 = max(f for _, f in bands)
sig = _build_source_signal(f0, dt, nt)
c_init = c_true.copy(); c_init[roi] = ex06.C_BRAIN0
rho_j = jnp.asarray(rho_true); mask_j = jnp.asarray(roi.astype(np.float32))
alpha_j = jnp.asarray(alpha_true)


def run_with_helmet(kind, n_elem, exclude_face=False):
    print(f"\n=== helmet={kind}, n_elem={n_elem}, exclude_face={exclude_face} ===", flush=True)
    src, recv = ex06.make_helmet(labels, dx, n_elem, n_src=16, kind=kind, freq=600e3,
                                 exclude_face=exclude_face)
    solid = (labels == 5) | np.isin(labels, ex06.B.BRAIN) | (labels == 1)
    in_solid = sum(solid[p] for p in zip(*recv))
    print(f"  {len(recv[0])} receivers ({in_solid} in solid), {len(src)} sources", flush=True)
    obs = generate_observed_data(
        jnp.asarray(c_true), jnp.asarray(rho_true), dx, src, recv, f0, pml_size=8,
        time_axis=ta, source_signal=sig, dt=dt, attenuation=alpha_j,
        alpha_power=ex06.Y_POWER, verbose=False)
    cfg = FWIConfig(
        freq_bands=bands, n_iters_per_band=10, shots_per_iter=8, learning_rate=30.0,
        c_min=ex06.C_MIN, c_max=ex06.C_MAX, pml_size=8, cfl=0.3, gradient_smooth_sigma=1.5,
        mask=mask_j, attenuation=alpha_j, alpha_power=ex06.Y_POWER, verbose=False)
    t0 = time.time()
    res = run_fwi(obs, jnp.asarray(c_init), rho_j, dx, src, recv, sig, dt, t_end,
                  config=cfg, key=jr.PRNGKey(0))
    recon = np.asarray(res.velocity)
    rmse = brain_roi_rmse(recon, c_true, roi)
    print(f"  brain RMSE {rmse:.2f} m/s  ({time.time()-t0:.0f}s)", flush=True)
    return recon, rmse, len(recv[0])


# Fair comparison: BOTH face-realistic (can't image through eyes/airway).
# Current cap = sparse generic-ellipsoid array; clinical = dense scalp-conformal.
cap, cap_rmse, cap_n = run_with_helmet("cap", 256, exclude_face=True)
clin, clin_rmse, clin_n = run_with_helmet("clinical", 768)

# figure
les = (labels == 1)
zc = int(np.argmax(les.sum(axis=(0, 1)))) if les.any() else int(np.argmax((labels == 2).sum(axis=(0, 1))))
sl = lambda a: a[:, :, zc].T
skull = sl(labels == 5).astype(float); lesm = sl(les).astype(float)
vmin, vmax = 1500, 1620
fig = plt.figure(figsize=(16.5, 5.6), facecolor="white")
sfs = fig.subfigures(1, 2, width_ratios=[3.25, 0.82], wspace=0.02)
axs = sfs[0].subplots(1, 3)
for a, img, t in zip(axs, [c_true, cap, clin],
                     ["Ground truth", f"Current cap ({cap_n} elem)\nbrain RMSE {cap_rmse:.1f} m/s",
                      f"Clinical helmet ({clin_n} elem)\nbrain RMSE {clin_rmse:.1f} m/s"]):
    im = a.imshow(sl(img), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    a.contour(skull, [0.5], colors="white", linewidths=1.0)
    if lesm.any(): a.contour(lesm, [0.5], colors="red", linewidths=1.5)
    a.set_title(t, fontsize=13, pad=6); a.set_xticks([]); a.set_yticks([])
cb = sfs[0].colorbar(im, ax=axs, location="right", fraction=0.035, pad=0.02)
cb.set_label("sound speed (m/s)", fontsize=10)
axb = sfs[1].subplots(1, 1)
gain = 100 * (cap_rmse - clin_rmse) / cap_rmse
bars = axb.bar(["current\ncap", "clinical\nhelmet"], [cap_rmse, clin_rmse],
               color=["#9e9e9e", "#2a9d3f"], width=0.6)
for b, v in zip(bars, [cap_rmse, clin_rmse]):
    axb.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
axb.set_ylabel("brain-ROI RMSE (m/s) ↓"); axb.set_title("Reconstruction error", fontsize=12)
axb.spines[["top", "right"]].set_visible(False); axb.tick_params(labelsize=9)
fig.suptitle(f"Future imaging helmet, reconstructed: conformal {clin_n}-element clinical helmet "
             f"vs current {cap_n}-element cap  (+{gain:.0f}%)", fontsize=14, y=1.04)
fig.text(0.5, -0.04, f"absorption-aware FWI, synthetic head {N}³ (matched coverage)  ·  "
         "JAX autodiff through j-Wave", ha="center", fontsize=10.5, color="#444")
out = "results/absorption_aware_fwi_3d/helmet_recon_compare.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\ncap {cap_rmse:.2f} -> clinical {clin_rmse:.2f} m/s ({gain:+.0f}%); saved {out}", flush=True)
np.savez("results/absorption_aware_fwi_3d/helmet_recon_compare.npz",
         c_true=c_true, cap=cap, clin=clin, labels=labels, zc=zc,
         cap_rmse=cap_rmse, clin_rmse=clin_rmse, cap_n=cap_n, clin_n=clin_n)
