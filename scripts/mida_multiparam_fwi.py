#!/usr/bin/env python
"""Scenario A: isotropic MULTIPARAMETER FWI on a real MIDA head — invert both
sound speed AND attenuation intracranially (skull known from CT), de-risked with
the machinery proven this session: c-first schedule, preconditioning, constitutive
c->alpha coupling, cfl<=0.2 (the 2800 m/s skull is NaN-unstable at higher cfl).

Unlike the 192^3 "pub" run (which FROZE alpha at truth), this inverts alpha too —
the honest clinical setup (brain attenuation is unknown).

  python scripts/mida_multiparam_fwi.py --smoke        # fast sanity (48^3)
  python scripts/mida_multiparam_fwi.py --n 96         # de-risked run
"""
from __future__ import annotations
import argparse, importlib.util, os, time
import numpy as np, jax.numpy as jnp, jax.random as jr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("ex06", "examples/06_absorption_aware_fwi_3d.py")
ex06 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex06)
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data, _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi
from brain_fwi.constitutive import speed_alpha_anchors
from brain_fwi.robustness import brain_roi_rmse

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--n", type=int, default=96)
ap.add_argument("--mida-path", default=None)
args = ap.parse_args()
N = 48 if args.smoke else args.n
Y = ex06.Y_POWER

c, rho, alpha, labels, dx = (ex06.load_mida_head(N, args.mida_path) if args.mida_path
                             else ex06.load_mida_head(N))
c = c.astype(np.float32); alpha = alpha.astype(np.float32)
roi = ex06.brain_roi(labels)                     # intracranial region
skull = labels == ex06.B.SKULL
C_MAX = 2900.0
print(f"MIDA {N}^3 dx={dx*1e3:.2f}mm intracranial={int(roi.sum())} skull={int(skull.sum())} "
      f"c[roi]={c[roi].min():.0f}-{c[roi].max():.0f} a[roi]={alpha[roi].min():.2f}-{alpha[roi].max():.2f}",
      flush=True)

n_elem = 400 if args.smoke else 800
src_pos, sensor_pos = ex06.make_helmet(labels, dx, n_elem, n_src=24, kind="clinical",
                                       freq=600e3, standoff=1.0 * dx)
print(f"helmet: {len(sensor_pos[0])} sensors, {len(src_pos)} sources", flush=True)

pml = 8
ref = build_medium(build_domain((N, N, N), dx), C_MAX, 1000.0, pml_size=pml)
t_end = 1.9 * (N * dx) / 1500.0
ta = build_time_axis(ref, cfl=0.2, t_end=t_end); dt, nt = float(ta.dt), int(ta.Nt)
bands = [(50e3, 100e3)] if args.smoke else [(50e3, 100e3), (100e3, 180e3)]
n_iters = 2 if args.smoke else 8
f0 = max(b for _, b in bands); sig = _build_source_signal(f0, dt, nt)

print(f"generating observed (absorption-aware), Nt={nt}...", flush=True)
observed = generate_observed_data(
    jnp.asarray(c), jnp.asarray(rho), dx, src_pos, sensor_pos, f0, pml_size=pml,
    time_axis=ta, source_signal=sig, dt=dt, attenuation=jnp.asarray(alpha), alpha_power=Y, verbose=False)

# Clinical init: skull/scalp known at truth; intracranial c homogeneous, alpha unknown(0).
c_init = c.copy(); c_init[roi] = 1552.0
a_init = alpha.copy(); a_init[roi] = 0.0
roi_j = jnp.asarray(roi.astype(np.float32))
cfg = FWIConfig(
    freq_bands=bands, n_iters_per_band=n_iters, shots_per_iter=min(8, len(src_pos)),
    learning_rate=25.0, c_min=ex06.C_MIN, c_max=C_MAX, pml_size=pml, cfl=0.2,
    gradient_smooth_sigma=1.0, mask=roi_j, precondition=True, precondition_floor=0.05,
    invert_attenuation=True, attenuation_init=jnp.asarray(a_init), attenuation_lr=1.5,
    attenuation_max=6.0, attenuation_mask=roi_j, attenuation_release_frac=0.4,      # c-first
    attenuation_speed_anchors=speed_alpha_anchors(), attenuation_speed_weight=0.4,
    verbose=True)
print("running multiparameter FWI (c-first + precondition + coupling, cfl 0.2)...", flush=True)
t0 = time.time()
res = run_fwi(observed, jnp.asarray(c_init), jnp.asarray(rho), dx, src_pos, sensor_pos,
              sig, dt, t_end, config=cfg, key=jr.PRNGKey(0))
c_rec = np.asarray(res.velocity); a_rec = np.asarray(res.attenuation)
print(f"done ({time.time()-t0:.0f}s)", flush=True)

c_rmse = brain_roi_rmse(c_rec, c, roi)
a_rmse = float(np.sqrt(np.mean((a_rec[roi] - alpha[roi]) ** 2)))
c_rmse0 = brain_roi_rmse(c_init, c, roi)
print(f"intracranial c RMSE {c_rmse0:.1f} -> {c_rmse:.1f} m/s  |  alpha RMSE (from 0) {a_rmse:.3f} "
      f"(true mean {alpha[roi].mean():.2f})", flush=True)

cz = int(np.where(roi)[2].mean())
fig, ax = plt.subplots(2, 3, figsize=(13, 8.4), facecolor="white")
sl = lambda A: A[:, :, cz].T
for j, (img, ttl, lo, hi, cm) in enumerate([
    (c, "c truth", 1490, 1620, "viridis"), (c_rec, "c recon", 1490, 1620, "viridis"),
    (c_rec - c, "c error", -60, 60, "coolwarm")]):
    im = ax[0, j].imshow(sl(img), origin="lower", cmap=cm, vmin=lo, vmax=hi)
    ax[0, j].set_title(ttl); ax[0, j].set_xticks([]); ax[0, j].set_yticks([]); fig.colorbar(im, ax=ax[0, j], fraction=0.046)
for j, (img, ttl, lo, hi, cm) in enumerate([
    (alpha, "alpha truth", 0, 1, "magma"), (a_rec, "alpha recon", 0, 1, "magma"),
    (a_rec - alpha, "alpha error", -0.6, 0.6, "coolwarm")]):
    im = ax[1, j].imshow(sl(img), origin="lower", cmap=cm, vmin=lo, vmax=hi)
    ax[1, j].set_title(ttl); ax[1, j].set_xticks([]); ax[1, j].set_yticks([]); fig.colorbar(im, ax=ax[1, j], fraction=0.046)
fig.suptitle(f"Multiparameter FWI on MIDA ({N}^3, clinical: skull known, invert intracranial c+alpha)\n"
             f"c RMSE {c_rmse0:.0f}->{c_rmse:.0f} m/s, alpha RMSE {a_rmse:.2f}", fontsize=12, y=0.99)
os.makedirs("results/absorption_aware_fwi_3d", exist_ok=True)
base = f"results/absorption_aware_fwi_3d/mida_multiparam{'_smoke' if args.smoke else f'_{N}'}"
fig.savefig(base + ".png", dpi=140, bbox_inches="tight", facecolor="white")
np.savez(base + ".npz", c_true=c, c_rec=c_rec, alpha_true=alpha, a_rec=a_rec, roi=roi, dx=dx,
         c_rmse=c_rmse, a_rmse=a_rmse)
print("saved", base + ".png", flush=True)
