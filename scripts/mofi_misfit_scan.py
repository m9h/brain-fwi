"""Diagnostic: does the j-Wave data misfit have a usable minimum vs skull pose?

The gradient-based MOFI failed (non-convex/flat landscape). Before building a
global pose search, scan the misfit along ONE translation DOF across frequency
bands: if there's a clear dip at the aligned pose in some band, a global search is
viable and we know the band; if it's flat everywhere, the data doesn't constrain
the pose at this setup. tx-only perturbation so the minimum is cleanly at dx=-4.
"""
import os, time, numpy as np
import jax, jax.numpy as jnp, jax.random as jr
jax.config.update("jax_enable_x64", False)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from brain_fwi.phantoms.birnbaum import to_velocity, SKULL, C_WATER, C_SKULL
from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis,
    simulate_shot_sensors, generate_observed_data)
from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet
from brain_fwi.inversion.losses import l2_loss
from brain_fwi.inversion.fwi import _bandpass_signal
import sys; sys.path.insert(0, os.path.dirname(__file__))
from mofi3d import rigid_warp_3d

S = 96; dx = 1.5e-3; pml = 8; F0 = 80e3; T_END = 1.0e-4; SHOTS = 6
TX_PERT = 4.0
crop = np.load("/tmp/subj2_crop_96.npy")
c_true = jnp.asarray(to_velocity(crop, with_skull=True))
skull_true = jnp.asarray((crop == SKULL).astype(np.float32))
rho = jnp.full((S, S, S), 1000.0, jnp.float32)
c = (S // 2) * dx; r = (S // 2 - 4) * dx
pos = helmet_array_3d(n_elements=80, center=(c, c, c), radius_ap=r, radius_lr=r, radius_si=r,
                      standoff=0.0, coverage_angle=3.1, exclude_face=False)
pg = tuple(np.asarray(z) for z in transducer_positions_to_grid(pos, dx, (S, S, S))); ne = len(pg[0])
allsrc = [(int(pg[0][i]), int(pg[1][i]), int(pg[2][i])) for i in range(ne)]
src = [allsrc[i] for i in np.linspace(0, ne - 1, SHOTS).astype(int)]
ta = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml), cfl=0.3, t_end=T_END)
dt = float(ta.dt); nsteps = int(T_END / dt); sig = ricker_wavelet(f0=F0, dt=dt, n_samples=nsteps)
obs = generate_observed_data(sound_speed=c_true, density=rho, dx=dx, src_positions_grid=src,
    sensor_positions_grid=pg, freq=F0, pml_size=pml, time_axis=ta, source_signal=sig, dt=dt, verbose=False)
template = rigid_warp_3d(skull_true, jnp.array([TX_PERT, 0.0, 0.0]), jnp.zeros(3))   # tx-only misalignment

# optionally put the TRUE brain (not water) under the warped skull -> isolates the
# skull-pose signal from the brain-mismatch floor (tests why the plateau is flat).
USE_TRUE_BRAIN = os.environ.get("BFWI_TRUE_BRAIN", "0") == "1"
brain_bg = jnp.where(jnp.asarray(crop == SKULL), C_WATER, c_true)   # true brain on water, no skull

def misfit(dx_search, bp, bobs):
    sk = rigid_warp_3d(template, jnp.array([dx_search, 0.0, 0.0]), jnp.zeros(3))
    skm = jnp.clip(sk, 0, 1)
    skv = (brain_bg if USE_TRUE_BRAIN else C_WATER) * (1 - skm) + C_SKULL * skm
    med = build_medium(build_domain((S, S, S), dx), skv, rho, pml_size=pml)
    tot = 0.0
    for k in range(SHOTS):
        pred = simulate_shot_sensors(med, ta, src[k], pg, bp, dt, checkpointed=True)
        mt = min(pred.shape[0], bobs[k].shape[0]); tot = tot + l2_loss(pred[:mt], bobs[k][:mt])
    return float(tot / SHOTS)

BANDS = [(20e3, 45e3), (40e3, 80e3), (70e3, 130e3)]
dxs = np.linspace(-8, 2, 11)                                   # min expected at dx=-4 (re-aligns)
fig, ax = plt.subplots(1, 3, figsize=(15, 4)); t0 = time.time()
for bi, (fmin, fmax) in enumerate(BANDS):
    bp = _bandpass_signal(sig, dt, fmin, fmax)
    bobs = jax.vmap(lambda d: jax.vmap(lambda col: _bandpass_signal(col, dt, fmin, fmax))(d.T).T)(obs)
    ms = [misfit(float(d), bp, bobs) for d in dxs]
    mn = dxs[int(np.argmin(ms))]
    print(f"band {fmin/1e3:.0f}-{fmax/1e3:.0f}kHz: misfit min at dx={mn:.1f} (true -4); "
          f"contrast {max(ms)/ (min(ms)+1e-30):.2f}x ({time.time()-t0:.0f}s)", flush=True)
    ax[bi].plot(dxs, ms, "-o"); ax[bi].axvline(-4, color="r", ls="--", label="aligned (dx=-4)")
    ax[bi].set_title(f"{fmin/1e3:.0f}-{fmax/1e3:.0f}kHz", fontsize=10); ax[bi].set_xlabel("search tx (vox)"); ax[bi].legend(fontsize=8)
plt.tight_layout(); plt.savefig("/tmp/mofi_misfit_scan.png", dpi=120); print("saved /tmp/mofi_misfit_scan.png", flush=True)
