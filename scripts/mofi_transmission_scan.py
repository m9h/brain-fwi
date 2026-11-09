"""Transmission traveltime misfit: does a DIFFERENT observable kill the plateau?

Every optimizer (grid, L2/AWI gradient, CMA-ES) failed pose recovery because the
early-window REFLECTION L2 misfit is a wide plateau (~6e-9, overlap 50-66%) with truth
a sub-voxel needle (project_mofi_pose_characterization). The blocker is the observable,
not the optimizer. The skull's TRANSMISSION delay, by contrast, is smooth and monotone
in pose -- the basis of traveltime tomography (Luo-Schuster 1991) and likely MOFI's
actual mechanism.

This builds an envelope cross-correlation TRAVELTIME misfit on the FULL waveform
(transmission arrivals, energy-weighted over receivers) and -- before any search --
1D-scans it along tx, ty, rz through truth. Decision diagnostic:
  smooth bowl, min at truth, no plateau, ~0 local minima -> gradient descent will land it.
  still flat/rough -> transmission at this geometry doesn't constrain pose either.
Single-warp consistent forward (po via same forward at true medium -> misfit(truth)=0).
"""
import os, time, numpy as np
import jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_enable_x64", False)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from brain_fwi.phantoms.birnbaum import to_velocity, roi_mask, SKULL, C_WATER, C_SKULL
from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis, simulate_shot_sensors)
from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet
import sys; sys.path.insert(0, os.path.dirname(__file__))
from mofi3d import rigid_warp_3d

S = 96; dx = 1.5e-3; pml = 8; F0 = 80e3; T_END = 1.4e-4; N_SHOTS = 12   # full waveform -> transmission arrivals
T_TRUE = jnp.array([4.0, 2.0, 0.0]); A_TRUE = jnp.array([0.0, 0.0, np.deg2rad(5.0)]); RZ_TRUE = 5.0
POSE_SHOTS = 6; MAXLAG = 60; BETA = 30.0

crop = np.load("/tmp/subj2_crop_96.npy")
c_true = jnp.asarray(to_velocity(crop, with_skull=True))
skull_true = jnp.asarray((crop == SKULL).astype(np.float32))
interior = jnp.asarray(roi_mask(crop))
rho = jnp.full((S, S, S), 1000.0, jnp.float32)
c = (S // 2) * dx; r = (S // 2 - 4) * dx
pos = helmet_array_3d(n_elements=80, center=(c, c, c), radius_ap=r, radius_lr=r, radius_si=r,
                      standoff=0.0, coverage_angle=3.1, exclude_face=False)
pg = tuple(np.asarray(z) for z in transducer_positions_to_grid(pos, dx, (S, S, S))); ne = len(pg[0])
allsrc = [(int(pg[0][i]), int(pg[1][i]), int(pg[2][i])) for i in range(ne)]
src = [allsrc[i] for i in np.linspace(0, ne - 1, N_SHOTS).astype(int)]
ta = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml), cfl=0.3, t_end=T_END)
dt = float(ta.dt); nsteps = int(T_END / dt); sig = ricker_wavelet(f0=F0, dt=dt, n_samples=nsteps)
skull_obs = jnp.clip(rigid_warp_3d(skull_true, T_TRUE, A_TRUE), 0, 1)
brain_no_skull = jnp.where(skull_true > 0.5, C_WATER, c_true)
c_obs = (brain_no_skull * (1 - skull_obs) + C_SKULL * skull_obs).astype(jnp.float32)
brain_est = brain_no_skull.astype(jnp.float32)                       # true brain (isolate the landscape)
ps = [src[i] for i in np.linspace(0, N_SHOTS - 1, POSE_SHOTS).astype(int)]
true_med = build_medium(build_domain((S, S, S), dx), c_obs, rho, pml_size=pml)
po = [simulate_shot_sensors(true_med, ta, ps[k], pg, sig, dt, checkpointed=False) for k in range(POSE_SHOTS)]
print(f"transmission scan: full waveform {nsteps} steps (~{nsteps*dt*1e6:.0f}us), {POSE_SHOTS} shots, "
      f"envelope-XC traveltime, maxlag {MAXLAG} (~{MAXLAG*dt*1e6:.1f}us)", flush=True)


def envelope(x):                                                     # (nt, nrec) analytic-signal envelope along time
    n = x.shape[0]; X = jnp.fft.fft(x, axis=0)
    h = jnp.zeros(n).at[0].set(1.0)
    h = (h.at[n // 2].set(1.0).at[1:n // 2].set(2.0)) if n % 2 == 0 else h.at[1:(n + 1) // 2].set(2.0)
    return jnp.abs(jnp.fft.ifft(X * h[:, None], axis=0))

def tt_misfit_pair(pred, obs):                                       # envelope XC soft-lag, energy-weighted
    ep = envelope(pred); ed = envelope(obs)
    en = jnp.sum(ed ** 2, axis=0)                                    # per-receiver transmitted energy (weights)
    epn = ep - jnp.mean(ep, axis=0); edn = ed - jnp.mean(ed, axis=0)
    epn = epn / (jnp.linalg.norm(epn, axis=0) + 1e-12); edn = edn / (jnp.linalg.norm(edn, axis=0) + 1e-12)
    nt = ep.shape[0]
    xc = jnp.fft.fftshift(jnp.fft.irfft(jnp.conj(jnp.fft.rfft(epn, axis=0)) * jnp.fft.rfft(edn, axis=0),
                                        n=nt, axis=0), axes=0)        # (nt, nrec), centre = zero lag
    ctr = nt // 2; C = xc[ctr - MAXLAG:ctr + MAXLAG + 1]              # (2L+1, nrec)
    lags = jnp.arange(-MAXLAG, MAXLAG + 1).astype(jnp.float32)
    w = jax.nn.softmax(BETA * C, axis=0)
    tau = jnp.sum(lags[:, None] * w, axis=0)                          # (nrec,) soft traveltime shift
    return jnp.sum(en * tau ** 2) / (jnp.sum(en) + 1e-30)

@eqx.filter_jit
def misfit(tx, ty, rz):
    skm = jnp.clip(rigid_warp_3d(skull_true, jnp.array([tx, ty, 0.0]), jnp.array([0.0, 0.0, jnp.deg2rad(rz)])), 0, 1)
    v = brain_est * (1 - skm) + C_SKULL * skm
    med = build_medium(build_domain((S, S, S), dx), v, rho, pml_size=pml)
    tot = 0.0
    for k in range(POSE_SHOTS):
        pred = simulate_shot_sensors(med, ta, ps[k], pg, sig, dt, checkpointed=False)
        mt = min(pred.shape[0], po[k].shape[0]); tot = tot + tt_misfit_pair(pred[:mt], po[k][:mt])
    return tot / POSE_SHOTS

def overlap(tx, ty, rz):
    a = rigid_warp_3d(skull_true, jnp.array([tx, ty, 0.0]), jnp.array([0.0, 0.0, np.deg2rad(rz)])) > 0.5
    return 100 * float(jnp.mean(a & (skull_obs > 0.5)) / (jnp.mean(skull_obs > 0.5) + 1e-9))

_f = lambda v: jnp.asarray(v, jnp.float32)
m_truth = float(misfit(_f(4.0), _f(2.0), _f(RZ_TRUE))); m_orig = float(misfit(_f(0.0), _f(0.0), _f(0.0)))
m_wrong = float(misfit(_f(-6.0), _f(0.0), _f(-2.5)))
print(f"SANITY: truth(4,2,5)={m_truth:.3e} ov{overlap(4,2,RZ_TRUE):.0f}% | origin={m_orig:.3e} | wrong={m_wrong:.3e} "
      f"-> truth {'MIN' if m_truth<m_orig and m_truth<m_wrong else 'NOT min'} "
      f"(contrast {max(m_orig,m_wrong)/(m_truth+1e-30):.1f}x)", flush=True)

TXg = np.arange(-2, 10.01, 0.5); TYg = np.arange(-4, 8.01, 0.5); RZg = np.arange(-3, 13.01, 0.5)
fig, ax = plt.subplots(1, 3, figsize=(15, 4)); t0 = time.time()
for ai, (g, truth, nm, ev) in enumerate([(TXg, 4.0, "tx", lambda v: misfit(_f(v), _f(2.0), _f(RZ_TRUE))),
                                          (TYg, 2.0, "ty", lambda v: misfit(_f(4.0), _f(v), _f(RZ_TRUE))),
                                          (RZg, 5.0, "rz", lambda v: misfit(_f(4.0), _f(2.0), _f(v)))]):
    ms = [float(ev(v)) for v in g]; ms = np.asarray(ms); mn = ms.min(); thr = mn + 0.1 * (ms.max() - mn)
    below = g[ms < thr]; w = (below.max() - below.min()) if len(below) else 0.0
    nmin = int(np.sum((ms[1:-1] < ms[:-2]) & (ms[1:-1] < ms[2:])))
    print(f"{nm}: argmin {g[int(np.argmin(ms))]:+.1f} (true {truth:+.1f}) basin-w {w:.1f} local-minima {nmin} "
          f"({time.time()-t0:.0f}s)", flush=True)
    ax[ai].plot(g, ms, "-o", ms=3); ax[ai].axvline(truth, color="r", ls="--")
    ax[ai].set_title(f"transmission TT  {nm} (argmin {g[int(np.argmin(ms))]:+.1f}/true {truth:+.1f}, {nmin} loc-min)", fontsize=9)
    ax[ai].set_xlabel(f"{nm}")
plt.tight_layout(); plt.savefig("/tmp/mofi_transmission_scan.png", dpi=120); print("saved /tmp/mofi_transmission_scan.png", flush=True)
