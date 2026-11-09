"""Transmission-traveltime pose recovery: gradient descent from origin.

mofi_transmission_scan.py (job 1691) showed the envelope-XC TRANSMISSION traveltime
misfit is a smooth bowl with the minimum EXACTLY at truth on every axis (tx/ty/rz),
O(1)->~0 contrast, sloping monotonically downhill from origin toward truth -- the exact
opposite of the reflection misfit's plateau+needle. So plain gradient descent from
origin should land the 4vox/5deg pose where grid / L2-grad / AWI / CMA-ES all failed.
This is the decisive test. true brain (mechanism) then generic brain (clinical).
Single-warp consistent forward (po via same forward at true medium).
"""
import os, time, numpy as np
import jax, jax.numpy as jnp, equinox as eqx, optax
jax.config.update("jax_enable_x64", False)
from brain_fwi.phantoms.birnbaum import to_velocity, roi_mask, SKULL, C_WATER, C_SKULL
from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis, simulate_shot_sensors)
from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet
import sys; sys.path.insert(0, os.path.dirname(__file__))
from mofi3d import rigid_warp_3d

S = 96; dx = 1.5e-3; pml = 8; F0 = 80e3; T_END = 1.4e-4; N_SHOTS = 12
T_TRUE = jnp.array([4.0, 2.0, 0.0]); A_TRUE = jnp.array([0.0, 0.0, np.deg2rad(5.0)])
POSE_SHOTS = 6; MAXLAG = 60; BETA = 30.0; STEPS = 60; LR = 0.5
TRUE_BRAIN = os.environ.get("BFWI_TRUE_BRAIN", "1") == "1"

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
brain_est = (brain_no_skull if TRUE_BRAIN else jnp.where(interior, 1560.0, C_WATER)).astype(jnp.float32)
ps = [src[i] for i in np.linspace(0, N_SHOTS - 1, POSE_SHOTS).astype(int)]
true_med = build_medium(build_domain((S, S, S), dx), c_obs, rho, pml_size=pml)
po = [simulate_shot_sensors(true_med, ta, ps[k], pg, sig, dt, checkpointed=False) for k in range(POSE_SHOTS)]
print(f"transmission grad-descent: brain={'TRUE' if TRUE_BRAIN else 'generic'}, full waveform {nsteps} steps, "
      f"{POSE_SHOTS} shots, from origin, {STEPS} Adam steps lr {LR}", flush=True)


def envelope(x):
    n = x.shape[0]; X = jnp.fft.fft(x, axis=0)
    h = jnp.zeros(n).at[0].set(1.0)
    h = (h.at[n // 2].set(1.0).at[1:n // 2].set(2.0)) if n % 2 == 0 else h.at[1:(n + 1) // 2].set(2.0)
    return jnp.abs(jnp.fft.ifft(X * h[:, None], axis=0))

def tt_misfit_pair(pred, obs):
    ep = envelope(pred); ed = envelope(obs); en = jnp.sum(ed ** 2, axis=0)
    epn = ep - jnp.mean(ep, axis=0); edn = ed - jnp.mean(ed, axis=0)
    epn = epn / (jnp.linalg.norm(epn, axis=0) + 1e-12); edn = edn / (jnp.linalg.norm(edn, axis=0) + 1e-12)
    nt = ep.shape[0]
    xc = jnp.fft.fftshift(jnp.fft.irfft(jnp.conj(jnp.fft.rfft(epn, axis=0)) * jnp.fft.rfft(edn, axis=0),
                                        n=nt, axis=0), axes=0)
    ctr = nt // 2; C = xc[ctr - MAXLAG:ctr + MAXLAG + 1]
    lags = jnp.arange(-MAXLAG, MAXLAG + 1).astype(jnp.float32)
    tau = jnp.sum(lags[:, None] * jax.nn.softmax(BETA * C, axis=0), axis=0)
    return jnp.sum(en * tau ** 2) / (jnp.sum(en) + 1e-30)

def loss(q):
    skm = jnp.clip(rigid_warp_3d(skull_true, jnp.array([q["tx"], q["ty"], 0.0]),
                                 jnp.array([0.0, 0.0, jnp.deg2rad(q["rz"])])), 0, 1)
    v = brain_est * (1 - skm) + C_SKULL * skm
    med = build_medium(build_domain((S, S, S), dx), v, rho, pml_size=pml)
    tot = 0.0
    for k in range(POSE_SHOTS):
        pred = simulate_shot_sensors(med, ta, ps[k], pg, sig, dt, checkpointed=True)
        mt = min(pred.shape[0], po[k].shape[0]); tot = tot + tt_misfit_pair(pred[:mt], po[k][:mt])
    return tot / POSE_SHOTS

def overlap(tx, ty, rz):
    a = rigid_warp_3d(skull_true, jnp.array([tx, ty, 0.0]), jnp.array([0.0, 0.0, np.deg2rad(rz)])) > 0.5
    return 100 * float(jnp.mean(a & (skull_obs > 0.5)) / (jnp.mean(skull_obs > 0.5) + 1e-9))

p = {"tx": jnp.array(0.0), "ty": jnp.array(0.0), "rz": jnp.array(0.0)}
opt = optax.adam(LR); st = opt.init(p); vg = eqx.filter_jit(jax.value_and_grad(loss))
print(f"  init overlap {overlap(0,0,0):.0f}%", flush=True); t0 = time.time()
for i in range(STEPS):
    l, g = vg(p); up, st = opt.update(g, st); p = optax.apply_updates(p, up)
    if (i + 1) % 5 == 0:
        tx, ty, rz = float(p["tx"]), float(p["ty"]), float(p["rz"])
        print(f"  step {i+1}: TT {float(l):.3e} t=({tx:.2f},{ty:.2f}) rz={rz:.2f} overlap {overlap(tx,ty,rz):.0f}% "
              f"err |dt|={np.hypot(tx-4,ty-2):.2f} |drz|={abs(rz-5):.2f} ({time.time()-t0:.0f}s)", flush=True)
ftx, fty, frz = float(p["tx"]), float(p["ty"]), float(p["rz"]); ov = overlap(ftx, fty, frz)
print(f"FINAL: t=({ftx:.2f},{fty:.2f},0) rz={frz:.2f}deg (true 4,2,5) OVERLAP {ov:.0f}% "
      f"=> pose {'LANDED' if ov >= 90 else 'PARTIAL' if ov >= 75 else 'did NOT land'}", flush=True)
np.save("/tmp/mofi_tt_pose.npy", np.array([ftx, fty, frz]))
