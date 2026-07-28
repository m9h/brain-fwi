"""MOFI 3D end-to-end: recover skull pose from acoustic data, then FWI the brain.

Skull-from-data fix (GB10). A misaligned skull TEMPLATE (the only thing available
clinically) catastrophically breaks the brain FWI (~-400% at 4vox/5deg). MOFI
recovers the 6-DOF pose by minimising the LOW-frequency data misfit through j-Wave
(4 vox << a 20-45 kHz wavelength -> smooth landscape), aligns the skull, then runs
the brain FWI. Compares: skull-at-truth (cheat) vs misaligned-frozen (penalty) vs
MOFI-aligned (the fix).   .venv/bin/python scripts/run_mofi3d_fwi.py
"""
import os, time, numpy as np
import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx, optax
jax.config.update("jax_enable_x64", False)
from brain_fwi.phantoms.birnbaum import to_velocity, roi_mask, LESION, SKULL, C_WATER, C_SKULL
from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis,
    simulate_shot_sensors, generate_observed_data)
from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet
from brain_fwi.inversion.losses import l2_loss
from brain_fwi.inversion.fwi import _smooth_gradient, _bandpass_signal
import sys; sys.path.insert(0, os.path.dirname(__file__))
from mofi3d import rigid_warp_3d

S = 96; dx = 1.5e-3; pml = 8; F0 = 80e3; T_END = 1.0e-4; N_SHOTS = 12
T_PERT = jnp.array([4.0, 2.0, 0.0]); A_PERT = jnp.array([0.0, 0.0, np.deg2rad(5.0)])   # known misalignment
POSE_SHOTS = 6; POSE_STEPS = int(os.environ.get("BFWI_POSE_STEPS", "30"))
# skull (~2-3 vox shell) is INVISIBLE at low freq (lambda>>shell) -> use a band where
# it's visible (80kHz: lambda~13 vox; a 4-vox shift is ~0.3 lambda, sub-cycle-skip).
POSE_BAND = (float(os.environ.get("BFWI_POSE_FMIN", "40e3")), float(os.environ.get("BFWI_POSE_FMAX", "100e3")))

crop = np.load("/tmp/subj2_crop_96.npy")
c_true = jnp.asarray(to_velocity(crop, with_skull=True))
skull_true = jnp.asarray((crop == SKULL).astype(np.float32))
interior = jnp.asarray(roi_mask(crop)); imask = interior.astype(jnp.float32)
rho = jnp.full((S, S, S), 1000.0, jnp.float32)
c = (S // 2) * dx; r = (S // 2 - 4) * dx
pos = helmet_array_3d(n_elements=80, center=(c, c, c), radius_ap=r, radius_lr=r, radius_si=r,
                      standoff=0.0, coverage_angle=3.1, exclude_face=False)
pg = tuple(np.asarray(z) for z in transducer_positions_to_grid(pos, dx, (S, S, S))); ne = len(pg[0])
allsrc = [(int(pg[0][i]), int(pg[1][i]), int(pg[2][i])) for i in range(ne)]
src = [allsrc[i] for i in np.linspace(0, ne - 1, N_SHOTS).astype(int)]
ta = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml), cfl=0.3, t_end=T_END)
dt = float(ta.dt); nsteps = int(T_END / dt); sig = ricker_wavelet(f0=F0, dt=dt, n_samples=nsteps)
obs = generate_observed_data(sound_speed=c_true, density=rho, dx=dx, src_positions_grid=src,
    sensor_positions_grid=pg, freq=F0, pml_size=pml, time_axis=ta, source_signal=sig, dt=dt, verbose=False)
model = eqx.tree_deserialise_leaves("/tmp/brain_score_3d_96.eqx",
    __import__("brain_fwi.inference.score_unet3d", fromlist=["UNet3DScore"]).UNet3DScore(S, C=16, key=jr.PRNGKey(0)))
nz = np.load("/tmp/brain_score_3d_96_norm.npz"); SM, SS = float(nz["mean"]), float(nz["std"])

# the misaligned skull TEMPLATE (what we're given): true skull warped by the perturbation
template = rigid_warp_3d(skull_true, T_PERT, A_PERT)
print(f"template overlap w/ true: {100*float(jnp.mean((template>0.5)&(skull_true>0.5))/(jnp.mean(skull_true>0.5)+1e-9)):.0f}%", flush=True)


def skull_field(mask_field):                                   # mask in [0,1] -> velocity (skull/water)
    return C_WATER + (C_SKULL - C_WATER) * jnp.clip(mask_field, 0.0, 1.0)

# ---- MOFI pose recovery (40-80kHz where skull is visible; GENERIC brain estimate
#      removes the brain-mismatch floor that flattened the landscape; angles in
#      DEGREES so adam treats t/ang on the same scale) -----------------------------
bp_lo = _bandpass_signal(sig, dt, *POSE_BAND)
obs_lo = jax.vmap(lambda d: jax.vmap(lambda col: _bandpass_signal(col, dt, *POSE_BAND))(d.T).T)(obs)
pose_src = [src[i] for i in np.linspace(0, N_SHOTS - 1, POSE_SHOTS).astype(int)]
pose_obs = [obs_lo[i] for i in np.linspace(0, N_SHOTS - 1, POSE_SHOTS).astype(int)]
brain_est = jnp.where(interior, 1560.0, C_WATER).astype(jnp.float32)   # generic uniform-brain (no FWI needed)

# early-time window: the first arrivals transit the high-contrast SKULL before the
# brain reverberations arrive -> windowing makes the pose misfit skull-dominated and
# far less brain-dependent (breaks the pose<->brain chicken-and-egg).
NW = int(float(os.environ.get("BFWI_POSE_TWIN", "0.45")) * nsteps)
print(f"pose misfit window: first {NW}/{nsteps} samples (~{NW*dt*1e6:.0f}us)", flush=True)

def pose_loss(pose):
    skm = jnp.clip(rigid_warp_3d(template, pose["t"], jnp.deg2rad(pose["ang_deg"])), 0.0, 1.0)
    v = brain_est * (1 - skm) + C_SKULL * skm                  # warped skull on the generic brain
    med = build_medium(build_domain((S, S, S), dx), v, rho, pml_size=pml)
    tot = 0.0
    for k in range(POSE_SHOTS):
        pred = simulate_shot_sensors(med, ta, pose_src[k], pg, bp_lo, dt, checkpointed=True)
        mt = min(pred.shape[0], pose_obs[k].shape[0], NW); tot = tot + l2_loss(pred[:mt], pose_obs[k][:mt])
    return tot / POSE_SHOTS

def overlap(pose):
    a = rigid_warp_3d(template, pose["t"], jnp.deg2rad(pose["ang_deg"])) > 0.5
    return 100 * float(jnp.mean(a & (skull_true > 0.5)) / (jnp.mean(skull_true > 0.5) + 1e-9))

print(f"=== MOFI pose recovery ({POSE_BAND[0]/1e3:.0f}-{POSE_BAND[1]/1e3:.0f}kHz, generic-brain est) ===", flush=True)
pose = {"t": jnp.zeros(3), "ang_deg": jnp.zeros(3)}; opt = optax.adam(0.5); st = opt.init(pose); t0 = time.time()
vg_pose = jax.value_and_grad(pose_loss)
for i in range(POSE_STEPS):
    l, g = vg_pose(pose); up, st = opt.update(g, st); pose = optax.apply_updates(pose, up)
    if (i + 1) % 5 == 0:
        print(f"  pose {i+1}/{POSE_STEPS}: misfit {float(l):.4e}  t={np.round(np.asarray(pose['t']),2)}  "
              f"ang(deg)={np.round(np.asarray(pose['ang_deg']),2)}  overlap {overlap(pose):.0f}% ({time.time()-t0:.0f}s)", flush=True)
aligned = rigid_warp_3d(template, pose["t"], jnp.deg2rad(pose["ang_deg"])) > 0.5
ov0 = 100 * float(jnp.mean((template > 0.5) & (skull_true > 0.5)) / (jnp.mean(skull_true > 0.5) + 1e-9))
ov1 = 100 * float(jnp.mean(aligned & (skull_true > 0.5)) / (jnp.mean(skull_true > 0.5) + 1e-9))
print(f"skull overlap with truth: misaligned {ov0:.0f}% -> MOFI-aligned {ov1:.0f}%", flush=True)

# ---- brain FWI with the chosen skull (frozen), annealed-t ----------------------
BANDS = [[20e3, 45e3], [40e3, 80e3], [70e3, 120e3]]; ITERS = 10

def fwi(skull_mask, prior_lr=8.0, lr=20.0, t_hi=0.25, t_lo=0.04, lr_decay=0.3, prior_ramp=1.5):
    sk = jnp.asarray(skull_mask) > 0.5
    x = jnp.where(sk, C_SKULL, C_WATER).astype(jnp.float32); total = len(BANDS) * ITERS; gi = 0; t0 = time.time()
    def shot_loss(xx, sp, ot, bp):
        med = build_medium(build_domain((S, S, S), dx), xx, rho, pml_size=pml)
        pred = simulate_shot_sensors(med, ta, sp, pg, bp, dt, checkpointed=True)
        mt = min(pred.shape[0], ot.shape[0]); return l2_loss(pred[:mt], ot[:mt])
    vg = jax.value_and_grad(shot_loss)
    for bi, (fmin, fmax) in enumerate(BANDS):
        bp = _bandpass_signal(sig, dt, fmin, fmax)
        bobs = jax.vmap(lambda d: jax.vmap(lambda col: _bandpass_signal(col, dt, fmin, fmax))(d.T).T)(obs)
        for _ in range(ITERS):
            frac = gi / max(total - 1, 1)
            t_eps = t_hi + (t_lo - t_hi) * frac; lr_t = lr * (1 - (1 - lr_decay) * frac); plr = prior_lr * (1 + (prior_ramp - 1) * frac)
            g = jnp.zeros_like(x); gsq = jnp.zeros_like(x)
            for k in range(N_SHOTS):
                _, gs = vg(x, src[k], bobs[k], bp); g += gs; gsq += gs ** 2
            g /= N_SHOTS; g = g / (jnp.sqrt(gsq / N_SHOTS) + 1e-12 * jnp.max(jnp.sqrt(gsq / N_SHOTS)))
            g = _smooth_gradient(g, 1.0) * imask; g = g / (jnp.max(jnp.abs(g)) + 1e-30); x = x - lr_t * g
            xi = jnp.where(interior, x, C_WATER)
            s = model(((xi - SM) / SS).reshape(-1), t_eps).reshape(S, S, S) * imask
            x = x + plr * s / (jnp.max(jnp.abs(s)) + 1e-30); x = jnp.clip(x, 1400.0, 2900.0); gi += 1
        x.block_until_ready(); print(f"    band {bi} {time.time()-t0:.0f}s", flush=True)
    return np.asarray(x)

m = np.asarray(interior) > 0.5; ct = np.asarray(c_true); les = (np.asarray(crop) == LESION) & m; brn = m & ~les
def rep(tag, rec):
    ri = np.sqrt(np.mean((C_WATER - ct[m]) ** 2)); rf = np.sqrt(np.mean((rec[m] - ct[m]) ** 2))
    rb = np.sqrt(np.mean((rec[brn] - ct[brn]) ** 2)); lv = float(rec[les].mean()) if les.sum() else float("nan")
    print(f"  {tag}: cerebrum RMSE {ri:.1f}->{rf:.1f} ({100*(1-rf/ri):+.0f}%)  brain-only {rb:.1f}  lesion {lv:.0f} ({100*(lv-1500)/160:.0f}%)", flush=True)
    return rec
print("=== brain FWI: MOFI-aligned skull ===", flush=True); t0 = time.time()
ra = rep("MOFI-aligned", fwi(aligned))
print(f"runtime {time.time()-t0:.0f}s", flush=True)
np.savez("/tmp/mofi3d_arrays.npz", c_true=ct, mofi=ra, aligned=np.asarray(aligned), template=np.asarray(template > 0.5),
         skull_true=np.asarray(skull_true > 0.5), interior=m, lesion=les, ov_before=ov0, ov_after=ov1)
print("saved /tmp/mofi3d_arrays.npz", flush=True)
