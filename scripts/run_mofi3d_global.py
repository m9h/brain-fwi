"""Global-search MOFI: land the skull pose via grid-search + gradient refine.

Gradient descent on the j-Wave misfit does NOT land a 4vox/5deg pose (under-
constrained out-of-plane DOF + shallow landscape). Fix: GLOBAL grid search over the
data-CONSTRAINED in-plane subspace (tx, ty, rz) -- forward-only, robust to the bad
landscape, no out-of-plane drift -- to get into the basin, then a short gradient
refine to land sub-voxel. Skull-dominated misfit: early-time window (short time axis)
+ 40-80kHz + generic-brain estimate. Gate the brain FWI on a landed pose.
"""
import os, time, itertools, numpy as np
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
T_PERT = jnp.array([4.0, 2.0, 0.0]); A_PERT = jnp.array([0.0, 0.0, np.deg2rad(5.0)])
POSE_SHOTS = 4; POSE_BAND = (40e3, 80e3); TWIN_US = 45e-6                # early window

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
template = rigid_warp_3d(skull_true, T_PERT, A_PERT)
print(f"template overlap {100*float(jnp.mean((template>0.5)&(skull_true>0.5))/(jnp.mean(skull_true>0.5)+1e-9)):.0f}%", flush=True)

# short time axis (early window) for the pose search -> skull-dominated + ~2x faster
NW = int(TWIN_US / dt); ta_p = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml),
                                               cfl=0.3, t_end=NW * dt)
sig_p = ricker_wavelet(f0=F0, dt=dt, n_samples=NW); bp_p = _bandpass_signal(sig_p, dt, *POSE_BAND)
ps = [src[i] for i in np.linspace(0, N_SHOTS - 1, POSE_SHOTS).astype(int)]
po = [jax.vmap(lambda col: _bandpass_signal(col, dt, *POSE_BAND))(obs[i][:NW].T).T for i in np.linspace(0, N_SHOTS - 1, POSE_SHOTS).astype(int)]
brain_est = jnp.where(interior, 1560.0, C_WATER).astype(jnp.float32)
print(f"pose search: {NW} steps (~{NW*dt*1e6:.0f}us), {POSE_BAND[0]/1e3:.0f}-{POSE_BAND[1]/1e3:.0f}kHz, {POSE_SHOTS} shots", flush=True)


@eqx.filter_jit
def misfit(tx, ty, rz):                                        # in-plane pose, forward-only, early window
    skm = jnp.clip(rigid_warp_3d(template, jnp.array([tx, ty, 0.0]), jnp.array([0.0, 0.0, jnp.deg2rad(rz)])), 0, 1)
    v = brain_est * (1 - skm) + C_SKULL * skm
    med = build_medium(build_domain((S, S, S), dx), v, rho, pml_size=pml)
    tot = 0.0
    for k in range(POSE_SHOTS):
        pred = simulate_shot_sensors(med, ta_p, ps[k], pg, bp_p, dt, checkpointed=True)
        mt = min(pred.shape[0], po[k].shape[0]); tot = tot + l2_loss(pred[:mt], po[k][:mt])
    return tot / POSE_SHOTS

def overlap(tx, ty, rz):
    a = rigid_warp_3d(template, jnp.array([tx, ty, 0.0]), jnp.array([0.0, 0.0, np.deg2rad(rz)])) > 0.5
    return 100 * float(jnp.mean(a & (skull_true > 0.5)) / (jnp.mean(skull_true > 0.5) + 1e-9))

# ---- coarse grid over (tx, ty, rz) ---------------------------------------------
print("=== global grid search (tx, ty, rz) ===", flush=True); t0 = time.time()
TXS = np.arange(-8, 3, 2.0); TYS = np.arange(-6, 3, 2.0); RZS = np.arange(-10, 3, 2.5)
best = (1e30, 0, 0, 0)
_f = lambda v: jnp.asarray(v, jnp.float32)                     # traced args -> misfit compiles ONCE
for i, (tx, ty, rz) in enumerate(itertools.product(TXS, TYS, RZS)):
    m = float(misfit(_f(tx), _f(ty), _f(rz)))
    if m < best[0]:
        best = (m, tx, ty, rz)
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(TXS)*len(TYS)*len(RZS)} evals, best misfit {best[0]:.3e} @ t=({best[1]},{best[2]}) rz={best[3]} "
              f"overlap {overlap(best[1],best[2],best[3]):.0f}% ({time.time()-t0:.0f}s)", flush=True)
bm, btx, bty, brz = best
print(f"COARSE best: misfit {bm:.3e} t=({btx},{bty}) rz={brz}  overlap {overlap(btx,bty,brz):.0f}% ({time.time()-t0:.0f}s)", flush=True)

# ---- gradient refine (in-plane only, from the coarse best -> sub-voxel) ---------
p = {"tx": jnp.array(float(btx)), "ty": jnp.array(float(bty)), "rz": jnp.array(float(brz))}
opt = optax.adam(0.3); st = opt.init(p); vg = jax.value_and_grad(lambda q: misfit(q["tx"], q["ty"], q["rz"]))
for i in range(25):
    l, g = vg(p); up, st = opt.update(g, st); p = optax.apply_updates(p, up)
    if (i + 1) % 5 == 0:
        print(f"  refine {i+1}: misfit {float(l):.3e} t=({float(p['tx']):.2f},{float(p['ty']):.2f}) rz={float(p['rz']):.2f} "
              f"overlap {overlap(float(p['tx']),float(p['ty']),float(p['rz'])):.0f}%", flush=True)
ftx, fty, frz = float(p["tx"]), float(p["ty"]), float(p["rz"])
ov = overlap(ftx, fty, frz)
print(f"FINAL pose: t=({ftx:.2f},{fty:.2f},0) rz={frz:.2f}deg (true t=(-4,-2,0) rz=-5)  OVERLAP {ov:.0f}%", flush=True)
aligned = rigid_warp_3d(template, jnp.array([ftx, fty, 0.0]), jnp.array([0.0, 0.0, np.deg2rad(frz)])) > 0.5
np.save("/tmp/mofi_global_aligned.npy", np.asarray(aligned))

# ---- brain FWI with the aligned skull (gated on a landed pose) ------------------
if ov < 75:
    print(f"pose did NOT land (overlap {ov:.0f}% < 75%) -- skipping brain FWI", flush=True); raise SystemExit(0)
print("=== pose landed -> brain FWI (annealed-t) ===", flush=True)
model = eqx.tree_deserialise_leaves("/tmp/brain_score_3d_96.eqx",
    __import__("brain_fwi.inference.score_unet3d", fromlist=["UNet3DScore"]).UNet3DScore(S, C=16, key=jr.PRNGKey(0)))
nz = np.load("/tmp/brain_score_3d_96_norm.npz"); SM, SS = float(nz["mean"]), float(nz["std"])
BANDS = [[20e3, 45e3], [40e3, 80e3], [70e3, 120e3]]; ITERS = 10
sk = jnp.asarray(aligned) > 0.5
def shot_loss(x, sp, ot, bp):
    med = build_medium(build_domain((S, S, S), dx), x, rho, pml_size=pml)
    pred = simulate_shot_sensors(med, ta, sp, pg, bp, dt, checkpointed=True)
    mt = min(pred.shape[0], ot.shape[0]); return l2_loss(pred[:mt], ot[:mt])
vg2 = jax.value_and_grad(shot_loss); x = jnp.where(sk, C_SKULL, C_WATER).astype(jnp.float32); total = len(BANDS) * ITERS; gi = 0; t0 = time.time()
for bi, (fmin, fmax) in enumerate(BANDS):
    bp = _bandpass_signal(sig, dt, fmin, fmax)
    bobs = jax.vmap(lambda d: jax.vmap(lambda col: _bandpass_signal(col, dt, fmin, fmax))(d.T).T)(obs)
    for _ in range(ITERS):
        frac = gi / max(total - 1, 1); t_eps = 0.25 + (0.04 - 0.25) * frac; lr_t = 20.0 * (1 - 0.7 * frac); plr = 8.0 * (1 + 0.5 * frac)
        g = jnp.zeros_like(x); gsq = jnp.zeros_like(x)
        for k in range(N_SHOTS):
            _, gs = vg2(x, src[k], bobs[k], bp); g += gs; gsq += gs ** 2
        g /= N_SHOTS; g = g / (jnp.sqrt(gsq / N_SHOTS) + 1e-12 * jnp.max(jnp.sqrt(gsq / N_SHOTS)))
        g = _smooth_gradient(g, 1.0) * imask; g = g / (jnp.max(jnp.abs(g)) + 1e-30); x = x - lr_t * g
        xi = jnp.where(interior, x, C_WATER); s = model(((xi - SM) / SS).reshape(-1), t_eps).reshape(S, S, S) * imask
        x = x + plr * s / (jnp.max(jnp.abs(s)) + 1e-30); x = jnp.clip(x, 1400.0, 2900.0); gi += 1
    x.block_until_ready(); print(f"    band {bi} {time.time()-t0:.0f}s", flush=True)
rec = np.asarray(x); m = np.asarray(interior) > 0.5; ct = np.asarray(c_true); les = (np.asarray(crop) == LESION) & m; brn = m & ~les
ri = np.sqrt(np.mean((C_WATER - ct[m]) ** 2)); rf = np.sqrt(np.mean((rec[m] - ct[m]) ** 2)); rb = np.sqrt(np.mean((rec[brn] - ct[brn]) ** 2))
lv = float(rec[les].mean()) if les.sum() else float("nan")
print(f"  MOFI-global brain FWI: cerebrum RMSE {ri:.1f}->{rf:.1f} ({100*(1-rf/ri):+.0f}%)  brain-only {rb:.1f}  lesion {lv:.0f} ({100*(lv-1500)/160:.0f}%)", flush=True)
print("(skull-at-truth baseline +38%; misaligned-frozen -400%)", flush=True)
