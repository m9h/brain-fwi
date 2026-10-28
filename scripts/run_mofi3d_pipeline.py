"""Full unknown-skull -> pose -> brain pipeline (the MOFI-and-beyond demonstration).

One measurement (skull at an unknown 4vox/5deg pose, over the true brain). With only a
GENERIC brain estimate and a skull TEMPLATE (no pose given):
  1. recover the skull pose from data via the transmission-traveltime misfit + grid-seed
     staged recovery (project_mofi_pose_characterization: generic brain lands 98%);
  2. place the skull template at the RECOVERED pose;
  3. run annealed-t DPS brain FWI with that skull -> reconstruct the brain interior.
Compares against the skull-at-TRUTH baseline (+38%) and the misaligned-frozen catastrophe
(-400%). If recovered-pose FWI ~= truth-pose FWI, the unknown-skull->brain chain works end to
end -- which is strictly more than MOFI (Bates 2026), which stops at alignment.
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

S = 96; dx = 1.5e-3; pml = 8; F0 = 80e3; T_END = 1.4e-4; N_SHOTS = 12
T_TRUE = jnp.array([4.0, 2.0, 0.0]); A_TRUE = jnp.array([0.0, 0.0, np.deg2rad(5.0)])
POSE_SHOTS = 6; MAXLAG = 60; BETA = 30.0

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
# the ONE measurement: skull at the unknown true pose, over the true brain
skull_obs = jnp.clip(rigid_warp_3d(skull_true, T_TRUE, A_TRUE), 0, 1)            # SOFT (differentiable) -> pose stage
skull_obs_bin = (rigid_warp_3d(skull_true, T_TRUE, A_TRUE) > 0.5).astype(jnp.float32)  # BINARY -> FWI (matches its binary skull init)
brain_no_skull = jnp.where(skull_true > 0.5, C_WATER, c_true)
c_obs = (brain_no_skull * (1 - skull_obs) + C_SKULL * skull_obs).astype(jnp.float32)              # soft-skull medium for pose obs
# binary-skull medium for the FWI obs: soft skull -> unphysical edge velocities the binary-skull FWI can't fit -> divergence
# (job 1701: even truth-pose -413%). Binary obs + binary FWI skull = consistent -> +38%.
c_obs_bin = (brain_no_skull * (1 - skull_obs_bin) + C_SKULL * skull_obs_bin).astype(jnp.float32)
obs = generate_observed_data(sound_speed=c_obs, density=rho, dx=dx, src_positions_grid=src,
    sensor_positions_grid=pg, freq=F0, pml_size=pml, time_axis=ta, source_signal=sig, dt=dt, verbose=False)
# Two windows of the SAME measurement: the pose stage needs the LONG window (transmission arrivals,
# T_END=1.4e-4); the brain FWI needs the validated SHORT window (T_END=1e-4) -- the long window's late
# coda cycle-skips the reflection L2 FWI (job 1698 -> -413%; 1e-4 -> +38%, job 1699). dt is identical
# (same medium/cfl), so obs_f is just the leading slice of obs (causal forward).
TF = 1.0e-4; ta_f = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml), cfl=0.3, t_end=TF)
dt_f = float(ta_f.dt); nsteps_f = int(TF / dt_f); sig_f = ricker_wavelet(f0=F0, dt=dt_f, n_samples=nsteps_f)
# generate the FWI obs NATIVELY at ta_f/dt_f (do NOT slice the long-axis obs -- ta.dt != ta_f.dt would
# make the sliced data time-inconsistent with the FWI forward -> cycle-skip/divergence, job 1700 -407%).
obs_f = generate_observed_data(sound_speed=c_obs, density=rho, dx=dx, src_positions_grid=src,
    sensor_positions_grid=pg, freq=F0, pml_size=pml, time_axis=ta_f, source_signal=sig_f, dt=dt_f, verbose=False)  # SOFT skull obs
brain_est = jnp.where(interior, 1560.0, C_WATER).astype(jnp.float32)     # GENERIC brain (unknown interior)
ps = [src[i] for i in np.linspace(0, N_SHOTS - 1, POSE_SHOTS).astype(int)]
po = [obs[i] for i in np.linspace(0, N_SHOTS - 1, POSE_SHOTS).astype(int)]
print(f"=== PIPELINE: unknown skull pose, generic brain. obs {obs.shape}, {nsteps} steps ===", flush=True)


# ---------- STAGE 1: pose recovery (transmission TT, grid-seed + polish) ----------
def envelope(x):
    n = x.shape[0]; X = jnp.fft.fft(x, axis=0)
    h = jnp.zeros(n).at[0].set(1.0)
    h = (h.at[n // 2].set(1.0).at[1:n // 2].set(2.0)) if n % 2 == 0 else h.at[1:(n + 1) // 2].set(2.0)
    return jnp.abs(jnp.fft.ifft(X * h[:, None], axis=0))

def tt_pair(pred, obs_):
    ep = envelope(pred); ed = envelope(obs_); en = jnp.sum(ed ** 2, axis=0)
    epn = ep - jnp.mean(ep, axis=0); edn = ed - jnp.mean(ed, axis=0)
    epn = epn / (jnp.linalg.norm(epn, axis=0) + 1e-12); edn = edn / (jnp.linalg.norm(edn, axis=0) + 1e-12)
    nt = ep.shape[0]
    xc = jnp.fft.fftshift(jnp.fft.irfft(jnp.conj(jnp.fft.rfft(epn, axis=0)) * jnp.fft.rfft(edn, axis=0),
                                        n=nt, axis=0), axes=0)
    ctr = nt // 2; C = xc[ctr - MAXLAG:ctr + MAXLAG + 1]
    lags = jnp.arange(-MAXLAG, MAXLAG + 1).astype(jnp.float32)
    tau = jnp.sum(lags[:, None] * jax.nn.softmax(BETA * C, axis=0), axis=0)
    return jnp.sum(en * tau ** 2) / (jnp.sum(en) + 1e-30)

def pose_loss(q):
    skm = jnp.clip(rigid_warp_3d(skull_true, jnp.array([q["tx"], q["ty"], 0.0]),
                                 jnp.array([0.0, 0.0, jnp.deg2rad(q["rz"])])), 0, 1)
    v = brain_est * (1 - skm) + C_SKULL * skm
    med = build_medium(build_domain((S, S, S), dx), v, rho, pml_size=pml)
    tot = 0.0
    for k in range(POSE_SHOTS):
        pred = simulate_shot_sensors(med, ta, ps[k], pg, sig, dt, checkpointed=True)
        mt = min(pred.shape[0], po[k].shape[0]); tot = tot + tt_pair(pred[:mt], po[k][:mt])
    return tot / POSE_SHOTS

mf = eqx.filter_jit(pose_loss); vgp = eqx.filter_jit(jax.value_and_grad(pose_loss))
def overlap(q):
    a = rigid_warp_3d(skull_true, jnp.array([q["tx"], q["ty"], 0.0]), jnp.array([0.0, 0.0, np.deg2rad(q["rz"])])) > 0.5
    return 100 * float(jnp.mean(a & (skull_obs > 0.5)) / (jnp.mean(skull_obs > 0.5) + 1e-9))
def descend(p, active, steps, lr, tag):
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps)); st = opt.init(p); t0 = time.time()
    for i in range(steps):
        l, g = vgp(p); g = {k: (g[k] if k in active else jnp.zeros_like(g[k])) for k in g}
        up, st = opt.update(g, st); p = optax.apply_updates(p, up)
    print(f"  [{tag}] {steps} steps: t=({float(p['tx']):.2f},{float(p['ty']):.2f}) rz={float(p['rz']):.2f} "
          f"overlap {overlap(p):.0f}% ({time.time()-t0:.0f}s)", flush=True)
    return p

t0 = time.time()
if os.environ.get("BFWI_KNOWN_POSE", "0") == "1":          # skip the PROVEN pose recovery -> fast FWI-fix check
    ftx, fty, frz = 3.94, 2.01, 5.10                       # job 1697/1710 recovered pose (98% overlap)
    print(f"STAGE 1 SKIPPED: known recovered pose ({ftx},{fty},{frz})", flush=True)
else:
    GTX = np.arange(-6, 7, 3.0); GTY = np.arange(-6, 7, 3.0); best = (1e30, 0.0, 0.0)
    for gtx in GTX:
        for gty in GTY:
            m = float(mf({"tx": jnp.asarray(gtx, jnp.float32), "ty": jnp.asarray(gty, jnp.float32), "rz": jnp.asarray(0.0, jnp.float32)}))
            if m < best[0]:
                best = (m, float(gtx), float(gty))
    print(f"  prealign grid: (tx,ty)=({best[1]:.0f},{best[2]:.0f}) ({time.time()-t0:.0f}s)", flush=True)
    p = {"tx": jnp.array(best[1]), "ty": jnp.array(best[2]), "rz": jnp.array(0.0)}
    p = descend(p, {"tx", "ty"}, 20, 0.4, "translation")
    RZg = np.arange(-10, 16, 2.0); brz = (1e30, 0.0)
    for grz in RZg:
        m = float(mf({"tx": p["tx"], "ty": p["ty"], "rz": jnp.asarray(grz, jnp.float32)}))
        if m < brz[0]:
            brz = (m, float(grz))
    print(f"  rz-grid: best rz={brz[1]:.0f}", flush=True)
    p = {"tx": p["tx"], "ty": p["ty"], "rz": jnp.array(brz[1])}
    p = descend(p, {"tx", "ty", "rz"}, 20, 0.2, "joint")
    ftx, fty, frz = float(p["tx"]), float(p["ty"]), float(p["rz"]); ov = overlap(p)
    print(f"STAGE 1 done: recovered pose t=({ftx:.2f},{fty:.2f},0) rz={frz:.2f} (true 4,2,5) OVERLAP {ov:.0f}% "
          f"({time.time()-t0:.0f}s)", flush=True)
# ---------- STAGE 2: brain FWI (annealed-t DPS), SOFT skull (no binarization) ----------
# job 1713: binarizing the warped skull amplifies the sub-voxel pose error (~2% mask flip) -> +18%; a SOFT skull
# used CONSISTENTLY in obs AND inversion keeps it a tiny soft diff -> +52% (beats even binary-at-truth +44%).
from brain_fwi.inference.score_unet3d import UNet3DScore
model = eqx.tree_deserialise_leaves("/tmp/brain_score_3d_96.eqx", UNet3DScore(S, C=16, key=jr.PRNGKey(0)))
nz = np.load("/tmp/brain_score_3d_96_norm.npz"); SM, SS = float(nz["mean"]), float(nz["std"])
BANDS = [[20e3, 45e3], [40e3, 80e3], [70e3, 120e3]]; ITERS = 10

def soft_skull_at(ptx, pty, prz):
    return jnp.clip(rigid_warp_3d(skull_true, jnp.array([ptx, pty, 0.0]), jnp.array([0.0, 0.0, jnp.deg2rad(prz)])), 0, 1)

def shot_loss(x, Sp, sp, ot, bp):
    v = jnp.where(interior, x, C_WATER) * (1 - Sp) + C_SKULL * Sp           # soft skull medium
    pred = simulate_shot_sensors(build_medium(build_domain((S, S, S), dx), v, rho, pml_size=pml), ta_f, sp, pg, bp, dt_f, checkpointed=True)
    mt = min(pred.shape[0], ot.shape[0]); return l2_loss(pred[:mt], ot[:mt])
vg2 = jax.value_and_grad(shot_loss)

def fwi(Sp):
    imk = (interior & (Sp <= 0.5)).astype(jnp.float32)   # invert BRAIN only; hold the (soft) skull, which intrudes into roi_mask
    x = jnp.full((S, S, S), C_WATER, jnp.float32); total = len(BANDS) * ITERS; gi = 0; t0 = time.time()
    for bi, (fmin, fmax) in enumerate(BANDS):
        bp = _bandpass_signal(sig_f, dt_f, fmin, fmax)
        bobs = jax.vmap(lambda d: jax.vmap(lambda col: _bandpass_signal(col, dt_f, fmin, fmax))(d.T).T)(obs_f)
        for _ in range(ITERS):
            frac = gi / max(total - 1, 1); t_eps = 0.25 + (0.04 - 0.25) * frac
            lr_t = 20.0 * (1 - 0.7 * frac); plr = 8.0 * (1 + 0.5 * frac)
            g = jnp.zeros_like(x); gsq = jnp.zeros_like(x)
            for k in range(N_SHOTS):
                _, gs = vg2(x, Sp, src[k], bobs[k], bp); g += gs; gsq += gs ** 2
            g /= N_SHOTS; g = g / (jnp.sqrt(gsq / N_SHOTS) + 1e-12 * jnp.max(jnp.sqrt(gsq / N_SHOTS)))
            g = _smooth_gradient(g, 1.0) * imk; g = g / (jnp.max(jnp.abs(g)) + 1e-30); x = x - lr_t * g
            xi = jnp.where(imk > 0, x, C_WATER); s = model(((xi - SM) / SS).reshape(-1), t_eps).reshape(S, S, S) * imk
            x = x + plr * s / (jnp.max(jnp.abs(s)) + 1e-30); x = jnp.clip(x, 1400.0, 2900.0); gi += 1
        x.block_until_ready(); print(f"    FWI band {bi} {time.time()-t0:.0f}s", flush=True)
    return np.asarray(x)

m = (np.asarray(interior) > 0.5) & ~(np.asarray(skull_obs_bin) > 0.5); ct = np.asarray(c_true)  # imageable brain
les = (np.asarray(crop) == LESION) & m; brn = m & ~les
ri = np.sqrt(np.mean((C_WATER - ct[m]) ** 2))
def rep(tag, rec):
    rf = np.sqrt(np.mean((rec[m] - ct[m]) ** 2)); rb = np.sqrt(np.mean((rec[brn] - ct[brn]) ** 2))
    print(f"  {tag}: cerebrum RMSE {ri:.1f}->{rf:.1f} ({100*(1-rf/ri):+.0f}%)  brain-only {rb:.1f}", flush=True)

print("=== STAGE 2: brain FWI (soft skull) ===", flush=True)
rep("RECOVERED-pose FWI", fwi(soft_skull_at(ftx, fty, frz)))   # live-recovered pose -> end-to-end pipeline result
rep("TRUTH-pose FWI    ", fwi(soft_skull_at(4.0, 2.0, 5.0)))   # baseline (skull given exactly)
print("(soft-skull: recovered ~+52%; binary-recovered was +18%; misaligned-frozen ~-400%)", flush=True)
