"""Joint brain + skull-pose refinement: close the +18% -> +44% gap.

Pipeline (job 1712): recovered-pose FWI +18% vs truth-pose +44% -- the gap is the residual
0.06-vox pose error (binarizing the near-truth pose flips ~2% of skull voxels; transcranial
FWI is that pose-sensitive). Fix: after the transmission pose recovery (98%), optimize the
BRAIN (FWI) and the POSE (jax.grad through the differentiable soft warp) JOINTLY against the
multi-band data misfit -- the pose gradient closes the last 0.06 vox while the brain rebuilds.
Two design points: (1) SOFT skull consistently in obs AND inversion -> the sub-voxel pose
error stays a tiny soft difference, not a 2% hard-mask flip; (2) start brain=water so it does
not pre-absorb the pose error (co-adaptation). imask excludes the (warped) skull from inversion.
Compares vs fixed-recovered (+18%) and truth (+44%).  BFWI_REFINE_POSE=0 -> fixed-pose control.
"""
import os, time, numpy as np
import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx, optax
jax.config.update("jax_enable_x64", False)
from brain_fwi.phantoms.birnbaum import to_velocity, roi_mask, LESION, SKULL, C_WATER, C_SKULL
from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis, simulate_shot_sensors)
from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet
from brain_fwi.inversion.losses import l2_loss
from brain_fwi.inversion.fwi import _smooth_gradient, _bandpass_signal
from brain_fwi.inference.score_unet3d import UNet3DScore
import sys; sys.path.insert(0, os.path.dirname(__file__))
from mofi3d import rigid_warp_3d

S = 96; dx = 1.5e-3; pml = 8; F0 = 80e3; T_END = 1.0e-4; N_SHOTS = 12
T_TRUE = jnp.array([4.0, 2.0, 0.0]); A_TRUE = jnp.array([0.0, 0.0, np.deg2rad(5.0)])
P0 = (3.94, 2.01, 5.10)                                    # recovered pose (transmission-staged, 98%)
BANDS = [[20e3, 45e3], [40e3, 80e3], [70e3, 120e3]]; ITERS = 10
REFINE = os.environ.get("BFWI_REFINE_POSE", "1") == "1"; POSE_LR = float(os.environ.get("BFWI_POSE_LR", "0.03"))
dom = build_domain((S, S, S), dx)

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
ta = build_time_axis(build_medium(dom, C_SKULL, 1000.0, pml_size=pml), cfl=0.3, t_end=T_END)
dt = float(ta.dt); nsteps = int(T_END / dt); sig = ricker_wavelet(f0=F0, dt=dt, n_samples=nsteps)

def soft_skull(ptx, pty, prz):                            # differentiable warp of the template
    return jnp.clip(rigid_warp_3d(skull_true, jnp.array([ptx, pty, 0.0]), jnp.array([0.0, 0.0, jnp.deg2rad(prz)])), 0, 1)
# the measurement: SOFT skull at the TRUE pose over the true brain
S_true = soft_skull(4.0, 2.0, 5.0)
brain_no_skull = jnp.where(skull_true > 0.5, C_WATER, c_true)
c_obs = (brain_no_skull * (1 - S_true) + C_SKULL * S_true).astype(jnp.float32)
med_true = build_medium(dom, c_obs, rho, pml_size=pml)
obs = [np.asarray(simulate_shot_sensors(med_true, ta, src[k], pg, sig, dt, checkpointed=False)) for k in range(N_SHOTS)]
model = eqx.tree_deserialise_leaves("/tmp/brain_score_3d_96.eqx", UNet3DScore(S, C=16, key=jr.PRNGKey(0)))
nz = np.load("/tmp/brain_score_3d_96_norm.npz"); SM, SS = float(nz["mean"]), float(nz["std"])
S_obs_bin = np.asarray(S_true > 0.5)
print(f"joint refine={REFINE} pose_lr={POSE_LR}; obs soft-skull@truth; start pose {P0}", flush=True)


def shot_loss(x, ptx, pty, prz, sp, ot, bp):
    Sp = soft_skull(ptx, pty, prz)
    v = jnp.where(interior, x, C_WATER) * (1 - Sp) + C_SKULL * Sp
    pred = simulate_shot_sensors(build_medium(dom, v, rho, pml_size=pml), ta, sp, pg, bp, dt, checkpointed=True)
    mt = min(pred.shape[0], ot.shape[0]); return l2_loss(pred[:mt], ot[:mt])
vgj = eqx.filter_jit(jax.value_and_grad(shot_loss, argnums=(0, 1, 2, 3)))

def overlap(ptx, pty, prz):
    a = np.asarray(soft_skull(ptx, pty, prz) > 0.5)
    return 100 * float((a & S_obs_bin).sum() / (S_obs_bin.sum() + 1e-9))

p = {"tx": P0[0], "ty": P0[1], "rz": P0[2]}; opt_p = optax.adam(POSE_LR); st_p = opt_p.init(p)
x = jnp.full((S, S, S), C_WATER, jnp.float32); total = len(BANDS) * ITERS; gi = 0; t0 = time.time()
for bi, (fmin, fmax) in enumerate(BANDS):
    bp = _bandpass_signal(sig, dt, fmin, fmax)
    bobs = [jax.vmap(lambda col: _bandpass_signal(col, dt, fmin, fmax))(jnp.asarray(o).T).T for o in obs]
    for _ in range(ITERS):
        frac = gi / max(total - 1, 1); t_eps = 0.25 + (0.04 - 0.25) * frac
        lr_t = 20.0 * (1 - 0.7 * frac); plr = 8.0 * (1 + 0.5 * frac)
        Sp = soft_skull(p["tx"], p["ty"], p["rz"]); imk = (interior & (Sp <= 0.5)).astype(jnp.float32)
        gx = jnp.zeros_like(x); gxsq = jnp.zeros_like(x); gp = {"tx": 0.0, "ty": 0.0, "rz": 0.0}; L = 0.0
        for k in range(N_SHOTS):
            l, (gxk, gtx, gty, grz) = vgj(x, p["tx"], p["ty"], p["rz"], src[k], bobs[k], bp)
            gx += gxk; gxsq += gxk ** 2; L += float(l)
            gp["tx"] += float(gtx); gp["ty"] += float(gty); gp["rz"] += float(grz)
        gx /= N_SHOTS; gx = gx / (jnp.sqrt(gxsq / N_SHOTS) + 1e-12 * jnp.max(jnp.sqrt(gxsq / N_SHOTS)))
        gx = _smooth_gradient(gx, 1.0) * imk; gx = gx / (jnp.max(jnp.abs(gx)) + 1e-30); x = x - lr_t * gx
        xi = jnp.where(imk > 0, x, C_WATER); s = model(((xi - SM) / SS).reshape(-1), t_eps).reshape(S, S, S) * imk
        x = x + plr * s / (jnp.max(jnp.abs(s)) + 1e-30); x = jnp.clip(x, 1400.0, 2900.0)
        if REFINE:                                        # pose step (jax.grad through the soft warp)
            gpn = {k2: jnp.asarray(gp[k2] / N_SHOTS) for k2 in gp}; up, st_p = opt_p.update(gpn, st_p); p = optax.apply_updates(p, up)
        gi += 1
        print(f"  b{bi} it{gi}: loss {L/N_SHOTS:.3e} pose=({p['tx']:.3f},{p['ty']:.3f},{p['rz']:.3f}) "
              f"ov {overlap(p['tx'],p['ty'],p['rz']):.0f}% err|dt|={np.hypot(p['tx']-4,p['ty']-2):.3f} ({time.time()-t0:.0f}s)", flush=True)

rec = np.asarray(x); m = (np.asarray(interior) > 0.5) & ~S_obs_bin; ct = np.asarray(c_true)
ri = np.sqrt(np.mean((C_WATER - ct[m]) ** 2)); rf = np.sqrt(np.mean((rec[m] - ct[m]) ** 2))
print(f"FINAL joint: pose=({p['tx']:.3f},{p['ty']:.3f},{p['rz']:.3f}) (true 4,2,5) overlap {overlap(p['tx'],p['ty'],p['rz']):.0f}%", flush=True)
print(f"  brain RMSE {ri:.1f}->{rf:.1f} ({100*(1-rf/ri):+.0f}%)   [fixed-recovered was +18%, truth-pose +44%]", flush=True)
np.save("/tmp/mofi_joint_rec.npy", rec)
