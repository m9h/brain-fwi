"""Standalone GB10 (local Slurm) 3D DPS-FWI runner — reliable fallback for beam.

Same compute as scripts/beam_dps_fwi_3d.py but no beam wrapper: reads the prior /
crop / norm from /tmp and runs on the local GPU (jax[cuda13]). Use this when the
beam serverless pool is in a flaky-cuInit window (see docs/dev/cloud-gpu-venues.md).
Includes the modeling-error knobs and source co-inversion.

    # prereqs: build_birnbaum_3d_dataset.py + modal_train_unet3d.py (96^3 prior),
    #          and /tmp/<subj>_crop_96.npy (cerebrum_volume_crop)
    BFWI_SUBJECT=subj2 BFWI_SRC_MISMATCH=1.15 BFWI_NOISE_DB=20 BFWI_SRC_INV=1 \
        .venv/bin/python scripts/local_fwi_3d.py
"""
import os, time
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx
jax.config.update("jax_enable_x64", False)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from brain_fwi.inference.score_unet3d import UNet3DScore
from brain_fwi.phantoms.birnbaum import to_velocity, roi_mask, LESION, SKULL, C_WATER, C_SKULL
from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis,
    simulate_shot_sensors, generate_observed_data)
from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet
from brain_fwi.inversion.losses import l2_loss
from brain_fwi.inversion.fwi import _smooth_gradient, _bandpass_signal

SUBJ = os.environ.get("BFWI_SUBJECT", "subj2")
S = int(os.environ.get("BFWI_S", "96")); dx = 1.5e-3 if S == 96 else 3.0e-3; pml = 8
F0 = 80e3; T_END = 1.0e-4
BANDS = [[20e3, 45e3], [40e3, 80e3], [70e3, 120e3]]; ITERS = 10; N_SHOTS = 12
SRC_MISMATCH = float(os.environ.get("BFWI_SRC_MISMATCH", "1.0"))
NOISE_DB = float(os.environ.get("BFWI_NOISE_DB", "0"))
SRC_INV = os.environ.get("BFWI_SRC_INV", "0") == "1"
OUT = os.environ.get("BFWI_OUT", f"/tmp/local_fwi_{SUBJ}.png")
sfx = "_96" if S == 96 else ""
print(f"S={S} subj={SUBJ} src_mismatch={SRC_MISMATCH} noise_db={NOISE_DB} src_inv={SRC_INV}", flush=True)

crop = np.load(f"/tmp/{SUBJ}_crop_{S}.npy")
c_true = jnp.asarray(to_velocity(crop, with_skull=True)); skull = jnp.asarray(crop == SKULL)
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
gen_f0 = F0 * SRC_MISMATCH; sig_gen = ricker_wavelet(f0=gen_f0, dt=dt, n_samples=nsteps)
OBS_FILE = os.environ.get("BFWI_OBS_FILE", "")
if OBS_FILE:                                   # invert externally-generated obs (e.g. k-Wave -> cross-solver test)
    obs = jnp.asarray(np.load(OBS_FILE).astype(np.float32))
    print(f"loaded external obs {obs.shape} from {OBS_FILE}", flush=True)
else:
    obs = generate_observed_data(sound_speed=c_true, density=rho, dx=dx, src_positions_grid=src,
        sensor_positions_grid=pg, freq=gen_f0, pml_size=pml, time_axis=ta, source_signal=sig_gen, dt=dt, verbose=False)
if NOISE_DB > 0:
    nstd = jnp.sqrt(jnp.mean(obs ** 2)) * 10 ** (-NOISE_DB / 20.0)
    obs = obs + nstd * jr.normal(jr.PRNGKey(7), obs.shape)
print(f"helmet {ne} elems, {N_SHOTS} shots, {nsteps} steps, data f0={gen_f0/1e3:.0f}kHz invert {F0/1e3:.0f}kHz", flush=True)

model = eqx.tree_deserialise_leaves(f"/tmp/brain_score_3d{sfx}.eqx", UNet3DScore(S, C=16, key=jr.PRNGKey(0)))
nz = np.load(f"/tmp/brain_score_3d{sfx}_norm.npz"); SM, SS = float(nz["mean"]), float(nz["std"])


def forward_pred(x, sp, bp):
    med = build_medium(build_domain((S, S, S), dx), x, rho, pml_size=pml)
    return simulate_shot_sensors(med, ta, sp, pg, bp, dt, checkpointed=True)

def shot_loss(x, sp, ot, bp):
    pred = forward_pred(x, sp, bp); mt = min(pred.shape[0], ot.shape[0]); return l2_loss(pred[:mt], ot[:mt])
vg = jax.value_and_grad(shot_loss)

# source co-inversion (Pratt variable projection): per-freq filter phi shared
# across shots+receivers; applied to the prediction (phi held constant).
def estimate_phi(x, bp, bobs):
    NT = bobs.shape[1]; num = 0.0; den = 0.0
    for k in range(N_SHOTS):
        pred = forward_pred(x, src[k], bp); mt = min(pred.shape[0], NT)
        P = jnp.fft.rfft(jnp.pad(pred[:mt], ((0, NT - mt), (0, 0))), axis=0); D = jnp.fft.rfft(bobs[k], axis=0)
        num = num + jnp.sum(jnp.conj(P) * D, axis=1); den = den + jnp.sum(jnp.abs(P) ** 2, axis=1)
    return num / (den + 1e-3 * jnp.max(den))

def shot_loss_si(x, sp, ot, bp, phi):
    pred = forward_pred(x, sp, bp); NT = ot.shape[0]; mt = min(pred.shape[0], NT)
    Pp = jnp.fft.rfft(jnp.pad(pred[:mt], ((0, NT - mt), (0, 0))), axis=0)
    return l2_loss(jnp.fft.irfft(Pp * phi[:, None], n=NT, axis=0), ot)
vg_si = jax.value_and_grad(shot_loss_si)


def fwi(prior_lr, anneal, lr=20.0, t_hi=0.25, t_lo=0.04, lr_decay=0.3, prior_ramp=1.5):
    x = jnp.where(skull, C_SKULL, C_WATER).astype(jnp.float32); total = len(BANDS) * ITERS; gi = 0; t0 = time.time()
    for bi, (fmin, fmax) in enumerate(BANDS):
        bp = _bandpass_signal(sig, dt, fmin, fmax)
        bobs = jax.vmap(lambda d: jax.vmap(lambda col: _bandpass_signal(col, dt, fmin, fmax))(d.T).T)(obs)
        for _ in range(ITERS):
            frac = gi / max(total - 1, 1)
            if anneal:
                t_eps = t_hi + (t_lo - t_hi) * frac; lr_t = lr * (1 - (1 - lr_decay) * frac); plr = prior_lr * (1 + (prior_ramp - 1) * frac)
            else:
                t_eps, lr_t, plr = 0.1, lr, prior_lr
            g = jnp.zeros_like(x); gsq = jnp.zeros_like(x)
            phi = estimate_phi(x, bp, bobs) if SRC_INV else None
            for k in range(N_SHOTS):
                _, gs = (vg_si(x, src[k], bobs[k], bp, phi) if phi is not None else vg(x, src[k], bobs[k], bp))
                g += gs; gsq += gs ** 2
            g /= N_SHOTS; g = g / (jnp.sqrt(gsq / N_SHOTS) + 1e-12 * jnp.max(jnp.sqrt(gsq / N_SHOTS)))
            g = _smooth_gradient(g, 1.0) * imask; g = g / (jnp.max(jnp.abs(g)) + 1e-30); x = x - lr_t * g
            if prior_lr > 0:
                xi = jnp.where(interior, x, C_WATER)
                s = model(((xi - SM) / SS).reshape(-1), t_eps).reshape(S, S, S) * imask
                x = x + plr * s / (jnp.max(jnp.abs(s)) + 1e-30)
            x = jnp.clip(x, 1400.0, 2900.0); gi += 1
        x.block_until_ready(); print(f"    band {bi} {time.time()-t0:.0f}s", flush=True)
    return np.asarray(x)


m = np.asarray(interior) > 0.5; ct = np.asarray(c_true); les = (np.asarray(crop) == LESION) & m; brn = m & ~les
def rep(tag, rec):
    ri = np.sqrt(np.mean((C_WATER - ct[m]) ** 2)); rf = np.sqrt(np.mean((rec[m] - ct[m]) ** 2))
    rb = np.sqrt(np.mean((rec[brn] - ct[brn]) ** 2)); lv = float(rec[les].mean()) if les.sum() else float("nan")
    print(f"  {tag}: cerebrum RMSE {ri:.1f}->{rf:.1f} ({100*(1-rf/ri):+.0f}%)  brain-only {rb:.1f}  "
          f"lesion {lv:.0f} ({100*(lv-1500)/160:.0f}% contrast)", flush=True); return rec

if __name__ == "__main__":
    print("=== 96^3 FWI (local GB10) ===", flush=True); t0 = time.time()
    rb = rep("no prior  ", fwi(0.0, False)); ra = rep("annealed-t", fwi(8.0, True))
    print(f"runtime {time.time()-t0:.0f}s", flush=True)
    np.savez(OUT.replace(".png", "_arrays.npz"), c_true=ct, no_prior=rb, annealed=ra, interior=m, lesion=les)
    zc = int(np.round(np.where(les)[0].mean())) if les.sum() else S // 2
    win = dict(vmin=1490, vmax=1700, cmap="turbo")
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, (img, ttl) in zip(ax, [(ct[zc], "TRUE"), (rb[zc], "no prior"), (ra[zc], "annealed-t"), (np.abs(ra - ct)[zc] * m[zc], "|err|")]):
        kw = dict(vmin=0, vmax=150, cmap="magma") if "err" in ttl else win
        a.imshow(np.where(m[zc] | ("err" in ttl), img, np.nan), **kw)
        if les[zc].any(): a.contour(les[zc], levels=[0.5], colors="cyan", linewidths=0.8)
        a.set_title(ttl, fontsize=10); a.axis("off")
    fig.suptitle(f"96^3 FWI src_inv={SRC_INV} (mismatch {SRC_MISMATCH}, {NOISE_DB:.0f}dB), {SUBJ}, z={zc}", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT, dpi=120); print(f"saved {OUT}", flush=True)
