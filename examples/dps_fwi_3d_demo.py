"""Diffusion-prior-guided FWI demo (3D, real Birnbaum cerebrum volume).

The 3D counterpart of ``examples/dps_fwi_demo.py``. 3D removes both 2D blockers:
the cerebrum is a clean 3D connected component (no face entanglement) and the
helmet gives full angular coverage. Skull is given at truth (MOFI-style); a
preconditioned, ROI-masked FWI reconstructs the brain interior, guided by a
learned 3D U-Net diffusion prior (DPS-style: data-misfit gradient + score).
Compares prior-off vs prior-on on the cerebrum ROI. Runs on the GB10 (j-Wave 3D).

Prereqs (see docs/design/diffusion_prior_fwi.md):
  1. build dataset:  python scripts/build_birnbaum_3d_dataset.py
  2. train prior:    modal run scripts/modal_train_unet3d.py  ->  brain_score_3d.eqx + norm
Then (on the GB10):  python examples/dps_fwi_3d_demo.py

RESULT (held-out subj2, 48^3/3mm): the prior beats both no-prior and the water
init on the cerebrum ROI (+7..16%, vs no-prior which degrades) and roughly halves
the brain-tissue RMSE — the first time in this project a diffusion prior improves
over the starting model. The focal lesion stays under-recovered (~23% contrast):
resolution-limited at 48^3/3mm (needs finer grid / higher freq), not prior-limited.
"""
import argparse
import time
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brain_fwi.inference.score_unet3d import UNet3DScore
from brain_fwi.phantoms.birnbaum import (
    label_files, cerebrum_volume_crop, roi_mask, to_velocity, prior_volume,
    LESION, SKULL, BRAIN, C_WATER, C_SKULL)
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, simulate_shot_sensors, generate_observed_data)
from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet
from brain_fwi.inversion.losses import l2_loss
from brain_fwi.inversion.fwi import _smooth_gradient, _bandpass_signal

jax.config.update("jax_enable_x64", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/brain_score_3d.eqx")
    ap.add_argument("--norm", default="/tmp/brain_score_3d_norm.npz")
    ap.add_argument("--subject", default="subj2", help="held-out subject id")
    ap.add_argument("--out", default="/tmp/brain_recon_dps_3d.png")
    ap.add_argument("--S", type=int, default=48)
    ap.add_argument("--dx", type=float, default=3.0e-3)
    ap.add_argument("--n-shots", type=int, default=20)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--prior-lr", type=float, default=8.0)
    args = ap.parse_args()

    S, dx, pml = args.S, args.dx, 8
    F0, T_END = 60e3, 1.0e-4   # 100 us covers transmission across the ~S*dx helmet
    BANDS = ((20e3, 45e3), (40e3, 80e3))

    # ---- held-out subject cerebrum crop ----------------------------------
    f = [x for x in label_files() if f"{args.subject}_label" in x][0]
    lab = np.asarray(nib.load(f).dataobj).astype(np.int16)
    crop = cerebrum_volume_crop(lab, S=S)
    c_true = jnp.asarray(to_velocity(crop, with_skull=True))
    skull = jnp.asarray(crop == SKULL)
    interior = jnp.asarray(roi_mask(crop)); imask = interior.astype(jnp.float32)
    rho = jnp.full((S, S, S), 1000.0, jnp.float32)
    print(f"3D crop {crop.shape}, lesion vox {int((crop == LESION).sum())}, "
          f"cerebrum {int(np.asarray(interior).sum())}, skull {int(np.asarray(skull).sum())}", flush=True)

    # ---- helmet array, fixed shot set (bounded per-source compiles) ------
    c = (S // 2) * dx; r = (S // 2 - 4) * dx
    pos = helmet_array_3d(n_elements=80, center=(c, c, c), radius_ap=r, radius_lr=r,
                          radius_si=r, standoff=0.0, coverage_angle=3.1, exclude_face=False)
    pg = transducer_positions_to_grid(pos, dx, (S, S, S))
    ne = len(pg[0])
    allsrc = [(int(pg[0][i]), int(pg[1][i]), int(pg[2][i])) for i in range(ne)]
    src = [allsrc[i] for i in np.linspace(0, ne - 1, args.n_shots).astype(int)]

    ta = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml),
                         cfl=0.3, t_end=T_END)
    dt = float(ta.dt); nsteps = int(T_END / dt); sig = ricker_wavelet(f0=F0, dt=dt, n_samples=nsteps)
    print(f"helmet {ne} elems, {args.n_shots} fixed shots, {nsteps} steps, dt {dt*1e9:.0f} ns", flush=True)

    t_obs = time.time()
    obs = generate_observed_data(sound_speed=c_true, density=rho, dx=dx, src_positions_grid=src,
        sensor_positions_grid=pg, freq=F0, pml_size=pml, time_axis=ta, source_signal=sig, dt=dt, verbose=False)
    print(f"observed {obs.shape} in {time.time()-t_obs:.0f}s", flush=True)

    # ---- 3D prior --------------------------------------------------------
    model = eqx.tree_deserialise_leaves(args.model, UNet3DScore(S, C=16, key=jr.PRNGKey(0)))
    nz = np.load(args.norm); SM, SS = float(nz["mean"]), float(nz["std"])

    def shot_loss(x, sp, ot, bp):
        med = build_medium(build_domain((S, S, S), dx), x, rho, pml_size=pml)
        pred = simulate_shot_sensors(med, ta, sp, pg, bp, dt)
        mt = min(pred.shape[0], ot.shape[0])
        return l2_loss(pred[:mt], ot[:mt])

    vg = jax.value_and_grad(shot_loss)

    def fwi(prior_lr, t_eps=0.1, lr=20.0):
        x = jnp.where(skull, C_SKULL, C_WATER).astype(jnp.float32)
        for fmin, fmax in BANDS:
            bp = _bandpass_signal(sig, dt, fmin, fmax)
            bobs = jax.vmap(lambda d: jax.vmap(lambda col: _bandpass_signal(col, dt, fmin, fmax))(d.T).T)(obs)
            for _ in range(args.iters):
                g = jnp.zeros_like(x); gsq = jnp.zeros_like(x)
                for k in range(args.n_shots):
                    _, gs = vg(x, src[k], bobs[k], bp); g += gs; gsq += gs ** 2
                g /= args.n_shots
                g = g / (jnp.sqrt(gsq / args.n_shots) + 1e-12 * jnp.max(jnp.sqrt(gsq / args.n_shots)))
                g = _smooth_gradient(g, 1.0) * imask
                g = g / (jnp.max(jnp.abs(g)) + 1e-30)
                x = x - lr * g
                if prior_lr > 0:
                    xi = jnp.where(interior, x, C_WATER)
                    s = model(((xi - SM) / SS).reshape(-1), t_eps).reshape(S, S, S) * imask
                    x = x + prior_lr * s / (jnp.max(jnp.abs(s)) + 1e-30)
                x = jnp.clip(x, 1400.0, 2900.0)
            x.block_until_ready()
        return np.asarray(x)

    m = np.asarray(interior) > 0.5; ct = np.asarray(c_true)
    les = (np.asarray(crop) == LESION) & m; brn = m & ~les

    def rep(tag, rec):
        ri = np.sqrt(np.mean((C_WATER - ct[m]) ** 2)); rf = np.sqrt(np.mean((rec[m] - ct[m]) ** 2))
        rb = np.sqrt(np.mean((rec[brn] - ct[brn]) ** 2))
        lv = rec[les].mean() if les.sum() else float("nan")
        print(f"  {tag}: cerebrum RMSE {ri:.1f}->{rf:.1f} ({100*(1-rf/ri):+.0f}%)  brain-only {rb:.1f}  "
              f"lesion {lv:.0f} (true 1660, {100*(lv-1500)/160:.0f}% contrast)", flush=True)
        return rec

    print("=== 3D DPS-FWI ===", flush=True)
    t0 = time.time(); rb = rep("no prior ", fwi(0.0))
    rd = rep("3D prior ", fwi(args.prior_lr, 0.1))
    print(f"runtime {time.time()-t0:.0f}s", flush=True)

    zc = int(np.round(np.where(les)[0].mean())) if les.sum() else S // 2
    win = dict(vmin=1490, vmax=1700, cmap="turbo")
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [(ct[zc], "TRUE (brain window)", win), (rb[zc], "no prior", win),
              (rd[zc], "3D prior", win),
              (np.abs(rd - ct)[zc] * m[zc], "|error| prior", dict(vmin=0, vmax=150, cmap="magma"))]
    for a, (img, ttl, kw) in zip(ax, panels):
        im = a.imshow(np.where(m[zc] | ("error" in ttl), img, np.nan), **kw)
        if les[zc].any():
            a.contour(les[zc], levels=[0.5], colors="cyan", linewidths=0.8)
        a.set_title(ttl, fontsize=10); a.axis("off"); plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"3D DPS-FWI, held-out {args.subject} cerebrum, lesion-centred axial z={zc} "
                 f"(skull given at truth, masked out)", fontsize=11)
    plt.tight_layout(); plt.savefig(args.out, dpi=120)
    print(f"saved {args.out} (z={zc})", flush=True)


if __name__ == "__main__":
    main()
