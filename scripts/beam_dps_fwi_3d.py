"""96^3 DPS-FWI (annealed-t) on beam.cloud — the focal-lesion test, off the GB10.

beam is the right venue for this long FWI: no function timeout (Modal's 2 h cap
killed FWI) and it doesn't block the GB10's gpu partition (which is mutually
exclusive with the cpu partition the hbn jobs use). j-Wave is dispatch-bound at
these grids, so A10G/RTX4090 (~0.008 s/step) run a 96^3 FWI in a few hours.

The entrypoint (.venv-beam, pure-python) ships the trained 96^3 prior, the
held-out subject's label crop, and norm stats as bytes; the GPU function rebuilds
everything from the image's brain_fwi and runs no-prior vs annealed-t DPS-FWI,
returning a PNG + npz + metrics. Run:

    python scripts/build_birnbaum_3d_dataset.py --out /tmp/birnbaum_3d_dataset_96.npz --S 96
    BFWI_DATASET=/tmp/birnbaum_3d_dataset_96.npz BFWI_GPU=A100 BFWI_TAG=96 \\
        .venv/bin/modal run scripts/modal_train_unet3d.py     # -> /tmp/brain_score_3d_96.eqx
    .venv/bin/python -c "import numpy as np,nibabel as nib; from brain_fwi.phantoms import birnbaum; \\
        f=[x for x in birnbaum.label_files() if 'subj2_label' in x][0]; \\
        np.save('/tmp/subj2_crop_96.npy', birnbaum.cerebrum_volume_crop(np.asarray(nib.load(f).dataobj).astype('int16'),S=96).astype('int16'))"
    (cd scripts && ../.venv-beam/bin/python beam_dps_fwi_3d.py)
"""
import os
from beam import function, Image

GIT_BRANCH = "feature/diffusion-prior-fwi"
GPU = os.environ.get("BFWI_BEAM_GPU", "RTX4090")   # A10G / RTX4090 (24 GB) — serverless tier

image = (
    Image(python_version="python3.11")
    .add_commands([
        "apt-get update && apt-get install -y git build-essential",
        f"git clone --depth 1 --branch {GIT_BRANCH} https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && pip install -e '.[cuda12]'",
    ])
    .with_envs(["JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9"])
)


@function(gpu=GPU, image=image, timeout=6 * 3600)
def run(prior_bytes, crop_bytes, norm_bytes, cfg):
    import io, sys, time
    sys.path.insert(0, "/opt/brain-fwi")
    import jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx
    jax.config.update("jax_enable_x64", False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from brain_fwi.inference.score_unet3d import UNet3DScore
    from brain_fwi.phantoms.birnbaum import (to_velocity, roi_mask, LESION, SKULL,
                                             C_WATER, C_SKULL)
    from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis,
        simulate_shot_sensors, generate_observed_data)
    from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
    from brain_fwi.utils.wavelets import ricker_wavelet
    from brain_fwi.inversion.losses import l2_loss
    from brain_fwi.inversion.fwi import _smooth_gradient, _bandpass_signal
    print("devices:", jax.devices(), flush=True)

    S, dx, pml = cfg["S"], cfg["dx"], 8
    F0, T_END = cfg["f0"], cfg["t_end"]
    BANDS = cfg["bands"]; ITERS = cfg["iters"]; N_SHOTS = cfg["n_shots"]

    crop = np.load(io.BytesIO(crop_bytes))
    c_true = jnp.asarray(to_velocity(crop, with_skull=True))
    skull = jnp.asarray(crop == SKULL)
    interior = jnp.asarray(roi_mask(crop)); imask = interior.astype(jnp.float32)
    rho = jnp.full((S, S, S), 1000.0, jnp.float32)
    print(f"crop {crop.shape}, lesion {int((crop==LESION).sum())}, "
          f"cerebrum {int(np.asarray(interior).sum())}, skull {int(np.asarray(skull).sum())}", flush=True)

    c = (S // 2) * dx; r = (S // 2 - 4) * dx
    pos = helmet_array_3d(n_elements=80, center=(c, c, c), radius_ap=r, radius_lr=r,
                          radius_si=r, standoff=0.0, coverage_angle=3.1, exclude_face=False)
    pg = transducer_positions_to_grid(pos, dx, (S, S, S)); ne = len(pg[0])
    allsrc = [(int(pg[0][i]), int(pg[1][i]), int(pg[2][i])) for i in range(ne)]
    src = [allsrc[i] for i in np.linspace(0, ne - 1, N_SHOTS).astype(int)]
    ta = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml),
                         cfl=0.3, t_end=T_END)
    dt = float(ta.dt); nsteps = int(T_END / dt); sig = ricker_wavelet(f0=F0, dt=dt, n_samples=nsteps)
    print(f"helmet {ne} elems, {N_SHOTS} shots, {nsteps} steps, dt {dt*1e9:.0f} ns", flush=True)

    t_obs = time.time()
    obs = generate_observed_data(sound_speed=c_true, density=rho, dx=dx, src_positions_grid=src,
        sensor_positions_grid=pg, freq=F0, pml_size=pml, time_axis=ta, source_signal=sig, dt=dt, verbose=False)
    print(f"obs {obs.shape} in {time.time()-t_obs:.0f}s", flush=True)

    model = eqx.tree_deserialise_leaves(io.BytesIO(prior_bytes), UNet3DScore(S, C=16, key=jr.PRNGKey(0)))
    nz = np.load(io.BytesIO(norm_bytes)); SM, SS = float(nz["mean"]), float(nz["std"])

    def shot_loss(x, sp, ot, bp):
        med = build_medium(build_domain((S, S, S), dx), x, rho, pml_size=pml)
        # segmented checkpointing: O(sqrt(N)) backward memory so the 96^3 grad
        # fits the 24 GB serverless card (full history would be ~6 GiB/tensor).
        pred = simulate_shot_sensors(med, ta, sp, pg, bp, dt, checkpointed=cfg.get("checkpointed", True))
        mt = min(pred.shape[0], ot.shape[0]); return l2_loss(pred[:mt], ot[:mt])
    vg = jax.value_and_grad(shot_loss)

    def fwi(prior_lr, anneal, data_gate=0.0, lr=20.0, t_hi=0.25, t_lo=0.04, lr_decay=0.3, prior_ramp=1.5):
        x = jnp.where(skull, C_SKULL, C_WATER).astype(jnp.float32)
        total = len(BANDS) * ITERS; gi = 0; t0 = time.time()
        for bi, (fmin, fmax) in enumerate(BANDS):
            bp = _bandpass_signal(sig, dt, fmin, fmax)
            bobs = jax.vmap(lambda d: jax.vmap(lambda col: _bandpass_signal(col, dt, fmin, fmax))(d.T).T)(obs)
            for _ in range(ITERS):
                frac = gi / max(total - 1, 1)
                if anneal:
                    t_eps = t_hi + (t_lo - t_hi) * frac
                    lr_t = lr * (1.0 - (1.0 - lr_decay) * frac)
                    plr = prior_lr * (1.0 + (prior_ramp - 1.0) * frac)
                else:
                    t_eps, lr_t, plr = 0.1, lr, prior_lr
                g = jnp.zeros_like(x); gsq = jnp.zeros_like(x)
                for k in range(N_SHOTS):
                    _, gs = vg(x, src[k], bobs[k], bp); g += gs; gsq += gs ** 2
                g /= N_SHOTS
                g = g / (jnp.sqrt(gsq / N_SHOTS) + 1e-12 * jnp.max(jnp.sqrt(gsq / N_SHOTS)))
                g = _smooth_gradient(g, 1.0) * imask
                g = g / (jnp.max(jnp.abs(g)) + 1e-30)         # |g| in [0,1]
                x = x - lr_t * g
                if prior_lr > 0:
                    xi = jnp.where(interior, x, C_WATER)
                    s = model(((xi - SM) / SS).reshape(-1), t_eps).reshape(S, S, S) * imask
                    nudge = s / (jnp.max(jnp.abs(s)) + 1e-30)
                    if data_gate > 0:
                        # measurement-guided: back off the prior where the data still
                        # strongly wants to change x (e.g. the focal lesion, which holds a
                        # residual data-gradient because prior & data disagree there).
                        nudge = nudge * (1.0 - jnp.abs(g)) ** data_gate
                    x = x + plr * nudge
                x = jnp.clip(x, 1400.0, 2900.0); gi += 1
            x.block_until_ready()
            print(f"    band {bi} [{fmin/1e3:.0f}-{fmax/1e3:.0f}kHz] {time.time()-t0:.0f}s", flush=True)
        return np.asarray(x)

    m = np.asarray(interior) > 0.5; ct = np.asarray(c_true)
    les = (np.asarray(crop) == LESION) & m; brn = m & ~les
    out = {}
    def rep(tag, rec):
        ri = np.sqrt(np.mean((C_WATER - ct[m]) ** 2)); rf = np.sqrt(np.mean((rec[m] - ct[m]) ** 2))
        rb = np.sqrt(np.mean((rec[brn] - ct[brn]) ** 2)); lv = float(rec[les].mean()) if les.sum() else float("nan")
        line = (f"{tag}: cerebrum RMSE {ri:.1f}->{rf:.1f} ({100*(1-rf/ri):+.0f}%)  brain-only {rb:.1f}  "
                f"lesion {lv:.0f} (true 1660, {100*(lv-1500)/160:.0f}% contrast)")
        print("  " + line, flush=True); out[tag.strip()] = line; return rec

    # (label, prior_lr, anneal, data_gate)
    modes = cfg.get("modes", [("no prior  ", 0.0, False, 0.0),
                              ("annealed-t", cfg["prior_lr"], True, 0.0),
                              ("ann+gate  ", cfg["prior_lr"], True, cfg.get("gate_exp", 1.0))])
    print("=== 96^3 DPS-FWI on beam ===", flush=True); t0 = time.time()
    recs = {}
    for label, plr0, anneal, gate in modes:
        recs[label.strip()] = rep(label, fwi(plr0, anneal=anneal, data_gate=gate))
    runtime = time.time() - t0; print(f"runtime {runtime:.0f}s", flush=True)

    nb = io.BytesIO(); np.savez(nb, c_true=ct, interior=m, lesion=les, **recs)

    zc = int(np.round(np.where(les)[0].mean())) if les.sum() else S // 2
    win = dict(vmin=1490, vmax=1700, cmap="turbo")
    last = list(recs.values())[-1]
    panels = [(ct[zc], "TRUE", win)]
    panels += [(recs[lbl.strip()][zc], lbl.strip(), win) for lbl, *_ in modes]
    panels += [(np.abs(last - ct)[zc] * m[zc], "|err| " + modes[-1][0].strip(), dict(vmin=0, vmax=150, cmap="magma"))]
    fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.2))
    for a, (img, ttl, kw) in zip(ax, panels):
        im = a.imshow(np.where(m[zc] | ttl.startswith("|err"), img, np.nan), **kw)
        if les[zc].any(): a.contour(les[zc], levels=[0.5], colors="cyan", linewidths=0.8)
        a.set_title(ttl, fontsize=10); a.axis("off"); plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"96^3 DPS-FWI (beam {GPU}), held-out {cfg.get('subj', '?')}, lesion-centred z={zc}", fontsize=11)
    plt.tight_layout(); pb = io.BytesIO(); plt.savefig(pb, dpi=120, format="png")
    return {"png": pb.getvalue(), "npz": nb.getvalue(), "metrics": out,
            "runtime_s": round(runtime, 0), "zc": zc, "gpu": GPU}


if __name__ == "__main__":
    subj = os.environ.get("BFWI_SUBJECT", "subj2")
    out = os.environ.get("BFWI_OUT", f"/tmp/brain_recon_dps_3d_96_{subj}.png")
    cfg = {"S": 96, "dx": 1.5e-3, "f0": 80e3, "t_end": 1.0e-4,
           "bands": [[20e3, 45e3], [40e3, 80e3], [70e3, 120e3]],
           "iters": 10, "n_shots": 12, "prior_lr": 8.0, "checkpointed": True,
           "gate_exp": float(os.environ.get("BFWI_GATE", "1.0")), "subj": subj}
    _mode = os.environ.get("BFWI_MODE")
    if _mode == "gate-only":          # data-gated prior vs no-prior (trade-off experiment)
        cfg["modes"] = [["no prior  ", 0.0, False, 0.0], ["ann+gate  ", cfg["prior_lr"], True, cfg["gate_exp"]]]
    elif _mode == "anneal-only":      # generalization: the proven-best scheme vs no-prior
        cfg["modes"] = [["no prior  ", 0.0, False, 0.0], ["annealed-t", cfg["prior_lr"], True, 0.0]]
    prior = open("/tmp/brain_score_3d_96.eqx", "rb").read()
    crop = open(f"/tmp/{subj}_crop_96.npy", "rb").read()
    norm = open("/tmp/brain_score_3d_96_norm.npz", "rb").read()
    print(f"=== launching 96^3 DPS-FWI on beam {GPU} (gate_exp={cfg['gate_exp']}) ===", flush=True)
    res = run.remote(prior, crop, norm, cfg)
    if not res:
        print("FAILED: null result"); raise SystemExit(1)
    open(out, "wb").write(res["png"])
    open(out.replace(".png", "_arrays.npz"), "wb").write(res["npz"])
    print(f"\nGPU {res['gpu']}  runtime {res['runtime_s']}s  slice z={res['zc']}")
    for k, v in res["metrics"].items():
        print("  " + v)
    print(f"saved {out} + _arrays.npz")
