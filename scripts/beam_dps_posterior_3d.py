"""Reverse-diffusion DPS (measurement-guided posterior sampling) for 96^3 FWI.

The principled attempt at the bulk-vs-lesion trade-off that MAP (annealed-t)
can't resolve. Instead of a fixed-t prior nudge, run the reverse VP-SDE with the
score augmented by the FWI data-likelihood gradient evaluated at the Tweedie
estimate x̂_0 (Chung et al. 2023). FWI is nonlinear (cycle-skips from a bad
start), so we use WARM-START DPS: init from the annealed-t MAP noised to t_start,
then reverse-diffuse to 0 (t_start=1.0 -> pure from-noise DPS). The data term
uses a low band + few fixed shots (cheap; avoids cycle-skip).

Compares MAP (loaded) vs DPS-refined on the cerebrum ROI + lesion. Runs on beam.
Ship the MAP recon, prior, crop, norm as bytes; all compute is remote.

    (cd scripts && BFWI_SUBJECT=subj2 BFWI_TSTART=0.6 BFWI_ZETA=0.3 \
        ../.venv-beam/bin/python beam_dps_posterior_3d.py)
"""
import os
from beam import function, Image

GIT_BRANCH = "feature/diffusion-prior-fwi"
GPU = os.environ.get("BFWI_BEAM_GPU", "RTX4090")

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
def run(prior_bytes, crop_bytes, norm_bytes, map_bytes, cfg):
    import io, sys, time
    sys.path.insert(0, "/opt/brain-fwi")
    import jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx
    jax.config.update("jax_enable_x64", False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from brain_fwi.inference.score_unet3d import UNet3DScore
    from brain_fwi.inference.diffusion import VPSDE
    from brain_fwi.phantoms.birnbaum import (to_velocity, roi_mask, LESION, SKULL, C_WATER, C_SKULL)
    from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis,
        simulate_shot_sensors, generate_observed_data)
    from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
    from brain_fwi.utils.wavelets import ricker_wavelet
    from brain_fwi.inversion.losses import l2_loss
    from brain_fwi.inversion.fwi import _bandpass_signal
    print("devices:", jax.devices(), flush=True)

    S, dx, pml = cfg["S"], cfg["dx"], 8
    F0, T_END = cfg["f0"], cfg["t_end"]
    BAND = cfg["band"]; N_SHOTS = cfg["n_shots"]
    sde = VPSDE()

    crop = np.load(io.BytesIO(crop_bytes))
    c_true = jnp.asarray(to_velocity(crop, with_skull=True))
    skull = jnp.asarray(crop == SKULL)
    interior = jnp.asarray(roi_mask(crop)); imask = interior.astype(jnp.float32)
    rho = jnp.full((S, S, S), 1000.0, jnp.float32)
    _mz = np.load(io.BytesIO(map_bytes), allow_pickle=True)               # MAP recon npz
    _mk = "annealed" if "annealed" in _mz else "annealed-t"
    x_map = jnp.asarray(_mz[_mk].astype(np.float32))                       # MAP recon (full volume)
    print(f"crop {crop.shape}, lesion {int((crop==LESION).sum())}, cerebrum {int(np.asarray(interior).sum())}", flush=True)

    c = (S // 2) * dx; r = (S // 2 - 4) * dx
    pos = helmet_array_3d(n_elements=80, center=(c, c, c), radius_ap=r, radius_lr=r,
                          radius_si=r, standoff=0.0, coverage_angle=3.1, exclude_face=False)
    # numpy positions: _build_sensors does int(pg[d][i]); under jit (the DPS step)
    # int() on a traced jax.Array raises ConcretizationTypeError, but numpy -> Python int.
    pg = tuple(np.asarray(x) for x in transducer_positions_to_grid(pos, dx, (S, S, S))); ne = len(pg[0])
    allsrc = [(int(pg[0][i]), int(pg[1][i]), int(pg[2][i])) for i in range(ne)]
    src = [allsrc[i] for i in np.linspace(0, ne - 1, N_SHOTS).astype(int)]
    ta = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml),
                         cfl=0.3, t_end=T_END)
    dt = float(ta.dt); nsteps = int(T_END / dt); sig = ricker_wavelet(f0=F0, dt=dt, n_samples=nsteps)
    bp = _bandpass_signal(sig, dt, BAND[0], BAND[1])
    obs = generate_observed_data(sound_speed=c_true, density=rho, dx=dx, src_positions_grid=src,
        sensor_positions_grid=pg, freq=F0, pml_size=pml, time_axis=ta, source_signal=bp, dt=dt, verbose=False)
    print(f"helmet {ne} elems, {N_SHOTS} shots, {nsteps} steps, obs {obs.shape}", flush=True)

    model = eqx.tree_deserialise_leaves(io.BytesIO(prior_bytes), UNet3DScore(S, C=16, key=jr.PRNGKey(0)))
    nz = np.load(io.BytesIO(norm_bytes)); SM, SS = float(nz["mean"]), float(nz["std"])
    score_fn = lambda th, t: model(th, t)

    # ---- FWI data log-likelihood at a denoised (normalised, flat) interior -----
    obs_n = obs / (jnp.sqrt(jnp.mean(obs ** 2)) + 1e-12)   # normalise data scale
    def log_lik(theta_hat_0):
        v_int = theta_hat_0 * SS + SM
        x = jnp.where(interior.reshape(-1), v_int, jnp.where(skull.reshape(-1), C_SKULL, C_WATER)).reshape(S, S, S)
        mis = 0.0
        for k in range(N_SHOTS):
            med = build_medium(build_domain((S, S, S), dx), x, rho, pml_size=pml)
            pred = simulate_shot_sensors(med, ta, src[k], pg, bp, dt, checkpointed=True)
            ot = obs[k]; mt = min(pred.shape[0], ot.shape[0])
            predn = pred[:mt] / (jnp.sqrt(jnp.mean(obs ** 2)) + 1e-12)
            mis = mis + l2_loss(predn, obs_n[k][:mt])
        return -0.5 * mis / N_SHOTS

    # ---- warm-start reverse-diffusion DPS (jit step + py loop; t is a traced ARG
    #      so the step compiles ONCE and j-Wave stays one scan-level deep — nesting
    #      the checkpointed j-Wave scan inside an outer lax.scan trips a Concretization
    #      error, so we mirror the proven MAP pattern instead).
    @jax.jit
    def step(theta, key, t, t_next):
        s_prior = score_fn(theta, t)
        def lik_through_denoise(tt_):
            s = score_fn(tt_, t)
            x0 = (tt_ + sde.sigma(t) ** 2 * s) / sde.alpha(t)
            return log_lik(x0)
        g = jax.grad(lik_through_denoise)(theta)
        g = g / (jnp.max(jnp.abs(g)) + 1e-30)              # normalise data-grad; zeta sets strength
        s_total = s_prior + cfg["zeta"] * g
        beta = sde.beta(t); dtau = t - t_next
        drift = -0.5 * beta * theta - beta * s_total
        key, sk = jr.split(key)
        theta = theta - drift * dtau + jnp.sqrt(beta * dtau) * jr.normal(sk, theta.shape)
        return theta, key

    def dps(x_init_norm, t_start, zeta, n_steps, key):
        ts = jnp.linspace(t_start, cfg["t_min"], n_steps + 1)
        key, sk = jr.split(key)
        theta = sde.alpha(t_start) * x_init_norm + sde.sigma(t_start) * jr.normal(sk, x_init_norm.shape)
        tstep = time.time()
        for i in range(n_steps):
            theta, key = step(theta, key, ts[i], ts[i + 1])
            theta.block_until_ready()
            print(f"    step {i+1}/{n_steps} t={float(ts[i]):.3f} ({time.time()-tstep:.0f}s)", flush=True)
        x0 = (theta + sde.sigma(cfg["t_min"]) ** 2 * score_fn(theta, cfg["t_min"])) / sde.alpha(cfg["t_min"])
        v = np.asarray(x0) * SS + SM
        full = np.where(np.asarray(interior).reshape(-1), v,
                        np.where(np.asarray(skull).reshape(-1), C_SKULL, C_WATER)).reshape(S, S, S)
        return full.astype(np.float32)

    m = np.asarray(interior) > 0.5; ct = np.asarray(c_true)
    les = (np.asarray(crop) == LESION) & m; brn = m & ~les
    out = {}
    def rep(tag, rec):
        ri = np.sqrt(np.mean((C_WATER - ct[m]) ** 2)); rf = np.sqrt(np.mean((rec[m] - ct[m]) ** 2))
        rb = np.sqrt(np.mean((rec[brn] - ct[brn]) ** 2)); lv = float(rec[les].mean()) if les.sum() else float("nan")
        line = (f"{tag}: cerebrum RMSE {ri:.1f}->{rf:.1f} ({100*(1-rf/ri):+.0f}%)  brain-only {rb:.1f}  "
                f"lesion {lv:.0f} (true 1660, {100*(lv-1500)/160:.0f}% contrast)")
        print("  " + line, flush=True); out[tag.strip()] = line; return rec

    x_map_norm = ((jnp.asarray(x_map) - SM) / SS).reshape(-1) * imask.reshape(-1)
    print(f"=== DPS posterior (t_start={cfg['t_start']}, zeta={cfg['zeta']}, {cfg['n_steps']} steps) ===", flush=True)
    t0 = time.time()
    rmap = rep("MAP (in) ", np.asarray(x_map))
    rdps = rep("DPS      ", dps(x_map_norm, cfg["t_start"], cfg["zeta"], cfg["n_steps"], jr.PRNGKey(0)))
    runtime = time.time() - t0; print(f"runtime {runtime:.0f}s", flush=True)

    nb = io.BytesIO(); np.savez(nb, c_true=ct, MAP=rmap, DPS=rdps, interior=m, lesion=les)
    zc = int(np.round(np.where(les)[0].mean())) if les.sum() else S // 2
    win = dict(vmin=1490, vmax=1700, cmap="turbo")
    panels = [(ct[zc], "TRUE", win), (rmap[zc], "MAP (annealed-t)", win), (rdps[zc], "DPS posterior", win),
              (np.abs(rdps - ct)[zc] * m[zc], "|err| DPS", dict(vmin=0, vmax=150, cmap="magma"))]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, (img, ttl, kw) in zip(ax, panels):
        a.imshow(np.where(m[zc] | ttl.startswith("|err"), img, np.nan), **kw)
        if les[zc].any(): a.contour(les[zc], levels=[0.5], colors="cyan", linewidths=0.8)
        a.set_title(ttl, fontsize=10); a.axis("off")
    fig.suptitle(f"96^3 reverse-diffusion DPS vs MAP (beam {GPU}), {cfg.get('subj','?')}, z={zc}", fontsize=11)
    plt.tight_layout(); pb = io.BytesIO(); plt.savefig(pb, dpi=120, format="png")
    return {"png": pb.getvalue(), "npz": nb.getvalue(), "metrics": out, "runtime_s": round(runtime, 0), "gpu": GPU}


if __name__ == "__main__":
    subj = os.environ.get("BFWI_SUBJECT", "subj2")
    S = int(os.environ.get("BFWI_S", "96"))
    sfx = "_96" if S == 96 else ""           # 48^3 prior is brain_score_3d.eqx; 96^3 is _96
    out = os.environ.get("BFWI_OUT", f"/tmp/brain_recon_dps_posterior_{S}_{subj}.png")
    cfg = {"S": S, "dx": 1.5e-3 if S == 96 else 3.0e-3, "f0": 80e3, "t_end": 1.0e-4, "band": [20e3, 60e3],
           "n_shots": int(os.environ.get("BFWI_SHOTS", "4")),
           "t_start": float(os.environ.get("BFWI_TSTART", "0.6")),
           "zeta": float(os.environ.get("BFWI_ZETA", "0.3")),
           "n_steps": int(os.environ.get("BFWI_STEPS", "30")),
           "t_min": 0.02, "subj": subj}
    prior = open(f"/tmp/brain_score_3d{sfx}.eqx", "rb").read()
    crop = open(f"/tmp/{subj}_crop_{S}.npy", "rb").read()
    norm = open(f"/tmp/brain_score_3d{sfx}_norm.npz", "rb").read()
    # MAP recon (annealed-t) from the saved arrays
    if S == 48:
        mp = "/tmp/brain_recon_dps_3d_ab_arrays.npz"      # 48^3 demo A/B output (key 'annealed')
    else:
        mp = "/tmp/brain_recon_dps_3d_96_arrays.npz" if subj == "subj2" else f"/tmp/brain_recon_dps_3d_96_{subj}_arrays.npz"
    map_npz = open(mp, "rb").read()                       # ship whole npz; extract array remotely
    print(f"=== launching reverse-diffusion DPS on beam {GPU} ({subj}) ===", flush=True)
    res = None                                            # retry transient flaky-node cuInit failures
    for attempt in range(4):
        try:
            res = run.remote(prior, crop, norm, map_npz, cfg)
        except Exception as e:
            print(f"attempt {attempt+1}/4 raised: {type(e).__name__}: {str(e)[:160]}", flush=True)
        if res:
            break
        print(f"attempt {attempt+1}/4 failed (likely flaky node); retrying...", flush=True)
    if not res:
        print("FAILED after retries"); raise SystemExit(1)
    open(out, "wb").write(res["png"]); open(out.replace(".png", "_arrays.npz"), "wb").write(res["npz"])
    print(f"\nGPU {res['gpu']}  runtime {res['runtime_s']}s")
    for v in res["metrics"].values():
        print("  " + v)
    print(f"saved {out} + _arrays.npz")
