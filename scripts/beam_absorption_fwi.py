"""Does modeling skull ABSORPTION improve transcranial brain FWI? (Phase-5 breakthrough eval)

Transcranial FWI today is sound-speed-only; the skull attenuates strongly (power-law
alpha ~ f^y). Ignoring it -> the inversion mis-attributes the missing amplitude to wrong
velocity = systematic brain bias. The time-domain absorption is installed (build_medium
accepts attenuation/alpha_power; jwave@feature/time-domain-absorption). 3-condition 96^3
plain FWI (no prior, to isolate the absorption effect), skull at truth:
  crime  : absorption-FREE obs, absorption-free inversion        (ideal reference)
  ignore : absorption-FUL  obs, absorption-free inversion        (what the pipeline does now)
  aware  : absorption-FUL  obs, absorption-AWARE inversion        (the Phase-5 fix)
ignore worse than crime = real damage; aware ~ crime = absorption modeling is the fix.
Returns cerebrum RMSE for each. Runs on beam (96^3 fits a 24GB 4090).
"""
import io, os
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
    .with_envs(["JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false", "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9"])
)


# defaults are cpu=0.125 / 128 MiB (jax import alone OOMs, exit 137) and timeout=3600; -1 = no cap
@function(gpu=GPU, image=image, cpu=4, memory="16Gi", timeout=-1)
def run(crop_bytes, cfg):
    import sys, time
    sys.path.insert(0, "/opt/brain-fwi")
    import jax, jax.numpy as jnp, numpy as np
    jax.config.update("jax_enable_x64", False)
    from brain_fwi.phantoms.birnbaum import to_velocity, roi_mask, LESION, SKULL, C_WATER, C_SKULL
    from brain_fwi.simulation.forward import build_domain, build_medium, build_time_axis, simulate_shot_sensors
    from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
    from brain_fwi.utils.wavelets import ricker_wavelet
    from brain_fwi.inversion.losses import l2_loss
    from brain_fwi.inversion.fwi import _smooth_gradient, _bandpass_signal
    print("devices:", jax.devices(), flush=True)

    S, dx, pml = cfg["S"], cfg["dx"], 8
    F0 = cfg["f0"]; N_SHOTS = cfg["n_shots"]; BANDS = cfg["bands"]; ITERS = cfg["iters"]
    AP = cfg["alpha_power"]; A_SKULL = cfg["a_skull"]; A_BRAIN = cfg["a_brain"]; A_WATER = cfg["a_water"]
    crop = np.load(io.BytesIO(crop_bytes))
    c_true = jnp.asarray(to_velocity(crop, with_skull=True))
    skull = jnp.asarray(crop == SKULL); interior = jnp.asarray(roi_mask(crop)); imask = interior.astype(jnp.float32)
    rho = jnp.full((S, S, S), 1000.0, jnp.float32)
    # per-voxel attenuation [dB/cm/MHz^y]: skull high, brain low, water ~0
    atten = jnp.where(skull, A_SKULL, jnp.where(interior, A_BRAIN, A_WATER)).astype(jnp.float32)
    dom = build_domain((S, S, S), dx)
    c = (S // 2) * dx; r = (S // 2 - 4) * dx
    pos = helmet_array_3d(n_elements=80, center=(c, c, c), radius_ap=r, radius_lr=r, radius_si=r,
                          standoff=0.0, coverage_angle=3.1, exclude_face=False)
    pg = tuple(np.asarray(z) for z in transducer_positions_to_grid(pos, dx, (S, S, S))); ne = len(pg[0])
    allsrc = [(int(pg[0][i]), int(pg[1][i]), int(pg[2][i])) for i in range(ne)]
    src = [allsrc[i] for i in np.linspace(0, ne - 1, N_SHOTS).astype(int)]
    ta = build_time_axis(build_medium(dom, C_SKULL, 1000.0, pml_size=pml), cfl=0.3, t_end=cfg["t_end"])
    dt = float(ta.dt); nsteps = int(cfg["t_end"] / dt); sig = ricker_wavelet(f0=F0, dt=dt, n_samples=nsteps)

    def med_of(x, with_atten):
        return build_medium(dom, x, rho, pml_size=pml,
                            attenuation=(atten if with_atten else None), alpha_power=AP)
    def gen_obs(with_atten):
        m = med_of(c_true, with_atten)
        return [np.asarray(simulate_shot_sensors(m, ta, src[k], pg, sig, dt, checkpointed=False)) for k in range(N_SHOTS)]

    t0 = time.time()
    obs_noabs = gen_obs(False); obs_abs = gen_obs(True)
    # absorption modeling-error magnitude (how different is the data?)
    rel = np.mean([np.linalg.norm(obs_abs[k] - obs_noabs[k]) / (np.linalg.norm(obs_noabs[k]) + 1e-9) for k in range(N_SHOTS)])
    print(f"absorption trace change: rel-L2 {rel:.3f} (skull {A_SKULL} dB/cm/MHz^{AP}); obs in {time.time()-t0:.0f}s", flush=True)

    def fwi(obs, inv_atten, tag):
        def shot_loss(x, sp, ot, bp):
            pred = simulate_shot_sensors(med_of(x, inv_atten), ta, sp, pg, bp, dt, checkpointed=True)
            mt = min(pred.shape[0], ot.shape[0]); return l2_loss(pred[:mt], ot[:mt])
        vg = jax.value_and_grad(shot_loss)
        x = jnp.where(skull, C_SKULL, C_WATER).astype(jnp.float32); t1 = time.time()
        for bi, (fmin, fmax) in enumerate(BANDS):
            bp = _bandpass_signal(sig, dt, fmin, fmax)
            bobs = [jax.vmap(lambda col: _bandpass_signal(col, dt, fmin, fmax))(jnp.asarray(o).T).T for o in obs]
            for _ in range(ITERS):
                g = jnp.zeros_like(x); gsq = jnp.zeros_like(x)
                for k in range(N_SHOTS):
                    _, gs = vg(x, src[k], bobs[k], bp); g += gs; gsq += gs ** 2
                g /= N_SHOTS; g = g / (jnp.sqrt(gsq / N_SHOTS) + 1e-12 * jnp.max(jnp.sqrt(gsq / N_SHOTS)))
                g = _smooth_gradient(g, 1.0) * imask; g = g / (jnp.max(jnp.abs(g)) + 1e-30)
                x = jnp.clip(x - 20.0 * g, 1400.0, 2900.0)
            x.block_until_ready()
        rec = np.asarray(x); m = np.asarray(interior) > 0.5; ct = np.asarray(c_true)
        ri = np.sqrt(np.mean((C_WATER - ct[m]) ** 2)); rf = np.sqrt(np.mean((rec[m] - ct[m]) ** 2))
        line = f"{tag}: cerebrum RMSE {ri:.1f}->{rf:.1f} ({100*(1-rf/ri):+.0f}%)  [{time.time()-t1:.0f}s]"
        print("  " + line, flush=True); return {"tag": tag, "rmse": float(rf), "impr_pct": float(100 * (1 - rf / ri))}

    print("=== absorption FWI experiment (96^3, plain) ===", flush=True)
    res = {"rel_trace_change": float(rel),
           "crime": fwi(obs_noabs, False, "crime  (no-abs obs, no-abs inv)"),
           "ignore": fwi(obs_abs, False, "ignore (abs obs, no-abs inv)"),
           "aware": fwi(obs_abs, True, "aware  (abs obs, abs-aware inv)")}
    print(f"VERDICT: ignore {res['ignore']['impr_pct']:+.0f}% vs crime {res['crime']['impr_pct']:+.0f}% "
          f"vs aware {res['aware']['impr_pct']:+.0f}%  -> absorption "
          f"{'MATTERS (aware recovers the loss)' if res['aware']['impr_pct'] - res['ignore']['impr_pct'] > 5 else 'minor at these freqs'}", flush=True)
    return res


if __name__ == "__main__":
    cfg = {"S": 96, "dx": 1.5e-3, "f0": 80e3, "t_end": 1.0e-4, "n_shots": 8,
           "bands": [[40e3, 80e3], [70e3, 120e3]], "iters": 8, "alpha_power": 1.1,
           "a_skull": 10.0, "a_brain": 0.6, "a_water": 0.002}
    crop = open("/tmp/subj2_crop_96.npy", "rb").read()
    print("=== launching absorption FWI experiment on beam ===", flush=True)
    res = None
    for attempt in range(4):
        try:
            res = run.remote(crop, cfg)
        except Exception as e:
            print(f"attempt {attempt+1}/4 raised: {type(e).__name__}: {str(e)[:160]}", flush=True)
        if res:
            break
    print("RESULT:", res)
