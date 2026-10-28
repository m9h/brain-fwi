"""Derivative-free global skull-pose search (CMA-ES) on beam.cloud.

The pose-recovery story so far (GB10 jobs 1685-1690): with a model-consistent
single-warp forward (misfit=0 at truth, contrast 1e11x), every LOCAL/coarse method
fails identically -- L2 coarse grid+refine and L2/AWI gradient descent all land in
spurious wells (~1e-8 misfit, overlap ~30-50%), in every band, true or generic brain.
Truth is the global minimum but a deep narrow notch amid a rough plateau: a genuinely
non-convex landscape that needs a DERIVATIVE-FREE GLOBAL optimizer (the fix named by
MOFI/Bates 2026 and our own status doc).

CMA-ES is forward-only (no backward pass -> ~3x cheaper per eval than the gradient runs)
and population-based (samples broadly, adapts covariance -> can see over the shallow
spurious wells toward the deep truth). beam's RTX4090 (~1 TB/s vs GB10's ~273 GB/s
LPDDR5X) is ~2-3x faster per forward on this FFT-bound sim, so the whole search runs in
minutes. Tests true-brain (can CMA-ES find the exact-zero global min at all?) then
generic-brain (does it survive the brain-mismatch floor?). IPOP restart (grow popsize)
if the first run doesn't reach truth's basin.
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
        "pip install cma",
    ])
    .with_envs(["JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9"])
)


# defaults are cpu=0.125 / 128 MiB (jax import alone OOMs, exit 137) and timeout=3600; -1 = no cap
@function(gpu=GPU, image=image, cpu=4, memory="16Gi", timeout=-1)
def run(crop_bytes, cfg):
    import sys, time
    sys.path.insert(0, "/opt/brain-fwi")
    import jax, jax.numpy as jnp, numpy as np, equinox as eqx, cma
    jax.config.update("jax_enable_x64", False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from jax.scipy.ndimage import map_coordinates
    from brain_fwi.phantoms.birnbaum import to_velocity, roi_mask, SKULL, C_WATER, C_SKULL
    from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis, simulate_shot_sensors)
    from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
    from brain_fwi.utils.wavelets import ricker_wavelet
    from brain_fwi.inversion.losses import l2_loss
    from brain_fwi.inversion.fwi import _bandpass_signal
    print("devices:", jax.devices(), flush=True)

    # --- inlined rigid warp (scripts/mofi3d.py) so we don't depend on remote scripts/ ---
    def rotation_matrix(ang):
        rx, ry, rz = ang
        cx, sx = jnp.cos(rx), jnp.sin(rx); cy, sy = jnp.cos(ry), jnp.sin(ry); cz, sz = jnp.cos(rz), jnp.sin(rz)
        Rx = jnp.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = jnp.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = jnp.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx
    def rigid_warp_3d(field, t, ang):
        Sz = field.shape[0]; ctr = (Sz - 1) / 2.0; R = rotation_matrix(ang)
        g = jnp.stack(jnp.meshgrid(jnp.arange(Sz), jnp.arange(Sz), jnp.arange(Sz), indexing="ij")).astype(jnp.float32)
        cen = g - ctr - t[:, None, None, None]
        srcc = jnp.einsum("ij,jxyz->ixyz", R.T, cen) + ctr
        return map_coordinates(field, [srcc[0], srcc[1], srcc[2]], order=1, mode="constant")

    S, dx, pml = cfg["S"], cfg["dx"], 8
    F0 = cfg["f0"]; N_SHOTS = cfg["n_shots"]; POSE_SHOTS = cfg["pose_shots"]
    BAND = tuple(cfg["band"]); TWIN_US = cfg["twin_us"]
    T_TRUE = jnp.array([4.0, 2.0, 0.0]); A_TRUE = jnp.array([0.0, 0.0, np.deg2rad(5.0)])

    crop = np.load(io.BytesIO(crop_bytes))
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
    ta = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml), cfl=0.3, t_end=1e-4)
    dt = float(ta.dt)
    skull_obs = jnp.clip(rigid_warp_3d(skull_true, T_TRUE, A_TRUE), 0, 1)
    brain_no_skull = jnp.where(skull_true > 0.5, C_WATER, c_true)
    c_obs = (brain_no_skull * (1 - skull_obs) + C_SKULL * skull_obs).astype(jnp.float32)

    NW = int(TWIN_US / dt)
    ta_p = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml), cfl=0.3, t_end=NW * dt)
    bp = _bandpass_signal(ricker_wavelet(f0=F0, dt=dt, n_samples=NW), dt, *BAND)
    ps = [src[i] for i in np.linspace(0, N_SHOTS - 1, POSE_SHOTS).astype(int)]
    true_med = build_medium(build_domain((S, S, S), dx), c_obs, rho, pml_size=pml)
    po = [simulate_shot_sensors(true_med, ta_p, ps[k], pg, bp, dt, checkpointed=False) for k in range(POSE_SHOTS)]
    skull_obs_b = np.asarray(skull_obs > 0.5); sob_sum = float(skull_obs_b.sum())
    print(f"setup: band {BAND[0]/1e3:.0f}-{BAND[1]/1e3:.0f}kHz, {POSE_SHOTS} shots, {NW} steps (~{NW*dt*1e6:.0f}us), "
          f"forward-only CMA-ES; true pose (4,2,5)", flush=True)

    def overlap(vec):
        a = np.asarray(rigid_warp_3d(skull_true, jnp.array([vec[0], vec[1], 0.0]),
                                     jnp.array([0.0, 0.0, np.deg2rad(vec[2])])) > 0.5)
        return 100.0 * float((a & skull_obs_b).sum()) / (sob_sum + 1e-9)

    def make_misfit(brain_est):
        @eqx.filter_jit
        def misfit(vec):                                   # vec = (tx, ty, rz_deg)
            skm = jnp.clip(rigid_warp_3d(skull_true, jnp.array([vec[0], vec[1], 0.0]),
                                         jnp.array([0.0, 0.0, jnp.deg2rad(vec[2])])), 0, 1)
            v = brain_est * (1 - skm) + C_SKULL * skm
            med = build_medium(build_domain((S, S, S), dx), v, rho, pml_size=pml)
            tot = 0.0
            for k in range(POSE_SHOTS):
                pred = simulate_shot_sensors(med, ta_p, ps[k], pg, bp, dt, checkpointed=False)
                mt = min(pred.shape[0], po[k].shape[0]); tot = tot + l2_loss(pred[:mt], po[k][:mt])
            return tot / POSE_SHOTS
        return misfit

    def cma_search(brain_est, tag):
        mf = make_misfit(brain_est)
        best_vec, best_f, traj = None, 1e30, []; t0 = time.time(); popsize = cfg["popsize"]
        for restart in range(cfg["restarts"]):                # IPOP: grow popsize on restart
            es = cma.CMAEvolutionStrategy([0.0, 0.0, 0.0], 1.0, {
                "popsize": popsize, "CMA_stds": [4.0, 4.0, 5.0],
                "bounds": [[-12, -12, -15], [12, 12, 15]],
                "maxiter": cfg["maxiter"], "tolfun": 1e-13, "tolx": 1e-3, "verbose": -9, "seed": 1 + restart})
            while not es.stop():
                cands = es.ask()
                vals = [float(mf(jnp.asarray(cc, jnp.float32))) for cc in cands]
                es.tell(cands, vals)
                fb = es.result.fbest; xb = np.asarray(es.result.xbest)
                if fb < best_f: best_f, best_vec = fb, xb
                traj.append((tag, restart, es.countiter, fb, overlap(xb)))
                print(f"  [{tag} r{restart}] gen {es.countiter:2d}: best {fb:.3e} "
                      f"@ ({xb[0]:+.2f},{xb[1]:+.2f},{xb[2]:+.2f}) overlap {overlap(xb):.0f}% "
                      f"({time.time()-t0:.0f}s)", flush=True)
            print(f"  [{tag} r{restart}] stop: {es.stop()}", flush=True)
            if best_f < cfg["target"]:                        # reached truth's basin -> done
                break
            popsize *= 2
        ov = overlap(best_vec)
        print(f"=> {tag}: best misfit {best_f:.3e} pose ({best_vec[0]:+.2f},{best_vec[1]:+.2f},{best_vec[2]:+.2f}) "
              f"(true 4,2,5) OVERLAP {ov:.0f}% => {'LANDED' if ov>=90 else 'PARTIAL' if ov>=75 else 'did NOT land'}",
              flush=True)
        return {"tag": tag, "best_f": best_f, "pose": [float(x) for x in best_vec], "overlap": ov, "traj": traj}

    results = []
    for tag, be in [("true", brain_no_skull.astype(jnp.float32)),
                    ("generic", jnp.where(interior, 1560.0, C_WATER).astype(jnp.float32))]:
        results.append(cma_search(be, tag))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for res in results:
        tr = res["traj"]; gens = list(range(len(tr)))
        ax[0].semilogy(gens, [t[3] for t in tr], "-o", ms=3, label=f"{res['tag']} (ov {res['overlap']:.0f}%)")
        ax[1].plot(gens, [t[4] for t in tr], "-o", ms=3, label=res["tag"])
    ax[0].set_title("CMA-ES best misfit"); ax[0].set_xlabel("generation"); ax[0].legend(); ax[0].axhline(cfg["target"], color="k", ls=":")
    ax[1].set_title("skull overlap %"); ax[1].set_xlabel("generation"); ax[1].axhline(90, color="r", ls="--"); ax[1].legend()
    plt.tight_layout(); pb = io.BytesIO(); plt.savefig(pb, dpi=120, format="png")
    return {"png": pb.getvalue(), "results": [{k: v for k, v in r.items() if k != "traj"} for r in results], "gpu": GPU}


if __name__ == "__main__":
    out = os.environ.get("BFWI_OUT", "/tmp/mofi3d_cmaes.png")
    cfg = {"S": 96, "dx": 1.5e-3, "f0": 80e3, "n_shots": 12, "pose_shots": 6,
           "band": [50e3, 90e3], "twin_us": 50e-6,
           "popsize": 12, "maxiter": 30, "restarts": 2, "target": 1e-9}
    crop = open("/tmp/subj2_crop_96.npy", "rb").read()
    print(f"=== launching CMA-ES pose search on beam {GPU} ===", flush=True)
    res = None
    for attempt in range(4):
        try:
            res = run.remote(crop, cfg)
        except Exception as e:
            print(f"attempt {attempt+1}/4 raised: {type(e).__name__}: {str(e)[:160]}", flush=True)
        if res:
            break
        print(f"attempt {attempt+1}/4 failed (likely flaky node); retrying...", flush=True)
    if not res:
        print("FAILED after retries"); raise SystemExit(1)
    open(out, "wb").write(res["png"])
    print(f"\nGPU {res['gpu']}")
    for r in res["results"]:
        print(f"  {r['tag']}: misfit {r['best_f']:.3e} pose {r['pose']} overlap {r['overlap']:.0f}%")
    print(f"saved {out}")
