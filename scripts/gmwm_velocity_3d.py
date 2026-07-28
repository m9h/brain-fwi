"""3-D GM/WM velocity FWI at imaging frequency (the Guasch-class deliverable).

Follows the 2-D result (`gmwm_velocity_resolution_test.md`): GM/WM resolved at
99% / AUC 1.000 without skull, 17% / AUC 0.752 through skull, using the REAL
measured ~17 m/s contrast (Mitcham 2025) instead of the ITRUSST GM==WM convention.

Staged by frequency (see `scope_3d_gmwm_run.md`, costs measured on GB10):
  stage 1: 300 kHz, N=112, res 2.60 mm   ~1.6-4 h   <- pipeline validation
  stage 2: 400 kHz, N=144, res 1.95 mm   ~5-14 h
  stage 3: 500 kHz, N=176, res 1.56 mm   ~14-36 h   <- headline

TWO GM/WM tests at different spatial scales, because resolution and contrast are
separate questions:
  * **deep-gray nuclei** (~14 mm, GM speed embedded in WM) -- RESOLVABLE at every
    stage. This is the pure CONTRAST test: can FWI see a 17 m/s GM/WM difference?
  * **cortical ribbon** (3 mm, realistic) -- resolution-limited at stage 1
    (2 voxels at dx 1.61 mm), expected to emerge as the ladder climbs. This is
    the pure RESOLUTION test.
Plus a **ventricle** (CSF, ~51 m/s) large-contrast positive control.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data,
    _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi

STAGES = {1: dict(N=112, bands=[(150e3, 225e3), (225e3, 300e3)], freq=250e3),
          2: dict(N=144, bands=[(200e3, 300e3), (300e3, 400e3)], freq=330e3),
          3: dict(N=176, bands=[(250e3, 375e3), (375e3, 500e3)], freq=400e3)}

ap = argparse.ArgumentParser()
ap.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
ap.add_argument("--phase", choices=["A", "B"], default="B")   # A=no skull, B=transcranial
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--shots-per-iter", type=int, default=4)
ap.add_argument("--n-src", type=int, default=16)
ap.add_argument("--n-rec", type=int, default=128)
ap.add_argument("--noise-db", type=float, default=None,
                help="additive white-noise SNR in dB (None = noiseless)")
ap.add_argument("--tag", type=str, default="")
args = ap.parse_args()
S = STAGES[args.stage]
N, BANDS, FREQ = S["N"], S["bands"], S["freq"]

OUT = "results/gmwm_velocity_3d"; os.makedirs(OUT, exist_ok=True)
TAG = f"s{args.stage}{args.phase}" + (f"_{args.tag}" if args.tag else "")
mm = 1e-3
FOV = 0.180
dx = FOV / N
PML = 8

# ---- real measured GM/WM contrast (Mitcham 2025 magnitude), mean 1560 ----
C_WM, C_GM, C_CSF, C_WATER, C_SKULL = 1551.5, 1568.5, 1500.0, 1500.0, 2800.0
CONTRAST = C_GM - C_WM

# ---- 3-D phantom ----
cx = N // 2
gx, gy, gz = np.meshgrid(*[np.arange(N)]*3, indexing="ij")
X, Y, Z = (gx-cx)*dx, (gy-cx)*dx, (gz-cx)*dx
r = np.sqrt(X**2 + Y**2 + Z**2)
R_BRAIN, RIBBON = 55*mm, 3*mm
lab = np.zeros((N, N, N), np.int32)              # 0 water
lab[r <= R_BRAIN] = 2                            # 2 = GM cortical ribbon
lab[r <= R_BRAIN - RIBBON] = 3                   # 3 = WM interior
# 4 = deep-gray nuclei (GM speed, ~14mm -> RESOLVABLE contrast test)
for off in (-16*mm, 16*mm):
    lab[np.sqrt((X-off)**2 + (Y+6*mm)**2 + Z**2) <= 7*mm] = 4
# 5 = ventricles (CSF, large-contrast positive control)
for off in (-8*mm, 8*mm):
    lab[np.sqrt((X-off)**2 + (Y-10*mm)**2 + (Z/2.2)**2) <= 5*mm] = 5
if args.phase == "B":
    lab[(r > R_BRAIN) & (r <= R_BRAIN + 1*mm)] = 0            # CSF gap
    lab[(r > R_BRAIN + 1*mm) & (r <= R_BRAIN + 7*mm)] = 6     # 6 mm skull

CMAP = {0: C_WATER, 2: C_GM, 3: C_WM, 4: C_GM, 5: C_CSF, 6: C_SKULL}
RMAP = {0: 1000.0, 2: 1040.0, 3: 1040.0, 4: 1040.0, 5: 1000.0, 6: 1900.0}
c_true = np.vectorize(CMAP.get)(lab).astype(np.float32)
rho = np.vectorize(RMAP.get)(lab).astype(np.float32)
brain = np.isin(lab, [2, 3, 4, 5])

# ---- full-encirclement spherical array (Fibonacci; transmission tomography) ----
def fib_sphere(n, R):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2*i/n); th = np.pi*(1 + 5**0.5)*i
    return np.stack([R*np.sin(phi)*np.cos(th), R*np.sin(phi)*np.sin(th),
                     R*np.cos(phi)], 1)
R_ARR = 70*mm
def to_idx(p):
    return tuple(np.clip((p[:, d]/dx + cx).round().astype(int), 0, N-1) for d in range(3))
src_p = fib_sphere(args.n_src, R_ARR); rec_p = fib_sphere(args.n_rec, R_ARR)
si = to_idx(src_p); ri = to_idx(rec_p)
src_list = [(int(si[0][k]), int(si[1][k]), int(si[2][k])) for k in range(args.n_src)]
recv = ri

# ---- time axis: MUST match what run_fwi builds from config.c_max ----
CFG_CMAX = 1600.0 if args.phase == "A" else 2900.0
ref = build_medium(build_domain((N,)*3, dx), CFG_CMAX, 1000.0, pml_size=PML)
t_end = 1.6 * FOV / 1500.0
ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt, nt = float(ta.dt), int(ta.Nt)
lam_hi = 1560.0 / BANDS[-1][1]
print(f"[{TAG}] N={N} dx={dx*1e3:.2f}mm FOV={FOV*1e3:.0f}mm  top band {BANDS[-1][1]/1e3:.0f}kHz "
      f"lambda={lam_hi*1e3:.2f}mm ppw={lam_hi/dx:.1f} res={lam_hi/2*1e3:.2f}mm", flush=True)
print(f"  ribbon {RIBBON*1e3:.0f}mm = {RIBBON/dx:.1f} vox | nuclei 14mm = {14*mm/dx:.1f} vox | "
      f"nt={nt} dt={dt*1e9:.1f}ns | {args.n_src} src / {args.n_rec} rec", flush=True)

t0 = time.time()
sig = _build_source_signal(FREQ, dt, nt)   # build ONCE; identical for obs and FWI
print("generating observed data ...", flush=True)
obs = generate_observed_data(jnp.asarray(c_true), jnp.asarray(rho), dx, src_list, recv,
                             FREQ, pml_size=PML, time_axis=ta, source_signal=sig,
                             dt=dt, verbose=False)
print(f"  done {time.time()-t0:.0f}s  obs {obs.shape}", flush=True)

# additive white Gaussian noise at a given SNR (per-trace RMS reference)
if args.noise_db is not None:
    obs_np = np.asarray(obs)
    rms = np.sqrt(np.mean(obs_np**2))
    sigma = rms * 10 ** (-args.noise_db / 20.0)
    rng_n = np.random.default_rng(0)
    obs = jnp.asarray(obs_np + sigma*rng_n.standard_normal(obs_np.shape).astype(obs_np.dtype))
    print(f"  added noise: SNR {args.noise_db:.0f} dB (sigma/rms = {sigma/rms:.3f})", flush=True)

c_init = np.where(brain, 1560.0, c_true).astype(np.float32)
obs_init = generate_observed_data(jnp.asarray(c_init), jnp.asarray(rho), dx, src_list, recv,
                                  FREQ, pml_size=PML, time_axis=ta, source_signal=sig,
                                  dt=dt, verbose=False)
rel0 = float(np.linalg.norm(np.asarray(obs_init)-np.asarray(obs))/np.linalg.norm(np.asarray(obs)))
print(f"  starting relative residual {rel0*100:.2f}%  "
      f"({'OK' if rel0 < 0.25 else 'TOO LARGE - forward/inversion inconsistency!'})", flush=True)

cfg = FWIConfig(freq_bands=BANDS, n_iters_per_band=args.iters,
                shots_per_iter=args.shots_per_iter, learning_rate=4.0,
                c_min=1480.0, c_max=CFG_CMAX, pml_size=PML, cfl=0.3,
                gradient_smooth_sigma=1.0, mask=jnp.asarray(brain.astype(np.float32)),
                precondition=True, verbose=True)
print(f"running 3-D FWI ({len(BANDS)} bands x {args.iters} it x {args.shots_per_iter} sh) ...", flush=True)
res = run_fwi(obs, jnp.asarray(c_init), jnp.asarray(rho), dx, src_list, recv, sig, dt,
              t_end, config=cfg)
c_rec = np.asarray(res.velocity)
print(f"  FWI done {(time.time()-t0)/60:.1f} min", flush=True)

# ---- metrics ----
def eroded(m, it):
    return m & ~ndi.binary_dilation(~m, iterations=it)
wm_m = eroded(lab == 3, 3)
nuc_m = eroded(lab == 4, 2)      # resolvable GM-in-WM  -> CONTRAST test
rib_m = eroded(lab == 2, 1)      # 3mm ribbon           -> RESOLUTION test
csf_m = eroded(lab == 5, 1)

def auc(pos, neg):
    v = np.concatenate([pos, neg]); y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    o = np.argsort(v); rk = np.empty_like(o, float); rk[o] = np.arange(1, v.size+1)
    return float((rk[y == 1].sum() - pos.size*(pos.size+1)/2)/(pos.size*neg.size))

wm_v = c_rec[wm_m]
out = {"stage": args.stage, "phase": args.phase, "N": N, "dx_mm": dx*1e3,
       "noise_db": args.noise_db,
       "top_band_kHz": BANDS[-1][1]/1e3, "res_mm": lam_hi/2*1e3, "ppw": lam_hi/dx,
       "true_contrast": CONTRAST, "start_residual_pct": rel0*100,
       "n_src": args.n_src, "n_rec": args.n_rec}
print("\n=========== 3-D RESULT: GM/WM via velocity ===========", flush=True)
print(f"  WM median {np.median(wm_v):.1f} (true {C_WM})", flush=True)
for name, m, truth, key in [("deep-gray nuclei (RESOLVABLE, contrast test)", nuc_m, C_GM, "nuclei"),
                            ("cortical ribbon 3mm (resolution test)", rib_m, C_GM, "ribbon"),
                            ("ventricle CSF (positive control)", csf_m, C_CSF, "csf")]:
    if m.sum() < 10:
        print(f"  {name}: too few voxels", flush=True); continue
    v = c_rec[m]
    rc = float(np.median(v) - np.median(wm_v)); tc = truth - C_WM
    a = auc(v, wm_v) if truth > C_WM else auc(wm_v, v)
    print(f"  {name}:\n     true {tc:+6.1f} -> recovered {rc:+6.1f} m/s ({100*rc/tc:5.0f}%)  AUC {a:.3f}",
          flush=True)
    out[key] = dict(true=tc, recovered=rc, pct=100*rc/tc, auc=a, n_vox=int(m.sum()))
verdict = ("RESOLVED" if out.get("nuclei", {}).get("auc", 0) > 0.9 else
           "PARTIAL" if out.get("nuclei", {}).get("auc", 0) > 0.7 else "NOT RESOLVED")
out["verdict"] = verdict
print(f"  VERDICT (on the resolvable GM/WM contrast): {verdict}", flush=True)
print("=" * 55, flush=True)

# ---- figure: central axial slice ----
k = cx
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="white")
vmin, vmax = 1540, 1580
for a_, img, ttl in [(ax[0], c_true, "truth"), (ax[1], c_init, "start (1560)"),
                     (ax[2], c_rec, f"3-D FWI @ {BANDS[-1][1]/1e3:.0f} kHz")]:
    im = a_.imshow(np.where(brain[:, :, k], img[:, :, k], np.nan).T, origin="lower",
                   cmap="turbo", vmin=vmin, vmax=vmax)
    a_.set_title(ttl); a_.set_xticks([]); a_.set_yticks([])
    plt.colorbar(im, ax=a_, fraction=.046)
nu = out.get("nuclei", {})
fig.suptitle(f"3-D GM/WM velocity FWI — stage {args.stage} "
             f"({'transcranial' if args.phase=='B' else 'no skull'}), res {lam_hi/2*1e3:.2f}mm | "
             f"deep-gray AUC {nu.get('auc', float('nan')):.3f}, {nu.get('pct', float('nan')):.0f}% contrast",
             y=1.02, fontsize=12)
fig.savefig(f"{OUT}/gmwm3d_{TAG}.png", dpi=140, bbox_inches="tight", facecolor="white")
np.save(f"{OUT}/c_rec_{TAG}.npy", c_rec)
json.dump(out, open(f"{OUT}/summary_{TAG}.json", "w"), indent=2)
print(f"saved {OUT}/gmwm3d_{TAG}.png", flush=True)
