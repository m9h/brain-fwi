"""Does GM/WM survive an UNKNOWN skull? — pose-error tolerance for 3-D velocity FWI.

Every GM/WM result so far (Run 1: AUC 0.998; Run 2: knee at 32 elements) assumed the
skull was **given at truth**. That is the last big caveat. The MOFI line recovers the
skull pose from data to |dt| = 0.07 vox, |drz| = 0.10 deg (transmission-traveltime
misfit + grid-seed-then-polish). This asks the coupling question directly:

    how accurate must the skull pose be for the GM/WM contrast to survive,
    and is MOFI's measured accuracy good enough?

Sweeps the pose error used to place the skull in the INVERSION (observed data always
has the skull at truth), from MOFI-level (0.07 vox / 0.1 deg) up to gross (4 vox / 5
deg, the misalignment the campaign measured as catastrophic, -400% on bulk recon).

TWO design points that make this a real test rather than a rigged one:
  * **Ellipsoidal head, not spherical.** Rotating a sphere is a no-op and translating
    one is nearly benign, so a spherical phantom would report false robustness. The
    head here is a realistic ellipsoid (semi-axes ~52/62/58 mm) with an ellipsoidal
    6 mm skull shell, so both translation and rotation genuinely perturb the model.
  * **Soft (anti-aliased) warp, never binarised** -- the campaign's key modelling
    insight: binarising a sub-voxel-accurate pose flips ~2% of skull voxels and that
    2% destroys the reconstruction. The skull enters as a float blend in BOTH the
    forward model and the inversion, and the warped skull is excluded from the
    inverted brain mask.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mofi3d import rigid_warp_3d
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data,
    _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi

ap = argparse.ArgumentParser()
ap.add_argument("--pose-vox", type=float, default=0.0, help="translation error (voxels)")
ap.add_argument("--pose-deg", type=float, default=0.0, help="rotation error (degrees, about z)")
ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--shots-per-iter", type=int, default=4)
ap.add_argument("--n-src", type=int, default=16)
ap.add_argument("--n-rec", type=int, default=64)
ap.add_argument("--tag", type=str, required=True)
args = ap.parse_args()

OUT = "results/gmwm_unknown_skull"; os.makedirs(OUT, exist_ok=True)
mm = 1e-3; FOV = 0.180; N = 112; dx = FOV / N; PML = 8
BANDS = [(150e3, 225e3), (225e3, 300e3)]; FREQ = 250e3
C_WM, C_GM, C_CSF, C_WATER, C_SKULL = 1551.5, 1568.5, 1500.0, 1500.0, 2800.0
CONTRAST = C_GM - C_WM
CFG_CMAX = 2900.0

# ---- ELLIPSOIDAL head (pose-sensitive, unlike a sphere) ----
cx = N // 2
gx, gy, gz = np.meshgrid(*[np.arange(N)]*3, indexing="ij")
X, Y, Z = (gx-cx)*dx, (gy-cx)*dx, (gz-cx)*dx
AX = np.array([52.0, 62.0, 58.0]) * mm        # brain semi-axes (AP/LR/SI-ish)
def ell(sc):                                   # normalised ellipsoid radius
    return np.sqrt((X/(AX[0]*sc))**2 + (Y/(AX[1]*sc))**2 + (Z/(AX[2]*sc))**2)
rb = ell(1.0)
RIBBON_SC = 1.0 - 3*mm/np.mean(AX)             # 3 mm ribbon
lab = np.zeros((N, N, N), np.int32)
lab[rb <= 1.0] = 2                                       # GM cortical ribbon
lab[ell(RIBBON_SC) <= 1.0] = 3                           # WM interior
for off in (-16*mm, 16*mm):                              # deep-gray nuclei (resolvable)
    lab[np.sqrt((X-off)**2 + (Y+6*mm)**2 + Z**2) <= 7*mm] = 4
for off in (-8*mm, 8*mm):                                # ventricles (control)
    lab[np.sqrt((X-off)**2 + (Y-10*mm)**2 + (Z/2.2)**2) <= 5*mm] = 5
brain = np.isin(lab, [2, 3, 4, 5])

# soft (anti-aliased) ellipsoidal skull shell, 6 mm, standing 1 mm off the brain
def smoothstep(e0, e1, x):
    t = np.clip((x - e0)/(e1 - e0), 0, 1); return t*t*(3 - 2*t)
sc_in = 1.0 + 1*mm/np.mean(AX); sc_out = 1.0 + 7*mm/np.mean(AX)
w = dx/np.mean(AX)                                        # ~1 voxel in normalised units
skull_soft = (smoothstep(sc_in - w, sc_in + w, rb) * (1.0 - smoothstep(sc_out - w, sc_out + w, rb))
              ).astype(np.float32)

def compose(skm):
    """velocity field = tissue blended with a SOFT skull mask (never binarised)."""
    base = np.where(brain, np.vectorize({2: C_GM, 3: C_WM, 4: C_GM, 5: C_CSF}.get)(
        np.where(brain, lab, 3)), C_WATER).astype(np.float32)
    return (base*(1-skm) + C_SKULL*skm).astype(np.float32)

def compose_rho(skm):
    base = np.where(brain, 1040.0, 1000.0).astype(np.float32)
    return (base*(1-skm) + 1900.0*skm).astype(np.float32)

c_true = compose(skull_soft); rho_true = compose_rho(skull_soft)

# ---- the skull as the INVERSION believes it to be: warped by the pose error ----
t_err = jnp.array([args.pose_vox, args.pose_vox*0.5, 0.0], jnp.float32)
a_err = jnp.array([0.0, 0.0, np.deg2rad(args.pose_deg)], jnp.float32)
skull_est = np.asarray(jnp.clip(rigid_warp_3d(jnp.asarray(skull_soft), t_err, a_err), 0, 1))
mismatch = float(np.abs(skull_est - skull_soft).sum()/max(skull_soft.sum(), 1))
rho_est = compose_rho(skull_est)
# brain interior held homogeneous; skull placed at the ESTIMATED pose
base_init = np.where(brain, 1560.0, C_WATER).astype(np.float32)
c_init = (base_init*(1-skull_est) + C_SKULL*skull_est).astype(np.float32)
# exclude the ESTIMATED skull from the inverted region (MOFI insight)
inv_mask = brain & (skull_est < 0.05)

ref = build_medium(build_domain((N,)*3, dx), CFG_CMAX, 1000.0, pml_size=PML)
t_end = 1.6 * FOV / 1500.0
ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt, nt = float(ta.dt), int(ta.Nt)
sig = _build_source_signal(FREQ, dt, nt)

def fib(n, R):
    i = np.arange(n) + 0.5; phi = np.arccos(1-2*i/n); th = np.pi*(1+5**0.5)*i
    return np.stack([R*np.sin(phi)*np.cos(th), R*np.sin(phi)*np.sin(th), R*np.cos(phi)], 1)
def to_idx(p):
    return tuple(np.clip((p[:, d]/dx + cx).round().astype(int), 0, N-1) for d in range(3))
R_ARR = 74*mm
si_ = to_idx(fib(args.n_src, R_ARR)); ri_ = to_idx(fib(args.n_rec, R_ARR))
src_list = [(int(si_[0][k]), int(si_[1][k]), int(si_[2][k])) for k in range(args.n_src)]
recv = ri_

print(f"[{args.tag}] pose error {args.pose_vox:.2f} vox / {args.pose_deg:.2f} deg -> "
      f"skull-mass mismatch {mismatch*100:.1f}%  | ellipsoidal head, soft warp", flush=True)
print(f"  N={N} dx={dx*1e3:.2f}mm nt={nt} | {args.n_src} src / {args.n_rec} rec | "
      f"inverted voxels {int(inv_mask.sum())}", flush=True)

t0 = time.time()
obs = generate_observed_data(jnp.asarray(c_true), jnp.asarray(rho_true), dx, src_list, recv,
                             FREQ, pml_size=PML, time_axis=ta, source_signal=sig, dt=dt,
                             verbose=False)
obs_init = generate_observed_data(jnp.asarray(c_init), jnp.asarray(rho_est), dx, src_list, recv,
                                  FREQ, pml_size=PML, time_axis=ta, source_signal=sig, dt=dt,
                                  verbose=False)
rel0 = float(np.linalg.norm(np.asarray(obs_init)-np.asarray(obs))/np.linalg.norm(np.asarray(obs)))
print(f"  obs {tuple(obs.shape)} in {time.time()-t0:.0f}s | starting residual {rel0*100:.2f}%", flush=True)

cfg = FWIConfig(freq_bands=BANDS, n_iters_per_band=args.iters,
                shots_per_iter=args.shots_per_iter, learning_rate=4.0,
                c_min=1480.0, c_max=CFG_CMAX, pml_size=PML, cfl=0.3,
                gradient_smooth_sigma=1.0, mask=jnp.asarray(inv_mask.astype(np.float32)),
                precondition=True, verbose=False)
res = run_fwi(obs, jnp.asarray(c_init), jnp.asarray(rho_est), dx, src_list, recv, sig, dt,
              t_end, config=cfg)
c_rec = np.asarray(res.velocity)
print(f"  FWI done {(time.time()-t0)/60:.1f} min", flush=True)

def eroded(m, it): return m & ~ndi.binary_dilation(~m, iterations=it)
wm_m = eroded(lab == 3, 3) & inv_mask
nuc_m = eroded(lab == 4, 2) & inv_mask
rib_m = eroded(lab == 2, 1) & inv_mask
csf_m = eroded(lab == 5, 1) & inv_mask
def auc(pos, neg):
    v = np.concatenate([pos, neg]); y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    o = np.argsort(v); rk = np.empty_like(o, float); rk[o] = np.arange(1, v.size+1)
    return float((rk[y == 1].sum() - pos.size*(pos.size+1)/2)/(pos.size*neg.size))

wm_v = c_rec[wm_m]
out = dict(tag=args.tag, pose_vox=args.pose_vox, pose_deg=args.pose_deg,
           skull_mismatch_pct=mismatch*100, start_residual_pct=rel0*100,
           n_src=args.n_src, n_rec=args.n_rec, wm_median=float(np.median(wm_v)))
print(f"  WM median {np.median(wm_v):.1f} (true {C_WM})", flush=True)
for nm, m, truth, key in [("deep-gray nuclei (GM/WM contrast)", nuc_m, C_GM, "nuclei"),
                          ("cortical ribbon 3mm", rib_m, C_GM, "ribbon"),
                          ("ventricle (control)", csf_m, C_CSF, "csf")]:
    if m.sum() < 10:
        print(f"  {nm}: too few voxels", flush=True); continue
    v = c_rec[m]; rc = float(np.median(v)-np.median(wm_v)); tc = truth - C_WM
    a = auc(v, wm_v) if truth > C_WM else auc(wm_v, v)
    print(f"  {nm}: true {tc:+.1f} -> {rc:+.1f} m/s ({100*rc/tc:.0f}%)  AUC {a:.3f}", flush=True)
    out[key] = dict(true=tc, recovered=rc, pct=100*rc/tc, auc=a, n_vox=int(m.sum()))
json.dump(out, open(f"{OUT}/summary_{args.tag}.json", "w"), indent=2)
np.save(f"{OUT}/c_rec_{args.tag}.npy", c_rec)
print(f"  saved summary_{args.tag}.json", flush=True)
