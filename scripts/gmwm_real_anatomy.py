"""GM/WM velocity FWI on REAL anatomy with a TRILAYER skull — the honest test.

Every GM/WM result so far used geometry I chose: a smooth spherical/ellipsoidal
brain with a 3 mm "cortical ribbon" shell and a UNIFORM 2800 m/s skull. Both are
far kinder than reality. This swaps in the MIDA head (IT'IS, 500 um, ITRUSST
reference) which supplies the two things the synthetic phantom was missing:

  * **Real folded cortex** — the GM/WM boundary is a convoluted surface with
    1-2 mm features and partial-volume mixing everywhere, not a smooth shell.
  * **A real trilayer skull** — MIDA separates the diploe (label 52, trabecular,
    2300 m/s, alpha 8.0) from the inner/outer tables (53/54, cortical, 2800 m/s,
    alpha 4.0). The diploe is the dominant source of transcranial aberration and
    the uniform-shell tests could not see it at all.

MIDA's own property table has the SAME GM==WM==1560 degeneracy as ITRUSST, so the
real measured contrast (Mitcham 2025, ~17 m/s) is applied on top — otherwise the
target would be invisible by construction.

Two GM/WM metrics, mirroring the synthetic design:
  * **deep GM vs WM** (heavily eroded GM = chunky subcortical structures) — the
    CONTRAST test, the analogue of the synthetic "deep-gray nuclei".
  * **all GM vs WM** (includes thin folded cortex) — the hard, honest test, which
    is partly a RESOLUTION test because real cortex is ~2.5-4 mm.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from brain_fwi.phantoms.mida import (
    load_mida_volume, center_crop_to_cube, resample_volume,
    MIDA_TISSUE_GROUPS, MIDA_ACOUSTIC_PROPERTIES)
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data,
    _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi

MIDA_PATH = "/data/datasets/MIDAv1-0/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii"
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=144)
ap.add_argument("--n-head", type=int, default=100, help="voxels across the MIDA head cube")
ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--shots-per-iter", type=int, default=4)
ap.add_argument("--n-src", type=int, default=16)
ap.add_argument("--n-rec", type=int, default=128)
ap.add_argument("--uniform-skull", action="store_true",
                help="collapse the trilayer skull to uniform cortical bone (ablation)")
# --- skull PROPERTY error: applied to the INVERSION's skull model only.
# Real sources: CT->acoustic mapping uncertainty (~5-10% on speed), subject-to-
# subject diploe porosity variation (large), and poorly-known bone attenuation.
ap.add_argument("--skull-dc", type=float, default=0.0,
                help="fractional sound-speed error on ALL skull bone (e.g. 0.05 = +5%%)")
ap.add_argument("--diploe-dc", type=float, default=0.0,
                help="fractional sound-speed error on the DIPLOE only (porosity error)")
ap.add_argument("--skull-dalpha", type=float, default=1.0,
                help="multiplicative error on skull attenuation (2.0 = 2x underestimate)")
ap.add_argument("--tag", type=str, default="real")
# --- interface handling (scope_3d_gmwm_run.md: a hard-frozen skull sign-inverts the sub-skull
# cortex; freeing it + smoothing at ~lambda/2 fixed the synthetic ribbon 0.85 -> 0.97)
ap.add_argument("--interface", choices=["frozen", "free", "soft"], default="frozen",
                help="frozen = invert brain only (campaign default); free = also invert bone+CSF "
                     "(start at the estimated skull); soft = bone+CSF at a low relative lr")
ap.add_argument("--soft-weight", type=float, default=0.2, help="relative lr of bone+CSF for --interface soft")
ap.add_argument("--smooth", type=float, default=1.0, help="gradient_smooth_sigma (grid pts); ~lambda/2 = 2")
args = ap.parse_args()

OUT = "results/gmwm_real_anatomy"; os.makedirs(OUT, exist_ok=True)
N, PML = args.N, 10
BANDS = [(120e3, 190e3), (190e3, 260e3)]; FREQ = 210e3
C_WM, C_GM = 1551.5, 1568.5          # real measured contrast (Mitcham 2025)
CONTRAST = C_GM - C_WM
CFG_CMAX = 2900.0

# ---- load MIDA, resample head, pad with water so the array has clearance ----
lab_native = load_mida_volume(MIDA_PATH)
lab_cube = center_crop_to_cube(lab_native)
HEAD_M = lab_cube.shape[0] * 0.5e-3                      # MIDA is 500 um isotropic
lab_head = resample_volume(lab_cube, (args.n_head,)*3, order=0)
dx = HEAD_M / args.n_head
pad = (N - args.n_head) // 2
BG = 50                                                   # MIDA background -> water
lab = np.full((N, N, N), BG, lab_head.dtype)
lab[pad:pad+args.n_head, pad:pad+args.n_head, pad:pad+args.n_head] = lab_head
cx = N // 2
print(f"[{args.tag}] MIDA {lab_native.shape} -> head {args.n_head}^3 in grid {N}^3, "
      f"dx={dx*1e3:.2f}mm, FOV={N*dx*1e3:.0f}mm (head {HEAD_M*1e3:.0f}mm)", flush=True)

# ---- group masks + properties (REAL GM/WM contrast on top of MIDA's table) ----
g2l = MIDA_TISSUE_GROUPS
def gmask(name): return np.isin(lab, g2l.get(name, []))
GM, WM = gmask("grey_matter"), gmask("white_matter")
CORT, TRAB = gmask("cortical_bone"), gmask("trabecular_bone")
CSF = gmask("csf")
c = np.full((N, N, N), 1500.0, np.float32)               # water default
rho = np.full((N, N, N), 1000.0, np.float32)
alpha = np.zeros((N, N, N), np.float32)
for name in g2l:
    m = gmask(name)
    if not m.any(): continue
    p = MIDA_ACOUSTIC_PROPERTIES.get(name)
    if p is None: continue
    c[m], rho[m], alpha[m] = p["sound_speed"], p["density"], p["attenuation"]
# internal air -> water (343 m/s is unresolvable at this dx and destabilises the solver)
air = gmask("air"); c[air], rho[air], alpha[air] = 1500.0, 1000.0, 0.0
# THE REAL CONTRAST (MIDA's own table has GM==WM==1560, same degeneracy as ITRUSST)
c[GM], c[WM] = C_GM, C_WM
if args.uniform_skull:                                    # ablation: kill the diploe
    c[TRAB], rho[TRAB], alpha[TRAB] = 2800.0, 1850.0, 4.0
brain = GM | WM
print(f"  GM {int(GM.sum())} WM {int(WM.sum())} | skull cortical {int(CORT.sum())} "
      f"diploe {int(TRAB.sum())} ({'UNIFORM ablation' if args.uniform_skull else 'TRILAYER'}) "
      f"| CSF {int(CSF.sum())}", flush=True)

# ---- array: spherical shell in the water gap ----
ref = build_medium(build_domain((N,)*3, dx), CFG_CMAX, 1000.0, pml_size=PML)
t_end = 1.6 * (N*dx) / 1500.0
ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt, nt = float(ta.dt), int(ta.Nt)
sig = _build_source_signal(FREQ, dt, nt)
head_vox = np.argwhere(lab != BG)
head_r = float(np.abs(head_vox - cx).max())               # head half-extent in voxels
r_arr = min(head_r + 6, N/2 - PML - 4)
def fib(n, R):
    i = np.arange(n) + 0.5; phi = np.arccos(1-2*i/n); th = np.pi*(1+5**0.5)*i
    return np.stack([R*np.sin(phi)*np.cos(th), R*np.sin(phi)*np.sin(th), R*np.cos(phi)], 1)
def to_idx(p): return tuple(np.clip((p[:, d] + cx).round().astype(int), 0, N-1) for d in range(3))
si_, ri_ = to_idx(fib(args.n_src, r_arr)), to_idx(fib(args.n_rec, r_arr))
src_list = [(int(si_[0][k]), int(si_[1][k]), int(si_[2][k])) for k in range(args.n_src)]
recv = ri_
lam = 1560.0/BANDS[-1][1]
print(f"  array r={r_arr:.0f} vox (head {head_r:.0f}, PML edge {N/2-PML:.0f}) | "
      f"top {BANDS[-1][1]/1e3:.0f}kHz lambda={lam*1e3:.2f}mm ppw={lam/dx:.1f} "
      f"res={lam/2*1e3:.2f}mm | nt={nt}", flush=True)

t0 = time.time()
obs = generate_observed_data(jnp.asarray(c), jnp.asarray(rho), dx, src_list, recv, FREQ,
                             pml_size=PML, time_axis=ta, source_signal=sig, dt=dt,
                             attenuation=jnp.asarray(alpha), alpha_power=1.5, verbose=False)

# ---- the skull as the INVERSION believes it: perturbed properties ----
bone = CORT | TRAB
c_est, alpha_est = c.copy(), alpha.copy()
if args.skull_dc:    c_est[bone] = c[bone] * (1.0 + args.skull_dc)
if args.diploe_dc:   c_est[TRAB] = c_est[TRAB] * (1.0 + args.diploe_dc)
if args.skull_dalpha != 1.0: alpha_est[bone] = alpha[bone] * args.skull_dalpha
serr = float(np.abs(c_est[bone]-c[bone]).mean()) if bone.any() else 0.0
print(f"  skull property error: dc={args.skull_dc:+.0%} diploe={args.diploe_dc:+.0%} "
      f"dalpha={args.skull_dalpha:.2g}x -> mean |dc| {serr:.0f} m/s", flush=True)

c_init = np.where(brain, 1560.0, c_est).astype(np.float32)  # brain homogeneous; skull PERTURBED
obs_i = generate_observed_data(jnp.asarray(c_init), jnp.asarray(rho), dx, src_list, recv, FREQ,
                               pml_size=PML, time_axis=ta, source_signal=sig, dt=dt,
                               attenuation=jnp.asarray(alpha_est), alpha_power=1.5, verbose=False)
rel0 = float(np.linalg.norm(np.asarray(obs_i)-np.asarray(obs))/np.linalg.norm(np.asarray(obs)))
print(f"  obs {tuple(obs.shape)} in {time.time()-t0:.0f}s | start residual {rel0*100:.2f}%", flush=True)

if args.interface == "frozen":
    fwi_mask = brain.astype(np.float32)
else:
    shell = (bone | CSF) & ~brain
    fwi_mask = brain.astype(np.float32) + (1.0 if args.interface == "free" else args.soft_weight) * shell
print(f"  interface={args.interface} smooth={args.smooth} | inverted brain {int(brain.sum())} "
      f"+ shell {int(((bone | CSF) & ~brain).sum()) if args.interface != 'frozen' else 0}", flush=True)
cfg = FWIConfig(freq_bands=BANDS, n_iters_per_band=args.iters,
                shots_per_iter=args.shots_per_iter, learning_rate=4.0,
                c_min=1480.0, c_max=CFG_CMAX, pml_size=PML, cfl=0.3,
                gradient_smooth_sigma=args.smooth, mask=jnp.asarray(fwi_mask),
                attenuation=jnp.asarray(alpha_est), alpha_power=1.5,  # absorption-aware (est.)
                precondition=True, verbose=False)
res = run_fwi(obs, jnp.asarray(c_init), jnp.asarray(rho), dx, src_list, recv, sig, dt,
              t_end, config=cfg)
c_rec = np.asarray(res.velocity)
print(f"  FWI done {(time.time()-t0)/60:.1f} min", flush=True)

# ---- metrics ----
def er(m, it): return m & ~ndi.binary_dilation(~m, iterations=it)
wm_m = er(WM, 2)
gm_all = er(GM, 1)
gm_deep = er(GM, 2) & ~ndi.binary_dilation(~brain, iterations=6)   # chunky subcortical GM
csf_m = er(CSF, 1)
def auc(pos, neg):
    v = np.concatenate([pos, neg]); y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    o = np.argsort(v); rk = np.empty_like(o, float); rk[o] = np.arange(1, v.size+1)
    return float((rk[y == 1].sum() - pos.size*(pos.size+1)/2)/(pos.size*neg.size))

wm_v = c_rec[wm_m]
out = dict(tag=args.tag, interface=args.interface, smooth=args.smooth, N=N, dx_mm=dx*1e3, res_mm=lam/2*1e3, ppw=lam/dx,
           trilayer=not args.uniform_skull, start_residual_pct=rel0*100,
           skull_dc=args.skull_dc, diploe_dc=args.diploe_dc,
           skull_dalpha=args.skull_dalpha, skull_mean_dc_ms=serr,
           n_src=args.n_src, n_rec=args.n_rec, wm_median=float(np.median(wm_v)),
           true_contrast=CONTRAST)
print(f"\n===== REAL ANATOMY ({'TRILAYER' if not args.uniform_skull else 'uniform'} skull) =====", flush=True)
print(f"  WM median {np.median(wm_v):.1f} (true {C_WM})", flush=True)
for nm, m, truth, key in [("deep GM vs WM  (CONTRAST test)", gm_deep, C_GM, "deep_gm"),
                          ("all GM vs WM   (incl. folded cortex)", gm_all, C_GM, "all_gm"),
                          ("CSF vs WM      (control)", csf_m, 1500.0, "csf")]:
    if m.sum() < 50:
        print(f"  {nm}: too few voxels ({int(m.sum())})", flush=True); continue
    v = c_rec[m]; rc = float(np.median(v)-np.median(wm_v)); tc = truth - C_WM
    a = auc(v, wm_v) if truth > C_WM else auc(wm_v, v)
    print(f"  {nm}: true {tc:+.1f} -> {rc:+.1f} m/s ({100*rc/tc:.0f}%)  AUC {a:.3f}  "
          f"[{int(m.sum())} vox]", flush=True)
    out[key] = dict(true=tc, recovered=rc, pct=100*rc/tc, auc=a, n_vox=int(m.sum()))
json.dump(out, open(f"{OUT}/summary_{args.tag}.json", "w"), indent=2)

k = cx
fig, ax = plt.subplots(1, 4, figsize=(19, 4.5), facecolor="white")
ax[0].imshow(lab[:, :, k].T, origin="lower", cmap="nipy_spectral"); ax[0].set_title("MIDA labels")
for a_, img, ttl in [(ax[1], c, "truth (real GM/WM contrast)"), (ax[2], c_init, "start"),
                     (ax[3], c_rec, f"FWI @ {BANDS[-1][1]/1e3:.0f} kHz")]:
    im = a_.imshow(np.where(brain[:, :, k], img[:, :, k], np.nan).T, origin="lower",
                   cmap="turbo", vmin=1545, vmax=1575)
    a_.set_title(ttl); plt.colorbar(im, ax=a_, fraction=.046)
for a_ in ax: a_.set_xticks([]); a_.set_yticks([])
dg = out.get("deep_gm", {})
fig.suptitle(f"GM/WM on REAL anatomy (MIDA) + {'TRILAYER' if not args.uniform_skull else 'uniform'} "
             f"skull — deep-GM AUC {dg.get('auc', float('nan')):.3f}, res {lam/2*1e3:.2f}mm", y=1.02)
fig.savefig(f"{OUT}/gmwm_real_{args.tag}.png", dpi=140, bbox_inches="tight", facecolor="white")
print(f"  saved {OUT}/gmwm_real_{args.tag}.png", flush=True)
