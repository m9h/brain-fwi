"""THE DECISIVE TEST: can velocity FWI at imaging frequency resolve GM/WM?

The project ruled out sound speed as a GM/WM discriminator on the premise
"GM = WM = 1560 m/s, velocity FWI is blind to it". That premise is an artifact of
the ITRUSST benchmark table (`phantoms/properties.py` assigns BOTH 1560), not
physics: real measurements separate them by ~17 m/s (Mitcham/Duric, Med. Phys.
2025, ex-vivo human: WM corpus callosum 1639 +/- 3 vs GM cerebellum 1656 +/- 6).

Everything in the campaign so far ran at 40-160 kHz, where the best-case
resolution (lambda/2) is 5-20 mm -- the cortical GM ribbon is 2.5-4 mm, i.e.
physically unresolvable REGARDLESS of contrast. So GM/WM via velocity was never
actually tested; it was ruled out at frequencies too low to see it.

This runs the test that was skipped: **500 kHz-class FWI (lambda 3.1 mm,
resolution ~1.6 mm) with the real measured ~17 m/s contrast**, and asks directly
whether the cortical ribbon appears in the reconstruction.

  phase A (default) -- brain only, no skull. The cheap decisive test: if a 17 m/s
      contrast in a 3 mm ribbon cannot be recovered even WITHOUT the skull, the
      whole GM/WM-via-velocity idea dies here.
  phase B -- + skull (given at truth, MOFI-style; invert brain only). The
      transcranial version, run only if A passes.

Sign caveat: the ex-vivo measurement has GM > WM while the repo's Birnbaum table
has WM > GM. Literature is not consistent on sign. Detection of a BOUNDARY does
not depend on the sign, so the test uses the measured MAGNITUDE (17 m/s) with the
measured ordering, rescaled to an in-vivo 1560 m/s mean (ex-vivo absolute values
are elevated by fixation; only the difference transfers).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data,
    _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi

ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=["A", "B"], default="A")
ap.add_argument("--iters", type=int, default=25)
ap.add_argument("--shots-per-iter", type=int, default=4)
args = ap.parse_args()

OUT = "results/gmwm_velocity_resolution"; os.makedirs(OUT, exist_ok=True)
mm = 1e-3
N, dx = 320, 0.5e-3                      # 160 mm FOV, lambda/dx = 6.2 at 500 kHz
FREQ = 400e3                             # Ricker centre; bands push to 550 kHz
BANDS = [(250e3, 400e3), (400e3, 550e3)]

# ---- real measured GM/WM contrast (magnitude from Mitcham 2025), mean 1560 ----
C_WM, C_GM = 1551.5, 1568.5              # 17 m/s apart
C_DEEP, C_CSF, C_WATER, C_SKULL = 1565.0, 1500.0, 1500.0, 2800.0
CONTRAST = C_GM - C_WM

# ---- phantom: WM interior, 3 mm cortical GM ribbon, deep-gray nuclei, ventricles
cx = N // 2
xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
r = np.hypot(xx - cx, yy - cx) * dx
R_BRAIN, RIBBON = 42*mm, 3*mm            # cortical ribbon thickness = 3 mm (~1 lambda)
lab = np.zeros((N, N), np.int32)         # 0 water
lab[r <= R_BRAIN] = 2                    # 2 = GM ribbon
lab[r <= R_BRAIN - RIBBON] = 3           # 3 = WM interior
# deep gray nuclei (two blobs) and CSF ventricles (positive control: 51 m/s)
for off in (-11*mm, 11*mm):
    lab[(np.hypot((xx-cx)*dx - off, (yy-cx)*dx + 4*mm) <= 7*mm)] = 4
for off in (-6*mm, 6*mm):
    lab[(np.hypot((xx-cx)*dx - off, (yy-cx)*dx - 8*mm) <= 3.5*mm)] = 5
if args.phase == "B":
    lab[(r > R_BRAIN) & (r <= R_BRAIN + 1*mm)] = 0          # CSF gap -> water
    lab[(r > R_BRAIN + 1*mm) & (r <= R_BRAIN + 7*mm)] = 6   # 6 mm skull

CMAP = {0: C_WATER, 2: C_GM, 3: C_WM, 4: C_DEEP, 5: C_CSF, 6: C_SKULL}
RMAP = {0: 1000.0, 2: 1040.0, 3: 1040.0, 4: 1040.0, 5: 1000.0, 6: 1900.0}
c_true = np.vectorize(CMAP.get)(lab).astype(np.float32)
rho = np.vectorize(RMAP.get)(lab).astype(np.float32)
brain = np.isin(lab, [2, 3, 4, 5])

# ---- ring array ----
n_src, n_rec = 24, 96
Rring = 58*mm
def ring(n):
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    return ((cx + Rring/dx*np.cos(a)).round().astype(int),
            (cx + Rring/dx*np.sin(a)).round().astype(int))
sx, sy = ring(n_src); rx, ry = ring(n_rec)
src_list = [(int(sx[i]), int(sy[i])) for i in range(n_src)]
recv = (rx, ry)

# CRITICAL: run_fwi builds its time axis from config.c_max (NOT the true medium),
# so the observed data MUST be generated on that same axis. A mismatch here is a
# few-percent time-base drift = >1 wave period over the record, which swamps a 1%
# GM/WM contrast entirely (flat loss + velocity runaway to the bound).
CFG_CMAX = 1600.0 if args.phase == "A" else 2900.0
ref = build_medium(build_domain((N, N), dx), CFG_CMAX, 1000.0, pml_size=16)
t_end = 1.7 * (N*dx) / 1500.0
ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt, nt = float(ta.dt), int(ta.Nt)
sig = _build_source_signal(FREQ, dt, nt)
print(f"phase {args.phase}: N={N} dx={dx*1e3:.2f}mm  lambda(500k)={1560/500e3*1e3:.2f}mm "
      f"= {1560/500e3/dx:.1f} vox  ribbon {RIBBON*1e3:.0f}mm={RIBBON/dx:.0f} vox", flush=True)
print(f"  true GM {C_GM} WM {C_WM} -> contrast {CONTRAST:.1f} m/s ; nt={nt} dt={dt*1e9:.1f}ns", flush=True)

t0 = time.time()
print("generating observed data ...", flush=True)
obs = generate_observed_data(jnp.asarray(c_true), jnp.asarray(rho), dx, src_list, recv,
                             FREQ, pml_size=16, time_axis=ta, source_signal=sig, dt=dt,
                             verbose=False)
print(f"  done {time.time()-t0:.0f}s  obs {obs.shape}", flush=True)

# ---- initial model: homogeneous brain at the MEAN speed (blind to GM/WM) ----
c_init = np.where(brain, 1560.0, c_true).astype(np.float32)   # skull/water at truth

# Sanity guard: with a consistent time axis the starting residual must be SMALL
# (the model differs from truth by only ~17 m/s + ventricles). A large residual
# means a forward/inversion inconsistency, not a hard inverse problem.
obs_init = generate_observed_data(jnp.asarray(c_init), jnp.asarray(rho), dx, src_list, recv,
                                  FREQ, pml_size=16, time_axis=ta, source_signal=sig, dt=dt,
                                  verbose=False)
rel0 = float(np.linalg.norm(np.asarray(obs_init) - np.asarray(obs)) / np.linalg.norm(np.asarray(obs)))
print(f"  starting relative data residual: {rel0*100:.2f}%  "
      f"({'OK - consistent' if rel0 < 0.25 else 'TOO LARGE - forward/inversion inconsistency!'})", flush=True)
cfg = FWIConfig(freq_bands=BANDS, n_iters_per_band=args.iters,
                shots_per_iter=args.shots_per_iter, learning_rate=4.0,
                c_min=1480.0, c_max=CFG_CMAX, pml_size=16, cfl=0.3,
                gradient_smooth_sigma=1.0, mask=jnp.asarray(brain.astype(np.float32)),
                precondition=True, verbose=True)
print(f"running FWI ({len(BANDS)} bands x {args.iters} iters) ...", flush=True)
res = run_fwi(obs, jnp.asarray(c_init), jnp.asarray(rho), dx, src_list, recv, sig, dt,
              t_end, config=cfg)
c_rec = np.asarray(res.velocity)
print(f"  FWI done {time.time()-t0:.0f}s", flush=True)

# ---- metrics: does GM/WM separate in the RECOVERED map? ----
# erode masks 1 voxel off the boundary so the metric is about tissue, not the edge
# The cortical ribbon abuts water (68 m/s jump) so FWI overshoots at that edge;
# erode hard and use MEDIANS so the contrast metric measures tissue, not Gibbs.
from scipy import ndimage as ndi
gm_m = (lab == 2) & ~ndi.binary_dilation(lab != 2, iterations=2)
wm_m = (lab == 3) & ~ndi.binary_dilation(lab != 3, iterations=3)
csf_m = (lab == 5) & ~ndi.binary_dilation(lab != 5, iterations=1)
gm_v, wm_v, csf_v = c_rec[gm_m], c_rec[wm_m], c_rec[csf_m]
rec_contrast = float(np.median(gm_v) - np.median(wm_v))
pooled = np.sqrt(0.5*(gm_v.var() + wm_v.var()))
d = float(rec_contrast/pooled) if pooled > 0 else float("nan")
# AUC: can a threshold on recovered speed classify GM vs WM?
lo = np.concatenate([gm_v, wm_v]); yy_ = np.r_[np.ones(gm_v.size), np.zeros(wm_v.size)]
order = np.argsort(lo); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, lo.size+1)
auc = float((ranks[yy_ == 1].sum() - gm_v.size*(gm_v.size+1)/2) / (gm_v.size*wm_v.size))
rec_frac = 100.0*rec_contrast/CONTRAST
csf_true_c = C_CSF - C_WM
csf_rec_c = float(np.median(csf_v) - np.median(wm_v))

print("\n================ RESULT: GM/WM via velocity at imaging frequency ================", flush=True)
print(f"  true GM-WM contrast        : {CONTRAST:6.1f} m/s", flush=True)
print(f"  RECOVERED GM-WM contrast   : {rec_contrast:6.1f} m/s   ({rec_frac:.0f}% of true)", flush=True)
print(f"  GM median {np.median(gm_v):.1f} (IQR {np.subtract(*np.percentile(gm_v,[75,25])):.1f}) | "
      f"WM median {np.median(wm_v):.1f} (IQR {np.subtract(*np.percentile(wm_v,[75,25])):.1f})", flush=True)
print(f"  separability Cohen's d     : {d:6.2f}   (>0.8 = large, >2 = clearly separable)", flush=True)
print(f"  GM-vs-WM classification AUC: {auc:6.3f}   (0.5 = blind, >0.9 = resolved)", flush=True)
print(f"  positive control (ventricle): true {csf_true_c:.1f} -> recovered {csf_rec_c:.1f} m/s "
      f"({100*csf_rec_c/csf_true_c:.0f}%)", flush=True)
verdict = ("RESOLVED - velocity FWI sees GM/WM at imaging frequency" if auc > 0.9 and rec_frac > 40
           else "PARTIAL - contrast recovered but not cleanly separable" if auc > 0.7
           else "NOT RESOLVED")
print(f"  VERDICT: {verdict}", flush=True)
print("=" * 80, flush=True)

# ---- figure ----
fig, ax = plt.subplots(1, 4, figsize=(19, 4.3), facecolor="white")
vmin, vmax = 1540, 1580
im0 = ax[0].imshow(np.where(brain, c_true, np.nan).T, origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
ax[0].set_title(f"truth (GM-WM = {CONTRAST:.0f} m/s)"); plt.colorbar(im0, ax=ax[0], fraction=.046)
im1 = ax[1].imshow(np.where(brain, c_init, np.nan).T, origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
ax[1].set_title("start (homogeneous 1560)"); plt.colorbar(im1, ax=ax[1], fraction=.046)
im2 = ax[2].imshow(np.where(brain, c_rec, np.nan).T, origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
ax[2].set_title(f"FWI @ {BANDS[-1][1]/1e3:.0f} kHz\nrecovered {rec_contrast:.1f} m/s, AUC {auc:.2f}")
plt.colorbar(im2, ax=ax[2], fraction=.046)
# radial profile across the cortical ribbon
rr = np.linspace(0, R_BRAIN + 2*mm, 160)
prof_t = [np.nanmean(c_true[(r >= a-0.4*mm) & (r < a+0.4*mm) & brain]) for a in rr]
prof_r = [np.nanmean(c_rec[(r >= a-0.4*mm) & (r < a+0.4*mm) & brain]) for a in rr]
ax[3].plot(rr*1e3, prof_t, "k-", lw=2, label="truth")
ax[3].plot(rr*1e3, prof_r, "r-", lw=1.6, label="FWI")
ax[3].axvspan((R_BRAIN-RIBBON)*1e3, R_BRAIN*1e3, color="#ffd", alpha=.7, label="GM ribbon (3mm)")
ax[3].set_xlabel("radius (mm)"); ax[3].set_ylabel("sound speed (m/s)")
ax[3].set_title("cortical ribbon profile"); ax[3].legend(fontsize=8)
for a in ax[:3]: a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"Does velocity FWI resolve GM/WM at imaging frequency? phase {args.phase} "
             f"({'brain only' if args.phase=='A' else 'transcranial, skull at truth'}) "
             f"- real measured 17 m/s contrast, 3 mm ribbon", y=1.03, fontsize=12)
fp = f"{OUT}/gmwm_velocity_phase{args.phase}.png"
fig.savefig(fp, dpi=140, bbox_inches="tight", facecolor="white")
np.save(f"{OUT}/c_rec_phase{args.phase}.npy", c_rec)
json.dump({"phase": args.phase, "true_contrast": CONTRAST, "recovered_contrast": rec_contrast,
           "pct_recovered": rec_frac, "cohens_d": d, "auc": auc, "verdict": verdict,
           "gm_mean": float(gm_v.mean()), "wm_mean": float(wm_v.mean()),
           "csf_control_pct": 100*csf_rec_c/csf_true_c, "freq_bands": BANDS,
           "dx_mm": dx*1e3, "ribbon_mm": RIBBON*1e3},
          open(f"{OUT}/summary_phase{args.phase}.json", "w"), indent=2)
print(f"saved {fp}", flush=True)
