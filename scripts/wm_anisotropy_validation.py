"""WM attenuation-anisotropy validation on REAL tissue.

Chains three real-tissue evidence sources into a bracketed, honest verdict:

  (1) REAL ex-vivo human WM substrate  -- STE00 hemisphere DTI (run
      ste00_dti_preprocess.py first): WM is strongly, coherently oriented
      (median FA ~0.42) and distinct from near-isotropic GM (~0.15). The
      anisotropy substrate and the GM/WM discriminator physically EXIST in real
      tissue.
  (2) REAL cross-modal anisotropy magnitude -- muscle longitudinal attenuation
      anisotropy (~2.6x, Nassiri 1979) and WM shear-loss anisotropy (~1.3x,
      Anderson MRE 2018) bracket the (unmeasured) WM compressional value.
  (3) DETECTABILITY through our own pipeline -- is a real-tissue-magnitude ratio
      (~1.3-1.6, NOT the 4x toy cases) recoverable above measurement noise?

Honest limit: no direct acoustic measurement of WM attenuation anisotropy exists;
(2) is borrowed from adjacent tissue/modality. This quantifies whether the effect,
at its real plausible magnitude, is large enough to detect through transcranial
transmission tomography -- the go/no-go for the whole GM/WM-via-anisotropy path.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brain_fwi.tissue.wm_anisotropy import (
    predicted_alpha_profile, muscle_reference, wm_ratio_bracket,
    ratio_from_fa, wm_acoustic_estimate)

OUT = "results/wm_anisotropy_validation"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

# ---------- (1) real ex-vivo WM substrate ----------
with open(f"{OUT}/ste00_dti_stats.json") as f:
    ste = json.load(f)
fa_hist = np.load(f"{OUT}/ste00_fa_hist.npy")
wm_fa = ste["wm_fa_median"]; gm_fa = ste["gm_fa_median"]
print("=== (1) REAL ex-vivo human WM (STE00) ===", flush=True)
print(f"  WM median FA {wm_fa:.2f} (p90 {ste['wm_fa_p90']:.2f}), GM median FA {gm_fa:.2f}", flush=True)
print(f"  WM fraction {ste['wm_frac_FA>=0.25']*100:.0f}%, orientation coherence {ste.get('wm_global_orientation_coherence',float('nan')):.2f}", flush=True)
print(f"  -> anisotropic WM population is distinct from isotropic GM: discriminator exists in real tissue", flush=True)

# ---------- (2) cross-modal anisotropy magnitude ----------
lo, mid, hi = wm_ratio_bracket()
mus = muscle_reference()
# real-tissue FA-scaled central estimate (couple diffusion FA -> acoustic ratio, muscle-capped)
ratio_fa = float(ratio_from_fa(wm_fa, ratio_max=hi))
print("\n=== (2) cross-modal anisotropy magnitude (real, borrowed) ===", flush=True)
print(f"  muscle (measured, longitudinal): ratio {mus['ratio']:.2f}", flush=True)
print(f"  WM MRE shear-loss (measured):    ratio {lo:.2f}", flush=True)
print(f"  WM compressional BRACKET:        [{lo:.2f}, {hi:.2f}], central {mid:.2f}", flush=True)
print(f"  FA-scaled at real WM FA={wm_fa:.2f}: ratio {ratio_fa:.2f}", flush=True)

# ---------- (3) detectability through transmission tomography ----------
# ring transmission, homogeneous anisotropic WM patch; recover (b,u,v) by LSQ.
# alpha(theta) = b - u cos2theta - v sin2theta ; anisotropy magnitude = 2*hypot(u,v)
n_tr = 24; R = 0.05
ang = np.linspace(0, 2*np.pi, n_tr, endpoint=False)
tx, ty = R*np.cos(ang), R*np.sin(ang)
pairs = [(i, j) for i in range(n_tr) for j in range(i+1, n_tr)]
th = np.array([np.arctan2(ty[j]-ty[i], tx[j]-tx[i]) for i, j in pairs])
L = np.hypot([tx[j]-tx[i] for i, j in pairs], [ty[j]-ty[i] for i, j in pairs]) * 100  # cm
A = np.stack([L, -L*np.cos(2*th), -L*np.sin(2*th)], 1)  # design: d = A @ [b,u,v]

def recover_ratio(ratio_true, alpha_iso, phi, noise, trials=400):
    a_perp = 2*alpha_iso/(1+ratio_true); a_par = ratio_true*a_perp
    alpha = a_par*np.cos(th-phi)**2 + a_perp*np.sin(th-phi)**2   # true dB/cm
    d0 = alpha*L
    out = []
    for _ in range(trials):
        d = d0*(1 + noise*rng.standard_normal(d0.shape))
        b, u, v = np.linalg.lstsq(A, d, rcond=None)[0]
        amag = 2*np.hypot(u, v)                                   # a_par - a_perp
        r = (b + amag/2)/max(b - amag/2, 1e-6)
        out.append(r)
    return np.array(out)

alpha_iso = 0.9        # dB/cm/MHz representative bulk WM attenuation (brain tissue)
phi_true = 0.6
print("\n=== (3) detectability (24-element ring, 5% amplitude noise) ===", flush=True)
verdict = {}
for label, r_true in [("isotropic (null)", 1.0), ("MRE-analog", lo),
                      ("FA-scaled real", ratio_fa), ("muscle-analog", hi)]:
    rr = recover_ratio(r_true, alpha_iso, phi_true, noise=0.05)
    p5, p50, p95 = np.percentile(rr, [5, 50, 95])
    excl = p5 > 1.02   # 90% CI excludes isotropy
    print(f"  {label:18s} true {r_true:.2f} -> recovered {p50:.2f} [90%CI {p5:.2f},{p95:.2f}]  "
          f"{'DETECTED (excludes isotropy)' if excl else 'not resolved from isotropy'}", flush=True)
    verdict[label] = dict(true=r_true, med=p50, ci=[p5, p95], detected=bool(excl))

# noise sweep: smallest detectable ratio vs measurement noise
print("\n  smallest detectable ratio vs noise:", flush=True)
noise_curve = {}
for nz in [0.02, 0.05, 0.10, 0.20]:
    det = None
    for r_true in np.linspace(1.02, 2.0, 50):
        rr = recover_ratio(r_true, alpha_iso, phi_true, noise=nz, trials=300)
        if np.percentile(rr, 5) > 1.02:
            det = r_true; break
    noise_curve[nz] = det
    print(f"    noise {nz*100:4.0f}%  ->  min detectable ratio {det:.2f}" if det else
          f"    noise {nz*100:4.0f}%  ->  none up to 2.0", flush=True)

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="white")
# (a) real FA substrate
edges = np.linspace(0, 1, len(fa_hist)+1); ctr = 0.5*(edges[:-1]+edges[1:])
ax[0].fill_between(ctr, fa_hist, color="#888", alpha=.3, step="mid")
ax[0].axvline(gm_fa, color="#2a9d3a", lw=2, label=f"GM median {gm_fa:.2f} (isotropic)")
ax[0].axvline(wm_fa, color="#c02a2a", lw=2, label=f"WM median {wm_fa:.2f} (anisotropic)")
ax[0].set_xlabel("fractional anisotropy (FA)"); ax[0].set_yticks([])
ax[0].set_title("(1) REAL ex-vivo human WM\nanisotropy substrate exists"); ax[0].legend(fontsize=8)
# (b) predicted alpha(theta) bracket
thp = np.linspace(0, np.pi, 200)
for r, c, lab in [(lo, "#3a6", "MRE-analog 1.3x"), (mid, "#e8a", "central 1.6x"), (hi, "#c33", "muscle-analog 2.0x")]:
    a_perp = 2*alpha_iso/(1+r); a_par = r*a_perp
    ax[1].plot(np.degrees(thp), predicted_alpha_profile(thp, a_par, a_perp), color=c, label=lab)
ax[1].set_xlabel("angle to fibre (deg)"); ax[1].set_ylabel("attenuation (dB/cm/MHz)")
ax[1].set_title("(2) predicted WM directional\nattenuation (real-tissue bracket)"); ax[1].legend(fontsize=8)
# (c) detectability
labels = list(verdict); meds = [verdict[k]["med"] for k in labels]
cis = np.array([verdict[k]["ci"] for k in labels]).T
cols = ["#c33" if verdict[k]["detected"] else "#999" for k in labels]
xp = np.arange(len(labels))
ax[2].errorbar(xp, meds, yerr=[np.array(meds)-cis[0], cis[1]-np.array(meds)],
               fmt="o", ecolor="#555", capsize=4, mfc="w", ms=7)
for i, c in enumerate(cols): ax[2].plot(xp[i], meds[i], "o", color=c, ms=8)
ax[2].axhline(1.0, color="k", ls="--", lw=1, label="isotropy (null)")
ax[2].set_xticks(xp); ax[2].set_xticklabels([l.split()[0] for l in labels], rotation=20, fontsize=8)
ax[2].set_ylabel("recovered anisotropy ratio")
ax[2].set_title("(3) detectability through\ntransmission (5% noise)"); ax[2].legend(fontsize=8)
fig.suptitle("WM attenuation-anisotropy validation on real tissue: substrate (real) -> magnitude (borrowed) -> detectability", y=1.02, fontsize=12)
fig.savefig(f"{OUT}/wm_anisotropy_validation.png", dpi=140, bbox_inches="tight", facecolor="white")
print(f"\nsaved {OUT}/wm_anisotropy_validation.png", flush=True)

json.dump({"ste00": ste, "bracket": {"lo": lo, "mid": mid, "hi": hi},
           "ratio_fa_scaled": ratio_fa, "detectability": verdict, "noise_curve": {str(k): v for k, v in noise_curve.items()}},
          open(f"{OUT}/validation_summary.json", "w"), indent=2)
print("saved validation_summary.json", flush=True)
