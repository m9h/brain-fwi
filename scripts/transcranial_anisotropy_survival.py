"""Does WM attenuation anisotropy survive the SKULL at transcranial frequency?

This is the experiment the earlier "validation" skipped: a full-wave 2-D
transcranial slice at 500 kHz -- water / SKULL ring / brain, with a white-matter
region carrying the real-tissue-magnitude anisotropy tensor (ratio ~1.4). Fire a
ring of sources through the skull and ask whether the anisotropy signature is
still there in the received field.

Key design: the skull is IDENTICAL in the isotropic-WM and anisotropic-WM runs,
so differencing isolates the WM-anisotropy contribution EXACTLY (this is
optimistic -- a real blind inversion does not have the isotropic reference; the
skull aberration is then the confound). Two honest metrics:

  A. signal magnitude -- RMS through-skull amplitude change from WM anisotropy,
     vs a 5% measurement-noise floor and vs how far the skull attenuates the field.
  B. orientation survival -- rotate the fibre 90 deg; does the through-skull
     signature rotate coherently (orientation information survives) or is it
     scrambled below detectability by skull aberration?

Real j-Wave forward via the validated anisotropic absorber (fork
feature/anisotropic-absorption). GPU.
"""
import os, time, numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from jwave.geometry import Medium
from brain_fwi.simulation.forward import (
    build_domain, build_time_axis, simulate_shot_sensors, _build_source_signal)
from brain_fwi.simulation.anisotropic_absorber import fibre_attenuation_tensor

OUT = "results/wm_anisotropy_validation"; os.makedirs(OUT, exist_ok=True)
FREQ = 500e3; Y = 1.2
dx = 3.5e-4; N = 320                       # ~112 mm slice, ~8.9 pts/wavelength (brain)
S = (N, N); cx = cy = N // 2
mm = 1e-3

# ---- geometry: water | skull ring | brain | WM patch ----
xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
r = np.hypot(xx - cx, yy - cy) * dx
Ro, Ri = 50*mm, 44*mm                       # skull outer/inner radius (6 mm bone)
skull = (r <= Ro) & (r > Ri)
brain = r <= Ri
# WM patch: offset disk, radius 16 mm
wm = (np.hypot(xx - cx, yy - (cy + 30)) * dx <= 16*mm) & brain

c = np.full(S, 1500.0, np.float32)          # water
rho = np.full(S, 1000.0, np.float32)
aiso = np.zeros(S, np.float32)              # isotropic attenuation dB/cm/MHz^y
c[brain] = 1560.0; rho[brain] = 1040.0; aiso[brain] = 0.6      # grey matter
aiso[wm] = 0.0                              # WM loss carried by the TENSOR, not iso
c[skull] = 2800.0; rho[skull] = 1900.0; aiso[skull] = 8.0     # bone (dominant loss)

# ---- WM anisotropy tensor (real-tissue magnitude), preserve angle-mean alpha ----
ALPHA_WM = 0.9                              # bulk WM attenuation dB/cm/MHz^y
def wm_tensor(ratio, phi):
    a_perp = 2*ALPHA_WM/(1+ratio); a_par = ratio*a_perp    # a_perp>=a_par (physical)
    Dxx, Dxy, Dyy = fibre_attenuation_tensor(phi, a_par, a_perp)
    D = np.zeros((*S, 3), np.float32)
    D[wm, 0] = float(Dxx); D[wm, 1] = float(Dyy); D[wm, 2] = float(Dxy)  # [Dxx,Dyy,Dxy]
    # everywhere else isotropic-equivalent tensor from the iso field so the
    # absorber is active in brain/skull too (diagonal = aiso)
    for k, ax in [(0, aiso), (1, aiso)]:
        m = ~wm; D[m, k] = ax[m]
    return jnp.asarray(D)

dom = build_domain(S, dx)
def medium(D):
    return Medium(domain=dom, sound_speed=jnp.asarray(c), density=jnp.asarray(rho),
                  attenuation=0.0, pml_size=16, alpha_power=Y, attenuation_tensor=D)
ta = build_time_axis(medium(wm_tensor(1.0, 0.0)), cfl=0.2, t_end=1.3e-4)
dt, nt = float(ta.dt), int(ta.Nt)
sig = _build_source_signal(FREQ, dt, nt)
print(f"grid {S} dx {dx*1e3:.2f}mm  dt {dt*1e9:.1f}ns nt {nt}  skull vox {int(skull.sum())} wm vox {int(wm.sum())}", flush=True)

# ---- transducer ring (in water, just outside skull) ----
n_tr = 24; Rt = 52*mm
ta_ang = np.linspace(0, 2*np.pi, n_tr, endpoint=False)
tix = (cx + Rt/dx*np.cos(ta_ang)).round().astype(int)
tiy = (cy + Rt/dx*np.sin(ta_ang)).round().astype(int)
recv = (tix, tiy)
src_idx = np.arange(0, n_tr, 3)             # 8 sources around the ring

# transmitted-arrival gating: for an opposite (through-head) src-recv pair the
# through-head pulse arrives BEFORE the water wave creeping around the ring
# (t_water = arc_length/c_water). Gate the peak to [0, 0.9*t_water] to capture
# the transcranial transmission and reject the water bypass. Near pairs (whose
# chord stays in water) are excluded via the through-head angular mask.
def arc_len(si, j):
    dth = np.abs(np.angle(np.exp(1j*(ta_ang[j]-ta_ang[si]))))   # [0,pi]
    return Rt * dth
THROUGH = np.radians(110)                    # min src-recv angular sep for through-head

def shot_gated(D):
    med = medium(D); out = np.full((len(src_idx), n_tr), np.nan)
    for k, si in enumerate(src_idx):
        tr = np.asarray(simulate_shot_sensors(med, ta, (int(tix[si]), int(tiy[si])), recv, sig, dt))
        for j in range(n_tr):
            dth = np.abs(np.angle(np.exp(1j*(ta_ang[j]-ta_ang[si]))))
            if dth < THROUGH:                 # not a through-head path
                continue
            ncut = min(nt, int(0.9 * arc_len(si, j)/1500.0 / dt))
            if ncut > 5:
                out[k, j] = np.max(np.abs(tr[:ncut, j]))
    return out                                # (n_src, n_recv), NaN off-mask

t0 = time.time()
print("simulating: isotropic WM ...", flush=True)
P_iso = shot_gated(wm_tensor(1.0, 0.0))
print(f"  done {time.time()-t0:.0f}s. anisotropic WM (phi=30 deg, ratio 1.4) ...", flush=True)
P_an0 = shot_gated(wm_tensor(1.4, np.radians(30)))
print(f"  done {time.time()-t0:.0f}s. anisotropic WM (phi=120 deg) ...", flush=True)
P_an90 = shot_gated(wm_tensor(1.4, np.radians(120)))
print(f"  done {time.time()-t0:.0f}s.", flush=True)

# through-head transmission paths only (gating leaves NaN elsewhere)
mask = np.isfinite(P_iso) & (P_iso > 0)
d0 = (P_an0 - P_iso) / P_iso                  # isolated anisotropy signal (phi=30)
d90 = (P_an90 - P_iso) / P_iso                 # phi=120
rms_sig = float(np.sqrt(np.nanmean(d0[mask]**2)))
a, b = d0[mask], d90[mask]                     # metric B: does signature rotate with fibre?
corr = float(np.corrcoef(a, b)[0, 1])
# skull burial: no-skull reference (same brain, remove bone), same gating
print("simulating: no-skull reference (burial factor) ...", flush=True)
c_ns = c.copy(); rho_ns = rho.copy(); aiso_ns = aiso.copy()
c_ns[skull] = 1500.0; rho_ns[skull] = 1000.0; aiso_ns[skull] = 0.0
def medium_ns():
    D = np.zeros((*S, 3), np.float32); D[..., 0] = aiso_ns; D[..., 1] = aiso_ns
    return Medium(domain=dom, sound_speed=jnp.asarray(c_ns), density=jnp.asarray(rho_ns),
                  attenuation=0.0, pml_size=16, alpha_power=Y, attenuation_tensor=jnp.asarray(D))
si0 = int(src_idx[0])
tr_ns = np.asarray(simulate_shot_sensors(medium_ns(), ta, (int(tix[si0]), int(tiy[si0])), recv, sig, dt))
P_ns = np.full(n_tr, np.nan)
for j in range(n_tr):
    dth = np.abs(np.angle(np.exp(1j*(ta_ang[j]-ta_ang[si0]))))
    if dth >= THROUGH:
        ncut = min(nt, int(0.9*arc_len(si0, j)/1500.0/dt))
        P_ns[j] = np.max(np.abs(tr_ns[:ncut, j]))
m0 = np.isfinite(P_ns) & np.isfinite(P_iso[0])
burial = float(np.median(P_ns[m0] / P_iso[0][m0]))

# coherent integration: because the signature is coherent across paths (metric B),
# a matched filter over N through-head paths beats down per-path noise ~sqrt(N).
NOISE = 0.05
n_paths = int(mask.sum())
mf_snr = float(np.sqrt(np.nansum((d0[mask]/NOISE)**2)))   # matched-filter detection SNR
print("\n=== TRANSCRANIAL anisotropy survival (500 kHz, through 6mm skull) ===", flush=True)
print(f"  skull burial factor (no-skull / with-skull amplitude): {burial:.1f}x", flush=True)
print(f"  [A] WM-anisotropy signal, RMS per-path amplitude change: {rms_sig*100:.1f}%  ({n_paths} through-head paths)", flush=True)
print(f"      vs 5% single-shot noise floor -> per path {'ABOVE' if rms_sig>NOISE else 'BELOW'} noise", flush=True)
print(f"  [B] orientation survival: corr(phi=30, phi=120) = {corr:+.2f}  "
      f"({'orientation PRESERVED through skull' if corr < -0.8 else 'scrambled'})", flush=True)
print(f"  [C] matched-filter detection SNR over {n_paths} paths (exploits coherence): {mf_snr:.1f}"
      f"  -> {'DETECTABLE' if mf_snr>3 else 'marginal' if mf_snr>1.5 else 'undetectable'} with this array", flush=True)
print(f"      (clinical 256-elem ring gives ~30-100x more paths -> SNR scales ~sqrt(N))", flush=True)

# figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), facecolor="white")
lbl = np.zeros(S); lbl[brain] = 1; lbl[wm] = 2; lbl[skull] = 3
ax[0].imshow(lbl.T, origin="lower", cmap="viridis"); ax[0].plot(tix, tiy, "r.", ms=4)
ax[0].set_title("transcranial slice\n(water/skull/brain/WM + ring)"); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[1].hist(d0[mask]*100, bins=25, color="#c33", alpha=.8)
ax[1].axvline(5, color="k", ls="--"); ax[1].axvline(-5, color="k", ls="--", label="+/-5% noise")
ax[1].set_xlabel("per-path amplitude change from WM anisotropy (%)")
ax[1].set_title(f"[A] signal vs noise\nRMS {rms_sig*100:.1f}% (burial {burial:.0f}x)"); ax[1].legend(fontsize=8); ax[1].set_yticks([])
ax[2].scatter(a*100, b*100, s=18, c="#2a6fdb"); lim = max(np.abs(np.r_[a, b]))*100*1.1
ax[2].plot([-lim, lim], [lim, -lim], "k--", lw=1, alpha=.5)
ax[2].set_xlabel("signal, fibre phi=30 (%)"); ax[2].set_ylabel("signal, fibre phi=120 (%)")
ax[2].set_title(f"[B] orientation survival\ncorr {corr:+.2f}")
fig.suptitle("Does WM attenuation anisotropy survive the skull? (500 kHz full-wave transcranial)", y=1.02, fontsize=12)
fig.savefig(f"{OUT}/transcranial_anisotropy_survival.png", dpi=140, bbox_inches="tight", facecolor="white")
print(f"\nsaved {OUT}/transcranial_anisotropy_survival.png", flush=True)
import json; json.dump({"burial": burial, "rms_signal_pct": rms_sig*100, "orientation_corr": corr,
    "n_paths": n_paths, "matched_filter_snr": mf_snr, "noise_floor": NOISE,
    "freq_hz": FREQ, "skull_mm": 6, "ratio": 1.4, "alpha_wm": ALPHA_WM},
    open(f"{OUT}/transcranial_summary.json", "w"), indent=2)
