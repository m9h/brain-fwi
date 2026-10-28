# Validating WM attenuation anisotropy on real tissue

**Question (the #1 production-readiness gap):** the entire GM/WM-via-anisotropy
path assumes white-matter acoustic attenuation is anisotropic, by a large-enough
margin to detect. Is that real?

**Honest constraint:** a *direct* empirical validation — benchtop directional
ultrasound on excised WM — is impossible here. That measurement **does not exist
in the literature** (confirmed, mid-2026) and we have no wet-lab. So this is the
strongest *real-tissue-grounded* validation available, not an empirical acoustic
measurement. Three real evidence sources are chained.

## (1) Real ex-vivo human WM substrate — MEASURED here

DTI fit of the **STE00 ex-vivo human brain hemisphere** (Aganj/Martinos,
stimulated-echo dMRI, real post-mortem tissue; `scripts/ste00_dti_preprocess.py`):

| quantity | value |
|---|---|
| WM median FA (FA≥0.25 population) | **0.42** (p90 0.76) |
| GM median FA | **0.15** |
| WM fraction of brain voxels | 46% |
| WM global orientation coherence | 0.45 |

The FA distribution is bimodal: a strongly, coherently oriented WM population
distinct from near-isotropic GM. **The anisotropy substrate and the GM/WM
discriminator physically exist in real tissue.** Ex-vivo fixation *lowers* FA, so
in-vivo WM is if anything more anisotropic — this is a conservative floor.

## (2) Cross-modal anisotropy magnitude — MEASURED (adjacent tissue/modality)

WM compressional attenuation anisotropy is unmeasured, so it is bracketed from the
two nearest real measurements (`brain_fwi/tissue/wm_anisotropy.py`):

| source | modality/tissue | anisotropy ratio |
|---|---|---|
| Nassiri 1979 / Topp & O'Brien 2000 | skeletal **muscle**, longitudinal **attenuation** | ~2.6× |
| Anderson 2018 | **WM** shear-loss (**MRE**) | ~1.3× |

→ **WM compressional bracket [1.3×, 2.0×]**, central 1.6×; FA-scaled at the real
WM FA (0.42) gives **~1.4×**. Discounted below full muscle because WM axons are
finer, less densely packed, and more dispersed than muscle fibres.

**This is the load-bearing caveat:** the ratio is *borrowed* from muscle /
WM-shear-MRE, not measured in WM acoustically. It is a tunable, bracketed
parameter (1.0 isotropic control → ~2.0 muscle-like), never a claim.

## (3) Detectability through transmission tomography — SIMULATED

Is a *real-tissue-magnitude* ratio (~1.3–1.6, **not** the 4× toy cases) recoverable
above measurement noise? 24-element ring, straight-ray transmission, LSQ recovery
of the angular attenuation, Monte-Carlo over noise:

| case | true ratio | recovered [90% CI] | verdict |
|---|---|---|---|
| isotropic null | 1.00 | 1.01 [1.00, 1.03] | not resolved (correct) |
| MRE-analog | 1.30 | 1.30 [1.28, 1.32] | **detected** |
| FA-scaled real | 1.42 | 1.42 [1.39, 1.44] | **detected** |
| muscle-analog | 2.00 | 2.00 [1.96, 2.04] | **detected** |

Smallest detectable ratio ~1.04 (2% noise) to ~1.06 (20% noise). So at its real
plausible magnitude the anisotropy is **comfortably above the noise floor** in
idealized transmission.

**Detectability caveat (important):** this is a best case — **no skull**,
homogeneous WM patch, dense angular coverage (276 rays average the noise down).
The real limiter is **skull aberration + finite angular coverage**, not the
anisotropy magnitude. The magnitude clears the bar; the acquisition physics is the
open risk.

## Verdict

- **Substrate: confirmed real** — WM is strongly anisotropic and separable from GM
  in real ex-vivo tissue (FA 0.42 vs 0.15).
- **Magnitude: plausible and bracketed** — [1.3×, 2.0×] from real adjacent
  measurements; ~1.4× at the real WM FA. **Still unmeasured acoustically in WM —
  the one experiment that would close this is a benchtop directional-attenuation
  measurement on excised WM.**
- **Detectability: yes, if acquisition allows** — the real-magnitude effect is
  well above the noise floor in idealized transmission; skull aberration and
  angular coverage are the true gating risks, to be tested next on a skull-in-loop
  phantom.

**Bottom line:** the anisotropy premise survives real-tissue scrutiny — the
substrate is real, the magnitude is plausible and detectable — but the definitive
acoustic measurement remains the missing empirical link, and the skull is the next
gate. Artefacts: `scripts/ste00_dti_preprocess.py`,
`scripts/wm_anisotropy_validation.py`, `src/brain_fwi/tissue/wm_anisotropy.py`,
`tests/test_wm_anisotropy.py`, figure
`results/wm_anisotropy_validation/wm_anisotropy_validation.png`.

## (4) THROUGH THE SKULL — the only transcranial-condition test

The prior tiers were NOT at transcranial conditions (muscle 1-14 MHz no skull;
MRE ~100 Hz; the detectability test deleted the skull). This closes that gap:
a full-wave 2-D transcranial slice at **500 kHz** — water / 6 mm skull ring /
brain / anisotropic WM patch (ratio 1.4) — through the validated j-Wave
anisotropic absorber, transducer ring outside the skull. The skull is identical
in the isotropic- and anisotropic-WM runs, so differencing isolates the WM
contribution. Transmitted arrival time-gated (rejects the water wave creeping
around the ring). `scripts/transcranial_anisotropy_survival.py`.

| metric | result | reading |
|---|---|---|
| skull burial | **4.8×** amplitude loss | transmission genuinely goes through bone |
| [A] per-path signal | **0.9% RMS** (72 paths) | **below** a 5% single-shot noise floor |
| [B] orientation survival | corr(φ=30°, φ=120°) = **−1.00** | skull attenuates/delays but does **not scramble** the angular signature |
| [C] matched-filter SNR | **1.6** over 72 paths | marginal here; ~√N with a denser array |

**The transcranial finding (nuanced, and the real answer to "does it survive the
skull"):** the anisotropy signal is **not destroyed** by the skull — its angular
signature is perfectly coherent (−1.00: a 90° fibre rotation flips it exactly).
But per single path it is **small (~1%, sub-noise)**. Detection is therefore a
**coherent-integration problem**: matched-filtering the 72 through-head paths of
this sparse ring gives SNR ~1.6 (marginal); a clinical 256-element ring provides
~30-100× more through-paths, scaling SNR ~√N into a detectable regime.

**Honest limits of this test:** amplitude-only (full-waveform FWI carries more,
so 0.9% is a conservative floor); the −1.00 coherence is measured by *differencing
against a known isotropic reference* — a real BLIND inversion must jointly estimate
the unknown skull aberration, which is the true confound; 2-D, single WM patch,
uniform-bone skull (real trilayer skull aberrates worse); noise is an assumed 5%,
not simulated.

**Revised verdict:** the skull is a **SNR/coverage/aberration-correction problem,
not a signal-destruction problem** — the anisotropy information survives it
structurally. The gating risk moves from "does the effect exist through bone" (it
survives) to (i) the missing benchtop acoustic measurement of the WM anisotropy
magnitude *at sub-MHz*, and (ii) blind skull-aberration correction with enough
through-head angular coverage. Figure
`results/wm_anisotropy_validation/transcranial_anisotropy_survival.png`.
