# GM/WM via velocity at imaging frequency: the test that was skipped

**Why this exists.** The project ruled out sound speed as a GM/WM discriminator on
the premise *"GM = WM = 1560 m/s — velocity FWI is structurally blind to it"*, and
pivoted to attenuation, then anisotropy. The top-down review (2026-07) found that
premise does not survive fact-checking:

1. **It is a property-table artifact.** `phantoms/properties.py` assigns GM and WM
   *both* 1560 m/s — faithful to the ITRUSST benchmark, but a benchmark
   convention, not physics. Real measurement: **~17 m/s (~1%)** separation
   (Mitcham et al., Med. Phys. 2025, DOI 10.1002/mp.18090 — ex-vivo human WM
   corpus callosum 1639 ± 3 vs GM cerebellum 1656 ± 6; tight error bars, a 3-5σ
   separation, *not* "at the noise floor" as older SD 10-14 m/s estimates implied).
2. **The real blocker was resolution, not contrast.** The whole 3D campaign ran at
   **40-160 kHz**, where λ/2 = 5-20 mm. The cortical ribbon is 2.5-4 mm — i.e.
   physically unresolvable at *any* contrast. GM/WM via velocity was ruled out at
   frequencies too low to see it.

So this runs the missing experiment: **550 kHz-class FWI (λ 3.1 mm, λ/2 ≈ 1.6 mm)
with the real measured 17 m/s contrast on a 3 mm cortical ribbon.**
`scripts/gmwm_velocity_resolution_test.py`.

## Setup

2-D axial slice, N=320, dx=0.5 mm (λ/dx = 6.2), ring of 24 sources / 96 receivers,
bands 250-400 then 400-550 kHz, 25 iters/band, from a **homogeneous 1560 m/s**
brain start (blind to GM/WM). Phantom: WM interior, 3 mm GM cortical ribbon, deep-
gray nuclei, CSF ventricles (a 51 m/s **positive control**). Metrics use eroded
masks and medians so the boundary Gibbs overshoot is not counted as contrast.

## Results

| | phase A (no skull) | phase B (transcranial, 6 mm skull) |
|---|---|---|
| true GM-WM contrast | 17.0 m/s | 17.0 m/s |
| **recovered contrast** | **16.8 m/s (99%)** | **2.9 m/s (17%)** |
| GM / WM medians | 1566.7 / 1549.9 | 1554.3 / 1551.4 |
| Cohen's d | **8.25** | 0.95 |
| **GM-vs-WM AUC** | **1.000** | **0.752** |
| ventricle control (51 m/s) | 74% | 74% |
| verdict | **RESOLVED** | **PARTIAL** |

## Findings

**1. The "velocity is blind to GM/WM" premise is falsified.** Without the skull,
FWI at imaging frequency recovers the real contrast at **99%, AUC 1.000** — GM and
WM medians land within 1.6 m/s of truth and the 3 mm ribbon is visible in the
radial profile. The blindness was never physical; it was a table convention
combined with running 3-12× too low in frequency.

**2. Transcranially it degrades to partial, not blind.** Through 6 mm of skull the
contrast compresses to 17% and AUC falls to 0.752. Real, informative, but not
clean separation.

**3. The skull is a small-contrast sensitivity floor, not a general failure.** The
decisive control: the **51 m/s ventricle recovers at 74% through the skull — the
same as the 75% without it.** So the skull does not break the inversion's ability
to see contrast; it specifically compresses *small* contrasts. GM/WM at 17 m/s
sits near that floor; a 51 m/s feature sails through.

## How this compares to the anisotropy road

The anisotropy programme was chosen as the GM/WM answer *because* velocity was
believed blind. Measured head-to-head, transcranially:

- **velocity**: AUC **0.752**, 17% contrast recovery — un-tuned first attempt.
- **anisotropy** (`transcranial_anisotropy_survival.py`): **0.9% per-path signal,
  below a 5% noise floor**, matched-filter SNR 1.6 over 72 paths.

Velocity is the **easier observable**, and it was skipped. Anisotropy remains
scientifically valuable (a genuinely novel j-Wave capability and an empty phantom
niche) but it is the *harder* road to GM/WM, and it was selected on a false premise.

## Honest limits of this test

2-D only; **skull given at truth** (perfect geometry/properties — real is worse and
this is exactly what the MOFI pose work exists to supply); uniform-bone skull, no
trilayer aberration; noiseless data; idealized concentric ribbon geometry; same
solver forward and inverse (the campaign's k-Wave cross-validation addresses this
elsewhere, not here). Phase B is therefore an **upper bound** on transcranial
velocity performance. The untuned settings (24 sources, 25 iters/band, no bound
tuning) mean it is also not an optimized one.

**Setup bug worth recording:** `run_fwi` builds its time axis from
`config.c_max`, *not* from the true medium. Generating observed data on a
different axis produced a ~2% time-base drift (>1 wave period over the record),
which swamped the 1% contrast entirely — flat loss, velocity runaway to the
bound, and a spurious "0% recovered / not resolved" verdict. The script now
builds both on `CFG_CMAX` and asserts a starting relative residual < 25% as a
guard. Any future FWI experiment here should keep that guard.

## Implication

The path to GM/WM runs through **frequency and coverage**, not through an exotic
observable. Next steps, in order: (i) 3D at 300-500 kHz (~22-170× the 48³ compute
— cluster-scale, not a physics barrier); (ii) push AUC up with more sources,
iterations, and a real array-design sweep; (iii) unknown-skull (MOFI pose) instead
of skull-at-truth; (iv) then decide whether anisotropy adds anything velocity
cannot already deliver.
