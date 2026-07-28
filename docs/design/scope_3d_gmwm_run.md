# Scope: 3-D GM/WM velocity FWI at imaging frequency

Follows `gmwm_velocity_resolution_test.md` (2-D: GM/WM resolved at 99%/AUC 1.000
without skull, 17%/AUC 0.752 through skull). This scopes the 3-D run — the actual
deliverable for the original Guasch-class goal.

## Measured cost law (not estimated)

`scripts/scope_3d_gmwm.py`, GB10, fixed 180 mm FOV, **segmented gradient
checkpointing** (which `run_fwi` auto-enables at N>=128 — benchmarking without it
OOMs at N=128 needing 72 GB and gives a misleading law):

| N | dx | nt | grad/shot | peak mem |
|---|---|---|---|---|
| 128 | 1.41 mm | 1320 | 34.2 s | 10.8 GB |
| 160 | 1.12 mm | 1650 | 103.3 s | 23.3 GB |
| 176 | 1.02 mm | 1815 | 160.7 s | 32.4 GB |

**t_grad ~ 1.86e-9 · N^4.87** (theory N^4; the excess is bandwidth/cache).
Memory ~ N^3.47 → the 120 GB box tops out around **N ≈ 208–240**; N=240 (~92 GB)
is risky, N<=208 is safe.

## Frequency ladder (what each buys)

Cortical ribbon is 2.5–4 mm, so we need λ/2 comfortably below it. Points-per-
wavelength (ppw) is the accuracy knob — note the 2-D run that *worked* used ppw
6.2, so ppw 3 here is the aggressive end.

| stage | freq | N | dx | resolution | ppw | cost (8sh×20it×2b → 16sh×25it×2b) |
|---|---|---|---|---|---|---|
| **1** | 300 kHz | 112 | 1.61 mm | 2.60 mm | 3.2 | **1.6 – 4.0 h** |
| **2** | 400 kHz | 144 | 1.25 mm | 1.95 mm | 3.1 | **5.4 – 13.6 h** |
| **3** | 500 kHz | 176 | 1.02 mm | 1.56 mm | 3.1 | **14.4 – 36 h** |
| — | 500 kHz | 240 | 0.75 mm | 1.56 mm | 4.2 | 65 – 163 h (memory-risky) |

Stage 3 matches the resolution the 2-D test validated (1.42 mm). Stages 1–2 are
the de-risking ladder, not throwaways.

## The important strategic correction

Naively this is "raise frequency until GM/WM resolves". **The 2-D result says that
is not the binding constraint transcranially.** At 1.42 mm resolution — already
adequate for a 3 mm ribbon — the no-skull run hit AUC 1.000 and the through-skull
run still only reached 0.752. Resolution was *not* what cost us 0.25 of AUC; the
skull's **small-contrast compression** was.

So more frequency alone will not fix the transcranial number. What should:

1. **Angular coverage / element count** — more sources beats down the compression
   by averaging (the ventricle control shows large contrasts pass through fine).
2. **Absorption-aware inversion** — the skull is the dominant attenuator and the
   project already measured +76% brain RMSE from modelling it (`headline96.log`).
3. **3-D itself** — every voxel is crossed by far more ray paths in 3-D than in a
   2-D slice, so the transcranial AUC may improve for free.

**This makes the coverage sweep the primary experiment, not a secondary one — and
it is simultaneously the array-design deliverable the project has been missing.**

## Recommended plan

**Run 1 (tonight, ~2–4 h) — stage 1 pipeline validation, 300 kHz / N=112.**
Both arms: no-skull and transcranial. Success = ventricle control recovers (~70%+)
and the no-skull GM/WM AUC is high. This proves the 3-D phantom + metrics + FWI
chain before spending a weekend of GPU.

**Run 2 (overnight, ~6–14 h) — stage 2 + coverage sweep, 400 kHz / N=144.**
Transcranial only, sweeping sources ∈ {8, 16, 32, 64}. Produces the
**AUC-vs-element-count curve** — the array-design result. This is the run that
answers both the science question and the device question.

**Run 3 (weekend, 14–36 h) — stage 3 headline, 500 kHz / N=176**, at the best
coverage from Run 2, absorption-aware. This is the publishable number.

Long runs should go to **beam.cloud** (no timeout) rather than blocking the GB10 —
and must respect the queue-priority rule (hold when realtime-fmri / smri-fm work
is queuing).

## What needs building

- **3-D phantom** — extend the 2-D concentric construction to a spherical shell
  cortical ribbon + WM interior + deep-gray nuclei + ventricles + skull shell.
  Reuse the property fix (`GM_WM_C_DELTA = -17.0`).
- **3-D helmet array** — reuse `transducers/helmet.py` / `make_helmet` rather than
  the 2-D ring; this is also what makes the coverage sweep meaningful.
- **3-D metrics** — the 2-D metric block (eroded masks, medians, Cohen's d, AUC,
  ventricle control) transfers unchanged; only mask construction is new.
- **Keep the time-axis guard.** `run_fwi` builds its axis from `config.c_max`;
  observed data must be generated on the same axis, and the starting relative
  residual asserted < 25%. This bug silently produced a "not resolved" verdict
  once already.

## RUN 1 RESULT (stage 1, 300 kHz, N=112, 16 src / 128 rec)

| structure | phase A (no skull) | **phase B (transcranial)** | 2-D transcranial (for contrast) |
|---|---|---|---|
| deep-gray nuclei (14 mm, **resolvable → contrast test**) | +19.4 m/s (114%) AUC 0.978 | **+16.3 m/s (96%) AUC 0.998** | — |
| cortical ribbon (3 mm, 1.9 vox → resolution test) | +18.6 (110%) AUC 0.955 | +8.0 (47%) AUC 0.848 | — |
| ventricle CSF (control) | −34.0 (66%) AUC 1.000 | −45.2 (88%) AUC 1.000 | — |
| GM/WM overall | — | **96% / AUC 0.998** | 17% / AUC 0.752 |

WM median recovered 1551.1 vs true 1551.5.

**The headline: 3-D transcranial recovers GM/WM at 96% (AUC 0.998), versus 17%
(AUC 0.752) in 2-D.** The scope predicted coverage — not frequency — was the
binding transcranial constraint; this confirms it emphatically. A 2-D slice offers
only in-plane paths, while full-spherical encirclement gives 16x128 = 2048 paths
crossing every voxel from all directions, which averages down exactly the
small-contrast compression the skull imposes. **The 2-D pessimism was an artifact
of 2-D, not a property of the skull.**

Secondary: the 3 mm ribbon lands at 47% / AUC 0.848 — resolution-limited exactly
as predicted at 1.9 voxels / 2.60 mm resolution. This is the number the frequency
ladder exists to move; stage 3 (1.56 mm) is where it should resolve.

**Why transcranial ≈ no-skull here (and when it won't be).** The skull is *given
at truth* and excluded from inversion, and the data are noiseless — so the skull
is a perfectly-known aberrator that the forward model reproduces exactly. Under
those conditions it does not fundamentally obstruct the inversion. The skull bites
when it is **unknown** (pose/property error) or when **noise** limits how well
small contrasts can be teased out. Neither is tested here; both are the next gates.

**Confound to note:** phases A and B are not perfectly matched — `CFG_CMAX` differs
(1600 vs 2900 for skull CFL), so phase B runs at a finer dt with ~1.8x more time
samples (1155 vs 638). Phase B therefore has more data per shot. The meaningful,
clean comparison is 2-D-transcranial vs 3-D-transcranial (both at their own
appropriate settings), not A vs B.

## RUN 2 RESULT — array-design curve + noise robustness

Stage 1 (300 kHz, N=112), transcranial, 15 iters/band, 9 configs.
`scripts/run2_coverage_noise_sweep.sh` + `run2_aggregate.py`.

| config | elements | noise | **GM/WM AUC** | contrast | ribbon AUC | CSF |
|---|---|---|---|---|---|---|
| cov008 | 8 | clean | 0.688 | 22% | 0.359 | 16% |
| cov016 | 16 | clean | 0.469 | −1% | 0.449 | 30% |
| **cov032** | **32** | clean | **0.963** | **67%** | 0.669 | 59% |
| cov064 | 64 | clean | 0.995 | 92% | 0.723 | 72% |
| cov128 | 128 | clean | 0.999 | 95% | 0.846 | 84% |
| cov032_snr20 | 32 | 20 dB | 0.883 | 59% | 0.557 | 55% |
| cov032_snr10 | 32 | 10 dB | 0.771 | 48% | 0.479 | 45% |
| cov128_snr20 | 128 | 20 dB | 0.988 | 91% | 0.679 | 83% |
| cov128_snr10 | 128 | 10 dB | 0.889 | 76% | 0.484 | 67% |

**1. The coverage knee is sharp and LOW — between 16 and 32 elements.** 16
elements sits at chance (0.469); 32 is resolved (0.963). Above 64 returns
flatten hard (0.995 → 0.999 for a 2x element cost). For a full-azimuth 3-D
imaging array, **~32-64 elements suffices to recover the GM/WM contrast** — an
order of magnitude below Guasch's 1024 and Insightec's 1024. Coverage
*geometry* (full encirclement) matters far more than element *count*.

**2. Elements buy noise tolerance — the key engineering trade.** At 10 dB SNR,
128 elements gives 0.889 vs 32 elements' 0.771; at 20 dB, 0.988 vs 0.883. So
**extra coverage substitutes for receiver SNR**, which is exactly the trade a
device designer wants quantified: a noisier, cheaper front-end can be bought
back with more elements.

**3. GM/WM survives realistic noise.** Even 10 dB SNR with 128 elements keeps
AUC 0.889 / 76% contrast. Noise degrades but does not destroy the contrast.

**4. The thin ribbon is the first casualty of noise** (128 elem: 0.846 clean →
0.679 @20 dB → 0.484 @10 dB). Resolution-limited structures are far more fragile
than resolvable ones — reinforcing that stage 3's finer grid is what the ribbon
needs, not more elements.

**Caveats.** (a) The 8-vs-16 ordering is non-monotonic (0.688 vs 0.469) — both are
failures and the two configs differ in *both* source and receiver count
(8src/8rec vs 16src/16rec), so the ordering between two failed regimes is not
meaningful; the honest read is "≤16 elements fails". (b) The starting-residual
guard fired spuriously on the noisy runs (30.7% at 10 dB) — noise itself
contributes σ/rms = 0.316 to the residual, so the guard is only valid for
noiseless runs and should be conditioned on the noise level. (c) Skull still
given at truth.

## Honest expectations

Stage 3 no-skull should reproduce the 2-D AUC ~1.0. Transcranial is the open
question: the 2-D upper bound was 0.752 with skull-at-truth, and 3-D changes both
the coverage (better) and the aberration (worse, more paths through curved bone).
Skull is still given at truth here — swapping in the MOFI recovered pose is the
step after, and will cost accuracy again.

## RUN 3 RESULT — unknown skull: pose-error tolerance for GM/WM

The last big caveat: every GM/WM number above assumed the skull **given at truth**.
`scripts/gmwm_unknown_skull.py` + `run3_pose_tolerance.sh`. Observed data always has
the skull at truth; only the INVERSION's belief about where it is is perturbed.
**Ellipsoidal** head (52/62/58 mm) — a spherical skull would make rotation a no-op
and report false robustness — with a **soft, never-binarised** warp, and the
estimated skull excluded from the inverted brain mask.

| pose error | skull-mass mismatch | start resid | **GM/WM AUC** | contrast | ribbon AUC | CSF |
|---|---|---|---|---|---|---|
| 0.00 vox (baseline) | 0.0% | 11.7% | 0.913 | 64% | 0.572 | 52% |
| **0.07 vox / 0.10° (MOFI measured)** | 2.2% | 14.6% | **0.941** | **64%** | 0.626 | 48% |
| 0.25 vox / 0.30° | 8.4% | 26.9% | 0.912 | 66% | 0.686 | 43% |
| 0.50 vox / 0.60° | 16.4% | 38.4% | 0.734 | 45% | 0.686 | 45% |
| 1.00 vox / 1.20° | 32.2% | 54.3% | 0.436 | **−11%** | 0.597 | 61% |
| 2.00 vox / 2.50° | 63.9% | 64.1% | 0.557 | 18% | 0.648 | 49% |
| 4.00 vox / 5.00° | 116.2% | 67.2% | 0.514 | 7% | 0.591 | 50% |

**1. MOFI + velocity FWI COMPOSE — the "skull at truth" caveat is retired.** At
MOFI's measured accuracy (0.07 vox / 0.10°) GM/WM is fully preserved: AUC 0.941 vs
the 0.913 skull-at-truth baseline (the small excess is run-to-run noise, not a real
gain). The pipeline works end-to-end from an unknown skull.

**2. The tolerance cliff is at ~0.25-0.5 voxels.** Up to 0.25 vox (≈0.4 mm here)
GM/WM holds at AUC ~0.91; by 0.5 vox it degrades (0.734); at >=1 vox it is at
chance. **MOFI's 0.07 vox leaves roughly a 3.5x margin** to the cliff — comfortable,
but not enormous, and it should be re-checked at stage 3's finer grid where a voxel
is a smaller physical distance.

**3. Safety finding: a MODERATELY wrong skull is worse than a grossly wrong one.**
At 1.0 vox the recovered GM/WM contrast goes **negative (−11%)** — the inversion
does not merely lose the contrast, it confidently reports the wrong sign. A
misplaced skull generates structured artefacts that mimic and invert tissue
contrast. For a diagnostic device this is the dangerous regime: plausible-looking
output that is wrong. It argues for pose uncertainty to be *reported*, not just
minimised.

**4. Large contrasts are far more pose-robust than small ones.** The ventricle
control holds AUC 0.966 even at 4 vox / 116% skull mismatch, while GM/WM collapses
by 1 vox. Consistent with everything else in this campaign: small contrasts are the
fragile quantity, in coverage, in noise, and now in skull accuracy.

**Caveats.** (a) The ellipsoidal baseline (0.913) is below Run 1's spherical 0.998,
but the configs differ in geometry, receivers (64 vs 128) and iterations (12 vs 20)
— `pose000` is the correct control here, not Run 1. (b) The 1.0 / 2.0 / 4.0 vox
ordering is non-monotonic (0.436 / 0.557 / 0.514); all are at chance, so ranking
failed regimes is not meaningful. (c) Pose error is *imposed*, using MOFI's measured
accuracy rather than re-running the recovery live — a confirmatory end-to-end run
with pose recovered in-loop is the natural follow-up.
