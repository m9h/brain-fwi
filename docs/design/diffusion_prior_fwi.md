# Diffusion-Prior-Guided FWI for Brain Reconstruction

**Status:** prototype built + characterised (2D); 3D is the next phase.
**Code:** `inference/score_unet.py`, `inference/diffusion.py` (Phase-3 VPSDE/DSM/DDIM/DPS),
`phantoms/birnbaum.py`, `scripts/modal_train_unet_score.py`, `examples/dps_fwi_demo.py`.

## Why

Transcranial brain FWI splits into two problems with very different difficulty:

1. **The skull** (high contrast, ~2800 vs 1500 m/s). The 192³ MIDA run (job 984) recovered
   **0% of the skull** — diagnosed (this investigation) as a **gradient-conditioning** failure,
   not cycle-skipping: without the pseudo-Hessian, near-transducer voxels dominate the gradient
   and the deep skull starves. **Fix: `FWIConfig(precondition=True)`** — 2D skull recovery 17%→41%,
   and 3D (GB10 job 1629) 0%→4.8%. (AWI, initially hypothesised, was a dead end — it underperformed
   plain L2; see the AWI PR.) The skull is otherwise handled as a prior via MOFI pose alignment
   (`inversion/mofi.py`).

2. **The brain interior** (low contrast: GM/WM ~60 m/s, lesions ~100–200 m/s). Physics-only FWI —
   even with skull-prior + preconditioning + TV — produces acquisition-footprint ring artifacts,
   not the brain. This is where a **learned anatomical prior** is needed (the current research
   frontier; cf. the Imperial "diffusion-guided FWI" direction). This doc covers that piece.

## Pipeline

```
anatomy data (Birnbaum stroke slices)                MOFI skull pose
        │ phantoms.birnbaum.build_slice_dataset            │
        ▼                                                   ▼
  UNetScore (inference.score_unet)                  skull prior (held fixed)
        │ train_score_matching  (Modal GPU, ~80 s)         │
        ▼                                                   ▼
  learned brain prior  s_phi(x,t) ────────► DPS-FWI: x ← x − lr·∇_data L + λ·s_phi(x_ROI, t)
                                            (preconditioned, masked to intracranial ROI)
```

- **Score net:** conv U-Net (`UNetScore`) over the field, reusing the Phase-3 machinery
  (`VPSDE`, `train_score_matching`, `ddim_sample`). A shallow CNN gave fragmented, globally
  incoherent samples; the multi-scale U-Net gives coherent brains. (`surrogate/uno.py` is the
  spectral-operator cousin; conv U-Net is the right diffusion arch at fixed resolution.)
- **Training venue:** Modal GPU is fine for NN training (U-Net: 73–88 s on A10G, loss→77).
  Modal is **not** viable for the j-Wave FWI itself (pathologically slow — 3 timeouts); FWI runs
  on GB10/Slurm or CPU. Clean split: **Modal trains the prior, GB10/CPU runs the physics.**
- **DPS integration:** per-iteration score nudge on the intracranial ROI (a RED/score-prior
  regulariser; proper Tweedie/DDIM measurement-guided posterior sampling is a TODO).

## Results (2D, held-out real slices)

Interior RMSE (m/s), water start, skull held as prior:

| run | interior RMSE | note |
|---|---|---|
| no prior | → 98–110 | acquisition-footprint artifacts (always) |
| water init | 65–78 | (reference) |
| shallow-CNN prior | → 69 | first to beat init (subj4) |
| **U-Net prior** | **→ 67** | best; beats init in the good case |

**The learned prior reliably REGULARISES** — without it FWI blows up to artifacts; with it the
recon is pulled back to near/below the init, U-Net best. **But clean low-contrast lesion recovery
was NOT achieved in 2D**, consistently, across all variants (synthetic & real priors; shallow &
U-Net; full-head & cerebrum-ROI; tiny & large lesion). Two intrinsic 2D blockers, neither a
prior-quality issue:

1. **Data under-constrains** the low-contrast interior at this frequency/aperture — the prior
   can't invent contrast the data doesn't see.
2. **2D entangles brain with face** — in an axial slice the brain connects to the anterior through
   the skull-base/orbit, so even a largest-connected-component ROI can't isolate the cerebrum, and
   the face/sinus region dominates the error.

## 3D (the fix) — RESULT

The **same** pipeline applied in 3D on the GB10 removes both 2D blockers and, for the first time
in this project, the diffusion prior **improves over the starting model** (in every 2D variant it
could only regularise back toward init). 3D works because the intracranial volume is a clean
connected component (`birnbaum.cerebrum_volume_crop` / `build_volume_dataset` — 3D largest-CC, no
face entanglement) and the helmet gives full angular coverage.

Pipeline (all GB10; Modal trains the NN fine but cannot run j-Wave FWI):
1. `python scripts/build_birnbaum_3d_dataset.py` → 120 cerebrum volumes at 48³ (holds out 4 subjects).
2. `modal run scripts/modal_train_unet3d.py` → trains `UNet3DScore` (`inference/score_unet3d.py`).
3. `python scripts/validate_3d_prior.py` (CPU) → Tweedie denoise of a held-out real cerebrum shrinks
   ROI-RMSE at every noise level (+12% @t=0.05 … +45% @t=0.5): the *local* score is informative even
   though small-N unconditional samples look fragmented — and the local score is what DPS uses.
4. `python examples/dps_fwi_3d_demo.py` (GB10) → 3D DPS-FWI on held-out subj2 (48³/3mm, skull given
   at truth MOFI-style, helmet 16–20 fixed shots, bands 20-45/40-80 kHz, t_end 100 µs).

Held-out subj2 cerebrum-ROI RMSE (init = water, 65.3):

| run | cerebrum RMSE | brain-only RMSE | lesion (true 1660) |
|-----|---------------|-----------------|--------------------|
| no prior (12 it) | 65.3 → 68.1 (−4%) | — | 1531 |
| **3D prior (12 it)** | **65.3 → 54.8 (+16%)** | — | 1536 |
| no prior (16 it) | 65.3 → 74.6 (−14%) | 69.9 | 1538 |
| **3D prior (16 it)** | **65.3 → 61.0 (+7%)** | **55.7** | 1537 |

The prior beats no-prior by ~18–20% relative in both runs (robust); the gain *over init* depends on
iteration count (an iteration sweet spot ~12), because the DPS here is a **fixed-t nudge**, not a
measurement-guided posterior — more FWI iters accumulate data-misfit artifact the simple nudge can't
fully balance. The clearest, most robust win is brain-tissue cleanup (brain-only RMSE 69.9 → 55.7).

**Remaining gap — the focal lesion** (~1537 of 1660, ≈23% contrast) is flat across iteration counts ⇒
**resolution-limited, not prior-limited**: a ~10 mm lesion at 48³/3 mm under a 40–80 kHz band (λ 18–37 mm)
is under-resolved. Next steps: finer grid (96³/1.5 mm, ~150 kHz; needs a 96³ prior + ~8× FWI cost),
**proper DPS** (measurement-guided/annealed-t, replacing the fixed-t nudge — should lift the over-init
gain at any resolution), a richer prior on **SHARM** (196 heads, not yet on disk), and MOFI for the
skull pose (here given at truth). Prior + preconditioning + MOFI all compose unchanged into the 3D loop.

## Scaling up: 96³, DPS, generalization, and credibility (2026-06)

The 48³ findings above were superseded by a 96³ campaign. Runners: `scripts/beam_dps_fwi_3d.py`
(MAP/annealed-t + modeling-error knobs + source co-inversion), `scripts/beam_dps_posterior_3d.py`
(reverse-diffusion DPS + uncertainty), `scripts/local_fwi_3d.py` (GB10 fallback when beam's serverless
pool is flaky — see `docs/dev/cloud-gpu-venues.md`).

**Annealed-t DPS + generalization.** Annealed-t (graduated-non-convexity t-schedule + lr-decay + prior
ramp) beats the fixed-t nudge decisively (48³ +31% vs +7%). At 96³ on held-out subj2 it gives cerebrum
RMSE **+38%** (brain-only 34.2). Across **all 4 held-out patients** (subj1–4, a separate cohort from the
GU/NC/NYU training subjects): annealed-t **+52/+38/+49/+46%** — the bulk reconstruction generalizes
robustly. Focal-lesion recovery scales monotonically with lesion size (334 vox → 0%, 9438 vox → 37%).

**Reverse-diffusion DPS** (`beam_dps_posterior_3d.py`): warm-start from the MAP, reverse VP-SDE with the
score augmented by the FWI data-likelihood gradient at the Tweedie estimate (Chung 2023). Best bulk in the
project — 96³ subj2 **+56%** single sample, **+60%** posterior mean (6 samples), brain-only RMSE 34→15.
But the lesion does **not** improve (24%), robust to ζ over a 30× sweep — a **prior** limitation (the
lesion-rare prior smooths focal outliers), not an inference one. Force-multiplier: the posterior **std**
flags the lesion region at **1.19×** the bulk uncertainty — the method says "unsure here" rather than
silently smoothing.

**Credibility — the inverse crime, and the source-co-inversion fix.** All the above generate the
"observed" data with the *same* solver used to invert (inverse crime). Breaking it (mismatched source
wavelet + measurement noise, `BFWI_SRC_MISMATCH`/`BFWI_NOISE_DB`):

| 96³ subj2 condition | no-prior | annealed-t cerebrum | brain-only | lesion |
|---|---|---|---|---|
| clean (inverse crime) | −1% | **+38%** | 34.2 | 27% |
| +20 dB noise only | −2% | **+38%** | 34.1 | 24% |
| +15% source mismatch only | −43% | +3% | 56 | −12% |
| mismatch + noise, **no** src-inv | −43% | +3% | 56 | −11% |
| mismatch + noise, **with src co-inversion** | +27%† | **+52%** | **23.5** | 34% |

Decomposition is clean: the method is **fully robust to 20 dB noise** (+38%, unchanged), and the entire
collapse is **source-signature mismatch** — the standard *fixable* error. Source co-inversion (Pratt
variable projection: a per-frequency filter φ = Σconj(P)·D / Σ|P|², shared across shots/receivers, applied
to the prediction with φ held constant) **rescues +3% → +52%** and even edges past the clean baseline
(the mismatched 92 kHz source carries more high-frequency information). **Conclusion: robust to realistic
data error — noise *and* unknown source — once you do standard source estimation.** (†The no-prior
"lesion 83%" under src-inv is artifact-dominated over-fitting, not a clean recovery; the trustworthy
lesion is annealed-t's 34%.)

**Honest remaining gaps:** (1) physics-model error is untested — the modeling-error study used the *same*
solver; the gold-standard is an **independent-solver swap** (k-Wave / Stride). (2) the skull is still
**given at truth** (MOFI / skull-FWI is the clinical step). (3) the focal lesion still needs a
**lesion-aware prior** (SHARM or synthetic-lesion augmentation).
