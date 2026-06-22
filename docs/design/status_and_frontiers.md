# 3D brain-FWI: status, frontiers, and the Imperial comparison (2026-06)

Executive synthesis of the diffusion-prior-guided 3D FWI campaign. Technical
detail in `diffusion_prior_fwi.md`; this is the high-level state + how it relates
to the Imperial MOFI line the project follows.

## What the pipeline does now (validated)

On real held-out Birnbaum stroke-patient cerebra at 96³ (skull given at truth,
helmet array, j-Wave forward + JAX autodiff):

- **Generalizes across patients** — annealed-t DPS recovers the brain at +38…+52%
  cerebrum-ROI RMSE on all 4 held-out subjects (a separate cohort from training).
- **Best reconstruction via reverse-diffusion DPS** — posterior mean +60%,
  brain-tissue RMSE 34→15.
- **Uncertainty quantification** — DPS posterior std flags the lesion region at
  1.19× the bulk uncertainty (it says "unsure here" rather than silently smoothing).
- **Credibility-tested** (the inverse-crime study):
  - robust to 20 dB measurement noise (+38%, unchanged);
  - robust to unknown source signature via **source co-inversion** (Pratt variable
    projection): +3% (collapsed) → +52% (rescued);
  - robust to an **independent forward solver** — j-Wave vs k-Wave (field-standard)
    agree to 0.97 forward correlation, and inverting k-Wave data with j-Wave gives
    +55% (≈ the j-Wave-self result). The +52–55% is not a self-consistency artifact.
- **Forward surrogate (PoC)** — `CToTraceFNO3D` trained on 224 samples reaches 0.99
  forward correlation on dominant arrivals; the cross-solver harness validates it
  like any other solver (the bridge to a fast DPS inner loop).

## Two characterized frontiers (open, but understood)

1. **Focal lesion recovery.** No-prior FWI sees the lesion (~36%) but is noisy; every
   prior-regularized scheme smooths it to 24–28%. Five approaches converge on this
   (fixed-t, annealed-t, data-gate, reverse-diffusion DPS, lesion-aware-augmented
   prior) — it is a **prior-vs-data tension**, not a tuning failure. The prior that
   cleans the bulk inherently smooths the statistical outlier. Real next bet: a
   **conditional/data-aware prior** (sharpen where the data indicates, not regress
   to the mean), or strong measurement-guided guidance.
2. **Skull pose from data.** A 4 vox/5° skull misalignment is **catastrophic**
   (−400%): transcranial FWI needs near-sub-voxel skull accuracy. The SE(3) warp is
   built and validated (exact 6-DOF recovery on an asymmetric object), but
   **gradient descent on the j-Wave data misfit does not converge** for a 4-vox
   misalignment — the pose-misfit landscape is non-convex and the thin skull is
   low-frequency-invisible. Real next bet: a **global/derivative-free pose search**
   (grid or CMA-ES, forward-only) and/or fine-only refinement atop a coarse register.

## The Imperial comparison (MOFI, Bates et al. 2026, arXiv:2601.14533)

MOFI = **skull-template alignment via FWI** — recover the SE(3) skull pose by
minimizing the predicted-vs-observed RF acoustic misfit, *without* an MRI guidance
image. Validated in silico + in vitro; it "accurately recovers the position of skull
templates." It is **narrowly scoped to geometric alignment** and explicitly does
*not* reconstruct the brain interior, handle lesions, quantify uncertainty, or use
learned priors.

**Where they are ahead:** the one piece MOFI nails — robust skull pose recovery —
is exactly the piece our naive gradient approach *failed* at. They evidently use an
optimization strategy that escapes the non-convex pose landscape (multiscale /
global / a pose-robust misfit); we've characterized why the naive version fails but
haven't yet matched their alignment.

**Where we do more / what they don't do (yet):**
- **Brain-interior reconstruction** — MOFI stops at alignment; we reconstruct the
  brain (+38–60%), which is the actual imaging goal MOFI is an upstream step *for*.
- **Learned diffusion priors + DPS** — the generative-prior-guided FWI direction the
  Imperial Scalable-SciML lab is *advertising* as future PhD work; we've built and
  characterized it end-to-end.
- **Uncertainty quantification** — per-voxel posterior std (flagging the lesion).
- **Credibility methodology** — inverse-crime breakdown, source co-inversion, and an
  independent-solver (k-Wave) cross-validation harness that *also* validates learned
  surrogates.
- **Stroke-lesion focus** — the clinical target; we've precisely characterized why
  it's hard.

**Net:** MOFI and this work are **complementary, not competing** — their skull
alignment + our diffusion-prior brain reconstruction would form a pipeline that does
what MOFI does *plus* the downstream brain imaging, with uncertainty and learned
priors. To "improve on / exceed" them concretely, the move is to **match their skull
alignment (global-search MOFI) and combine it with our brain-recon + DPS + UQ** — a
strictly larger capability than either alone.

## Next
Push forward on **global-search MOFI** (close the one gap with Imperial), which —
combined with the credibility-tested, generalizing diffusion-prior brain recon —
yields the complete unknown-skull → brain-image pipeline. The lesion's conditional-
prior fix and the surrogate's Phase-0 scale-up are the subsequent research bets.
