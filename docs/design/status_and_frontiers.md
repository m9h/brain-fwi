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
2. **Skull pose from data — SOLVED (2026-06).** A 4 vox/5° misalignment is catastrophic
   (−400%): transcranial FWI needs near-sub-voxel skull accuracy. Naive gradient descent,
   coarse grid, AWI, AND derivative-free CMA-ES all fail — the *reflection* misfit is a
   plateau with a sub-voxel needle at truth. The fix was the **observable, not the
   optimizer**: a **transmission-traveltime misfit** (envelope cross-correlation soft-lag,
   full waveform) is a smooth bowl, and **grid-seed-then-polish** (2D tx,ty grid →
   translation refine → 1D rz grid → joint refine) lands the pose at **98% overlap,
   |dt|=0.07 vox, |drz|=0.10°** from a cold start, for both known and unknown brain.
   `scripts/run_mofi3d_transmission_staged.py`.

## The complete unknown-skull → brain pipeline (2026-06, validated)

From a single measurement with an *unknown* skull pose AND unknown brain, one end-to-end run
(`scripts/run_mofi3d_pipeline.py`, GB10 job 1714): recover the pose live (transmission TT → 98%),
then reconstruct the brain (soft-skull annealed-t DPS-FWI). At 96³: **recovered-pose +52% ≈
truth-pose +53%** cerebrum RMSE — the recovered pose reconstructs **as well as the exact skull** —
vs −400% if the misalignment is ignored.

**Key modeling insight — never binarize the warped skull.** Binarizing a sub-voxel-accurate pose
flips ~2% of skull voxels and that 2% destroys the recon (+18%); a SOFT/anti-aliased warp used
consistently in the forward AND inversion (and excluded from the brain imask — the warped skull
intrudes into roi_mask) keeps it a tiny soft difference (+52%, beating even binary-at-truth +44%).
This is strictly more than MOFI: alignment **and** brain reconstruction at exact-skull quality.

## The Imperial comparison (MOFI, Bates et al. 2026, arXiv:2601.14533)

MOFI = **skull-template alignment via FWI** — recover the SE(3) skull pose by
minimizing the predicted-vs-observed RF acoustic misfit, *without* an MRI guidance
image. Validated in silico + in vitro; it "accurately recovers the position of skull
templates." It is **narrowly scoped to geometric alignment** and explicitly does
*not* reconstruct the brain interior, handle lesions, quantify uncertainty, or use
learned priors.

**Where they were ahead — now matched and exceeded:** robust skull pose recovery, which
our naive gradient approach failed at, is now **solved** via the transmission-traveltime
misfit (98%, sub-voxel) — and coupled with brain reconstruction in one end-to-end pipeline
(+52% ≈ exact-skull). We do what MOFI does *plus* the downstream imaging, at exact-skull quality.

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
The unknown-skull → brain pipeline is **done** (+52% ≈ exact-skull, end-to-end). Subsequent bets:
(1) ✅ **Treeby–Cox time-domain absorption** (Phase 5) — **done + k-Wave-validated (2026-06-27).**
The forward AND the FWI checkpointed/adjoint path now apply the canonical Treeby-Cox absorbing
equation of state (loss folded into the constitutive p(ρ,u), not a non-accumulating per-step decay
of the diagnostic pressure — the earlier bug). j-Wave matches k-Wave **bit-for-bit** on homogeneous
power-law decay across absorption strengths (absorption-only and with-dispersion), with `jax.grad`
flowing through — the differentiable-FUS edge over k-Wave/Stride/BabelBrain. See
`docs/design/phase5_treeby_cox_absorption.md` + `scripts/kwave_absorption_xcheck.py`. Next here:
an absorption-aware FWI demo (invert a known-α skull). (2) the lesion **conditional/data-aware
prior** (the remaining recon-quality frontier); (3) a fresh **192³ high-res** MIDA reconstruction
with the soft-skull recipe.

Parked with clear findings: **FNO surrogate** (forward generalizes 0.989, but adjoint/gradient fails
for FWI — needs gradient-aware training at scale, not a swap).
