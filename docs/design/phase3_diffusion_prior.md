# Phase 3: Score-Based / Diffusion Prior over Anatomy

Status: **design draft — v1**, 2026-04-24
Owner: Morgan Hough
Depends on: Phase 0 dataset (✅ scaffolded), Phase 1 SIREN parameterisation (✅ on main), Phase 2 NPE baseline (🟡 in-flight)

This document specifies the score-based prior trained on Phase-0
`θ`-samples and the two ways we plan to consume it: as a regulariser
inside the existing MAP FWI loop, and as a sampling engine for
posterior inference via **Diffusion Posterior Sampling** (Chung et al.
2023). It sits alongside Phase 2 (amortised NPE) rather than
replacing it — the two have complementary strengths and the long-run
plan is to compose them.

---

## 1. Motivation

MAP FWI on main uses hand-tuned regularisation (Gaussian gradient
smoothing) that is agnostic to anatomy. It reduces high-frequency
artefacts but cannot express any of the structural priors that a
human radiologist uses when reading brain images: skull topology,
tissue-interface smoothness, hemispheric symmetry, realistic
sound-speed contrast ratios.

Phase 2 NPE addresses this at the *inference* level — it learns
`q(θ | d)` directly, amortised across future `d`. But:

- NPE needs retraining whenever transducer geometry changes, since
  `d`-dimension and its distribution both shift. A clinical helmet
  variant is effectively a new dataset.
- NPE's output is a single distribution, not a manipulable artefact.
  We cannot combine it with other priors, other likelihoods, or
  hand-designed constraints without retraining.

A score-based prior over `θ` alone is *geometry-agnostic*: it models
anatomy, not acquisition. The same trained prior can then drive MAP
regularisation, posterior sampling under any helmet configuration, or
compose with future domain priors. This makes Phase 3 an asset, not
just a model.

---

## 2. What we learn

A **score network** `s_φ(θ, t) : ℝ^d × ℝ → ℝ^d` approximating the
score (gradient of log-density) of the Phase-0 `θ`-distribution at a
continuum of noise levels `t ∈ [0, 1]`:

```
s_φ(θ_t, t) ≈ ∇_{θ_t} log p_t(θ_t)
```

where `θ_t` is `θ` perturbed by Gaussian noise with a `t`-dependent
variance schedule. At `t = 0` this is the score of the true prior
over SIREN weights; at `t = 1` it is (by construction) the score of
pure noise. The intermediate noise levels are what make the prior
controllable at inference time.

We train on `θ` alone (discarding `d` from each Phase-0 sample). The
prior therefore depends only on the Phase-0 anatomy distribution, not
on any acquisition specifics.

### Parameterisation

Target θ is the flat SIREN-weight vector produced by
`brain_fwi.inference.dataprep.theta_from_sample` (~5×10⁴ dims for the
default architecture). No further processing. If dimensionality
becomes a training bottleneck, we consider:

- PCA / SVD reduction on training θ to a rank-`r` subspace with
  ≥99% variance preserved. Likely r ≈ 10³–10⁴.
- Group-wise score networks (one per SIREN layer), inspired by
  parameter-generating hypernetwork priors.

Neither is MVP; baseline is score over the raw flat vector.

### Network

Four plausible architectures in order of increasing ambition:

1. **MLP with sinusoidal time embedding.** Simplest, matches the
   structure of θ (not spatial). Starting point. ~10 MLP blocks of
   width 512.
2. **U-Net over reshaped θ** (e.g. SIREN layer matrices as 2D
   tensors). Adds locality but assumes layer weights have spatial
   structure — they don't inherently.
3. **Transformer with per-weight tokens.** Each θ element is a token
   with a positional embedding indicating layer + position. Expensive
   but flexible.
4. **Conditioning on the voxel-grid reconstruction from θ.** The
   network sees `SIRENField(θ).to_velocity(...)` as an auxiliary input
   — gives anatomical context the weight vector alone lacks.

MVP: (1). Keep (4) on the shortlist as an easy post-baseline
improvement that reuses our existing SIREN decoder.

---

## 3. Training objective

Denoising score matching (Vincent 2011, Song & Ermon 2019):

```
L(φ) = E_{θ ~ p_data, t ~ U(0,1), ε ~ N(0, I)} [
    λ(t) · ‖ s_φ(θ_t, t) + ε / σ(t) ‖²
]
```

where `θ_t = α(t) · θ + σ(t) · ε` under a chosen variance-preserving
or variance-exploding schedule. We adopt the **variance-preserving
(VP) SDE** schedule from Song et al. 2021 for numerical stability at
high noise levels:

```
α(t) = exp(-½ · ∫₀ᵗ β(s) ds)
σ²(t) = 1 - α²(t)
β(t) = β_min + t · (β_max - β_min)   (linear)
```

`β_min = 0.1`, `β_max = 20.0` as a starting pair; both are tunable.

### Minimum dataset size

Literature intuition for images: `10³–10⁴` samples per 10⁴ θ-dim to
avoid collapse. Our `θ` dim is ~5×10⁴, so we want at least **10⁴ θ
samples** before training a serious prior. This matches the Phase-0
first-pass target in `data_pipeline.md`.

### Compute

MLP score net at width 512, 10 blocks → ~5M params. At batch size
256 with 10⁴ samples, one epoch is ~40 gradient steps; 200 epochs
(≈8000 steps) fits comfortably inside an A100-hour. Budget: **one
A100-hour per training run**, expect to iterate architectures 5–10×
before publication.

---

## 4. Consumption mode A — regulariser in MAP FWI

Drop-in replacement for `gradient_smooth_sigma` in the voxel path
and for SIREN's architectural-smoothness reliance. During each FWI
iteration, the effective gradient becomes:

```
∇_θ L_{data}(θ | d) − λ · s_φ(θ, ε)
```

where `ε` is a small fixed noise level (e.g. 0.01) that keeps the
prior evaluation in a well-trained regime and `λ` balances fidelity
vs. prior. The minus sign ascends toward higher-density θ regions.

Implementation:

- New `FWIConfig.score_prior_path: Optional[str]` and
  `score_prior_weight: float = 0.0`.
- Load the prior checkpoint once outside the FWI loop; `eqx.Module`
  serialisation via `eqx.tree_serialise_leaves` (same pattern as
  Phase-0 SIREN weights).
- Compose `∇_θ L` with `λ · s_φ(θ, ε)` after the existing grad
  max-norm normalisation but before clip.
- Voxel path: the prior is over SIREN-weight θ, so this mode only
  applies when `parameterization="siren"`. Good — it forces a clean
  compositional story.

### Why this works

SIREN pretrain already projects the initial velocity into a valid θ
region. The score prior stops FWI from leaving that region as it
optimises. The trade-off is bias toward the training distribution —
if the test subject is anatomically atypical, the prior will fight
the data. We manage this with `λ`.

### Validation

Regression against the current main's FWI on the coupled synthetic
head: score-prior-enabled FWI should give lower regional RMSE in
skull while preserving brain RMSE. Compare at 96³ and 192³.

---

## 5. Consumption mode B — posterior sampling via DPS

Chung et al. 2023, "Diffusion Posterior Sampling for General Noisy
Inverse Problems": during the reverse-diffusion trajectory from `t=1`
to `t=0`, inject a likelihood-gradient correction computed by
calling the forward model (j-Wave) on the current denoised θ.

Sketch of one reverse step:

```python
# prior term (learned)
score_prior = score_net(theta_t, t)

# likelihood term (physics)
theta_hat_0 = denoise(theta_t, t, score_prior)      # Tweedie's estimate
velocity    = SIRENField(siren_from_theta(theta_hat_0)).to_velocity()
pred        = simulate_shot_sensors(medium_from(velocity), ...)
log_lik     = -0.5 * ‖pred - d_observed‖²
score_lik   = jax.grad(lambda t: log_lik(t))(theta_t)

# combined reverse step (Euler-Maruyama)
drift = -½ β(t) (theta_t + score_prior + ζ · score_lik)
theta_next = theta_t + drift · dt + √(β(t) dt) · noise
```

`ζ` is the guidance strength — balances prior vs likelihood. `T`
reverse steps per sample, each calling j-Wave once. T ≈ 1000 for
Song et al.-style schedules; we'll likely cut to 100–200 with
fewer-step solvers (Karras 2022 or DDIM).

### Cost

Each sample from `p(θ | d)` costs `T` j-Wave calls. At 1s/call and
T=200, one posterior sample ≈ 3.3 min. For UQ we want 100 samples
per subject → ~5 GPU-hours. Expensive but feasible per subject; not
feasible at population scale without a neural-operator surrogate
(Phase 4).

### Advantage over NPE

- No retraining when helmet geometry changes — `score_lik` is the
  only helmet-aware term and it's a forward-model evaluation, not a
  trained network.
- The prior is reusable across entirely different modalities
  (MRI, X-ray FWI) — anything that admits a differentiable forward
  model.
- Gives samples, not a density. Trivially combinable with
  hard constraints via rejection.

### Disadvantage

- Slow. Every sample is a full reverse-diffusion run with physics
  evaluations.
- Guidance-strength tuning is unstable at high noise levels; literature
  has many patches (Chung et al. 2024 "Decomposed DPS", MCG, etc.).
  Expect empirical iteration.

---

## 6. Relationship to NPE (Phase 2)

Not competitive. Three compositions we plan to evaluate:

1. **NPE-only**: fast inference, retrains per helmet geometry.
2. **Diffusion-only**: geometry-agnostic, expensive inference.
3. **NPE proposes → diffusion refines**: NPE gives a fast initial
   posterior sample, diffusion takes 10–50 refinement steps to
   polish (Graikos et al. 2022, "Diffusion models as plug-and-play
   priors"). Cheap per-sample and geometry-aware only at inference.

The design-doc commitment is that Phase 2 and Phase 3 share the θ
representation (flat SIREN weights via `theta_from_sample`), the
dataset (`Phase-0_v1`), and the validation harness (regional RMSE,
simulation-based calibration). Composition (3) becomes a concrete
follow-up PR once both baselines are stable.

---

## 7. Validation plan

Three gates, in order:

1. **Toy prior.** Train score net on a 2D Gaussian mixture θ, recover
   sampling and score estimates to within known-analytic error.
   Filtering unit test, catches silent implementation bugs.
2. **Unconditional samples.** Trained on Phase-0 θ, samples from the
   prior should produce valid SIREN weights whose decoded
   velocity fields pass the "looks like a brain" sanity check: head
   outline, skull ring, interior tissue contrast roughly matching
   BrainWeb/MIDA distribution. Statistical check via Fréchet-style
   distance on a held-out subset of Phase-0 θ.
3. **Posterior coverage (SBC).** Once DPS is wired up, run
   simulation-based calibration (Talts 2018) on ≥1000 synthetic
   (θ*, d*) test pairs. Credible-interval coverage must be
   well-calibrated (rank histogram uniform). Blocks any downstream
   clinical claim.

---

## 8. Open decisions

- **Score network: MLP first, or skip straight to conditional U-Net
  over voxel-decoded θ.** The latter aligns with image-diffusion
  infrastructure we may want to reuse (HuggingFace diffusers,
  flow-matching libraries). Trade-off: MLP is simpler to debug; U-Net
  is closer to the eventual production architecture.
- **Noise schedule.** VP SDE is the stable default. VE SDE (Song
  2019) has been reported to give sharper details on image tasks but
  needs aggressive gradient clipping. Keep VP as MVP; revisit if
  prior samples are too smooth.
- **Guidance mechanism.** DPS vs classifier-free guidance conditioned
  on `d`-summary statistics. DPS is simpler and doesn't require
  retraining if `d` space changes; classifier-free would be an NPE-
  adjacent trained approach. Defer until we have a prior and a
  working likelihood gradient.
- **When to invest in the neural-operator surrogate (Phase 4).**
  Diffusion consumption mode B costs `T × one_jwave_call` per
  sample. If that's prohibitive at population scale, Phase 4's FNO
  surrogate at 100×–1000× speedup changes DPS from "per-subject
  feasible" to "routine." Decide after the first real DPS runs —
  don't pre-optimise.

---

## 9. Workstream kickoff

Tracked implementation steps once this design is accepted:

1. **Data loader over Phase-0 θ** — `ShardedReader` + `theta_from_sample` (both on main) + batching/shuffling helper. Tiny PR.
2. **Score-matching training loop** — `src/brain_fwi/inference/diffusion.py` with VP SDE schedule, MLP score net, denoising-score-matching loss. Reuses the Adam + `eqx.filter_value_and_grad` pattern from `train_npe`.
3. **Toy-prior gate** — 2D Gaussian mixture. Fast unit test, blocks landing until score matches analytic within 5%.
4. **Real Phase-0 training run** — A100 on Modal, one hour, save checkpoint.
5. **Reverse sampler** — DDIM-style deterministic solver first (cheap), then EM stochastic for DPS. Validate unconditional samples against §7 gate 2.
6. **MAP-mode regulariser integration** — `FWIConfig.score_prior_path` + composite gradient. Regression test against main's FWI.
7. **DPS sampler** — likelihood gradient via `jax.grad` through j-Wave (already differentiable). SBC gate (§7 gate 3).
8. **Compose with NPE** (§6 option 3) — "NPE proposes, diffusion refines". Final phase-3 PR.

Each step is its own PR, each landed behind its own test. Gate 3
blocks clinical use.

---

## 10. What this does NOT do

- Does not model measurement noise. The forward likelihood is
  Gaussian-L2 for now; richer noise models land in a future Phase
  alongside hardware impulse-response calibration.
- Does not model attenuation/density inversion. Phase-0 holds
  density fixed; inverting density is a Phase 5+ concern.
- Does not do joint inference across subjects. One subject, one
  posterior — population-scale priors are a follow-up.

This keeps Phase 3 scoped: a learned anatomy prior over SIREN
weights, consumed two ways, against a fixed acquisition and fixed
density. Everything else is deliberately out.
