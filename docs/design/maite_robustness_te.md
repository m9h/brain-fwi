# MAITE-style robustness test & evaluation for FWI ultrasound imaging

A framing note: how the **AI test-&-evaluation (T&E)** paradigm behind Kitware's
**MAITE** ("Modular AI Trustworthy Engineering") and **NRTK** ("Natural
Robustness Toolkit") maps onto transcranial FWI ultrasound imaging — and,
honestly, where it does *not*.

## What MAITE/NRTK are, and the honest fit

- **NRTK** generates *operationally-realistic perturbations* of **camera / EO-IR
  imagery** (via the pyBSM optical sensor model — lens blur, aperture, focal
  length, photometric/sensor noise, radial distortion) and measures how a
  computer-vision model's performance degrades as you sweep perturbation
  strength. **MAITE** is the surrounding interface standard (datasets, models,
  metrics, perturbers) so robustness suites compose across tools.

- **Direct fit to the FWI core: low.** brain-fwi's measurements are **acoustic
  RF time-series** at transducers, and its outputs are **sound-speed fields** —
  not optical images. NRTK's optical-sensor perturbers are not physically
  meaningful here. *Do not* bolt NRTK's perturber library onto the wave solver.

- **Fit of the *paradigm*: high.** The T&E pattern — *enumerate realistic
  perturbations → sweep strength → plot performance-vs-strength → identify the
  break-down boundary* — is exactly the credibility analysis this project
  already does ad hoc. Adopting the **vocabulary and structure** (not the
  optical perturbers) is the win, especially for the learned components.

## The FWI-US perturber suite (domain-specific)

The MAITE analogue of NRTK's optical perturbers is a set of **acoustic /
acquisition / model perturbers**. Most already exist in the credibility work
(`docs/design/status_and_frontiers.md`); the value is collecting them behind a
common sweep-and-report interface:

| Perturber (MAITE "augmentation") | Physical meaning | Status in brain-fwi |
|---|---|---|
| Measurement-noise SNR sweep | receiver/electronic noise | done (robust to 20 dB) |
| Source-signature error | unknown emitted wavelet | done (Pratt source co-inversion) |
| **Skull pose / registration error** | CT→acoustic misalignment | characterized (sub-voxel needed; MOFI transmission-TT fix) |
| Forward-model / solver mismatch | discretization, physics gaps | done (j-Wave↔k-Wave cross-validation) |
| **Absorption / tissue-property error** | wrong α, ρ, c priors | partially (absorption-aware A/B; α-error sweep is open) |
| Out-of-distribution anatomy / novel lesion | prior generalization | partially (held-out subjects, lesion frontier) |

The "model under test" is the **inversion pipeline** — and most usefully the
**learned** pieces (diffusion-prior DPS, FNO surrogate, NPE), where "AI
robustness" genuinely applies. The metric is reconstruction error (brain-ROI
RMSE), lesion detectability, or posterior calibration (SBC), as a function of
perturbation strength.

## How to operationalize (optional, scoped)

1. **Lightweight, recommended first:** standardize the existing sweeps into one
   harness that emits robustness curves (error vs SNR, vs pose error, vs α
   error) + a one-page T&E report. No new dependency; reuse `inversion/` +
   `validation/`.
2. **MAITE-compatible (if ecosystem alignment is a goal):** implement the FWI
   perturbers behind MAITE's `Augmentation`/`Metric` protocols so they compose
   with `nrtk-explorer` (dataset viz) and the broader Kitware/XAITK T&E stack.
   This buys interoperability and a recognized credibility story — worth it only
   if a Kitware/MAITE partnership or T&E-flavored funding is in play.

## Why this matters for support

Framing brain-fwi's credibility work in the MAITE/NRTK T&E vocabulary positions
the differentiable-FWI + learned-prior pipeline as **trustworthy-AI-ready** for
funders who speak that language (DoD/IQT-adjacent, Kitware medical-imaging
collaborations). The substance already exists; this is about *naming* it as
operational robustness T&E and producing the robustness curves to match.

**Guardrail:** keep perturbers domain-specific (acoustic/acquisition/model).
NRTK's optical perturbers are the wrong physics; the borrowed asset is the
*evaluation methodology*, not the camera-sensor models.
