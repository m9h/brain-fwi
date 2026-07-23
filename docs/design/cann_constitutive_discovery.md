# CANN constitutive discovery for acoustic attenuation (Living Matter Lab method)

Inspiration: Ellen Kuhl's Living Matter Lab (Stanford) **Constitutive Artificial
Neural Networks**. We adapt their automated *constitutive model discovery* from
hyperelasticity (stress–strain) to **frequency-domain acoustic dissipation**
α(ω), to discover interpretable per-tissue attenuation laws and characterise
gray vs white matter.

## Source method (what we mirror)

- **Architecture** — Linka & Kuhl, *CMAME* 403:115731 (2023), arXiv:2210.02202.
  A CANN's nodes *are* constitutive building blocks; admissibility is hard-wired
  via **non-negative weights + reference-vanishing, monotone activations**, and
  the model is **linear in interpretable outer weights** (every classical model
  is a sparse corner).
- **Discovery** — McCulloch, St. Pierre, Linka & Kuhl, *IJNME* 125:e7481 (2024),
  arXiv:2310.06872. **L0** (subset selection) is preferred: it counts terms
  without biasing retained magnitudes (L1 biases; L2 never zeroes). Sweep the
  accuracy/sparsity frontier; use loss-landscape geometry (ellipsoid = keep,
  ridge = collinear, drop).
- **Brain GM/WM** — Linka, St. Pierre & Kuhl, *Acta Biomater.* 160:134 (2023).
  **The crux:** across cortex, basal ganglia, corona radiata, corpus callosum,
  the CANN discovers the *same* functional law (an I₂-only, ~two-term model);
  **GM and WM differ only in magnitude** (shear modulus GM ≈ 1.8/0.9 kPa vs
  WM ≈ 0.9/0.5 kPa), not in form. Their study is **isotropic**.
- **Dissipative extension (iCANN)** — Holthusen et al., *CMAME* 428:117063
  (2024), arXiv:2311.06380; JAX dual-potential arXiv:2502.17490 (2025). A convex,
  non-negative, origin-minimum **dissipation potential** enforces the 2nd law
  structurally; the viscoelastic specialisation is a **generalized-Maxwell /
  Prony relaxation spectrum** — the time-domain dual of a frequency-domain
  α(ω)/complex-modulus law. This is the right precedent for attenuation.

## What this means for GM/WM (honest)

1. **Isotropic ⇒ magnitude, not shape.** Kuhl's own brain result says GM/WM share
   the functional law and differ in magnitude. Our tables agree (same exponent
   1.3; α 0.6 vs 0.9). So an *isotropic* α(ω) CANN does **not** give a new
   orthogonal discriminator — GM/WM discrimination stays magnitude/SNR-limited
   (consistent with the #48 study: α recovery tops out ~50%).
2. **The shape discriminator is ANISOTROPY.** White matter is fibre-oriented and
   its attenuation is direction-dependent; gray matter is isotropic. That is the
   structural-tensor (I₄/I₅-analog) extension Kuhl explicitly did *not* activate
   — and the one axis where WM and GM differ in *form*. **Multi-angle**
   transmission FWI could resolve it. This is the CANN-native path to a genuine
   GM/WM separation, and the most promising open direction.

## The three gaps (and status)

| # | Gap | Status |
|---|---|---|
| 1 | No sparse **model discovery** (only fixed-2-mode Adam fit) | **DONE** — `discovery.py` |
| 2 | CANN is **not the FWI parameterisation** (scalar α, fixed exponent) | open |
| 3 | Forward **can't evaluate α(ω) from a CANN** (`build_medium` takes one scalar `alpha_power`) | open |

### #1 done — `constitutive/discovery.py`
`alpha_basis_library(omega)` builds an over-complete, **non-negative, DC-vanishing**
library (power laws `|ω|^y` + Debye/relaxation `ωτ/(1+(ωτ)²)`), each normalised
(mode-normalisation), **linear in outer weights**. `discover_alpha_law` runs **L0
greedy forward selection** (non-negative least squares per candidate) to a minimal
interpretable law. Tests (`test_cann_discovery.py`) confirm: DC-vanishing library,
sparse single-power-law recovery, non-negative weights, and — the headline —
**GM and WM select the same term and differ ~1.5× in magnitude**, reproducing the
Living Matter Lab brain finding in our domain.

### #2/#3 next — invert through the CANN
Replace the scalar-α parameterisation with CANN material parameters, and make the
forward evaluate frequency-dependent α(ω) from the CANN (the Treeby–Cox absorber
currently assumes a single global power-law exponent). Add the **Kramers–Kronig**
causality constraint (the one thing the hyperelastic CANNs never needed; already
in `kk.py`) so α(ω) and its dispersion partner c(ω) stay consistent. Then the
**anisotropic** extension (fibre-direction-dependent α, multi-angle data) is the
route to shape-level GM/WM discrimination.

## What does NOT translate (flagged)

Invariants / polyconvexity / objectivity / F-decomposition are hyperelastic
notions with no scalar-α(ω) analogue — don't force them. Our relevant convexity
is just **non-negativity + causal (KK) admissibility**. Dissipation is intrinsic
here (their Papers 1–3 are conservative; only the iCANN is the right structural
precedent). **Causality is the extra constraint we must add ourselves.**
