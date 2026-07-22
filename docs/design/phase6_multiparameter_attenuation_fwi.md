# Phase 6 — Multiparameter (velocity + attenuation) FWI: the road to gray/white contrast

**Status:** active (this branch). Supersedes the "invert velocity, freeze α"
regime of Phase 5.

**One-line thesis.** Gray/white-matter differentiation is *not* primarily a
resolution problem and *not* a learned-prior problem — it is a **contrast-channel
problem**. The GM/WM signal lives in **attenuation**, which the pipeline
currently freezes. Inverting α(x) jointly with c(x), regularised by the
`constitutive/` CANN, is the change most likely to surface tissue structure.

---

## 1. Why velocity-only FWI cannot resolve GM/WM (evidence)

Two independent walls, established by the repo audit and the 2024–2026
literature scan (see `status_and_frontiers.md` and the citations in §5):

1. **Sound speed barely separates GM from WM.** Direct measurement (Kang et al.,
   *Ultrasonics* 2022) puts whole-brain SOS at **1532–1541 m/s with SD only
   10–14 m/s** — the GM/WM speed difference sits at the measurement noise floor
   (<1%). Every phantom in this repo encodes this reality by assigning GM and WM
   the **identical 1560 m/s** (`phantoms/properties.py:45-46`,
   `phantoms/mida.py:268-269`). A c(x)-only inversion therefore has *no acoustic
   channel* in which GM and WM differ.

2. **Attenuation does separate them.** The same measurement study finds **white-
   matter attenuation ≈ 1.5× gray-matter** — a clean, reliable discriminator.
   ITRUSST/Aubry (2022) benchmark values, already encoded in
   `phantoms/properties.py` and fit by the CANN
   (`tests/test_constitutive.py::test_fit_tissue_alpha_curves_*`), carry this
   contrast. Ex-vivo transcranial UCT that *did* separate GM/WM (Duric group,
   Mitcham et al., *Med. Phys.* 2025, 300–700 kHz) reports WM 1639 / GM 1656 m/s
   **and** leans on attenuation + high-resolution geometry, not speed alone.

**Corollary.** At 50–280 kHz the λ/2 diffraction limit is ~2.7–9.4 mm — near the
cortical-ribbon scale but not the binding constraint. Even at infinite
resolution, velocity-only FWI has nothing to invert for GM vs WM. **The missing
degree of freedom is α(x), not more Hz.**

---

## 2. What we have to build on

- `build_medium(..., attenuation=<field>, alpha_power=y)` already accepts α as a
  **differentiable FourierSeries field** (`simulation/forward.py:44`). The Treeby-
  Cox absorbing EoS is k-Wave-validated bit-for-bit (Phase 5). So the forward
  *already* depends smoothly on α — nothing in the physics blocks `jax.grad`
  w.r.t. α.
- `constitutive/cann.py::AttenuationCANN` — a monotone, non-negative, DC-
  vanishing α(ω) model that fits every ITRUSST tissue to <2% median rel-RMSE.
- `constitutive/kk.py` — Kramers–Kronig α(ω)→c(ω), i.e. a **causality coupling**
  between the two channels we are about to invert jointly.
- The absorption-aware A/B: freezing α at *truth* is worth ~4× on brain RMSE
  (6.1 vs 25.7 at 48³). Inverting α is the generalisation of that win to the
  clinical case where α is *unknown*.

The only thing missing is that `single_shot_loss` differentiates w.r.t. velocity
only (`inversion/fwi.py:555-568`). Phase 6 makes α a co-inverted parameter.

---

## 3. Design

### 3.1 Joint parameterisation (this branch)
- Voxel path: optimisation state becomes a pytree `{"c": velocity, "a": alpha}`
  when `FWIConfig.invert_attenuation=True`; **byte-identical to today when
  False** (params stays a bare velocity array). Default path is untouched — the
  192³ production run is not at risk.
- α is a power-law coefficient field (dB/cm/MHz^`alpha_power`), clipped to
  `[0, attenuation_max]` (non-negativity = energy dissipation, the same physics
  the CANN enforces via softplus).
- Per-leaf gradient handling: velocity uses `config.mask`; α uses
  `config.attenuation_mask`. Each leaf is preconditioned → smoothed → masked →
  max-normalised independently, then α is scaled by
  `attenuation_lr / learning_rate` so a single `optax.sgd(learning_rate)` gives
  a max c-step of `learning_rate` m/s and a max α-step of `attenuation_lr`
  dB/cm/MHz per iteration. Distinct physical scales, distinct step sizes.

### 3.2 CANN as the α regulariser (issue #45, implemented)
Voxel α inversion is ill-posed (attenuation is weakly constrained by phase). The
CANN is the fix: instead of a free per-voxel α coefficient, tie the inverted α
to the **CANN's low-dimensional α(ω) manifold** per tissue class — "invert a few
coefficients + a spatial field" instead of a noisy image.

**Implemented** (`constitutive/manifold.py`): `tissue_alpha_coefficients()` reads
the physical archetype α levels straight from `TISSUE_PROPERTIES` (water≈0, brain
0.6, cortical 4.0, trabecular 8.0 dB/cm/MHz — note GM=WM=0.6 today, the zero
contrast #47 fixes); `manifold_proximal(α, archetypes, β)` projects the field
onto the nearest archetype; `manifold_prior_grad` is the differentiable soft
analogue. Wired into `run_fwi` via `attenuation_archetypes` /
`attenuation_prior_weight` / `attenuation_prior_ramp`, applied as a **late-ramped
proximal step** after each α update.

**Constitutive c→α coupling (implemented, `constitutive/coupling.py`).** The
second lever: the well-resolved velocity predicts α through the same-tissue
(c, α) relation (`speed_alpha_anchors` + `alpha_from_speed`, wired via
`attenuation_speed_anchors`/`attenuation_speed_weight`, ramped). It is
**degenerate for GM/WM** (equal c → equal predicted α), so it aids c-contrasted
tissues (skull, CSF, brain-vs-background) but not the GM/WM split.

**Crosstalk finding (measured, blocks the naive coupling demo).** On a phantom
where the anomaly has *both* a c and an α contrast, a free-from-iter-0 α
**greedily absorbs the amplitude misfit and starves the velocity update**:
single-band 32³, α reached in-blob 3.05 while c stayed 1506 m/s (true 2000) — c
was not recovered at all. Since the coupling predicts α *from* c, it needs c to
be good first. ⇒ multiparameter FWI needs **c-first / hierarchical scheduling**
(freeze or down-weight α until c is recovered, then release α with the coupling).
The coupling machinery is in place and unit-tested; demonstrating its in-loop win
requires that hierarchical schedule (next step) — an honest dependency, not a
bug.

**Decisive win once c is recovered (measured).** Giving the inversion the
well-recovered-c regime a c-first schedule provides (c started at truth), the
coupling is decisive on the skull-like blob (c 2000 / α 6 in water):

| | α RMSE | α in-blob (true 6.0) |
|---|---|---|
| free-voxel | 0.916 | 0.26 |
| + constitutive coupling | **0.030** | **5.99** |

With c correct, the weak α data-gradient barely moves α (0.26), but the coupling
reads the correct c → predicts α = 6 → fills it almost perfectly (**30× lower
RMSE**). So the recipe is clear: **recover c first, then release α with the
constitutive coupling.** This closes #45's second half for c-contrasted tissues —
GM/WM still excepted (degenerate c).

**Honest regime finding (measured).** The proximal prior snaps toward the
*nearest* archetype, so it only pulls α *up* once the data has recovered it past
the archetype **midpoint (~50 % of the true value)**. In the weak single-band
regime the joint inversion recovers α to only ~25 % (in-blob 1.4 vs true 6.0,
`test_multiparameter_fwi`), i.e. **below** midpoint — so the nearest archetype is
water and the prior *cannot rescue* it there. The manifold prior therefore
**sharpens an already-adequate α** (field-level test: RMSE↓, spread↓), and its
in-loop benefit is contingent on stronger data — multi-band / higher-frequency
(#48) or real tissue contrast (#47). The complementary escape for the weak regime
is the **Kramers–Kronig coupling** (`kk_consistency_loss`, `constitutive/kk.py`),
which lets the well-constrained velocity *inform* α through causal dispersion;
integrating that coupling into the 3-D loop is the remaining half of #45.

### 3.3 Contrast comes from α, structure comes from the prior
The Phase-4 diffusion prior stays the detail multiplier — but it must be
retrained on data with real per-tissue α/ρ contrast (Issue #4) and, ideally, on
the α channel too. The honest pipeline is: **α inversion supplies the GM/WM
contrast channel; the prior sharpens its geometry.**

---

## 4. Milestones / tracked issues

| # | Issue | Unblocks |
|---|---|---|
| 6.1 | **Joint velocity+attenuation inversion** (this branch) | the contrast channel itself |
| 6.2 | CANN-regularised α inversion + KK causality coupling | well-posedness of α |
| 6.3 | CFL-safe / stability-controlled gradient + DPS guidance | InverseBench failure mode |
| 6.4 | Per-tissue α/ρ contrast phantoms (encode WM≈1.5×GM α) | a target to resolve at all |
| 6.5 | 500–700 kHz high-frequency regime (0.5 mm, ~384³) | sharpen once α is inverted |

(6.3–6.5 filed as GitHub issues; 6.1 is implemented here TDD-first, 6.2 is the
immediate follow-on.)

---

## 5. Literature anchors (2024–2026 scan)

- **GM/WM contrast is attenuation, not speed** — Kang et al., *Ultrasonics* 2022
  (SOS SD 10–14 m/s; WM attenuation 1.5× GM). Duric group / Mitcham et al.,
  *Med. Phys.* 2025 (ex-vivo GM/WM separation at 300–700 kHz).
- **Stability is the binding constraint for learned-prior FWI** — InverseBench
  (Zheng et al., ICLR 2025): DPS/DAPS fail on FWI when guidance steps violate the
  solver CFL condition. Validates the preconditioned/annealed-t direction (6.3).
- **Neural-operator adjoints don't transfer to skull FWI** — DeFINO
  (arXiv:2509.13620), DINO (JMLR 2025): forward-accurate ≠ gradient-accurate;
  transcranial ML (BrainPuzzle, WAM-Net) routes around surrogate adjoints. The
  FNO surrogate stays parked; j-Wave autodiff remains the gradient source.
- **Nobody has coupled a diffusion/score prior to full nonlinear *transcranial*
  wave-equation FWI with an inverted attenuation channel.** That is this
  project's differentiated position.

---

## 6. Honest limits

- Attenuation is weakly constrained by phase-only misfit; without the CANN/KK
  coupling (6.2), voxel α inversion will be smeared and possibly trade off
  against velocity. 6.1 proves the plumbing + gradient; 6.2 makes it useful.
- α checkpointing is not yet wired (voxel-α is not saved/resumed). Tracked in 6.1.
- GM/WM demonstration is blocked on 6.4 — today's phantoms have zero GM/WM α
  contrast, so recovery cannot be *shown* until a contrasted phantom exists.
