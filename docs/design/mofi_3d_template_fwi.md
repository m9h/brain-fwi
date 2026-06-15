# SE(3) MOFI + 3D Template-Prior FWI

**Status:** proposed (design only)
**Author:** drafted 2026-06-15
**Motivates:** the path from "3D pipeline runs" to "3D pipeline reconstructs the brain."

## 1. Why

Plain FWI from a homogeneous water start does **not** recover the skull. Evidence: the
latest 192³ MIDA run (`brain_usct_192_mida_voxel_984.h5`, ~20 h GPU) recovered **0% of
the skull contrast** (true skull ≈2707 m/s, recon ≈1503) and improved global RMSE by
~0% (373→372). The high-contrast skull cycle-skips; the inversion never locks on.

This is the known hard problem. Guasch et al. avoid it: they do **not** invert the skull
from scratch. Instead:

1. take a **skull template** (from CT) as the starting model,
2. **rigidly align** it to the patient (guidance-free = MOFI, now implemented in 2D),
3. run FWI to reconstruct the **brain interior** with the skull as a prior.

Our 2D MOFI Stage 3 test reproduces the effect (skull-from-water fails; skull-as-prior
unblocks the interior, RMSE 277→174). This doc scopes lifting that to 3D.

**But the 984 recipe was also under-powered vs the one proven to work.** It used
**256 elements, 72 sources, ≤300 kHz, plain L2 multiscale**, from a water start. Guasch
et al. (2020) reconstructed the brain in-silico with **~1024 transducers, frequencies to
~850 kHz, and adaptive waveform inversion (AWI)** — a cycle-skip-robust objective — also
from a simple start, *without* a template. So the gap is partly **acquisition density +
frequency reach + objective (AWI)**, not only the starting model. (The 16-element
geometry in the 2D MOFI tests is test-only; it is not the production assumption.)

## 1.5 Two paths to functional 3D — and the role of AWI

**Path 1 — dense + AWI, from scratch (the Guasch-2020 recipe).** More elements (→512–1024),
more sources, higher frequency (as far as the grid allows), and **AWI instead of L2**.
Reconstructs skull *and* brain with no template — proven in-silico. Cost: high-frequency
3D needs fine grids (850 kHz ⇒ ~0.3 mm ⇒ ≫192³, ~10–50× the 984 compute) — the paper used
HPC. Realistic single-GPU ceiling ≈ 300–500 kHz at 256³–384³; Modal A100:4 pushes further.

**Path 2 — template + MOFI (this doc, §§3–6).** Skull from CT/template, align (MOFI),
reconstruct the brain interior. Cheaper FWI (the skull is never inverted) and
clinical-realistic, but more to build.

**The single highest-leverage missing piece for EITHER path is almost certainly AWI.**
The project has L2 and envelope losses, not AWI. AWI's wide basin of attraction (a Wiener
matching filter penalised away from zero lag; Warner & Guasch 2016) is precisely what
avoids the skull cycle-skip that sank job 984 — plain multiscale L2 from water cannot
climb the high-contrast skull. Adding AWI is a well-defined, framework-agnostic
enhancement and likely buys more than denser acquisition or the template alone.

**Recommended sequencing:** try the cheap recipe upgrade *first* (AWI + denser acquisition
+ extended frequency on the existing 192³ MIDA setup, from a simple start). If skull/brain
begin to recover, Path 1 is viable and no template is needed. If it plateaus (compute
ceiling, or genuine ill-posedness through the skull), fall back to Path 2 (template+MOFI),
which is also the clinical-realistic route since real patients won't be reconstructed
skull-from-scratch.

## 2. Architecture

```
MIDA phantom ──► skull-template extraction ──► canonical skull shell c_skull(x), ρ_skull(x)
                                                      │
observed data (patient/ground-truth pose) ──► SE(3) MOFI (coarse, low-freq) ──► pose φ̂ ∈ SE(3)
                                                      │
                                       warp(c_skull, φ̂) at full 192³  ──► starting model
                                                      │
                              masked FWI (skull = fixed prior, interior reconstructed) ──► brain image
```

Three new components (A, B, C) + validation. `run_fwi` already supports `mask` and an
arbitrary `initial_velocity`, so the FWI engine needs little change — most work is in
MOFI's 3D extension, skull extraction, and orchestration.

## 3. Component A — SE(3) MOFI

Generalise `warp_se2` / `run_mofi` (`src/brain_fwi/inversion/mofi.py`) from SE(2) (3 DOF)
to SE(3) (6 DOF: rotation vector ω∈ℝ³, translation t∈ℝ³).

### A.1 `warp_se3(template_3d, pose6, centre, cval)`
- Rotation via the **exponential map / Rodrigues' formula** on the rotation vector ω
  (analytic, differentiable, no gimbal lock; the SE(3) tangent is the natural
  optimisation space):
  `R = I + sinθ/θ · K + (1-cosθ)/θ² · K²`, `K = skew(ω)`, `θ = ‖ω‖` (with the θ→0 limit
  handled by a `jnp.where`/series guard).
- Inverse-warp resampling, identical pattern to the 2D version:
  `x_src = Rᵀ((x_out - centre) - t) + centre`, then **trilinear**
  `jax.scipy.ndimage.map_coordinates(..., order=1)` on the 3D field.
- Same JAX win: `jax.grad(loss(warp_se3(template, pose6)))` gives ∂f/∂φ end-to-end.

### A.2 Generalise the driver
- `run_mofi` already loops over bands/iters with single-norm gradient + line search.
  Make the pose dimension and the scale metric N-D: `param_scale` = rotation components
  scaled by a characteristic radius (so rotation/translation are commensurate, as in 2D),
  translation = 1.
- Expect the rotation↔translation coupling (Stage 2 finding) to be **worse** in 3D
  (6-DOF valley) — budget the same convergence tuning (single-norm stable; `rotation_scale`
  the lever; low frequency for robustness under model mismatch — Stage 3 finding).

### A.3 Multi-resolution pose (the cost key)
3D forward sims at 192³ are ~minutes each; MOFI needs O(100s) of them. **Pose is only 6
global DOF and low-frequency information**, so estimate it on a **downsampled grid**
(e.g. 48³–96³) at low frequency, then apply φ̂ at full 192³ for FWI. This is the single
most important performance decision — it makes 3D MOFI affordable (minutes, not days).

## 4. Component B — skull-template extraction

New `phantoms/templates.py` (or extend `phantoms/mida.py`):
- Map MIDA bone labels → a **skull-only** sound-speed/density shell on a water (1500/1000)
  background — the canonical template. (`phantoms/mida.py` already has the tissue→acoustic
  mapping incl. cortical vs trabecular bone.)
- Produce the **interior mask** (intracranial region) used both as the FWI update region
  and the brain-RMSE region.
- Clinical analog = CT-derived skull; for validation we use MIDA's own skull (inverse
  crime first, then a perturbed/mismatched template).

## 5. Component C — 3D template-prior FWI

- Starting model = `warp_se3(c_skull, φ̂)` + interior at 1500 (brain unknown).
- **Masked FWI** (proven in Stage 3): `FWIConfig(mask=interior_mask)` — skull held fixed
  as prior, FWI updates only the intracranial interior. No `run_fwi` change needed beyond
  passing the mask + skull-template init.
- Density: warp ρ_skull with the same φ̂ (`warp_medium` already does this in 2D).
- **Later (stretch):** adaptive waveform inversion (AWI; Warner & Guasch 2016) to absorb
  imperfect skull templates — a different loss, not a blocker for first results.

## 6. Validation (staged, mirrors the 2D MOFI suite)

Small grids (48³–64³) for tests; 192³ only for the final run.

- **V0** `warp_se3` unit tests: identity, integer translation = roll, pure-axis rotation
  vs analytic, differentiability. (CPU, fast.)
- **V1** FD gradient check of ∂f/∂φ⁶ through j-Wave on a tiny 3D grid. (GPU.)
- **V2** in-silico SE(3) pose recovery (inverse crime): warp MIDA head by known φ, recover.
  Contract |Δrot|, |Δtrans| small.
- **V3** 3D MOFI+masked-FWI unblock: skull-prior aligned vs mis-posed → interior RMSE
  ratio < 1 (the 3D Stage 3).
- **V4** the real test: 192³ MIDA, skull template (slightly perturbed pose), reconstruct
  brain, **compare against the from-water 984 baseline**. Success = recovers brain
  structure 984 could not.

## 7. Performance strategy

- MOFI on coarse/low-freq grid (§A.3) — pose est. in minutes.
- Reuse the segmented checkpointed forward (`simulation/checkpointed_scan.py`) for 192³ FWI.
- Modal (A100) for the heavy 192³ V4; Slurm/GB10 for smaller V0–V3.
- Tests deterministic-ish but GPU float jitter exists (Stage 2 finding) — tolerances allow it.

## 8. Risks / open questions

- **Does masked FWI recover the brain at clinical resolution with this acquisition?**
  Unknown — genuine research. De-risk with V2/V3 before the expensive V4.
- **6-DOF pose convergence** — coupling worse than 2D; may need per-axis tuning or a
  coarse-to-fine pose schedule.
- **Template mismatch** — real skulls ≠ template; AWI (§5 stretch) is the answer but adds scope.
- **Rotation centre / intracranial mask definition** in 3D from MIDA needs care.

## 9. Phasing & rough effort

| Phase | Deliverable | Effort |
|---|---|---|
| A | `warp_se3` + run_mofi N-D + V0/V1 tests | ~2–3 days |
| B | skull-template extraction from MIDA + V2 | ~1–2 days |
| C | multi-resolution pose + 3D masked template-FWI orchestration + V3 | ~3–4 days |
| D | 192³ MIDA validation (V4) vs 984 baseline | ~1 day work + GPU time |
| E (stretch) | AWI skull loss | ~3–5 days |

**Total to a first real 3D brain image (V4): ~1.5–2 focused weeks + GPU.** It is research,
not a guaranteed outcome — V2/V3 are the go/no-go gates before committing 192³ compute.

## 10. First step

Given §1.5, the highest-leverage first move is **not** SE(3) MOFI but a **recipe upgrade
experiment** on the existing 192³ MIDA setup: implement an **AWI objective** in
`inversion/losses.py`, bump elements/sources (256→512, use more of them as sources), and
extend the frequency ladder as far as 192³ allows (~400 kHz). Run from a simple start and
see whether the skull/brain begin to recover vs the 984 baseline. This is the cheapest
test of "are we recipe-limited or starting-model-limited?" and it gates everything:

- **If recovery improves** → Path 1 is alive; invest in AWI + denser acquisition (+ finer
  grid on Modal A100:4 for higher frequency). Template may be unnecessary in-silico.
- **If it plateaus** → commit to Path 2; start with **Phase A** (`warp_se3` + N-D
  `run_mofi` + V0 CPU unit tests — cheap, GPU-free, and the 2D code factors cleanly, so
  it's mostly generalising the rotation matrix and pose dimension, not a rewrite).

Either way, **AWI is on the critical path** — build it first.
