# The path forward for GM/WM: multi-angle anisotropic attenuation

After ruling out the alternatives this session, one axis survives with real
leverage for gray/white-matter discrimination — **attenuation anisotropy**.

## Why the other axes fail (established this session)

| Discriminator | Verdict |
|---|---|
| Sound speed | **Degenerate** — GM = WM = 1560 m/s. Velocity FWI is blind to it. |
| Attenuation magnitude | Small (WM ≈ 1.5× GM). Pure α inversion resolves the *ordering* (WM > GM, measured) but not quantitatively; recovery caps ~50%. |
| Attenuation spectral shape (exponent y) | **Not band-resolvable** — over the narrow transcranial band (~half a decade, skull forces sub-MHz), a relaxation spectrum can't pin y ≠ 1 (measured ~9%). |
| **Attenuation anisotropy** | **The leverage.** WM is fibre-oriented ⇒ α depends on propagation angle; GM is isotropic. The leverage is **angular**, and transmission tomography crosses every voxel at many angles. |

## The result (proven)

`inversion/anisotropic_atten.py` — a differentiable straight-ray tomography with
`alpha(theta) = alpha_iso + alpha_aniso * sin^2(theta - phi)` (minimal along
fibre, maximal across). On a phantom where **GM and WM have identical isotropic
α** and differ *only* in anisotropy (WM `alpha_aniso = 0.5`, GM = 0), given the
fibre direction (DTI) and a **bulk-α estimate** (from the isotropic FWI), the
recovered anisotropy map separates them at **ratio ~10×** (WM 0.42 vs GM 0.04,
true WM 0.5). Robust to ~50% error in the bulk estimate (ratio still ~4.8).
Figure: `results/absorption_aware_fwi_3d/anisotropy_gmwm.png`;
test `test_anisotropic_atten.py`.

**This is the path**: a discriminator that works when speed, bulk α, and spectral
shape all fail — because the leverage is angular, which the narrow band does not
constrain but the acquisition geometry does.

## Blind joint recovery — SOLVED (homogeneous-bulk prior)

The **fully-blind joint** inversion of bulk α *and* anisotropy (no supplied bulk)
is a genuine tomographic identifiability challenge: the angle-mean of α is
`alpha_iso + 0.5*alpha_aniso`, so a per-voxel *field* bulk is weakly constrained
and crosstalks (the bulk absorbs the WM structure, anisotropy leaks to GM).
Measured failure modes: free-field bulk (ratio ~1), TV/L1 sparsity on the
anisotropy (kills it — bulk wins), even a *smoothed-field* bulk (ratio 0.2 — a
smooth field still adapts locally).

**Fix (the Living Matter Lab insight):** tame the ill-posed inversion by baking a
strong **structural constraint** into the parameterisation — not on the
anisotropy (sparsity there backfires), but on the **bulk**: constrain
`alpha_iso` to be **homogeneous (a single scalar)**, matching the physical fact
that bulk attenuation is smooth. Then the bulk *cannot* absorb sharp WM structure,
so it is forced into the anisotropy channel where it belongs. Measured (blind, no
bulk supplied): WM/GM anisotropy **ratio ~12x**, scalar bulk recovers ~0.6
(true). `invert_anisotropic(a_iso=None)`, `test_anisotropic_atten.py`.

Generalisation for real (slowly-varying) tissue: a **low-rank** bulk (a coarse
grid or low-order polynomial, a few DOF) rather than a strict scalar — enough
freedom for real bulk variation, too little to mimic sharp anisotropy structure.
That is the next refinement (the scalar proves identifiability; low-rank makes it
realistic).

## Sequenced plan

1. **DONE** — straight-ray anisotropy tomography; proves the angular leverage
   (ratio 10 given bulk α).
2. **DONE** — blind joint recovery via the homogeneous-bulk structural prior
   (ratio ~12, bulk recovered). Next refinement: **low-rank** bulk for realistic
   slowly-varying tissue.
3. **Fibre orientation from ultrasound (no DTI)** — **DONE (magnitude+direction).**
   `alpha(theta) = b - u*cos2theta - v*sin2theta` with the anisotropy vector
   `(u,v) = (0.5 a_aniso cos2phi, 0.5 a_aniso sin2phi)` is *linear*, so inverting
   `(u,v)` gives `a_aniso = 2|(u,v)|` **and** `phi = 0.5 atan2(v,u)`.
   `invert_orientation` recovers the **fibre direction inside WM to ~0.6-0.9 deg**
   from the angular pattern alone — acoustic tractography, removing the DTI
   dependency for orientation. (`test_orientation_recovery.py`.) The blind (u,v)
   magnitude is leakier (ratio ~1.8) than the known-phi form (ratio ~12); the
   **two-stage** recipe — recover phi via (u,v), then the clean sin^2-form
   magnitude with that phi — sharpens it (ratio ~2.7 and rising). Remaining:
   per-voxel *varying* phi (curved tracts) and recovering phi where a_aniso is
   weak.

   **Real curved/crossing tracts (DiSCo diffusion phantom) — DONE (first result).**
   Using the sbi4dwi DiSCo numerical phantom (known curving/crossing fibre
   strands): DTI-fit → per-voxel fibre direction phi(x) + FA; paint
   `alpha_aniso ~ FA*(in-plane)`, `phi = fibre angle`; recover blindly.
   Measured: fibre direction to **~12 deg median** and anisotropy-magnitude
   correlation **0.58** on the 7-arm pinwheel — the strand geometry is clearly
   reproduced. Degrades at the central **crossing** hub (single-fibre model and
   DTI V1 both average crossings). `scripts/anisotropy_disco_demo.py`
   (+ `_extract_disco_fiber.py`), figure `anisotropy_disco.png`. The physical
   basis: acoustic-attenuation anisotropy and diffusion anisotropy share the same
   white-matter fibre microstructure, so DTI FA/orientation is the ground truth.
   Next: a **multi-fibre** (two-direction) model for crossings; 3D.

4. **Crossing-fibre model — DONE (angular-harmonic ODF).** The DiSCo hub failed
   because a single-fibre `sin^2(theta-phi)` profile is pure **2-theta**: two
   crossing fibres sum to one 2-theta sinusoid (2 measurements, 4 unknowns) —
   fundamentally unresolvable, the exact acoustic analogue of **DTI cannot do
   crossings**. Crossings resolve only when the single-fibre profile is SHARPER
   (`sin^{2s}`, s>1), carrying higher (4-theta) harmonics — the acoustic analogue
   of **HARDI / high-order ODFs**. `crossing_fibres.py`: `fibre_profile`,
   `two_fibre_profile`, `angular_harmonics` (attenuation-ODF), `crossing_index`
   (`|4theta|/|2theta|`, the crossing detector — small for one fibre, large for a
   sharp orthogonal crossing, ~0 for a soft crossing which has no 4-theta), and
   `fit_two_fibres` (grid-refine, recovers BOTH directions to <10 deg for a sharp
   crossing). Honest limit: **soft (sin^2) crossings are never resolvable**
   (orthogonal soft crossing even degenerates to isotropic); resolvability needs a
   physically sharp attenuation profile.

   **Per-voxel attenuation-ODF tomography — DONE.** `forward_ray_decay_odf` /
   `invert_odf` invert the full per-voxel angular ODF (a0 homogeneous bulk +
   2-theta AND 4-theta harmonic *fields*), so the tomography *carries* the
   crossing information the single-phi model discards. On a crossing phantom
   (central patch of two orthogonal sharp fibres vs a single-fibre surround) the
   recovered 4-theta crossing map (`crossing_index_map`) is ~2x elevated in the
   crossing patch — where the single-phi model sees LOW anisotropy (2-theta
   cancels) and mis-reads the crossing as near-isotropic. `sin4_odf_coeffs` builds
   the ground-truth ODF; `test_odf_tomography.py`. The 4-theta channel is
   higher-order/weaker (light smoothing), so recovered contrast is modest (~2x,
   not the true near-infinite ratio) — honest 4-theta SNR. `resolve_odf_crossings` gives the per-voxel two-fibre decomposition (explicit
   crossing directions — the tomography's "acoustic HARDI" output;
   `test_odf_resolve.py`). **These ODF harmonic fields are exactly the anisotropic-
   attenuation unknowns a full-wave FWI would invert — the ray tomography validates
   the parameterisation before the expensive j-Wave build.** Next: 3D; the
   anisotropic/Prony absorber to carry this into j-Wave FWI on real anatomy.
4. **Full-wave multi-angle — anisotropic absorber OPERATOR done (prototype).**
   Directional acoustic loss as a per-voxel symmetric attenuation **tensor**
   `D(x)`, applied via `sum_ij D_ij d_i d_j (field)`
   (`simulation/anisotropic_absorber.py`). Key properties (all tested,
   `test_anisotropic_absorber.py`): (a) **local** — spectral second derivatives
   contracted with a local tensor, so spatially-varying anisotropy works (unlike
   the isotropic fractional-Laplacian's global exponent); (b) for a plane wave the
   loss coefficient is `k^T D k/|k|^2 = a_par cos^2(theta-phi) + a_perp
   sin^2(theta-phi)` — **exactly the sandbox `sin^2(theta-phi)` model, so this
   absorber IS the full-wave implementation of the whole anisotropy program**;
   (c) **differentiable** w.r.t. the tensor fields (the FWI unknowns); (d) the
   **acoustic analogue of the DTI diffusion tensor** (same rank-2 symmetric form),
   which is why diffusion FA/orientation is the physical ground truth. Remaining:
   integrate this tensor loss into the j-Wave fork's Treeby-Cox pressure update
   (replacing the isotropic `|k|^p` term), k-Wave-validate, and invert the tensor
   fields end-to-end in a full-wave anisotropic FWI. This is the fork build the
   sandbox now fully specifies.
5. **3D + fibre crossings** — the real anatomy; the 2D `sin^2` becomes a
   structural-tensor form (the CANN I4/I5 analogue).

## Bottom line

GM/WM is **not** out of reach — but the reachable signal is **anisotropy**, not
speed, magnitude, or spectrum. The angular leverage is real and recoverable
(proven); the open work is the blind bulk/anisotropy identifiability and carrying
it into the full-wave solver with realistic fibre fields.
