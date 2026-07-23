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
4. **Full-wave multi-angle** — carry the leverage into j-Wave: the anisotropic
   absorber (route A of `cann_forward_scoping.md`, but *anisotropic* moduli) or,
   cheaper first, per-angle isotropic α inversions assembled into α(θ) and fed to
   the **anisotropic CANN discovery already built**
   (`alpha_basis_library_anisotropic`) — closing the loop to the constitutive
   model.
5. **3D + fibre crossings** — the real anatomy; the 2D `sin^2` becomes a
   structural-tensor form (the CANN I4/I5 analogue).

## Bottom line

GM/WM is **not** out of reach — but the reachable signal is **anisotropy**, not
speed, magnitude, or spectrum. The angular leverage is real and recoverable
(proven); the open work is the blind bulk/anisotropy identifiability and carrying
it into the full-wave solver with realistic fibre fields.
