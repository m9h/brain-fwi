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

## The remaining hard problem (honest)

The **fully-blind joint** inversion of bulk α *and* anisotropy (without a supplied
bulk estimate) is a genuine tomographic identifiability challenge: the angle-mean
of α is `alpha_iso + 0.5*alpha_aniso`, so the split is weakly constrained, and
naive gradient descent / alternating minimisation lands in a crosstalk basin
(the bulk channel absorbs the WM structure, anisotropy leaks to GM). Measured:
given a bulk estimate it is clean (ratio 10); blind, it is not.

Known routes to fix (not yet implemented):
- **Cross-gradient / structural** coupling tying anisotropy structure to an
  independent map (the DTI fibre field, or the velocity FWI's tissue boundaries).
- **Iterative bulk de-contamination** with a strong low-resolution prior on the
  bulk (it must be too smooth to absorb sharp WM structure) — my quick attempts
  under-converged; needs a proper multi-scale / bound-constrained solver.
- **A bulk estimate that structurally excludes anisotropy**, e.g. from the
  isotropic FWI at a frequency/geometry chosen to minimise the anisotropic bias.

## Sequenced plan

1. **DONE** — straight-ray anisotropy tomography; proves the angular leverage
   (ratio 10 given bulk α).
2. **Blind joint identifiability** — cross-gradient with the DTI/velocity
   structure; multi-scale bulk prior. The core open problem above.
3. **Realistic fibre fields** — per-voxel `phi` from DTI (not uniform); recover
   `phi` too, or take it from co-registered MRI (transcranial patients have it).
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
