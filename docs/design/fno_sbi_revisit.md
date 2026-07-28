# Revisiting the parked FNO via SBI over anisotropy (scoped + prototyped)

## Why the FNO was parked
`CToTraceFNO3D` (velocity->traces) forward generalises to unseen subjects (corr
0.989) but its GRADIENT/adjoint fails for FWI (~1/3 of subjects give cosine ~-0.95
even for tiny perturbations; forward-MSE training doesn't constrain the
derivative). Fatal for gradient-based FWI over a high-dim sharp velocity field.

## The revisit (this session's progress makes it viable)
Two changes reframe the FNO:
1. The anisotropy program produced a LOW-DIMENSIONAL inverse problem
   `(a_par, a_perp, phi)` -- a few smooth parameters, not a sharp field.
2. The original vision was SBI / neural posterior estimation, and the infra exists
   (`inference/flow.py` NPE, `inference/sbc.py`).

**SBI is likelihood-free -- it uses only the FORWARD, never the adjoint.** So the
FNO's adjoint wall (the reason it was parked) is IRRELEVANT; its proven forward is
exactly what SBI needs, and SBI needs millions of forwards that only a fast
surrogate can afford at 3D scale.

## Prototype + evaluation (`scripts/anisotropy_sbi_prototype.py`)
NPE flow over `(a_par, a_perp, phi)`, forward = the anisotropy amplitude model
(the exact model the validated j-Wave absorber / an FNO surrogate produces), 12000
sims, trained in 25 s. Evaluated at truth (2.0, 8.0, 0.60 rad):

| param | posterior mean +/- std | 90% CI | true |
|---|---|---|---|
| a_par | 2.03 +/- 0.03 | [1.97, 2.09] | 2.0 |
| a_perp | 7.98 +/- 0.07 | [7.88, 8.09] | 8.0 |
| phi | 0.58 +/- 0.01 | [0.57, 0.60] | 0.60 |

**Fibre orientation to 1.0 deg with a 0.5 deg credible width -- recovered WITH
uncertainty** (the point-estimate FWI cannot give this). SBC calibration (max CDF
deviation from uniform): a_par 0.092 (well-calibrated), a_perp 0.110, phi 0.145
(slightly OVER-confident -- an honest finding; the phi posterior is a touch too
tight, truth sits at its edge). Figure: `results/absorption_aware_fwi_3d/anisotropy_sbi.png`.

## Verdict
The revisit is validated in principle: SBI over the low-dim anisotropy parameters
gives a calibrated, uncertainty-aware fibre-orientation posterior, FORWARD-ONLY --
un-parking the FNO by using it where the adjoint doesn't matter. Clinically this
is the missing piece (a device must report confidence in its fibre estimate).

## Next (scale-up)
1. Fix the mild over-confidence: better noise model / more sims / ensemble or
   sequential (SNPE-C) rounds.
2. Train an **anisotropy-FNO** (params -> traces) so the 3D forward is fast enough
   for SBI (j-Wave is too slow for 1e4+ sims in 3D; the FNO forward at 0.989 is
   the accurate fast replacement). This is the FNO's actual un-parking.
3. Cheap parallel check: does the FNO *adjoint* work for the LOW-DIM smooth
   anisotropy params (it failed for high-dim sharp velocity)? If yes -> also
   FNO-accelerated gradient FWI. Try DeFINO Fisher-projected VJP-matching.
4. Spatially-varying phi field posterior (combine with the per-voxel absorber).
