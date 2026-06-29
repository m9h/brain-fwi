# Clinically-realistic imaging helmet: design plan

Goal: replace the current Kernel-Flow-inspired cap (`transducers/helmet.py
::helmet_array_3d`) with a helmet grounded in the actual transcranial-FWI and
FUS-MRI hardware, and parameterize coverage/element-count by physics rather
than a fixed 256.

## Reference devices

- **Imperial / Guasch (FWI imaging target)** — the stated clinical form factor is
  a *single fixed helmet array of low-frequency transducers surrounding the head
  in 3D at all azimuths*, **1024 sources × 1024 receivers** (every element both
  source and receiver), low frequency (~hundreds of kHz). This is an *imaging*
  array: it needs **transmission paths crossing the brain**, so it encircles the
  head — not just a superior cap. (npj Digital Medicine 2020; dual-probe study
  UMB 2023 is the 2-probe stepping stone toward this helmet.)
- **Insightec ExAblate Neuro (FUS-MRI therapy reference)** — a **hemispherical**
  phased array, ~**1024 elements**, ~30 cm diameter, 220–650 kHz, degassed-water
  coupling, covering the **superior hemisphere**. The proven manufacturable form
  factor — but therapy-focused (superior, convergent), so as an *imaging* array
  it under-samples the anterior/inferior transmission paths FWI wants.

**Takeaway:** adopt Insightec's manufacturable hemispherical-shell + water-
coupling form factor, but extend coverage toward Guasch's full-azimuth
encirclement for transmission tomography.

## Coverage relative to the head (the device geometry)

A skull-conforming ellipsoidal **vault helmet**, sitting a coupling standoff off
the scalp:

- **Shape / size**: ellipsoid matched to the adult skull outer surface —
  semi-axes ≈ AP 95 mm, LR 80 mm, SI 100 mm — plus a **5–10 mm degassed-water /
  gel standoff** (element sits ~scalp + standoff).
- **Polar coverage**: from the **vertex (0°)** down to **~140°** (just above the
  ears laterally / occipital base posteriorly), i.e. the upper ~2/3 of the head.
- **Azimuth**: **full 360°** at the upper levels (this is the key change — it's
  what gives crossing transmission paths). Only a **face aperture** is removed:
  a ~40° anterior-inferior cutout over the eyes/nose/airway, plus the inferior
  (neck) cutout. *Not* the whole anterior hemisphere (the current
  `exclude_face` removes too much, killing anterior↔posterior transmission).
- **Element count / spacing**: sample at ~**λ/2** at the top FWI frequency.
  In water at 300 kHz, λ ≈ 5 mm → ~2.5 mm pitch; over the ~500 cm² covered scalp
  that is several thousand at full Nyquist. **1024** (Guasch) is the practical
  sub-Nyquist compromise (~5 mm pitch). Plan: derive `n` from frequency +
  covered area, default ~1024, allow downsampling for compute.

## What changes vs the current model

| | current `helmet_array_3d` | planned `clinical_helmet_3d` |
|---|---|---|
| elements | 256 (fixed) | ~1024, derived from λ/2 at f_max |
| azimuth | cap, `exclude_face` drops the whole anterior | full 360°, only a face *aperture* cutout |
| polar | ~160° generic | vertex→~140°, tied to skull base |
| conformity | generic ellipsoid | **project onto the real scalp surface** (MIDA/Birnbaum label) + standoff |
| src/recv | subset src, all recv | every element src+recv (1024²), src subsampled for compute |

## Implementation path (`transducers/helmet.py`)

1. `clinical_helmet_3d(freq, scalp_mask=None, dx=..., standoff=7e-3,
   face_aperture_deg=40, polar_max_deg=140, n_elements=None)`:
   - Fibonacci-sphere sample the ellipsoid; keep polar ≤ polar_max; remove the
     anterior-inferior face aperture (azimuth within ±face/2 AND below equator).
   - If `n_elements is None`, set it from λ/2 spacing at `freq` over the kept area.
2. **Conformal projection** (the real upgrade): when a `scalp_mask` (from the
   MIDA/Birnbaum head) is given, ray-cast each sampled direction from the head
   centroid to the outermost scalp voxel, then add `standoff` — so the helmet
   *hugs the patient's head* instead of a generic ellipsoid. Reuses the
   `make_helmet` water-snap already in `examples/06`.
3. **Coverage check** (validation): for a head phantom, count brain voxels with
   ≥K crossing transmitted ray paths (source–receiver chords through the voxel).
   Target near-uniform interior coverage — the metric the current cap fails on
   anteriorly. This becomes a robustness "perturber" too: vary element count /
   aperture → recon-quality curve (ties into the MAITE harness).

## Why it matters

The anterior transmission gap in the current cap is exactly where through-
transmission FWI loses interior illumination (cf. the 192³ periphery-ring: a
preconditioning *and* a coverage problem). A full-azimuth ~1024-element helmet
both matches the Imperial clinical target and should reduce the interior
illumination non-uniformity that the preconditioner has been fighting — i.e.
better hardware coverage and better-conditioned inversion, together.
