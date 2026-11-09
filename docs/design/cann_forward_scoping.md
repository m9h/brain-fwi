# Scoping: a per-voxel α(ω) forward for single-stage CANN FWI

Goal: invert **CANN constitutive parameters** (per-voxel/per-tissue α(ω), and
eventually direction-dependent α(ω,θ)) directly, in a single-stage FWI — the full
realisation of the Living Matter Lab method in imaging (gaps #2/#3 of
`cann_constitutive_discovery.md`). This doc scopes the forward-model change,
grounded in the actual absorber code.

## The precise blocker

The Treeby–Cox power-law absorber (`~/dev/jwave/jwave/acoustics/time_varying.py::
absorbing_pressure_from_density`) folds the exponent **y = `medium.alpha_power`
(a single scalar, line 364)** into two **fractional Laplacians**, evaluated as a
global FFT multiply (lines 396–400):

```
nabla1 = |k|^(y-2)   # absorption operator
nabla2 = |k|^(y-1)   # dispersion operator
absorb = tau * IFFT(FFT(rho0_div_u) * nabla1)
disp   = eta * IFFT(FFT(rho_sum)   * nabla2)
```

- **Per-voxel α *magnitude* is already supported** — it enters via the fields
  `tau = -2 α₀ c^{y-1}`, `eta = 2 α₀ c^y tan(πy/2)` (lines 381–382), with `α₀` the
  per-voxel attenuation field. This is what Phase 6 inverts.
- **Per-voxel α *shape* is NOT.** The exponent is the *order* of the fractional
  Laplacian; a spatially-varying `y(x)` (or a full CANN α(ω) per voxel) makes it a
  variable-order pseudodifferential operator that **does not diagonalise in
  Fourier** — you cannot FFT-multiply by `|k|^(y(x)-2)`. Same obstruction, worse,
  for **anisotropy** (direction-dependent loss).

So the CANN's *magnitude* is reachable today; its *spectral shape* and
*anisotropy* need a different absorber. The staged **FWI→CANN bridge** (already
built) sidesteps this by using a per-band effective `y`; single-stage needs one
of the routes below.

## Three routes

### A. Multi-relaxation (generalized-Maxwell / Prony) absorber — recommended
Replace the single fractional power law with a **sum of relaxation mechanisms**.
Each mechanism `m` carries a per-voxel modulus `g_m(x)` and a fixed relaxation
time `τ_m`, with an auxiliary memory field evolving by a **local first-order ODE**
(`∂_t ξ_m = -ξ_m/τ_m + …`). No fractional Laplacian ⇒ **no FFT-order constraint**
⇒ per-voxel spectrum is natural and cheap; fully differentiable (the ODEs are
just extra state in the JAX scan).

Why this is the *right* route, not just a workaround: a generalized-Maxwell/Prony
spectrum is exactly the time-domain image of the **iCANN dissipation potential**
(Holthusen et al. 2024) and of our **Debye/relaxation basis** already in
`constitutive/discovery.py`. The per-voxel relaxation moduli `g_m(x)` **are** the
CANN's non-negative outer weights. Discovering how many mechanisms and their
weights *is* Kuhl's model discovery — now inside the solver. A few (2–4)
mechanisms approximate tissue α(ω) over the 50–300 kHz band to well within FWI
error (standard result in viscoelastic wave modelling).

- **Cost:** `N_mech` auxiliary fields (memory ↑ by ~N_mech×) and a local update
  per step (cheap vs the FFTs). Manageable at 96³–192³ with the checkpointed scan.
- **Measured result (2026-07-22, milestone 1 done — `relaxation.py`,
  `test_relaxation_spectrum.py`).** Fitting the relaxation spectrum with **both**
  moduli and relaxation times optimised (nonlinear Adam, bounded in-band freqs):
  **constant-Q (α ∝ f, y = 1) fits to <3 %**, but a **y = 1.3 power law only to
  ~9 %** over the 50–300 kHz band — and this does **not** improve with more
  mechanisms, more steps, or a relative-error loss (the mechanisms flee to the
  band edges). It is not a fitting bug: a Debye relaxation spectrum is naturally
  constant-Q, and the **narrow transcranial band (~half a decade, forced sub-MHz
  by skull attenuation) gives too little frequency leverage to pin an exponent
  ≠ 1.** Implications: (i) a Prony absorber is an excellent *constant-Q* tissue
  model (a defensible approximation); (ii) representing — let alone *inverting* —
  a per-tissue y ≠ 1 **spectral shape** over this band is leverage-limited, which
  **reinforces that GM/WM discrimination must lean on magnitude and ANISOTROPY
  (angular leverage), not spectral shape over frequency.** The forward change
  buys physically-correct broadband α and the *anisotropic* extension — not a
  spectral-shape GM/WM separator (the narrow band cannot supply one).
- **Causality:** each relaxation mechanism has an analytic Kramers–Kronig
  dispersion partner — causality is structural (no penalty needed), unlike the
  fractional form's explicit `disp` term.
- **Risk:** implementing + k-Wave-validating a new absorber in the jwave fork
  (the fractional one took real effort — see `project_absorption_treeby_cox`).

### B. Staged per-band bridge — DONE (fallback / today's capability)
Keep the fractional absorber, invert per band, discover α(ω) via
`discover_tissue_alpha_law`. Two-stage; per-band `y` is a band-effective scalar,
not per-voxel. Good enough to *characterise* tissue α(ω) and rank GM/WM now.

### C. Frequency-domain (Helmholtz) FWI
Complex wavenumber `k(x,ω) = ω/c(x) + i·α(x,ω)` makes per-voxel α(ω) trivial and
each frequency independent. But it is a **different solver** (jaxdf has some
Helmholtz support) and a large architectural change; broadband transcranial FWI
would need many frequencies. Park unless a narrowband CW study motivates it.

## Anisotropy (the actual GM/WM shape discriminator)

The isotropic absorber (A or the current one) cannot represent direction-dependent
loss. Two options, mirroring the routes above:
- **Staged (tractable now):** multi-angle acquisition → per-angle regional α
  estimates → the **anisotropic CANN discovery already built**
  (`alpha_basis_library_anisotropic`). Extend the bridge to assemble α(ω,θ) across
  source–receiver angles. This resolves WM's `sin²θ` term without an anisotropic
  solver — recommended first.
- **Single-stage (research):** direction-dependent relaxation moduli
  `g_m(x, n̂·fibre)` in route A — a genuinely anisotropic absorber. Hard; defer.

## Recommended plan (staged, each a shippable increment)

1. **Fit the relaxation spectrum first (constitutive, no solver).** **DONE** —
   `fit_relaxation_spectrum` (moduli + times, nonlinear). Result: constant-Q <3%,
   y = 1.3 ~9% (band-leverage-limited; see the measured result above). Verdict:
   proceed with a **constant-Q Prony absorber** as the tissue model; do **not**
   rely on spectral shape for GM/WM.
2. **Prototype the multi-relaxation absorber** (route A) as a brain-fwi forward
   variant (not yet the jwave fork): per-voxel moduli, the milestone-1 times.
   Validate vs the fractional absorber and k-Wave (reuse
   `scripts/kwave_absorption_xcheck.py`).
3. **Make the moduli invertible** — wire the per-voxel relaxation moduli into
   `FWIConfig`/`run_fwi` as an inverted parameter group (they compose with the
   existing `{c, a}` pytree). Single-stage CANN FWI on synthetic multi-tissue.
4. **CANN sparsity in the loop** — L0 across mechanisms so the inversion
   *discovers* the minimal per-tissue spectrum (Kuhl in the solver).
5. **Multi-angle anisotropy (staged, B/anisotropy)** — extend the bridge to
   α(ω,θ); demonstrate WM `sin²θ` recovery on synthetic fibre-oriented tissue.
6. **Upstream** the multi-relaxation absorber to the jwave fork + k-Wave validate;
   scale to 96³/192³.

Milestones 1–5 live in brain-fwi (no fork change, lower risk); 6 is the fork.
Route A is the through-line; B is already in hand for characterisation today.

## Honest risks

- A new time-domain absorber is the same class of effort as the fractional one we
  already validated — budget for careful k-Wave cross-checking (bit-for-bit was
  hard-won). CFL/stability of stiff relaxation times needs care.
- Even with per-voxel spectra, GM/WM *isotropic* discrimination stays magnitude-
  limited (Kuhl); the real payoff of route A is (i) physically-correct broadband
  α(ω) and (ii) enabling the anisotropic single-stage extension. Manage the
  expectation: the forward change is necessary for the *shape* discriminator, not
  a magic separator by itself.
