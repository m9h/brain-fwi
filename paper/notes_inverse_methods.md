# Research Notes: Inverse Methods for Brain USCT Paper

## Narrative Arc

The paper introduction tells this story as a natural progression:

1. **Ill-posed problems require regularization** (Hadamard → Tikhonov → Backus-Gilbert)
2. **Every imaging modality faces the same mathematical challenge** (EEG → EIT → DOT → USCT)
3. **FWI brought wave-equation inversion to geophysics** (Lailly → Tarantola → Pratt), then medicine (Guasch 2020)
4. **Automatic differentiation eliminates the adjoint bottleneck** (JAX, j-Wave)
5. **Our contribution**: fully differentiable JAX FWI for transcranial brain USCT

The unifying equation:

    m_est = argmin_m { ||G(m) - d||² + λ R(m) }

---

## 1. Pseudoinverse and Birth of Inverse Methods

- **Hadamard (1902)** — well-posedness: existence, uniqueness, stability
- **Moore (1920), Penrose (1955)** — pseudoinverse A⁺ (minimum-norm least-squares)
- **Phillips (1962)** — first numerical regularization for integral equations
- **Tikhonov (1963)** — general regularization theory: m_λ = (G^T G + λI)⁻¹ G^T d
- **Backus & Gilbert (1968)** — resolution matrix R = (G^T G + λI)⁻¹ G^T G
  - Each row is an averaging kernel; FWHM = spatial resolution
  - Fundamental trade-off: resolution vs variance amplification
- **Tikhonov & Arsenin (1977)** — standard monograph

Connection: our `resolution.py` computes exactly this R for the USCT helmet geometry.

## 2. EEG Source Imaging (ESI)

Forward: quasi-static Poisson equation ∇·(σ∇Φ) = -Iv
Inverse: d = Js (lead field J, source amplitudes s)

- **Hämäläinen & Ilmoniemi (1994)** — minimum norm estimate (MNE): s = J^T(JJ^T + λI)⁻¹d
- **Dale & Sereno (1993)** — anatomically-constrained distributed inverse
- **Pascual-Marqui (1994)** — LORETA (Laplacian smoothness constraint)
- **Pascual-Marqui (2002)** — sLORETA (standardized by resolution matrix diagonal)
  - s_sLORETA(i) = s_MNE(i) / √R(i,i) → zero localization error for test dipoles

**Structural analogy EEG ↔ USCT:**

| | EEG | USCT |
|---|---|---|
| Jacobian | J = lead field (∂V/∂s) | J = ∂p/∂c |
| Unknown | source amplitude s | sound speed c(x) |
| Data | electrode voltages | sensor time series |
| Geometry | electrodes on scalp | transducers on helmet |
| Depth bias | Yes (1/r decay) | Different (wave coverage) |

## 3. Electrical Impedance Tomography (EIT)

Forward: ∇·(σ∇u) = 0, recover σ(x) from boundary V

- **Calderón (1980)** — uniqueness of inverse conductivity problem
- **Sylvester & Uhlmann (1987)** — global uniqueness in 3D
- **Cheney, Isaacson & Newell (1999)** — SIAM Review survey
- **Adler & Lionheart (2006)** — EIDORS open-source toolkit
- **D-bar methods** (Siltanen et al. 2000) — direct reconstruction, no iteration

EIT Gauss-Newton: δσ = (J^T J + λR)⁻¹ J^T (d_meas - d_pred) — identical to FWI step.

## 4. Diffuse Optical Tomography (DOT) as Bridge

Forward: -∇·(D∇Φ) + μₐΦ = S (photon diffusion)

- **Arridge (1999)** — comprehensive review, Inverse Problems 15:R41-R93
- Adjoint Jacobian: J_{d,s,n} = -Φ_s(n)·Φ_d(n)·V_n
  - Same structure as acoustic "banana-doughnut" kernel
- dot-jax implements this: forward.py, spectral.py, recon.py
- Kernel Flow (Ban et al. 2022) = hardware parallel to US helmet

DOT bridges EIT (same boundary measurement geometry) and USCT (same adjoint Jacobian structure).

## 5. Full Waveform Inversion Heritage

- **Lailly (1983)** — adjoint state = seismic migration
- **Tarantola (1984)** — FWI formulation, adjoint gradient with 2 simulations/source
  - ∂L/∂m(x) = ∫ [∂²p/∂t²] λ(x,t) dt (forward × adjoint cross-correlation)
- **Pratt (1999)** — frequency-domain FWI, natural multi-scale
- **Virieux & Operto (2009)** — definitive review
- **Guasch et al. (2020)** — first brain USCT via FWI, npj Digital Medicine

Cycle-skipping mitigations (all in brain-fwi):
1. Multi-frequency banding (`FWIConfig.freq_bands`)
2. Envelope loss (`envelope_loss()`) — Hilbert envelope, convex basin
3. Multiscale loss combination
4. Optimal transport (Wasserstein) distance

## 6. SCICO and Proximal Splitting

- **SCICO** — LANL, Balke et al. (2022) JOSS, JAX-based
- **Rudin-Osher-Fatemi (1992)** — Total Variation regularization
- **Chambolle & Pock (2011)** — PDHG algorithm
- **Boyd et al. (2011)** — ADMM survey
- **Venkatakrishnan et al. (2013)** — Plug-and-Play priors (denoisers as proximal operators)

ADMM splits physics (x-update) from regularization (z-update via proximal).
TV preserves edges (skull boundary) better than gradient smoothing.
PnP could use brain anatomy denoisers trained on MRI as implicit priors.

## 7. PINNs and Neural Differential Equations

- **Raissi et al. (2019)** — PINNs: L = L_data + λ L_PDE
- **Chen et al. (2018)** — Neural ODEs (NeurIPS Best Paper)
  - Adjoint sensitivity method = Tarantola's adjoint = backprop
- **Rackauckas et al. (2020)** — Universal Differential Equations
- **Baydin et al. (2018)** — AD survey: reverse-mode AD IS the adjoint method

JAX differentiable physics ecosystem:
- j-Wave (acoustics), JAX-MD (molecular), JAX-Fluids (CFD), Diffrax (ODE/SDE)
- dot-jax (DOT), brain-fwi (USCT)

brain-fwi is NOT a PINN — it solves the PDE exactly via j-Wave, with JAX autodiff for gradients.
PINNs/neural operators could serve as learned initial models or learned regularizers.

## 8. The Unifying Thread

**Computational evolution:**

| Era | Method | Gradient computation |
|-----|--------|---------------------|
| 1920-1970 | Pseudoinverse | Explicit matrix (SVD) |
| 1983-2015 | Adjoint-state | Hand-coded adjoint per physics |
| 2015-present | Autodiff | jax.grad through any forward solver |
| 2019-present | Learned | Backprop through neural operators |

**All share:** forward operator G, Jacobian J = ∂G/∂m, resolution R = (J^T J + λI)⁻¹ J^T J

---

## Key References (by section)

### Foundations
- Hadamard (1902) Princeton University Bulletin
- Penrose (1955) Math Proc Cambridge Phil Soc 51:406
- Tikhonov (1963) Soviet Math Doklady 4:1035
- Backus & Gilbert (1968) Geophys J R Astr Soc 16:169

### EEG
- Hämäläinen & Ilmoniemi (1994) Med Biol Eng Comput 32:35
- Pascual-Marqui (2002) Methods Find Exp Clin Pharmacol 24D:5
- Grech et al. (2008) J NeuroEngineering Rehab 5:25

### EIT
- Calderón (1980) Seminar on Numerical Analysis, Rio
- Cheney et al. (1999) SIAM Review 41:85
- Adler & Lionheart (2006) Physiol Meas 27:S25

### DOT
- Arridge (1999) Inverse Problems 15:R41
- Boas et al. (2001) IEEE Signal Proc Mag 18:57

### FWI
- Tarantola (1984) Geophysics 49:1259
- Virieux & Operto (2009) Geophysics 74:WCC127
- Guasch et al. (2020) npj Digital Medicine 3:28
- Stanziola et al. (2023) SoftwareX 22:101338 (j-Wave)

### Optimization
- Rudin, Osher & Fatemi (1992) Physica D 60:259
- Chambolle & Pock (2011) J Math Imaging Vision 40:120
- Balke et al. (2022) JOSS (SCICO)

### PINNs/NDE
- Raissi et al. (2019) J Comput Phys 378:686
- Chen et al. (2018) NeurIPS
- Baydin et al. (2018) JMLR 18:1
