# Phase 4 Production: High-Fidelity Neural Operator Surrogate

Status: **Production Draft - v1**
Date: 2026-05-08
Focus: 128³ Resolution, UNO Architecture, and ITRUSST Physics Alignment

## 1. Executive Summary
This document formalizes the transition from experimental architecture search (AgenticSciML) to production implementation for the Brain-FWI surrogate model. Based on results from 32 generations of automated discovery, we are moving to a "Thinner Brush" approach (128³ grid) using a U-shaped Neural Operator (UNO) to break the 52 m/s RMSE plateau.

## 2. Core Specifications
### 2.1 Resolution & Grid
- **Voxel Resolution:** 1.0mm (Targeting cortical bone thickness)
- **Grid Size:** 128³
- **Domain Size:** 128mm³ (Preserved via dx=0.001m)
- **CFL Constraint:** dt must be recalculated for 128³ to ensure stability at max velocity (3200 m/s).

### 2.2 Physical Priors (ITRUSST BM3)
All future training data and simulations must align with the ITRUSST transcranial benchmark:
- **Cortical Bone:** c=2800 m/s, rho=1850 kg/m3, alpha=4.0 dB/cm/MHz
- **Trabecular Bone:** c=2300 m/s, rho=1700 kg/m3, alpha=8.0 dB/cm/MHz
- **Brain Tissue:** c=1560 m/s, rho=1040 kg/m3, alpha=0.6 dB/cm/MHz
- **Anatomy:** Forced trilayer skull model (Outer/Diploe/Inner) to prevent 10-15mm feature shifts.

## 3. Architecture: U-shaped Neural Operator (UNO)
The "Classic FNO" is retired in favor of the `UNONet` implementation:
- **Backbone:** Multi-scale encoder/decoder with spectral convolutions.
- **Skip Connections:** Concatenative skips to preserve high-frequency interface details.
- **Activation:** SiLU/Sine (SIREN-inspired) to handle sharp skull gradients.
  Pinned by `tests/test_uno.py::test_uno_default_activation_is_silu`.
- **Readout head — point-sample at receiver coords, NOT global pool.**
  The earlier global-average-pool readout collapsed the entire
  `(hidden, D, H, W)` feature volume to `hidden_channels` numbers
  before fanning back out to `(N_t, N_recv)` outputs, capping the
  expressible trace patterns at `~hidden²` and forcing the model to
  predict-zero everywhere as the L2-optimal solution. The current
  readout reads `features[:, rx, ry, rz]` per receiver and runs a
  shared MLP per receiver. Pinned by
  `tests/test_phase4_fno_pipeline.py::TestPointSampleHead`. **Do not
  revert to global pool at any resolution — the collapse reproduces
  identically at 128³.**
- **Optimization:** Gradient Accumulation (Batch Size >= 8) is required to stabilize the spectral loss.

## 4. Accelerated Execution (Modal)
- **Hardware:** Dispatched to `A100:4` (4x GPU cluster).
- **Parallelization:** JAX `vmap` and `DeviceMesh` used to parallelize ultrasound shots across devices.
- **Timeout:** 3 hours per generation allowed for deep convergence.

## 5. Closed-Loop Strategy (Track B)
The Direct FWI swarm outputs will now be used as **Active Learning** targets:
1.  Identify p95 failure cases in FNO trace-fidelity.
2.  Task the FWI engine with generating 128³ "Extreme Skull" phantoms.
3.  Weight these cases 3x in the FNO training loss to break the 1% error barrier.

## 6. Success Metric
- **Target:** < 1% relative L2 error on held-out MIDA traces.
- **Gate:** Surrogate must be accurate enough to replace the `j-Wave` forward solver in downstream NPE/DPS tasks.
