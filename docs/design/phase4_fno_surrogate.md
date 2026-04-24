# Phase 4: Neural-Operator Surrogate for j-Wave

Status: **design draft — v1**, 2026-04-24
Owner: Morgan Hough
Depends on: Phase 0 dataset (🟡 small-scale; scales to 10⁴+ to train the surrogate), Phase 1 SIREN (✅ optional input path), Phase 2 NPE (🟡 consumer), Phase 3 diffusion (🟡 consumer via DPS)
Blocks: Phase 0b at scale, Phase 3 DPS at population scale

This document specifies the neural-operator surrogate that replaces
j-Wave inside the inner loop for speed-sensitive consumers. It is the
economic lever: every other phase's cost profile assumes j-Wave is the
forward model, and j-Wave at 96³ costs ~2s per forward sim. A surrogate
at 100×–1000× speedup turns "barely feasible" Phase 3 DPS into
"routine", and reduces Phase 0b's 1100-A100-hour budget to something
affordable.

It sits after Phase 2/3 baselines exist because we want to measure
the surrogate's effect on *downstream quality* — NPE posterior
calibration, DPS regional RMSE — not just raw fidelity to j-Wave. A
surrogate that's 99% accurate on traces but catastrophically wrong on
NPE-relevant summary statistics is a net loss. The end-to-end
validation gate (§7.4) is the real bar.

---

## 1. Goals and non-goals

**Goals.**

- Approximate the j-Wave forward operator `c → d` such that downstream
  NPE (Phase 2) and DPS (Phase 3) consumers produce indistinguishable
  posteriors from their j-Wave-backed counterparts on held-out data.
- 100× speedup over j-Wave at 96³; 1000× stretch target at 128³+.
- Differentiable end-to-end (JAX-native) so DPS can use the surrogate's
  likelihood gradient directly without finite-difference approximation.
- Reusable across Phase 0b generation, Phase 2 real-time inference, and
  Phase 3 sampling — one surrogate, three consumers.
- Calibrated: trace-level error bounds known + distribution-level SBC
  equivalence verified.

**Non-goals (for Phase 4 v1).**

- Arbitrary transducer geometry. **V1 fixes source/receiver positions**
  to the canonical helmet from `transducers/helmet.py`; changing the
  helmet requires retraining. A geometry-conditional surrogate is
  Phase 4.5.
- Arbitrary media. V1 assumes the Phase-0 tissue-label distribution
  (BrainWeb + MIDA + augmentation). Out-of-distribution skulls
  (pediatric, pathological) get no accuracy guarantee.
- Attenuation / density variation. V1 treats ρ as fixed (same as j-Wave
  FWI today) and α as a fixed function of labels. Jointly varying
  (c, ρ, α) is Phase 5 territory.
- Time-varying source pulses. V1 uses the canonical Ricker wavelet.
  Pulse-conditional operators are a post-baseline extension.

---

## 2. What we learn

The forward operator at fixed geometry:

```
F_φ : ℝ^{Z×Y×X}  →  ℝ^{N_src × N_t × N_recv}
      sound speed    helmet traces
```

**Input.** Voxel-grid `c` normalised to `(c - c_min) / (c_max - c_min)`
for O(1) range. Density and attenuation are *implicit in the labels*
mapped to `c` — we don't learn the mapping separately because Phase 0
already draws `(c, ρ, α)` jointly and the label-to-property table is
deterministic. Out-of-distribution `c` (e.g., lesions not seen in
training) degrades gracefully.

**Output.** Full trace tensor at helmet-fixed sensor positions. Keeps
the surrogate a drop-in replacement for `simulate_shot_sensors`.
Bandpass filtering and summary statistics are computed downstream on
the trace output so the surrogate isn't tied to any particular NPE
d-summary method.

**Why voxel-grid input, not θ.** SIREN weights θ are not the natural
domain of the wave equation; the operator acts on `c(x)`. Training on
the decoded voxel grid keeps the surrogate agnostic to the
parameterisation choice. Consumers that live in θ-space (e.g., DPS)
apply `SIRENField.to_velocity(θ)` before calling the surrogate.

---

## 3. Architecture candidates

Ordered by our current expectation of fit / cost:

1. **FNO (Fourier Neural Operator; Li et al. 2020).**
   Mixes channels via point-wise MLP and spatial modes via truncated
   FFT. Natural fit for wave equations — the Fourier mixing roughly
   matches what a pseudospectral solver does each step. Proven on 3D
   PDE benchmarks. Parameter count depends on mode truncation; at 64
   modes per axis and 32 channels, ~5M params. **Baseline.**

2. **U-FNO / UNO (U-shaped FNO; You et al. 2022).**
   Multi-scale FNO with skip connections. Better on problems with
   structures at multiple scales — skull (thin ring), brain
   (continuous volume), helmet traces (fine time axis). Expected to
   help trace fidelity near sensor positions where spatial resolution
   matters.

3. **Transformer-operator.**
   Tokenise the input grid into patches, process with a transformer,
   decode to trace space. Flexible but expensive; likely too memory-
   hungry at 96³ unless we use linear attention variants. **Stretch.**

4. **Multi-grid / hierarchical FNO (MG-FNO; Liu-Schiaffini et al. 2023).**
   Explicit multiscale cascade. Best published results on 3D acoustic
   wave problems. **Post-baseline candidate.**

5. **Physics-informed approaches (PINNs, neural ODE).**
   Deprioritised. Ill-suited to production forward models — typically
   slower than FNO at comparable accuracy for wave equations.

**MVP architecture: classic 3D FNO with mode-truncation schedule 24/24/24,
channel width 32, 4 Fourier blocks.** Upgrade path to UNO or MG-FNO
gated on the §7.2 accuracy gate.

---

## 4. Training objective

Two terms, weighted:

```
L(φ) = E_{c ~ p_data} [
    ‖ F_φ(c) - F_{j-Wave}(c) ‖_rel_L2²            (trace fidelity)
  + λ_spec ‖ FFT(F_φ(c)) - FFT(F_{j-Wave}(c)) ‖_rel_L2²   (spectral loss)
]
```

where relative-L2 is normalised per-sample to prevent high-amplitude
samples dominating the loss. The spectral term penalises frequency-
content drift that the time-domain MSE can miss — particularly
important because FWI is sensitive to phase errors that look small in
time domain but sabotage the misfit.

`λ_spec = 0.3` as a starting point; tunable once we see the per-band
error distribution.

**Consumer-aware gradient loss (stretch).** If Phase 3 DPS is the
dominant consumer, add a gradient term matching `∂F_φ/∂c` to
`∂F_{j-Wave}/∂c` on held-out points. Expensive to evaluate during
training (second-order autodiff) but guarantees downstream DPS quality.
Keep gated until first DPS runs show measurable gradient-accuracy
degradation.

---

## 5. Data requirements

**Scale target.**

| Purpose | Sample count | Why |
|---|---|---|
| MVP training | 1,000 (c, d) pairs | Enough for a 5M-param FNO on 96³ |
| Production training | 10,000 (c, d) pairs | Matches the Phase-0 design-doc 10⁴ target; no extra cost |
| Held-out test | 500 pairs | Separate subjects (held-out at Phase 0 time) |

Phase-0 already produces `(c, d)` pairs — the surrogate trains on the
same dataset, **no separate generation needed**. This is a deliberate
alignment: Phase 0 pays the j-Wave cost once, Phase 4 amortises it
across many consumers.

**Data loader.** `ShardedReader` with `fields=["sound_speed_voxel",
"observed_data"]` gives `(c, d)` pairs without hauling θ or SIREN
weights through memory. Already implemented on main; just a new
consumer.

**Held-out discipline.** `manifest.json` reserves held-out `sample_id`s
before training starts. FNO-training code reads the reserved-list and
refuses to include those samples — same mechanism as Phase 0's
validation gate hold-out. Tests enforce.

---

## 6. Consumption modes

Each mode has its own accuracy tolerance:

### 6A. Phase 0b acceleration

Use `F_φ` in place of `simulate_shot_sensors` inside
`generate_observed_data` when generating the non-held-out samples of
Phase 0b.

- **Accuracy bar**: trace relative-L2 < 2% on held-out `c`. Error
  propagates into NPE training noise, but NPE is amortised — a bit
  of extra noise in `d` just regularises the flow.
- **Speedup target**: 100× reduces Phase 0b cost from ~1100 A100-hours
  to ~11 hours. Transformative.
- **Safety net**: keep j-Wave for the held-out split so validation
  never touches surrogate output. Phase 0b becomes mostly-surrogate
  with a small j-Wave "gold" subset.

### 6B. Phase 3 DPS likelihood

Replace j-Wave inside the per-reverse-step likelihood gradient of DPS.

- **Accuracy bar**: stricter. DPS gradient depends on `∂F/∂c`, so the
  surrogate's differentiability matters. Target: gradient cosine
  similarity > 0.95 vs j-Wave's gradient on held-out `c`.
- **Speedup target**: 100× turns a 100-sample posterior from
  ~5 GPU-hours to ~3 GPU-minutes. Moves DPS from "per-subject
  feasible" to "routine".
- **Failure mode**: gradient-accuracy failure is subtle — DPS
  trajectories drift, posterior shape distorts, SBC catches it but
  only after expensive sampling. Gate 7.3 catches this upstream.

### 6C. Real-time FWI loop

Use `F_φ` in `run_fwi` for fast-iteration workflows (interactive
clinical use, debugging).

- **Accuracy bar**: good enough to converge to a reconstruction close
  to j-Wave FWI's; final refinement with 5-10 j-Wave iterations.
- **Speedup target**: 100× — 192³ FWI from hours to minutes.

### 6D. Monte Carlo UQ

Marginalise uncertainty sources (source-position jitter, density
miscalibration, noise) by Monte Carlo sampling over the surrogate.

- **Accuracy bar**: statistically unbiased across the sampling
  distribution, not per-sample accurate.

---

## 7. Validation plan

Four gates, in ascending importance:

1. **Unit-test gate.** Surrogate wraps cleanly in a module with
   consistent input/output shapes, loads/saves weights, runs on a
   tiny toy problem. Standard unit-test coverage. Blocks PR merge.

2. **Trace-level accuracy gate.** On held-out `c` samples:
   - Median relative-L2 per trace < **1%**
   - 95th-percentile relative-L2 per trace < **5%**
   - No systematic bias in frequency content (spectral ratio within
     ±10% across all bands).

3. **Gradient-accuracy gate** (if 6B is on the roadmap).
   - Cosine similarity of `∂F_φ/∂c` and `∂F_{j-Wave}/∂c` > 0.95
     on held-out `c`, averaged over all output elements.
   - Catches latent failures that trace-level accuracy misses.

4. **Downstream gate — the real bar.**
   - Surrogate-accelerated Phase 0 + NPE reproduces the SBC p-value
     of j-Wave-backed Phase 0 + NPE within ±0.1 on 500 held-out pairs.
   - Surrogate DPS posterior samples produce regional RMSE within
     ±10% of j-Wave DPS on the same held-out subjects.
   - Blocks any clinical claim.

Gates 1–3 are engineering; gate 4 is the scientific bar. Don't ship
unless 4 passes.

---

## 8. Open decisions

- **Input normalisation.** `(c - c_min) / (c_max - c_min)` is the
  cheap default. Richer options: label-one-hot input (skips the
  float-vs-tissue ambiguity), or a learned embedding of the tissue
  label. The one-hot variant adds robustness to out-of-distribution
  sound speeds that share labels with known tissues.
- **Fourier mode truncation.** 24 modes per axis is a guess. Wave
  equation has broadband content — high-frequency skull reflections
  matter. If truncating kills trace fidelity near skull, go to 32
  or 48 modes at ~4× parameter cost.
- **Multi-physics extension.** Current plan: fix ρ, α as label
  functions. Worth exploring `(c, ρ, α)` joint input for very early
  consumers? Probably not until Phase 5; the marginal
  accuracy-at-extra-cost trade is almost certainly worse than
  investing the same training budget in a better `c`-only surrogate.
- **Online vs offline training.** Pre-train once on Phase-0b, or
  continue training as Phase 0b generates new samples? Online has
  appeal (surrogate gets better as data grows) but adds training-
  infrastructure complexity. V1: offline, Phase-0b-fixed.
- **Patch-based training for 192³.** 96³ fits comfortably; 192³ may
  not on a single A100. UNO's multi-scale structure naturally shards,
  classic FNO needs explicit spatial patching with overlap. Open
  until we measure memory at 192³.

---

## 9. Evidence wishlist

Before committing to MVP implementation, data I want to see. Each
item is either produced by existing in-flight work or cheap to run:

### 9.1. Hard prerequisites

- **Modal forward-sim benchmark (#10 output).** Without real per-call
  cost numbers across grid sizes, "100× speedup" is a guess. This
  tells us the baseline we're beating and confirms Phase 4 is
  economically justified.
- **DGX SIREN-vs-voxel validation (#11 outcome).** Confirms that the
  voxel `c` is the right surrogate input (vs learning `F(θ)` directly).
  If SIREN reconstructions underperform voxel on real data, the
  surrogate-on-voxel assumption becomes more important.

### 9.2. Pre-MVP experiments (1–2 days each)

- **Toy 2D FNO on acoustic wave.** 32² grid, single-source single-
  receiver, 500 training samples. Measures the achievable relative-L2
  on our simplest problem and tells us whether the MVP architecture
  is in the right ballpark. If a 2D FNO can't hit 1% relative-L2 on
  a toy problem, something's wrong before we scale.
- **Phase-0 NPE noise sensitivity study.** Take the existing Phase-0
  pipeline and inject Gaussian noise of varying amplitude into `d`
  before training NPE. Measure at what noise level the posterior
  calibration degrades. This sets the real accuracy bar for 6A — if
  NPE tolerates 5% trace noise, our surrogate target tightens to
  <3% to leave headroom.
- **Gradient-accuracy sensitivity.** If 6B is a serious goal, measure
  how DPS posteriors degrade as a function of synthetic gradient
  error injected into j-Wave's backward pass. Without this we can't
  set gate 7.3's threshold rigorously — 0.95 cosine similarity is a
  literature-informed guess.

### 9.3. Literature anchor

- **State-of-the-art 3D FNO numbers on acoustic wave problems.**
  Pathak 2022 (spherical-FNO on weather), Kovachki 2023 (FNO survey),
  Azizzadenesheli 2024 (neural-op review). Know the published
  accuracy-speedup Pareto before picking our target.

---

## 10. Workstream kickoff

Once this design is accepted, tracked implementation steps:

1. **Toy 2D FNO prototype** (§9.2 experiment 1). New module
   `src/brain_fwi/surrogate/fno2d.py` + unit tests on a synthetic
   wave dataset. Blocks everything; if 2D FNO can't hit 1%, we
   redesign before 3D.
2. **3D FNO module**. `src/brain_fwi/surrogate/fno3d.py` with Fourier
   blocks, tests at 32³ on homogeneous water.
3. **Training loop** using `build_c_d_matrix` (to be written,
   analogue of `build_theta_d_matrix`). Reuses `ShardedReader`,
   `eqx.filter_value_and_grad`.
4. **Validation harness**. Regional RMSE per-tissue on traces, plus
   full gate 7.2 stats. Reuses `brain_fwi.validation.compare`
   pattern.
5. **Modal runner**. `scripts/modal_train_fno.py` on A100, ~2h run
   time budget at 96³ with 5M params.
6. **Phase 0b integration** (gate 6A). Swap the forward sim call in
   `gen_phase0.py` behind a flag; validate the downstream NPE
   produces the same SBC stats within ±0.1.
7. **DPS integration** (gate 6B, after Phase 3 baseline exists).

Each step lands as its own PR with its own failing test. No
surrogate code lands on main without gate 7.1 passing; no
"surrogate-enabled default" anywhere without gate 7.4 passing.

---

## 11. What this does NOT do

- Does not claim to replace j-Wave. The surrogate is a *consumer-
  specific* approximation; j-Wave remains the ground truth for
  validation, final refinement, and any clinical decision pathway.
- Does not model variable helmet geometry. Geometry-conditional
  operators are a post-baseline extension.
- Does not produce uncertainty over the forward operator itself.
  Heteroscedastic / Bayesian neural operators are a future phase.
- Does not guarantee extrapolation outside the Phase-0 tissue
  distribution. Out-of-distribution skulls (pediatric, pathological
  thinning, implants) need dedicated validation or retraining.

Everything else is deliberately in.
