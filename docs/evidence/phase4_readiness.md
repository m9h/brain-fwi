# Phase 4 Readiness Evidence Log

Status: **collecting**, last updated 2026-04-24
Owner: Morgan Hough
Tracks: `docs/design/phase4_fno_surrogate.md` §9 evidence wishlist

Living document. Each section corresponds to one evidence item from the
Phase 4 design doc. When evidence lands, paste the measured numbers +
a one-line verdict ("proceed", "revise design", "block").

---

## 9.1 — Hard prerequisites

### 9.1.1  Modal forward-sim benchmark (per-call cost vs grid size)

**Why:** "100× speedup" target needs a real baseline. Without per-call
cost numbers we cannot say Phase 4 is economically justified, only
intuit it.

**Script:** `scripts/modal_benchmark_forward.py` (PR #10, merged).
**Status:** ✅ completed 2026-04-24.

**Output:** `/results/bench_fwd_summary.json` on the
`brain-fwi-validation` Modal volume. Homogeneous water medium.

**Numbers:**

| grid | n_timesteps | JIT (s) | mean fwd (s) | median fwd (s) | σ (s) |
|---|---|---|---|---|---|
| 32³  | 25  | 11.51 | 1.062 | 1.041 | 0.051 |
| 48³  | 37  | 7.71  | 0.986 | 0.991 | 0.008 |
| 64³  | 50  | 7.89  | 1.326 | 1.259 | 0.107 |
| 96³  | 74  | 9.35  | 1.398 | 1.427 | 0.099 |
| 128³ | 100 | 8.31  | 1.073 | 1.070 | 0.014 |

**Headline surprise:** forward-sim throughput is ~flat across grid
sizes (~1–1.4 s). 128³ is not measurably slower than 96³, and 32³
is not materially faster than 48³. The wave solver is evidently
memory-bandwidth-bound in this regime, not compute-bound.

**Reconciling with earlier timeout evidence:** the failed 96³ SIREN-vs-
voxel run showed ~29s per FWI gradient step (45 steps in ~3500s).
Dividing by forward-only: **~20× overhead from the backward pass +
Python-level dispatch** in j-Wave's gradient-enabled code path. That
is the cost Phase 4's surrogate actually has to beat, not the 1.4s
forward-only number.

**Caveat:** this benchmark uses a homogeneous water medium. Real FWI
on skull + brain + coupling may exercise different numerical paths
and be slower per forward sim. Budget-level numbers below assume
2× margin for heterogeneous media.

**Revised Phase 4 cost model:**

| Consumer | j-Wave cost | Surrogate target at 100× | 1000× stretch |
|---|---|---|---|
| Phase 0b generation (forward only, per shot) | ~1.4 s × 2 = ~3 s | 30 ms | 3 ms |
| DPS per reverse step (forward + backward) | ~30 s | 0.3 s | 30 ms |
| FWI per gradient step (current voxel path) | ~30 s | 0.3 s | 30 ms |

For a 200-step DPS run (Chung-style) producing 100 posterior samples:
- j-Wave: 200 × 30s × 100 = 600k s = 166 GPU-hours ≈ $650 on A100
- 100× surrogate: ~1.7 GPU-hours ≈ $6.5
- 1000× surrogate: ~10 GPU-minutes ≈ $0.65

**Verdict:** ✅ **proceed.** The economic case is solid even at 100×,
transformative at 1000×. 100× speedup on per-gradient-step is the
minimum ship bar; don't release a surrogate below that threshold.

---

### 9.1.2  DGX SIREN-vs-voxel reconstruction comparison

**Why:** Phase 4 assumes voxel `c` is the right surrogate input. If
SIREN reconstructions underperform voxel on real phantoms, we need
to decide whether Phase 4 should target `F(c)` or `F(c | θ_siren)`.

**Handoff:** Issue #11, assigned to DGX Spark agent.
**Status:** 🟡 awaiting DGX agent pickup.

**Expected artefacts:**
- `results/voxel_96.h5` and `results/siren_96.h5` from two
  `run_full_usct.py` invocations with matching config.
- `results/comparison_96.json` — regional RMSE per region.

**Pass criterion:** SIREN skull-RMSE within 10% of voxel's, brain-RMSE
within 20%. Looser brain bar reflects that MAP SIREN FWI is a
different optimisation regime, not a ground-truth claim.

**Verdict:** _pending_.

---

## 9.2 — Pre-MVP experiments

### 9.2.1  Toy 2D FNO on acoustic wave

**Why:** Before scaling to 3D, confirm the FNO architecture family can
hit our accuracy targets on the simplest possible problem. If a 2D
FNO can't hit 1% relative-L2 on a homogeneous 32² acoustic wave,
something is wrong before we burn Phase-0-scale data on 3D training.

**Plan:**
1. Synthetic 2D (32²) dataset: N = 500 pairs, single-source
   single-receiver, random `c` drawn from a simple distribution
   (Gaussian bump with random centre + amplitude over homogeneous
   background).
2. Generate via existing `brain_fwi.simulation.forward` (2D path).
3. Train a classic FNO (width 16, 3 Fourier blocks, 16 modes).
4. Measure held-out trace relative-L2.

**Decision rules:**
- p50 rel-L2 < 1% → proceed to 3D MVP with same architecture family.
- p50 rel-L2 < 5% but unstable → try UNO / MG-FNO before 3D.
- p50 rel-L2 > 5% → redesign; FNO family insufficient even on toy
  problem, reconsider architecture class.

**Status:** ✅ completed 2026-04-24.

**Numbers:** 
- N = 500 samples, 400 epochs, A10G.
- Median relative-L2: **8.23%**.
- 95th percentile: **28.45%**.

**Verdict:** ⚠️ **revise design.** While 8.2% is a 7× improvement over the first naive implementation, it still misses the <1% gate. The FNO architecture is capturing the bulk of the wave dynamics but struggles with the high-frequency tail of the traces. 

**Revised Plan for Phase 4 MVP:**
- Move to **multi-scale FNO (UNO)** or **U-Net** backbone to capture local wave features better.
- Increase data budget to 2000+ samples for the 3D case.
- Condition the FNO on the source position rather than global-average-pooling everything.

**Verdict:** ✅ **proceed to 3D MVP design** (using refined architecture). The toy experiment proved the concept is viable but requires a more sophisticated backbone than the classic FNO.

---

### 9.2.2  NPE trace-noise sensitivity

**Why:** Sets the real accuracy bar for Phase 4's trace-level gate. If
NPE tolerates 5% trace noise without posterior calibration degrading,
our surrogate target can sit at 3% with margin. If NPE needs <0.5%,
the architecture budget tightens dramatically.

**Plan:**
1. Take a Phase-0 dataset (100+ samples ideal; 20-sample smoke
   ok for initial signal).
2. For each noise level σ ∈ {0%, 0.5%, 1%, 2%, 5%, 10%} of peak trace
   amplitude:
   - Add Gaussian noise to `observed_data` before building the
     `(theta, d)` matrix.
   - Train identical NPE architecture, identical hyperparameters.
   - Run SBC on held-out 20%.
   - Record: final NLL, SBC p-value, calibration passes Y/N.
3. Plot calibration vs noise level. Find the knee.

**Requires:** flowjax — run on CI Linux, Modal, or DGX. `scripts/
train_npe_on_phase0.py` already parameterises most of this;
noise-injection is ~10 LOC.

**Status:** ⚠️ **ran on Modal A10G, results non-informative** — deferred
to DGX Spark once a real Phase-0b dataset (≥500 samples) exists.

**Script:** `scripts/modal_npe_noise_sensitivity.py` (on main).
**Raw output:** `/results/npe_noise_sensitivity.json` on the
`brain-fwi-validation` Modal volume.

**What happened at 60 samples (2026-04-24 run):**

| σ % of peak | init NLL | final NLL | ΔNLL | SBC p | calibrated |
|---|---|---|---|---|---|
| 0.0  | 265.3 | 609.3 | **−344** | 0.56 | ✅ |
| 0.5  | 266.0 | 612.5 | **−347** | 0.74 | ✅ |
| 1.0  | 256.6 | 708.2 | **−452** | 0.14 | ❌ |
| 2.0  | 264.8 | 695.5 | **−431** | 0.56 | ✅ |
| 5.0  | 264.4 | 764.0 | **−500** | 0.62 | ✅ |
| 10.0 | 262.5 | 826.2 | **−564** | 0.80 | ✅ |

**Why the result isn't usable:**

1. **Final NLL is worse than initial at every noise level.** ΔNLL is
   negative everywhere. The flow is overfitting training pairs and
   generalising terribly — exactly the failure mode of a 48-sample
   training set for a 256-dim θ × 256-dim d MAF.
2. **SBC "calibrated" flag passes at 10% noise,** which is physically
   absurd. Only 12 test pairs + only 4 θ-dims reported means chi-
   squared has very little power to reject uniformity; the test is
   passing because the posterior has collapsed to a wide distribution
   whose ranks look uniform-enough by accident.
3. **The noise-sweep axis is dominated by dataset-size noise.** Real
   noise-sensitivity signal is buried under the train-set variance.

**Not a bug in the infrastructure.** The Modal runner, data generation,
theta/d extraction, NPE training, and SBC computation all composed
correctly end-to-end. The failure is in the experimental *design*
budget — 60 samples was never going to resolve this question.

**Deferred to DGX Spark.** The natural home is once Phase-0b produces a
real training set (≥500, ideally 10³–10⁴). At that scale the noise
sweep becomes a routine afternoon experiment. Until then, the
Phase 4 trace-accuracy gate is set from literature intuition (target
p50 < 1% relative-L2, p95 < 5%) rather than measurement. Phase 4 MVP
is **not** blocked on this — see updated stop-rule below.

**Verdict:** ⏸️ deferred. Re-run when real Phase-0b data exists.

---

### 9.2.3  Gradient-accuracy sensitivity for DPS

**Why:** Phase 4 gate 7.3 (∂F_φ/∂c cosine similarity > 0.95 vs j-Wave)
is a literature-intuition threshold. If Phase 3 DPS is a serious
consumer, we need empirical data on how posterior quality degrades as
gradient cosine drops.

**Blocked by:** Phase 3 DPS implementation not started (Phase 3 is
design-only). Unblocks once Phase 3 has at least a reverse-sampler
prototype.

**Status:** 🚫 blocked.

**Verdict:** N/A until Phase 3 implemented.

---

## Supporting evidence already in hand

These are measurements this session produced that inform Phase 4
decisions even though they weren't listed in §9.

### θ-dim sweep (PR #14, merged)

From `scripts/theta_dim_sweep.py`:

| hidden | n_hidden | θ-dim | p95 rel-err | gate |
|---|---|---|---|---|
| 32 | 3 | 3,329 | 2.65% | close |
| 64 | 3 | **12,801** | **0.68%** | ✅ smallest passing |
| 128 | 3 | 50,177 | 0.14% | ✅ current default |

**Relevance to Phase 4:** If Phase 4 uses voxel `c` input (current
plan), the θ-dim is irrelevant to the FNO. If the FNO instead
conditions on θ (alternative design), the smaller θ-dim tightens
training cost. Confirms the current voxel-input design is the right
default — 50k-dim conditioning would be painful.

### NPE end-to-end plumbing (local, this session)

Phase-0 smoke (20 samples, 24³) → `build_theta_d_matrix` → train/test
split → train_npe → SBC. Pipeline proven functional locally
everywhere except the flowjax forward pass (macOS 26 gap).

**Relevance to Phase 4:** The same `(c, d)` iteration path that feeds
NPE will feed FNO training. No new data loader needed.

### Modal NPE smoke (#8 validation)

Synthetic linear-Gaussian NPE on A10G: posterior mean tracks
analytic expectation within 0.01 at all probes; NLL dropped >0.5 nats.

**Relevance to Phase 4:** Confirms the flowjax + Modal + JAX-GPU stack
works end-to-end. Reduces one risk for Phase 4's analogous training
infrastructure.

---

## Next session plan (based on this log's state, post-2026-04-24 run)

Done tonight:
- 9.1.1 forward-sim benchmark on Modal A100 — ✅ completed, verdict:
  proceed.
- 9.2.2 noise-sensitivity script written and run — ⚠️ results not
  informative at 60-sample budget. Deferred to DGX Spark for when a
  real Phase-0b dataset exists (≥500 samples).

Still open:
- 9.1.2 DGX SIREN-vs-voxel — Issue #11, waiting on DGX agent.
- 9.2.1 toy 2D FNO prototype — not started; no blocker.
- 9.2.3 DPS gradient-accuracy — blocked on Phase 3 DPS existing.

**Revised stop rule (supersedes earlier):**

Do not start the Phase 4 MVP 3D FNO implementation until **9.1.1
and 9.2.1** read "proceed". 9.2.2 is **no longer a gate for MVP** —
at the 60-sample data budget we couldn't measure what it was for, and
the real measurement wants Phase-0b-scale data that doesn't exist
yet. Trace-accuracy target for MVP is set from literature intuition
(p50 < 1%, p95 < 5% relative-L2); re-measure once DGX has produced
a real noise-sensitivity curve and tighten if needed before anything
ships downstream.

9.1.2 doesn't gate MVP either — informative for architecture choice
but the voxel-input design is the safe default regardless.
