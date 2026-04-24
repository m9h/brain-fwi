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

**Status:** 📋 not started (targeted for the next working session).

**Verdict:** _pending_.

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

**Status:** 📋 not started.

**Verdict:** _pending_.

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

## Next session plan (based on this log's state)

1. Fill in 9.1.1 numbers once the Modal benchmark lands (~25 min).
2. Write the 9.2.2 noise-sensitivity script (1–2 hours; mostly a
   wrapper around existing `train_npe_on_phase0.py`).
3. Run 9.2.2 on Modal or DGX.
4. Start 9.2.1 toy 2D FNO if 9.1.1 numbers confirm the speedup
   target is economically meaningful.

9.1.2 remains on the DGX agent; no blocker in principle but also
not controlling any decision we can make tonight.

**Stop rule:** do not start the Phase 4 MVP 3D FNO implementation
until at least 9.1.1, 9.2.1, and 9.2.2 verdicts read "proceed".
