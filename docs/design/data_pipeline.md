# Phase-0 Data Pipeline for Learning-Based Brain FWI

Status: **design draft — v1**, 2026-04-22
Owner: Morgan Hough

This document specifies the training-data factory that feeds the
learning-based augmentations roadmap (SIREN / SBI / score priors / neural
operators). The pipeline produces pairs of anatomical sound-speed maps and
the corresponding helmet-recorded traces simulated by j-Wave.

It is Phase 0 because every downstream component — posterior networks,
diffusion priors, FNO surrogates — consumes the same dataset. Building it
first pays off across all phases and forces early commitments on
parameterization and sensor geometry.

---

## 1. Goals and non-goals

**Goals.**

- Generate `(θ, d)` pairs at a scale sufficient for amortized simulation-
  based inference: target **10⁴ pairs** for first-pass experiments,
  **10⁵ pairs** for publication.
- Store `θ` in a parameterization suited to learned components
  (SIREN weights), not only as voxel grids.
- Cover realistic anatomical variability (inter-subject, registration
  noise, tissue-property uncertainty).
- Be reproducible: every sample tagged with the RNG seed, phantom
  source, acoustic-property version, and j-Wave commit hash.
- Be resumable: partial runs must be continuable without re-simulating
  prior samples.

**Non-goals (for Phase 0).**

- Clinical realism (skin coupling, transducer impulse response, noise
  models) — deferred to Phase 3+.
- In-vivo imaging — phantom-only.
- Multi-frequency / multi-physics variants — single Ricker source family
  per dataset version.

---

## 2. Dataset schema

One sample = one subject × one acoustic-property draw × one transducer
configuration.

| Field | Shape / dtype | Notes |
|---|---|---|
| `sample_id` | `str` | `{subject}_{seed}` |
| `subject_id` | `str` | `brainweb_04`, `mida_01`, … |
| `grid_shape` | `(3,) int32` | e.g. `(96, 96, 96)` |
| `dx` | `float32` | grid spacing in metres |
| `sound_speed_voxel` | `(Z, Y, X) float16` | canonical voxel grid (lossy) |
| `sound_speed_siren` | `(n_weights,) float32` | packed SIREN weights |
| `sound_speed_siren_arch` | attrs | `hidden_dim`, `n_hidden`, `omega_0` |
| `density_voxel` | `(Z, Y, X) float16` | kept fixed during FWI but needed for forward sim |
| `tissue_labels` | `(Z, Y, X) uint8` | integer label map for atlas-conditioned priors |
| `transducer_positions` | `(N_src, 3) float32` | helmet element centres (m) |
| `sensor_positions` | `(N_recv, 3) float32` | receiver positions (often ≡ transducer) |
| `source_signal` | `(N_t,) float32` | Ricker wavelet (base pulse, unbandpassed) |
| `dt` | `float32` | time step (s) |
| `freq_hz` | `float32` | nominal source frequency |
| `observed_data` | `(N_src, N_t, N_recv) float16` | helmet traces |
| `seed` | `int64` | PRNG seed used for all stochastic choices |
| `phantom_version` | `str` | `brainweb@v1.4` etc. |
| `acoustic_version` | `str` | tag for `phantoms/properties.py` values |
| `jwave_sha` | `str` | pinned j-Wave commit |

`sound_speed_voxel` + `sound_speed_siren` are stored **both**:

- voxel for FWI baselines and ground-truth comparisons;
- SIREN weights for networks that consume the compact representation.

`float16` keeps voxel storage manageable at 10⁵ scale (see §5).

---

## 3. Storage format

**Primary: Zarr v3 with sharded chunks.**

- Consumers are JAX programs; Zarr + `zarrs`/`numcodecs` streams cleanly
  into `jax.numpy` without the HDF5 GIL pain.
- Sharded chunks align with "batch = one subject" access patterns.
- Append-only: new samples can be added without rewriting prior shards.
- Works on local disk and cloud buckets interchangeably (DGX Spark run
  → S3 sync → laptop analysis).

**Fallback: HDF5** (single file per "chunk" of 1000 samples) for
environments where Zarr tooling is unavailable.

Directory layout:

```
data/
  phase0_v1/
    manifest.json                # version, seed map, sample count
    shards/
      00000/     # samples 0..999
      00001/     # ...
    metadata/
      phantom_catalog.json
      acoustic_table.json
```

The `manifest.json` is authoritative for "is sample X already computed?"
resumability checks.

---

## 4. Generation strategy

### 4.1 Anatomical sources

| Source | Count | Notes |
|---|---|---|
| BrainWeb (20 subjects × label maps) | ~400 after augmentation | already wired via `brainweb-dl` |
| MIDA | 1 subject, high-detail | for calibration / held-out eval |
| Atlas warps | 10³–10⁴ | thin-plate spline / SyN deformations |
| Tissue-property jitter | ∞ | draw `(c, ρ)` per tissue from ITRUSST-consistent Gaussians (Aubry 2022) |

Combine as: for each anatomical subject, produce N augmentations; for
each augmentation, produce M tissue-property draws. Target
N=10, M=10 per subject → ~2000 samples per BrainWeb subject.

### 4.2 Transducer configuration

Start with the existing helmet geometry (`transducers/helmet.py`). For
each sample, randomly sub-sample active elements (e.g., 256 of 1024) to
cover aperture variability that a trained model should be robust to.

### 4.3 Forward simulation

Per sample:

1. Load anatomy → voxel `c, ρ, labels`.
2. Pretrain a SIREN on `c` (200 steps — enough to capture coarse
   structure, inexpensive). Store voxel `c` for ground-truth use and
   SIREN weights for ML consumers.
3. Run `generate_observed_data()` with the current helmet config.
4. Write sample to the current shard.

Gradient checkpointing already enabled in `simulate_shot_sensors`.

### 4.4 Reproducibility and provenance

- Global seed → per-subject seeds → per-sample seeds (split via
  `jax.random.split`).
- All versions (jwave, jaxdf, brain_fwi, phantom sources) recorded in
  `manifest.json`.
- One Python script `scripts/gen_phase0.py` drives the full pipeline;
  parameters in a YAML config under `configs/phase0/`.

---

## 5. Compute budget

Estimates for DGX Spark (GB10, unified memory), based on the existing
96³ run:

| Quantity | Current 96³ | Phase-0 target | Notes |
|---|---|---|---|
| Single forward sim | ~3 s | ~3 s | per shot, post-JIT |
| Shots per sample | ~32 | ~128 | helmet aperture |
| Seconds per sample | ~100 | ~400 | dominated by shots |
| Samples per GPU-hour | ~36 | ~9 | |
| 10⁴ samples | — | ~1100 GPU-hours | ~46 days single GPU |
| 10⁵ samples | — | ~11k GPU-hours | multi-GPU essential |

**Implication:** before scaling to 10⁵, invest in:

1. **Multi-shot batching** (vmap over sources where memory allows).
2. **FNO surrogate warm-start** — Phase 4 actually comes back as a
   Phase-0 accelerator once trained on a 10⁴ seed set.
3. **Mixed precision** in j-Wave — reported 2× with minimal accuracy
   loss in similar PSTD codes.

Storage:

- `float16` voxel grid at 96³ = 1.8 MB/sample; 10⁵ samples = 180 GB.
- Helmet traces at 256 shots × 4096 samples × 256 receivers × fp16
  = 536 MB/sample. **This is the storage wall — not the grid.** Consider
  storing bandpass-decimated traces (halve sample rate per band) or
  only sparse representations (first-arrival picks + amplitudes).

---

## 6. Validation plan

A dataset is only as good as the checks it passes. Mandatory before
any ML training consumes `phase0_v1`:

1. **Round-trip FWI on a held-out subset** (100 samples).
   For each, run 30 iters of classical voxel FWI initialized from a
   smoothed version of the true `c`. Median L2 error should be within
   known-benchmark ranges (e.g., <3% in skull, <1% in brain).
2. **SIREN fidelity.** For each sample, compare `SIREN → voxel` against
   stored ground-truth voxel `c`. 95th-percentile error must be <2%.
3. **Simulation-based calibration (SBC)** — run a prototype NPE on a
   1000-sample subset, check rank histograms are uniform per parameter
   (Talts et al. 2018). Non-uniform rank histograms indicate
   distribution shift or a broken simulator, not a network problem.
4. **Anatomy coverage.** PCA of stored fields; confirm variance
   dimensionality matches augmentation intent (not collapsed to a
   single subject).

Failures in (1) block dataset release. Failures in (3) block Phase 2.

---

## 7. Open decisions

- **Parameterization of `θ`.** SIREN is the current plan, but the
  weight-space posterior is hard to interpret for clinicians. Open
  question: also store a low-rank / PCA basis on the voxel grid for
  interpretable posterior summaries.
- **Transducer variation strategy.** Fixed helmet vs. randomized
  aperture vs. parameterized aperture (input to the network). Affects
  whether the network amortizes over subjects only or over (subject,
  aperture) jointly.
- **Observation-noise model.** Phase 0 stores clean traces. Adding
  measurement noise at training time is cheap and correct; hardware-
  realistic impulse responses are deferred.
- **Held-out split.** MIDA + 2 BrainWeb subjects reserved for
  evaluation; never enter training shards. Enforced by `manifest.json`.

---

## 8. Workstream kickoff

Tracked implementation steps once this design is accepted:

1. Add `scripts/gen_phase0.py` skeleton + YAML config.
2. Write the anatomy augmentation module
   (`src/brain_fwi/phantoms/augment.py`) — TPS warps + property jitter.
3. Extend `generate_observed_data` to emit per-sample metadata.
4. Zarr writer with manifest update + resume.
5. Validation harness (items 1–4 in §6) as pytest `-m phase0_data`.

Review gate: checkpoint the design doc + validation results before
running the full 10⁴ batch on DGX.
