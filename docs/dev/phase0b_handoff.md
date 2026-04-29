# Phase-0b dataset generation — handoff brief

Notes for whichever agent / human picks this up next (DGX, Modal, or a
parallel session). Captures the on-disk state of the job as of
2026-04-29 so the next driver doesn't re-derive it.

## Why this is the highest-leverage thing right now

Three of the open phases — **Phase 2 (NPE training), Phase 3 (score
prior), Phase 4 (FNO/UNO at scale)** — all wait on the same artefact:
a Phase-0 dataset of (θ, d) pairs at design scale. None of them can
flip from 🟡 to 🟢 until this run completes.

Phase 5 is shipped (frequency- and time-domain absorption integrated;
gating xfail flipped to PASS) and doesn't need this dataset.

## Where the job lives

All on `origin/main` — **not yet merged into `phase5/cann-attenuation-scaffold`**:

| File | Role |
|---|---|
| `scripts/gen_phase0.py` | Pure-Python driver. CPU-runnable smoke at small grid, GPU-required at production grid. Resumable via `ShardedWriter` manifest. |
| `scripts/modal_gen_phase0.py` | Modal A100 wrapper around `gen_phase0.py`. Outsources off DGX. |
| `scripts/run_runpod_gen_phase0.py` | RunPod fallback for >24 h runs. |
| `slurm_gen_phase0.sh` | DGX/SLURM submission script. |
| `src/brain_fwi/data/sharded_{reader,writer}.py` | I/O for the produced shards. |

Design reference: `docs/design/data_pipeline.md`.

## Defaults baked into `modal_gen_phase0.py`

```
PHANTOM               = "mida"
GRID_SIZE             = 96
DX_M                  = 0.002          # 2 mm
N_ELEMENTS            = 128            # helmet element count
N_SUBJECTS            = 1              # MIDA has 1 subject
N_AUGMENTS            = 100            # ← produces 100 samples per run
FREQ_HZ               = 5.0e5          # 500 kHz Ricker
SIREN_PRETRAIN_STEPS  = 400
DATASET_VERSION       = "phase0_v1"
GIT_BRANCH            = "main"
WORK_VOL_NAME         = "brain-fwi-phase0"   # ← Modal volume for output
MIDA_VOL_NAME         = "mida-data"          # input, already exists
```

Output written to `/output/phase0_v1_mida_96/` on the
`brain-fwi-phase0` Modal volume. Estimated wall ≈ 8 h on a single A100
(~100 s/sample at 96³, per the script's comment block).

## Two open decisions before launching

1. **SIREN architecture.** Defaults are `hidden=128, layers=3` →
   ~50 k θ-dims. The Phase-2 audit flagged this as a likely MAF
   bottleneck. `scripts/theta_dim_sweep.py` already has results:
   `hidden=64, layers=3 → 12.8 k dims @ 0.68 % reconstruction error`,
   which passes the Phase-0 §6 SIREN-fidelity gate. **Pick a config
   before generating 10 k samples** — otherwise Phase 2 has to
   regenerate the whole dataset.

2. **§6 validation gates not wired into the generator.**
   `data_pipeline.md` lists round-trip FWI, SIREN fidelity, SBC, and
   anatomy-coverage PCA as quality gates; none are integrated into
   `gen_phase0.py`. Two paths:
   - **Run-and-validate-after**: cheap (no code), but you only
     discover dataset issues after spending the GPU hours.
   - **Wire-then-run**: half day of work, gives quality assurance
     in-flight, can resume if a gate trips.

## What "Phase-0b at scale" actually means

Default `N_AUGMENTS=100` per run gives **100 samples**, not the design
target.

| Target | Required ratio | Path |
|---|---|---|
| 10⁴ pairs (first-pass design target) | 100× current run | bump `N_AUGMENTS=10000`, single ~33 days A100 *or* shard across 100 jobs |
| 10⁵ pairs (publication target) | 1000× current run | needs subject diversity beyond MIDA (BrainWeb? Phase-1 SIREN-driven phantoms?) |

A pragmatic first run of **N_AUGMENTS = 1024** (~85 h on one A100,
roughly $30 of compute) gets to a Phase-2-trainable scale and is
sufficient for an initial NPE / surrogate sanity check before
committing to the full 10⁴.

## Commands to start it (Modal CLI required)

```bash
# One-time: create the output volume
modal volume create brain-fwi-phase0

# Run with defaults (100 samples)
modal run scripts/modal_gen_phase0.py

# Or override
modal run scripts/modal_gen_phase0.py \
    --n-augments 1024 --grid-size 96 --version phase0_v1_mida_96
```

Pull shards back to DGX once it finishes:

```bash
modal volume get brain-fwi-phase0 /output/phase0_v1_mida_96 ./
```

The Modal runner registers `brain-fwi-validation` as a sibling volume
for related artefacts (forward-sim benchmarks, SIREN validation,
NPE noise sensitivity); keep `brain-fwi-phase0` separate so the
dataset doesn't get mixed in with one-shot results.

## What unblocks once the dataset lands

| Phase | Action |
|---|---|
| 2 (NPE) | `scripts/train_npe_on_phase0.py --data <dataset_root>` produces a trained `ConditionalFlow` + SBC report. |
| 3 (score) | DGX implementation can start: training data is the SIREN-θ subset of the same dataset. |
| 4 (FNO/UNO) | `scripts/train_fno_on_phase0.py --data <dataset_root>` (already scripted) replaces synthetic-only weights with real-data weights. |
| 5 (CANN) | Already shipped. Optional: re-run `modal_npe_noise_sensitivity.py` with the real dataset + synthetic skull-attenuation noise to test how well the time-domain absorption work integrates with NPE. |

## Where to leave breadcrumbs after the run

Append run results (sample count, wall time, GPU hours, validation-gate
outcomes) to `docs/evidence/phase4_readiness.md` (it's already the
evidence-collection home for §9 of the Phase-4 doc and the only
"results" doc in the repo).

If the SIREN config from decision (1) above is changed from defaults,
update `docs/design/data_pipeline.md` §5 with the chosen config so
the schema stays single-sourced.
