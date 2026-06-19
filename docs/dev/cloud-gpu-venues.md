# Cloud GPU venues for brain-fwi

Where to run each workload, and why. Two workload classes with very different
needs: **NN/diffusion training** (short, GPU-flexible) and **j-Wave FWI** (long,
many sequential solves, autodiff through the solver).

## TL;DR

| Workload | Venue | Why |
|----------|-------|-----|
| Score/diffusion + FNO/NPE training | **Modal** (A10G/A100) or **beam** (A10G/RTX4090) | Short (~10 min); both fine. Modal already wired. |
| Small FWI (≤64³) | **GB10** if free, else **beam** | Both quick. |
| Large FWI (96³+) | **beam** (RTX4090/A10G), or **GB10** when the node is free | beam: no timeout, doesn't block the GB10 (see below). GB10: faster but contends with hbn. |
| Huge FWI (192³, multi-GPU) | **GB10**, or beam on-demand A100/H100 (needs account upgrade) | Memory + speed. |

## The three venues

### GB10 (local, Slurm) — fastest, but shared
- Blackwell GB10, jax[cuda13]; ~1.2 s/gradient at 48³, est. ~5 s at 96³.
- **Gotcha:** the `gpu` and `cpu` Slurm partitions are **mutually exclusive** on
  the single node. A GPU FWI job cannot start while a cpu-partition job (the
  `hbn_*` jobs) runs — and, symmetrically, a long GPU FWI job **blocks hbn from
  starting**. So per the queue-priority rule, don't hog the GB10 for multi-hour
  FWI; use it only when the node is free or for short jobs.

### Modal — great for NN training, NOT for FWI
- jax[cuda12] + j-Wave: usable for short NN training (e.g. 96³ score net, ~620 s
  on A100). **2 h function timeout** killed every FWI run (96³, 64³ all timed
  out). Use Modal for NN training only.
- Invoke with the **venv's** modal (`.venv/bin/modal run ...`); the global modal
  CLI's Python lacks numpy and the local entrypoint fails before the GPU runs.

### beam.cloud — the FWI venue that works off-GB10
- Auth already configured (`~/.beam/config.ini`); client in `.venv-beam`
  (`uv pip install beam-client`). API: `@function(gpu=..., image=Image(...).add_commands([...]))`
  + `.remote()`.
- **Serverless `@function` GPUs on this account: A10G and RTX4090 only** (24 GB).
  A100/H100/L40S are on-demand-pod only (separate mechanism / account upgrade).
- **No documented function timeout** — the key advantage over Modal for FWI.
- The jax[cuda12] + j-Wave image builds cleanly (`pip install -e '.[cuda12]'`).

#### beam forward-sim benchmark (`scripts/beam_benchmark_forward.py`)
Homogeneous-water forward, warm median (j-Wave is **dispatch-bound** at these
grids — time scales with timesteps, barely with grid size or GPU tier):

| GPU | 48³ (37 steps) | 64³ (50) | 96³ (74) |
|-----|------|------|------|
| A10G | 0.51 s | 0.51 s | 0.65 s |
| RTX4090 | 0.57 s | 0.73 s | 0.59 s |

⇒ ~0.008 s/step. A 96³ FWI (100 µs ≈ 625 steps, +backward ≈ 3×, 16 shots ×
10 iters × 3 bands × 2 configs) ≈ **~4–5 h** — fine on beam (no timeout), and it
**doesn't touch the GB10**, so hbn jobs are never blocked.

#### beam gotchas
- `.beamignore` is required: by default beam syncs the whole CWD. Put a
  `scripts/.beamignore` whitelisting only the beam entry files (`*` then
  `!beam_*.py`), and **run from inside `scripts/`** so the handler module loads
  as top-level (running from the repo root makes beam look for a `scripts`
  package → `ModuleNotFoundError: No module named 'scripts'`).
- The `.venv-beam` has only the beam client (no numpy/jax). Keep entrypoints
  pure-Python: ship inputs as **bytes**, do all numpy/jax/matplotlib work in the
  remote function, and return PNG/npz **bytes**.

## Scripts
- `scripts/beam_benchmark_forward.py` — forward-sim micro-benchmark (A10G/RTX4090).
- `scripts/beam_dps_fwi_3d.py` — 96³ DPS-FWI (annealed-t) on beam, the off-GB10 lesion test.
- `scripts/modal_train_unet3d.py` — 3D score-net training on Modal (48³ A10G / 96³ A100).
