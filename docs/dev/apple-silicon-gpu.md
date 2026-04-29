# Apple Silicon GPU dev venv (`.venv-mps`)

Optional sidecar venv that runs JAX on the Apple Silicon GPU via
[jax-mps](https://github.com/tillahoffmann/jax-mps) (MLX-backed). Use
it for the parts of brain-fwi that benefit from GPU dispatch on this
laptop (large CANN sweeps, batched K-K transforms). The default
`.venv` stays CPU-only and is fine for everything else.

## When to use which venv

| Task | Venv |
|---|---|
| Phase 5 unit tests (`tests/test_constitutive.py`) | either — both pass |
| ICL loader tests (`tests/test_icl_dual_probe_loader.py`) | either — both pass |
| Forward-sim / FWI tests using `lax.scatter` (e.g. `tests/test_bandpass.py`) | **`.venv` only** (MPS backend has a known scatter coverage gap, see below) |
| Real GPU FWI on MIDA-scale heads or full ICL passes | neither — use Modal / RunPod / DGX |

Phase 5 unit tests are *slower* on MPS (37 s vs 7 s on CPU) because
their per-op compute is dominated by jit-dispatch overhead. The GPU
benefit shows up at ≥1024² problem sizes and on batched sweeps.

## Setup

The default `.venv` is Python 3.14 + jax 0.9.x **CPU**. The MPS venv
needs Python 3.13 because `jax-mps` only ships `cp313` wheels.

```bash
uv venv --python 3.13 .venv-mps
UV_PROJECT_ENVIRONMENT=.venv-mps uv sync --no-dev
uv pip install --python .venv-mps/bin/python jax-mps pytest
```

Three gotchas worth knowing:

1. `uv pip install jax-mps` without `--python` resolves against the
   project's pinned interpreter (currently 3.14). On a 3.14 host this
   fails with `no wheels with a matching Python ABI tag (e.g., cp314)`.
   Always pass `--python .venv-mps/bin/python` so the resolver picks
   the cp313 wheel.
2. `pytest` is a dev dep and `--no-dev` skips it; install it
   explicitly into `.venv-mps`.
3. `.venv*/` is in `.gitignore` so this directory will not be tracked.

Verify the backend:

```bash
.venv-mps/bin/python -c "import jax; print(jax.default_backend(), jax.devices())"
# mps [MpsDevice(id=0)]
```

JAX prints a one-time warning on first import:
> Platform 'mps' is experimental and not all JAX functionality may be
> correctly supported!

That's normal. Take it seriously when ops fail (see below).

## Running tests

```bash
# Phase 5
.venv-mps/bin/python -m pytest tests/test_constitutive.py -v

# ICL dual-probe loader (needs the dataset on disk)
ICL_DUAL_PROBE_PATH=/path/to/icl-dual-probe-2023 \
  .venv-mps/bin/python -m pytest tests/test_icl_dual_probe_loader.py -v
```

## Known MPS gaps in this repo

These tests fail on MPS today and are expected to keep working on the
CPU `.venv`:

- `tests/test_bandpass.py::TestMultiFrequencyFWI::test_bandpass_data_nonzero`
  — `stablehlo.scatter` dispatch fails with a broadcast-shape mismatch
  on the source-seeding pattern j-Wave uses. Affects forward sims with
  point sources via `vmap`d scatter.

If you add a new test that needs GPU but can't run on MPS, gate it
with the existing pattern in `tests/conftest.py` (or with a CUDA-only
mark) rather than skipping per-platform — the cloud runners exist for
exactly this.

## See also

- `~/.claude/projects/-Users-mhough-Workspace/memory/reference_jax_apple_silicon.md`
  — global notes on jax-mps vs jax-metal vs applejax, with empirical
  pass-rate data from sibling projects.
- `run_runpod.py`, `run_dgx.sh`, `scripts/modal_*.py` — NVIDIA paths
  for everything `.venv-mps` cannot run.
