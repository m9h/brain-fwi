# Contributing to Brain FWI

Thank you for considering a contribution to brain-fwi. This document explains how
to set up a development environment, run the tests, and submit changes.

## 1. Development environment

Brain FWI uses **uv** for dependency management. Do not use pip or conda.

```bash
# Clone and set up
git clone https://github.com/mhough/brain-fwi.git
cd brain-fwi
uv sync                          # install all deps including dev extras
uv run pytest tests/ -v          # verify everything passes (106 tests)
```

### GPU requirements

- **2D examples** run on CPU (48x48 -- 256x256 grids).
- **3D examples** require a CUDA GPU with at least 16 GB VRAM.
- Install the CUDA extras for GPU support:

```bash
uv sync --extra cuda12   # CUDA 12.x
uv sync --extra cuda13   # CUDA 13.x (DGX Spark)
```

JAX will fall back to CPU automatically if no GPU is detected.

## 2. Project structure

```
src/brain_fwi/
  phantoms/       BrainWeb, MIDA, synthetic head models + acoustic properties
  transducers/    Ring (2D) and helmet (3D) array geometry
  simulation/     j-Wave forward solver wrapper with sensor recording
  inversion/      FWI engine: losses, optimizer, multi-frequency banding
  utils/          Ricker wavelet, toneburst generators
tests/            106 tests (unit + integration + end-to-end)
examples/         4 runnable examples (2D, 3D, recovery)
docs/             Sphinx documentation (MyST Markdown)
```

## 3. Coding conventions

- **Python 3.10+** with type hints on all public functions.
- **NumPy-style docstrings** (parsed by Napoleon). Every public function and class
  must have a docstring.
- **SI units throughout**: metres, seconds, m/s, kg/m^3, Pa, Hz.
- Grid spacing is always `dx` (not `h`, `ds`, etc.).
- Acoustic properties follow the **ITRUSST benchmark** (Aubry et al. 2022, JASA):
  skull cortical = 2800 m/s, trabecular = 2300 m/s, CSF = 1500 m/s.

### j-Wave integration notes

The forward operator wraps j-Wave's `simulate_wave_propagation`. Key gotchas:

1. **TimeAxis must be pre-computed** outside any `jax.jit`-traced scope.
   `TimeAxis.from_medium()` calls `float()`, which triggers a
   `ConcretizationTypeError` inside JAX tracing.

2. **FourierSeries fields** must have a trailing singleton dimension
   (`arr[..., jnp.newaxis]`). Forgetting this produces silent shape mismatches.

3. **Gradient checkpointing** (`TimeWavePropagationSettings(checkpoint=True)`) is
   essential for 3D to avoid OOM. Always enable it for FWI.

## 4. Running tests

```bash
# All tests (fast ones only, ~30 s)
uv run pytest tests/ -v -m "not slow"

# Full suite including end-to-end FWI (~2 min on CPU)
uv run pytest tests/ -v

# Single test file
uv run pytest tests/test_forward.py -v

# With coverage
uv run pytest tests/ --cov=brain_fwi --cov-report=term-missing
```

### Test categories

| Marker   | Description                                    | Time    |
|----------|------------------------------------------------|---------|
| (none)   | Unit tests: properties, reparameterization     | < 1 s   |
| (none)   | Integration tests: forward sim on 48x48 grid   | ~10 s   |
| `slow`   | End-to-end FWI on 48x48 grid                   | ~2 min  |

### Phantom validation

When adding or modifying acoustic property tables:

1. Verify values against ITRUSST benchmark Table III (Aubry 2022).
2. Run `tests/test_properties.py` and `tests/test_mida.py` to check that
   tissue-to-acoustic mappings are consistent.
3. If adding a new phantom source, ensure label 0 (background) maps to water
   (1500 m/s, 1000 kg/m^3) for acoustic coupling.

## 5. Adding new features

### New loss function

1. Add the function to `src/brain_fwi/inversion/losses.py` with a NumPy-style
   docstring.
2. Register it in `_get_loss_fn()` in `fwi.py`.
3. Add tests in `tests/test_losses.py`.
4. The loss must be differentiable by JAX (`jax.grad`-compatible).

### New head phantom

1. Add a loader in `src/brain_fwi/phantoms/`.
2. Map tissue labels to acoustic properties using the `TISSUE_PROPERTIES` table
   or create a new mapping (reference your source: Duck 1990, Aubry 2022, etc.).
3. Export from `phantoms/__init__.py`.
4. Add tests in `tests/test_phantom.py` or a dedicated test file.

### New transducer geometry

1. Add a generator in `src/brain_fwi/transducers/helmet.py` or a new file.
2. Provide both physical-coordinate and grid-index outputs.
3. Test that all grid indices fall within valid bounds (see `test_transducers.py`).

## 6. Building documentation

```bash
# Install doc dependencies
uv pip install -r docs/requirements.txt

# Build HTML
cd docs && make html

# View locally
open _build/html/index.html
```

Documentation uses MyST Markdown with Sphinx autodoc. All public API is
auto-generated from docstrings via `sphinx-apidoc`.

## 7. Submitting changes

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** before implementation (red-green-refactor).
3. **Run the full test suite** (`uv run pytest tests/ -v`).
4. **Update CHANGELOG.md** with a summary of your changes under `[Unreleased]`.
5. **Open a pull request** with a clear description of what and why.

### Commit message format

```
<type>: <short summary>

<optional body explaining the "why">
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `ci`.

### What we look for in reviews

- Tests pass (including `slow` markers for FWI changes).
- Docstrings on all public functions (NumPy-style).
- SI units with clear variable names (`sound_speed`, not `v`; `density`, not `r`).
- No hardcoded paths or magic numbers without comments.
- Acoustic property values cite a published source.
