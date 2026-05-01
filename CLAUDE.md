# CLAUDE.md - Brain FWI

## What is this project?

**brain-fwi** is a JAX-based Full Waveform Inversion (FWI) pipeline for
transcranial ultrasound brain imaging. It uses j-Wave's pseudospectral
solver for forward simulation and JAX autodiff for gradient computation,
following the approach of Guasch et al. (2020, npj Digital Medicine) but
reimplemented in a fully differentiable JAX framework.

Key innovation: replaces Stride/Devito adjoint-state gradients with
JAX automatic differentiation through j-Wave, enabling easy integration
with neural posterior estimation (SBI) and modern optimization.

## Architecture

```
brain_fwi/
  phantoms/       # Head model loading (BrainWeb, MIDA) + acoustic properties
  transducers/    # Ultrasound array geometry (ring, helmet)
  simulation/     # j-Wave forward solver wrapper
  inversion/      # FWI loop (losses, optimizer, multi-frequency)
  utils/          # Source wavelets, visualization helpers
```

## Key dependencies

| Library | Role |
|---------|------|
| j-Wave  | Pseudospectral acoustic wave solver (JAX) |
| jaxdf   | Domain/field discretization for j-Wave |
| JAX     | Autodiff + GPU acceleration |
| Equinox | Pytree-compatible modules |
| Optax   | Adam optimizer for FWI |
| brainweb-dl | BrainWeb phantom downloading |

## Conventions

- **uv only** for package management (no pip/conda)
- Units: SI (metres, seconds, m/s, kg/m^3, Pa)
- Grid spacing: determined by frequency (>10 points per wavelength)
- Acoustic properties: ITRUSST benchmark values (Aubry et al. 2022, JASA)

## Related projects (same dev tree)

- **sbi4dwi**: Provides acoustic.py tissue properties, j-Wave adapter, TUS optimizer
- **openlifu-python**: Transducer array classes, phase correction, skull segmentation
- **dot-jax**: Kernel Flow helmet geometry, atlas mesh generation
- **Stride**: Reference FWI implementation (breast examples, brain FWI paper)

## Running

```bash
uv sync
uv run python examples/01_2d_axial_fwi.py      # 2D demo (CPU ok)
uv run python examples/02_3d_brain_fwi.py       # 3D (needs GPU)
uv run pytest tests/ -v                          # tests
```

Apple Silicon GPU is available via an optional `.venv-mps` sidecar venv.
See `docs/dev/apple-silicon-gpu.md` for setup and the list of tests
that won't run there (notably the `scatter`-using forward-sim tests).
