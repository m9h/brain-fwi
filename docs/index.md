# Brain FWI Documentation

**Full Waveform Inversion for transcranial ultrasound brain imaging**

Brain FWI is a JAX-based toolkit that implements Full Waveform Inversion (FWI) for
transcranial ultrasound computed tomography (USCT). It uses
[j-Wave](https://github.com/ucl-bug/jwave)'s pseudospectral time-domain solver
as the forward engine and JAX automatic differentiation for gradient computation,
following the approach of [Guasch et al. (2020)](https://doi.org/10.1038/s41746-020-0240-8)
but reimplemented in a fully differentiable framework.

## Key features

- **JAX autodiff gradients** through j-Wave pseudospectral solver (no hand-coded adjoints)
- **Multi-frequency banding** (Stride pattern): coarse-to-fine frequency progression
- **Envelope loss** for robustness to cycle-skipping through thick skull
- **Sigmoid reparameterization** for bounded velocity optimization
- **Multiple head phantoms**: BrainWeb (12-class), MIDA (153-structure), and synthetic
- **ITRUSST benchmark values** for acoustic tissue properties (Aubry et al. 2022)

## Quick start

```bash
uv sync
uv run python examples/01_2d_axial_fwi.py --synthetic   # 2D demo (CPU ok)
uv run python examples/02_3d_brain_fwi.py --synthetic    # 3D (needs GPU)
uv run pytest tests/ -v                                   # 106 tests
```

## Contents

```{toctree}
:maxdepth: 2

tutorials/forward_simulation
tutorials/fwi_reconstruction
tutorials/head_phantoms
api/brain_fwi
contributing
changelog
```

## References

- Guasch et al. (2020). Full-waveform inversion imaging of the human brain.
  *npj Digital Medicine* 3:28.
- Stanziola et al. (2022). j-Wave: An open-source differentiable wave simulator.
  *arXiv:2207.01499*.
- Aubry et al. (2022). Benchmark problems for transcranial ultrasound simulation.
  *JASA* 152(2):1003--1019.
- Iacono et al. (2015). MIDA: A multimodal imaging-based detailed anatomical model
  of the human head and neck. *PLOS ONE*.
