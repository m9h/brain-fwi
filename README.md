# Brain FWI

Full Waveform Inversion for transcranial ultrasound brain imaging using
j-Wave (JAX pseudospectral solver) with automatic differentiation.

## Quick Start

```bash
uv sync
uv run python examples/01_2d_axial_fwi.py --synthetic
```

## Architecture

- **phantoms/** — BrainWeb head model loading + ITRUSST acoustic properties
- **transducers/** — Ring (2D) and helmet (3D) array geometry
- **simulation/** — j-Wave forward solver wrapper
- **inversion/** — FWI engine (multi-frequency, autodiff gradients, Adam)

## References

- Guasch et al. (2020). Full-waveform inversion imaging of the human brain. npj Digital Medicine.
- Stanziola et al. (2022). j-Wave: differentiable acoustic simulations. arXiv:2207.01499.
- Aubry et al. (2022). ITRUSST benchmark for transcranial ultrasound. JASA 152(2).
