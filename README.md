# Brain FWI

Full Waveform Inversion for transcranial ultrasound brain imaging using
j-Wave (JAX pseudospectral solver) with automatic differentiation.

## Quick Start

```bash
uv sync
uv run python examples/01_2d_axial_fwi.py --synthetic      # 2D demo (CPU ok)
uv run python examples/02_3d_brain_fwi.py --synthetic       # 3D (needs GPU)
uv run pytest tests/ -v                                      # 106 tests
```

## Architecture

```
brain_fwi/
  phantoms/         BrainWeb, MIDA, synthetic head models + ITRUSST acoustic properties
  transducers/      Ring (2D) and helmet (3D) array geometry (Kernel Flow-inspired)
  simulation/       j-Wave forward solver wrapper with sensor recording
  inversion/        FWI engine: multi-frequency banding, autodiff gradients, Adam
  utils/            Ricker wavelet, toneburst generators
```

## Head Models

| Model | Resolution | Tissues | Skull Layers | License |
|-------|-----------|---------|-------------|---------|
| **BrainWeb** | 1mm iso | 12 classes | 1 (no cortical/trabecular) | Open |
| **MIDA** | 500um iso | 153 structures | 3 (outer, diploe, inner) | IT'IS license |
| **Synthetic** | Configurable | 6 layers | 1 | Built-in |

## Key Design Decisions

- **JAX autodiff** through j-Wave pseudospectral solver (not adjoint-state like Stride/Devito)
- **TimeAxis pre-computed** outside traced scope (j-Wave's `float()` concretization trap)
- **ITRUSST benchmark values** (Aubry et al. 2022): skull cortical 2800 m/s, trabecular 2300 m/s
- **Multi-frequency banding** (Stride pattern): 50-100 kHz -> 100-200 kHz -> 200-300 kHz
- **Envelope loss** for robustness to cycle-skipping through thick skull
- **Sigmoid reparameterization** for bounded velocity optimization

## Comparison with Related Projects

| | brain-fwi | [Sonus](https://github.com/neurotech-berkeley/Sonus) | [Stride](https://github.com/trustimaging/stride) |
|---|---|---|---|
| Solver | j-Wave (pseudospectral, JAX) | Stride/Devito (FD) | Devito (FD) |
| Gradients | JAX autodiff | Adjoint-state | Adjoint-state |
| GPU | JAX native | Devito OpenACC | Devito OpenACC |
| Tissue properties | Full (c, rho, alpha) | Velocity only | Full |
| Head model | BrainWeb + MIDA + synthetic | MIDA (pre-baked) | MIDA |
| Brain FWI status | Working (tested) | OOM at forward step | Published (Guasch 2020) |
| Tests | 106 passing | None | CI |

## Relevant Datasets

- **BrainWeb 20 Normal Models** — 12-class tissue maps, 1mm iso, open access
- **MIDA** (IT'IS Foundation) — 153 structures, 500um, requires license
- **Dryad MIDA US dataset** (10.5061/dryad.nzs7h44n7) — pre-computed US simulation, CC0
- **Colin27** (BIC McGill) — single-subject MNI template with tissue segmentation
- **ITRUSST benchmark geometries** (Aubry et al. 2022) — 9 validation phantoms

## Related Infrastructure

- [sbi4dwi](../sbi4dwi) — Acoustic properties, j-Wave adapter, TUS optimizer
- [openlifu-python](../openlifu-python) — Transducer arrays, skull segmentation, phase correction
- [dot-jax](../dot-jax) — Kernel Flow helmet geometry, atlas mesh generation

## References

- Guasch et al. (2020). Full-waveform inversion imaging of the human brain. *npj Digital Medicine* 3:28.
- Stanziola et al. (2022). j-Wave: differentiable acoustic simulations. *arXiv:2207.01499*.
- Aubry et al. (2022). ITRUSST benchmark for transcranial ultrasound. *JASA* 152(2):1003-1019.
- Iacono et al. (2015). MIDA head model. *PLOS ONE*.
- Cueto et al. (2022). Stride: high-performance ultrasound CT. *Comp. Meth. Prog. Biomed.*
