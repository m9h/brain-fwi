# Brain FWI

### Full Waveform Inversion for Transcranial Ultrasound Brain Imaging

> *Seeing through the skull with sound — one gradient at a time*

A JAX-based pipeline for recovering the speed-of-sound map inside the
human head from ultrasound transmission data, using **Full Waveform
Inversion** (FWI) with automatic differentiation through
[j-Wave](https://github.com/ucl-bug/jwave)'s pseudospectral acoustic
solver.

---

## Why FWI for the Brain?

Conventional ultrasound imaging of the brain is severely limited by the
skull, which distorts, attenuates, and scatters acoustic waves.  FWI
sidesteps this by solving the *inverse problem*: given measurements of
how sound travels through the head, reconstruct a complete 3D map of
acoustic tissue properties.

This is the same technique that revolutionised exploration geophysics in
the 2000s, now repurposed for medical imaging following the landmark
demonstration by [Guasch et al. (2020)](#key-references).

---

## A Brief History of Full Waveform Inversion

FWI has deep roots in **exploration geophysics**, where the goal is to
image the Earth's subsurface from seismic reflection data.

**1980s — Theoretical foundations.** Lailly (1983) and Tarantola (1984)
independently formulated the seismic inverse problem as iterative
least-squares minimisation of the waveform misfit between observed and
simulated seismograms.  Tarantola showed that the gradient of the
misfit functional could be computed by cross-correlating the forward
wavefield with the *adjoint* (time-reversed) wavefield — an elegant
result, but one that required storing or recomputing the entire
space-time forward solution.

**1990s — Frequency-domain methods.** Pratt (1999) demonstrated
frequency-domain FWI, which reduces storage by solving one frequency at
a time and naturally enables multi-scale inversion (low frequencies
first for convexity, then higher frequencies for resolution).  This
"frequency banding" strategy remains standard practice.

**2000s — Industrial adoption.** Virieux & Operto (2009) published the
definitive review as FWI moved from academic curiosity to production
tool in the oil and gas industry.  Industrial-scale 3D FWI required
massive HPC clusters — a single inversion might need thousands of
forward simulations, each propagating waves through billion-cell grids.

**2010s — Breast USCT.** The medical ultrasound community began
applying FWI to *ultrasound computed tomography* (USCT) of the breast,
where the absence of bone makes the problem more tractable.  Wiskin et
al. (2012) and others demonstrated high-resolution sound-speed maps
from ring-array transducers.

**2020 — Brain FWI.** Guasch et al. (2020) showed that FWI could image
*through the skull*, recovering brain tissue properties from simulated
transcranial ultrasound data.  Their implementation used
[Stride](https://github.com/trustimaging/stride) built on
[Devito](https://www.devitoproject.org/) (finite-difference, C code
generation) with hand-coded adjoint-state gradients.  Follow-up work
from the same Imperial College London group developed hardware
prototypes (Cudeiro-Blanco et al. 2022), transducer calibration methods
(Cueto et al. 2021, 2022a), and experimental phantom validation (Robins
et al. 2023).

**2022 — OpenFWI.** Deng et al. (2022) released
[OpenFWI](https://openfwi-lanl.github.io/), a large-scale benchmark
(12 datasets, ~2.1 TB) for data-driven FWI from the geosciences.
Originally developed at Los Alamos National Laboratory, OpenFWI
catalysed machine-learning approaches to FWI by providing standardised
training and evaluation data — something the medical USCT community
still lacks.

**2023–present — Differentiable physics.** This project represents the
next step: replacing hand-coded adjoint operators with **JAX automatic
differentiation** through a pseudospectral solver.  This eliminates the
adjoint implementation burden, enables higher-order optimisers, and
opens a direct path to integration with neural posterior estimation and
simulation-based inference.

---

## The Computational Challenge

FWI is among the most computationally demanding inverse problems in
applied physics.  Each iteration requires:

1. **Forward simulation** — propagating an acoustic wave through a 3D
   heterogeneous medium for thousands of time steps
2. **Gradient computation** — differentiating the misfit through the
   entire simulation to obtain the sensitivity of the loss to every
   voxel in the model
3. **Repeat** for multiple source positions, frequency bands, and
   optimisation iterations

At publication-quality resolution (192^3 grid, 1 mm spacing, 256
transducer elements, 3 frequency bands, 20 iterations per band):

| Component | Cost |
|-----------|------|
| Single forward shot | ~35 s on GPU |
| Forward data generation (72 shots) | ~42 min |
| Single FWI iteration (8 shots + gradients) | ~50 min |
| Full inversion (3 bands x 20 iterations) | **~23 hours** |
| Peak memory (naive autodiff) | **218 GB** (carries) + 31 GB (outputs) |
| Peak memory (with checkpointing) | **~13 GB** |

The memory problem is fundamental: naive reverse-mode autodiff through
a time-stepping loop stores every intermediate state for the backward
pass.  At 192^3 with 1102 time steps, each carrying pressure, velocity,
and density fields, this exceeds 200 GB — more than any single GPU.

The geophysics community solved this decades ago with **gradient
checkpointing** (Griewank & Walther 2000): store only every Nth state,
recompute the rest during the backward pass.  This project implements a
two-level nested scan that reduces memory from O(N) to O(sqrt(N))
carries, making 192^3 FWI feasible on a single NVIDIA GB10 with 128 GB
unified memory — though it takes over a day to complete.

---

## Quick Start

```bash
uv sync
uv run python examples/01_2d_axial_fwi.py      # 2D demo (CPU ok, ~2 min)
uv run python examples/02_3d_brain_fwi.py       # 3D (needs GPU, ~1 hr)
uv run pytest tests/ -v                          # 116 tests
```

### Running at scale

```bash
# DGX Spark / any GPU box
./run_dgx.sh              # medium: 96^3, ~45 min
./run_dgx.sh --full       # full: 192^3, ~24 hr

# Slurm cluster
sbatch slurm_usct.sh                                        # medium
sbatch --export=GRID=192,ELEM=256,ITERS=20,SHOTS=8 slurm_usct.sh  # full
```

Results are written to `/data/datasets/brain-fwi/` (HDF5 volumes +
comparison figures).  FWI state is checkpointed to disk after each
frequency band, so preempted jobs resume automatically on resubmission.

---

## Architecture

```
brain_fwi/
  phantoms/         BrainWeb, MIDA, synthetic head models + ITRUSST acoustic properties
  transducers/      Ring (2D) and helmet (3D) array geometry (Kernel Flow-inspired)
  simulation/       j-Wave forward solver wrapper with sensor recording
    forward.py        Standard and checkpointed forward operators
    checkpointed_scan.py   O(sqrt(N)) memory gradient checkpointing
  inversion/        FWI engine: multi-frequency banding, autodiff gradients, Adam
    fwi.py            Core loop with disk checkpointing for preemption resilience
    losses.py         L2, envelope, multiscale loss functions
  utils/            Ricker wavelet, toneburst generators
```

---

## Head Models

| Model | Resolution | Tissues | Skull Layers | License |
|-------|-----------|---------|-------------|---------|
| **BrainWeb** | 1 mm iso | 12 classes | 1 (no cortical/trabecular) | Open |
| **MIDA** | 500 um iso | 153 structures | 3 (outer, diploe, inner) | IT'IS license |
| **Synthetic** | Configurable | 6 layers | 1 | Built-in |

---

## Key Design Decisions

- **JAX autodiff** through j-Wave pseudospectral solver (not
  adjoint-state like Stride/Devito)
- **Segmented gradient checkpointing** — two-level nested scan reduces
  backward-pass memory from O(N) to O(sqrt(N)) carries
- **Disk checkpointing** — FWI state saved after each frequency band
  for resilience to job preemption
- **Sensors passed into scan** — record at transducer positions during
  simulation, not full 3D field (saves 31 GB at 192^3)
- **ITRUSST benchmark values** (Aubry et al. 2022): skull cortical
  2800 m/s, trabecular 2300 m/s
- **Multi-frequency banding** (Stride pattern): 50-100 kHz -> 100-200
  kHz -> 200-300 kHz
- **Envelope loss** for robustness to cycle-skipping through thick skull
- **Sigmoid reparameterisation** for bounded velocity optimisation
- **TimeAxis pre-computed** outside JAX-traced scope (j-Wave's
  `float()` concretisation trap)

---

## Comparison with Related Projects

| | brain-fwi | [Stride](https://github.com/trustimaging/stride) | [Sonus](https://github.com/neurotech-berkeley/Sonus) | [OpenFWI](https://openfwi-lanl.github.io/) |
|---|---|---|---|---|
| Domain | Medical (brain) | Medical (brain, breast) | Medical (brain) | Geoscience |
| Solver | j-Wave (pseudospectral, JAX) | Devito (FD, C codegen) | Stride/Devito (FD) | Various |
| Gradients | JAX autodiff | Adjoint-state | Adjoint-state | Data-driven (ML) |
| GPU | JAX native | Devito OpenACC | Devito OpenACC | PyTorch |
| Checkpointing | Segmented scan + disk | Devito built-in | None | N/A |
| Head model | BrainWeb + MIDA + synthetic | MIDA | MIDA (pre-baked) | Velocity maps |
| Tests | 116 passing | CI | None | Benchmarks |

---

## Relevant Datasets

- **BrainWeb 20 Normal Models** — 12-class tissue maps, 1 mm iso, open
  access
- **MIDA** (IT'IS Foundation) — 153 structures, 500 um, requires
  license
- **Dryad MIDA US dataset** (10.5061/dryad.nzs7h44n7) — pre-computed
  US simulation, CC0
- **OpenFWI** (Deng et al. 2022) — 12 geoscience benchmark datasets,
  ~2.1 TB, open access
- **ITRUSST benchmark geometries** (Aubry et al. 2022) — 9 validation
  phantoms

---

## Related Projects

- [sbi4dwi](../sbi4dwi) — Acoustic properties, j-Wave adapter, TUS
  optimiser
- [openlifu-python](../openlifu-python) — Transducer arrays, skull
  segmentation, phase correction
- [dot-jax](../dot-jax) — Kernel Flow helmet geometry, atlas mesh
  generation

---

## Key References

### Brain FWI (Imperial College London group)

- Guasch L, Calderon Agudo O, Tang M-X, Nachev P, Warner M (2020).
  Full-waveform inversion imaging of the human brain. *npj Digital
  Medicine* 3:28.
  [doi:10.1038/s41746-020-0240-8](https://doi.org/10.1038/s41746-020-0240-8)

- Cueto C, Cudeiro J, Calderon Agudo O, Guasch L, Tang M-X (2021).
  Spatial response identification for flexible and accurate ultrasound
  transducer calibration and its application to brain imaging. *IEEE
  Trans. UFFC* 68(1):143-153.
  [doi:10.1109/TUFFC.2020.3015583](https://doi.org/10.1109/TUFFC.2020.3015583)

- Robins T, Camacho J, Calderon Agudo O, Herraiz JL, Guasch L (2021).
  Deep-learning-driven full-waveform inversion for ultrasound breast
  imaging. *Sensors* 21(13):4570.
  [doi:10.3390/s21134570](https://doi.org/10.3390/s21134570)

- Cueto C, Guasch L, Cudeiro J, et al. (2022). Spatial response
  identification enables robust experimental ultrasound computed
  tomography. *IEEE Trans. UFFC* 69(1):27-37.
  [doi:10.1109/TUFFC.2021.3104342](https://doi.org/10.1109/TUFFC.2021.3104342)

- Cueto C, Bates O, Strong G, et al. (2022). Stride: a flexible
  software platform for high-performance ultrasound computed tomography.
  *Comp. Meth. Prog. Biomed.* 221:106855.
  [doi:10.1016/j.cmpb.2022.106855](https://doi.org/10.1016/j.cmpb.2022.106855)

- Cudeiro-Blanco J, Cueto C, Bates O, et al. (2022). Design and
  construction of a low-frequency ultrasound acquisition device for 2-D
  brain imaging using full-waveform inversion. *Ultrasound Med. Biol.*
  48(10):1995-2008.
  [doi:10.1016/j.ultrasmedbio.2022.05.023](https://doi.org/10.1016/j.ultrasmedbio.2022.05.023)

- Robins TC, Cueto C, Cudeiro J, et al. (2023). Dual-probe
  transcranial full-waveform inversion: a brain phantom feasibility
  study. *Ultrasound Med. Biol.* 49(10):2302-2315.
  [doi:10.1016/j.ultrasmedbio.2023.06.001](https://doi.org/10.1016/j.ultrasmedbio.2023.06.001)

### FWI foundations (geoscience)

- Lailly P (1983). The seismic inverse problem as a sequence of before
  stack migrations. *Conf. on Inverse Scattering*, SIAM.

- Tarantola A (1984). Inversion of seismic reflection data in the
  acoustic approximation. *Geophysics* 49(8):1259-1266.

- Pratt RG (1999). Seismic waveform inversion in the frequency domain,
  Part 1. *Geophysics* 64:888-901.

- Virieux J, Operto S (2009). An overview of full-waveform inversion in
  exploration geophysics. *Geophysics* 74:WCC127-WCC152.

### Benchmarks and datasets

- Deng C, Feng S, Wang H, et al. (2022). OpenFWI: Large-scale
  multi-structural benchmark datasets for full waveform inversion.
  *NeurIPS 2022 Datasets and Benchmarks*.
  [doi:10.48550/arXiv.2111.02926](https://doi.org/10.48550/arXiv.2111.02926)

- Aubry J-F, et al. (2022). Benchmark problems for transcranial
  ultrasound simulation. *JASA* 152(2):1003-1019.

### Simulation tools

- Stanziola A, Arridge SR, Cox BT, Treeby BE (2023). j-Wave: an
  open-source differentiable wave simulator. *SoftwareX* 22:101338.

- Treeby BE, Cox BT (2010). k-Wave: MATLAB toolbox for the simulation
  and reconstruction of photoacoustic wave fields. *J. Biomed. Optics*
  15(2):021314.

### Checkpointing

- Griewank A, Walther A (2000). Algorithm 799: revolve: an
  implementation of checkpointing for the reverse or adjoint mode of
  computational differentiation. *ACM Trans. Math. Softw.*
  26(1):19-45.
  [doi:10.1145/347837.347846](https://doi.org/10.1145/347837.347846)

---

## License

MIT
