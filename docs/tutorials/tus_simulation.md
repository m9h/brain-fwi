# Transcranial Ultrasound (tFUS) Simulation Tutorial

This tutorial describes how to perform individualized forward acoustic simulations of Transcranial Ultrasound Stimulation (tFUS) using `brain-fwi` and `j-Wave`.

## Overview

The goal of this pipeline is to:
1.  **Individualize**: Use a SimNIBS (CHARM) segmentation of a subject's MRI.
2.  **Map Properties**: Assign acoustic properties (speed of sound, density, absorption) using the ITRUSST benchmark values.
3.  **Target**: Place a focused bowl transducer at a specific anatomical target.
4.  **Simulate**: Use the JAX-based PSTD solver (`j-Wave`) to compute the 3D peak pressure field.

## 1. Setup

Ensure you have a SimNIBS `m2m` directory for your subject. If you don't have one, the simulation script can fall back to a synthetic three-layer head model.

## 2. Running a Simulation

Use the provided example script `examples/04_tus_forward_simulation.py`.

```bash
# Run with a synthetic head model (default)
uv run python examples/04_tus_forward_simulation.py --grid-size 64 --target 0.0 0.0 0.0

# Run with a SimNIBS m2m directory
uv run python examples/04_tus_forward_simulation.py \
    --m2m /path/to/m2m_subject \
    --target 0.02 -0.01 0.0 \
    --frequency 500000 \
    --grid-size 128 \
    --dx 0.001
```

### Parameters:
- `--m2m`: Path to the SimNIBS output directory.
- `--target`: Focal target $(x, y, z)$ in metres, relative to the head center.
- `--frequency`: Transducer center frequency in Hz (e.g., 500,000 for 500 kHz).
- `--grid-size`: Number of voxels per dimension.
- `--dx`: Grid spacing in metres (e.g., 0.001 for 1 mm).

## 3. Acoustic Properties (ITRUSST Benchmarks)

The simulation uses the following reference values:
- **Water/CSF**: 1500 m/s, 1000 kg/m³
- **Grey/White Matter**: 1560 m/s, 1040 kg/m³, 0.6 dB/cm/MHz
- **Cortical Bone**: 2800 m/s, 1850 kg/m³, 4.0 dB/cm/MHz
- **Trabecular Bone**: 2300 m/s, 1700 kg/m³, 8.0 dB/cm/MHz

## 4. Convergence & PPW

For accurate results, the grid resolution should provide at least **6-10 points per wavelength (PPW)**. The script performs an automatic convergence check at startup. 

If your PPW is too low:
1.  Increase `--grid-size`.
2.  Decrease `--dx`.
3.  Ensure the frequency is appropriate for the resolution.

## 5. Outputs

- **`tus_simulation.png`**: A summary plot showing the acoustic medium, the transducer placement, and the resulting peak pressure field in Axial and Coronal slices.
- **Peak Pressure Map**: The script computes `jnp.maximum(peak, jnp.abs(p))` at every timestep, providing the maximum pressure reached at every voxel during the simulation.
