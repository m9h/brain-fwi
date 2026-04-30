# Brainlife.io App: Transcranial USCT Forward Simulation

This directory contains the configuration and wrapper scripts to run Brain-FWI simulations on [brainlife.io](https://brainlife.io).

## Overview

This App performs a full-wave acoustic simulation of a transcranial ultrasound computed tomography (USCT) scan. It uses a SimNIBS (CHARM) head segmentation to define the acoustic properties (sound speed, density, attenuation) of the head.

## Inputs

- **SimNIBS (neuro/m2m)**: A directory containing the `tissue_labeling.nii.gz` file produced by SimNIBS/CHARM.

## Configurable Parameters

- `grid_size`: The dimension of the computational grid (NxNxN). Default is 96.
- `dx`: Grid spacing in meters. Default is 0.002 (2mm).
- `n_elements`: Number of transducers in the helmet array. Default is 128.
- `frequency`: Source center frequency in Hz. Default is 500,000 (500 kHz).
- `shots`: Number of source positions to simulate.

## Outputs

- `simulation.h5`: HDF5 file containing the simulated sensor data, acoustic property volumes, and sensor positions.
- `summary.png`: Visual summary of the phantom and recorded traces.

## How to Register on Brainlife

1. Push this repository to GitHub.
2. Log in to [brainlife.io](https://brainlife.io).
3. Go to "Apps" and click "Register App".
4. Point to your GitHub repository and the `brainlife/manifest.json` file.
5. Set the Docker image to a built version of `brainlife/Dockerfile` or let Brainlife build it from the repo.
