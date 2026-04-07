# Changelog

All notable changes to brain-fwi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Sphinx documentation with Furo theme, autodoc, and MyST Markdown
- CONTRIBUTING.md with development setup, testing, and submission guidelines
- Tutorials: forward simulation, FWI reconstruction, head phantoms
- `.readthedocs.yaml` for Read the Docs hosting
- `[docs]` optional dependency group in pyproject.toml

## [0.1.0] - 2025-01-01

### Added
- Forward acoustic simulation via j-Wave pseudospectral solver
  (`brain_fwi.simulation.forward`)
- Full Waveform Inversion engine with multi-frequency banding
  (`brain_fwi.inversion.fwi`)
- Loss functions: L2, Hilbert envelope, multiscale
  (`brain_fwi.inversion.losses`)
- Resolution matrix and PSF analysis
  (`brain_fwi.inversion.resolution`)
- BrainWeb phantom loading and synthetic head generator
  (`brain_fwi.phantoms.brainweb`)
- MIDA 153-label head model loading and acoustic property mapping
  (`brain_fwi.phantoms.mida`)
- ITRUSST benchmark acoustic property table for 12 BrainWeb tissue classes
  (`brain_fwi.phantoms.properties`)
- Ring (2D) and helmet (3D) transducer array generators
  (`brain_fwi.transducers.helmet`)
- Ricker wavelet and toneburst source signal generators
  (`brain_fwi.utils.wavelets`)
- 106 tests (unit, integration, end-to-end)
- 4 examples: 2D axial FWI, 3D brain FWI, 3D brain recovery, quick test
