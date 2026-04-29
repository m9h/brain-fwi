# ICL dual-probe transcranial USCT dataset (Robins et al. 2023)

**Citation.** Robins TC, Cueto C, Cudeiro J, Bates O, Calderon Agudo O,
Strong G, Guasch L, Warner M, Tang M-X. *Dual-probe transcranial
full-waveform inversion: a brain phantom feasibility study.*
Ultrasound Med. Biol. 49(10):2302–2315 (2023).
PubMed [37474432](https://pubmed.ncbi.nlm.nih.gov/37474432/) ·
PMC [PMC7616382](https://pmc.ncbi.nlm.nih.gov/articles/PMC7616382/).

**Data.** [doi.org/10.17632/rbx3ybd5zx.1](https://doi.org/10.17632/rbx3ybd5zx.1)
(Mendeley Data, CC BY 4.0). 7.7 GB unzipped, 11 files.

## Why we use it

Real PVA-skull-phantom transmission traces, plus matching synthetic
traces over the same dual-probe ring geometry. Lets us test FWI on
real attenuated transcranial data — the experimental measurements
encode skull absorption and dispersion that no purely-numerical
benchmark (k-Wave / j-Wave forward) reproduces.

## Acquisition geometry

| Parameter | Value |
|---|---|
| Probe | 2× P4-1 cardiac, 24 active elements per shot |
| Rotation positions | 16 about target |
| Total source positions | 384 |
| Receivers per shot | 1152 (of 3072 elements) |
| Total traces | 442 368 |
| Trace length | 3 328 samples |
| Trace sampling rate (inferred) | 22.727 MHz → `dt = 44 ns` |
| Source wavelet length | 2 730 samples |
| Element list size | 3 072 (xyz, metres) |
| Tank | 0.7 × 0.6 × 0.6 m, water |

## Files in the archive

| File | Format | Variable | What |
|---|---|---|---|
| `01_Experimental_USCT_Brain.mat` | HDF5 (v7.3) | `dataset` `(442368, 3328)` f64 | Real PVA-skull-phantom traces |
| `02_Synthetic_USCT_Brain.mat` | HDF5 (v7.3) | **`WaterShot`** `(442368, 3328)` f64 | Synthetic brain traces (variable name is wrong, see below) |
| `03_Experimental_USCT_WaterShot.mat` | HDF5 (v7.3) | `dataset` `(442368, 3328)` f64 | Water-only reference shot |
| `signal_filtered.mat` | MAT v5 | `signal` `(2730, 384)` f64 | 384 source wavelets, one per source position |
| `elementList.txt` | CSV | `EleID,x,y,z` | 3 072 calibrated element positions, metres |
| `elementRelas.txt` | CSV | `TraceID,Src,Rcv` | 442 368 src/rcv element-ID pairs (1-based) |
| `Dual_Probe_Ring_Array.png` | PNG | — | Ring geometry diagram |
| `Experimental_Imaging_Setup.png` | PNG | — | Tank photo |
| `Source_Signal_Filtered.png` | PNG | — | Source wavelet plot |
| `Experimental_FWI_Reconstruction_Brain.png` | PNG | — | Authors' final reconstruction |
| `Synthetic_FWI_Reconstruction_Brain.png` | PNG | — | Authors' final synthetic reconstruction |

## Issues with the published archive

### 1. `TrueVp.mat` is missing

The Mendeley dataset description lists a file:

> `TrueVp.mat` — true velocity model of the numerical brain phantom
> (dx = 0.186 mm, dimensions = [957, 958])

**This file was never uploaded.** The Mendeley public file API
(`/api/datasets/rbx3ybd5zx/files?version=1`) enumerates exactly the 11
files above; `TrueVp` is not among them. We confirmed this against the
public listing on 2026-04-28. Re-downloading does not help.

**Consequence:** synthetic-mode users have no ground-truth velocity
to compare reconstructions against. The loader returns no
`true_velocity` key. Downstream code that wants ground truth must
either (a) email the authors, (b) reconstruct the phantom from the
paper's geometric description, or (c) skip ground-truth comparisons
on this dataset.

### 2. Wrong root variable name in `02_Synthetic_USCT_Brain.mat`

The synthetic-brain file stores its trace matrix under the root key
`WaterShot`, not `dataset` (matching the real-brain and water-shot
files). The shape `(442368, 3328)` and content match the synthetic
brain expectation. This is almost certainly a copy-paste bug from the
authors' upload script — the `WaterShot` variable name was kept from
file 03's template.

**Consequence:** loaders that hard-code `f["dataset"]` fail.
`brain_fwi.data.icl_dual_probe.load_icl_dual_probe` reads whichever
single key is at the root of the file.

### 3. `dt` is not stated

Neither the paper nor the archive metadata gives the temporal
sampling rate of the saved traces. The paper §2.x quotes
`dt = 5.00 × 10⁻² µs` (50 ns) for the **forward simulation** time
step, but this is a separate quantity from the acquisition rate of
the stored experimental data.

**Inferred value: `dt = 11 / 250 × 10⁶ s ≈ 44 ns`** (i.e.
22.727 MHz). Two converging lines of evidence:

- Verasonics systems clock ADC sampling at 250 MHz / N. The lowest
  N giving a rate compatible with 2.5 MHz P4-1 probes that matches
  published Verasonics documentation is N=11 → 22.727 MHz.
- The element table places all 3 072 elements on a ring of radius
  ≈ 0.11 m. Maximum cross-tank acoustic round-trip:
  `2 · 0.11 m / 1500 m s⁻¹ ≈ 146.7 µs`. Trace duration at 44 ns:
  `3328 · 44 ns = 146.4 µs` (0.2 % match). At 50 ns the trace would
  cover 166.4 µs — 13 % too long for the geometry.

**Consequence:** the loader exposes `dt = 44 ns` as
`brain_fwi.data.icl_dual_probe.DT_SECONDS`. Users who later confirm
the rate from the authors should update that constant.

### 4. Source signal length ≠ trace length

`signal_filtered.mat` stores wavelets of length 2 730 samples; traces
are 3 328 samples. The wavelet ends before the trace does. The
loader returns the wavelet at its native length and leaves zero-pad
(or projection onto the simulation grid) to the FWI driver.

### 5. h5py reads the trace matrix transposed vs. MATLAB

MATLAB's logical layout for these files is `(n_time, n_traces) =
(3328, 442368)`. h5py exposes the underlying HDF5 storage as
`(442368, 3328)`. Both views are correct; they describe the same
bytes. The loader returns the h5py orientation —
`traces[trace_idx, time_idx]` — which is the natural numpy idiom.

## Local layout expected by the loader

Unzip the Mendeley archive flat (no enclosing folder) and point the
loader at it:

```
$DATA_ROOT/icl-dual-probe-2023/
├── 01_Experimental_USCT_Brain.mat
├── 02_Synthetic_USCT_Brain.mat
├── 03_Experimental_USCT_WaterShot.mat
├── elementList.txt
├── elementRelas.txt
├── signal_filtered.mat
└── … (PNGs, optional)
```

```bash
export ICL_DUAL_PROBE_PATH=$DATA_ROOT/icl-dual-probe-2023
uv run pytest tests/test_icl_dual_probe_loader.py -v
```
