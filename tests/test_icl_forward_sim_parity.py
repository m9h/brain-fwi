"""Forward-simulation parity check against the ICL dual-probe geometry.

This test does *not* compare j-Wave traces directly against the ICL
synthetic traces — that comparison would mix three solver-specific
choices (Stride/Devito vs. j-Wave, the authors' source wavelet vs.
ours, finite-volume vs. pseudospectral). Instead it does the simpler
self-consistency check the loader actually needs:

    Given two transducer positions from `transducer_positions_m`, and
    homogeneous water at 1500 m/s in the brain-fwi forward solver,
    does the simulated peak arrival match the geometric TOF?

This catches: position-table unit errors (mm vs m), grid-mapping
mistakes (xz vs xy), CFL miscalculation, and sensor-extraction bugs
in `simulate_shot_sensors`. It does *not* validate the ICL synthetic
trace data itself — that's covered by the speed-of-sound fit in
`tests/test_icl_dual_probe_loader.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    jax.default_backend() == "mps",
    reason=(
        "j-Wave's source-seeding uses lax.scatter, which has a known "
        "broadcast-shape gap on the MPS backend. CPU and NVIDIA work. "
        "See docs/dev/apple-silicon-gpu.md."
    ),
)

DEFAULT_PATH = Path(
    os.environ.get(
        "ICL_DUAL_PROBE_PATH",
        "/data/datasets/icl-dual-probe-2023",
    )
)

REQUIRED_FILES = [
    "02_Synthetic_USCT_Brain.mat",
    "elementList.txt",
    "elementRelas.txt",
    "signal_filtered.mat",
]

needs_data = pytest.mark.skipif(
    not all((DEFAULT_PATH / f).exists() for f in REQUIRED_FILES),
    reason=f"ICL dual-probe data not at {DEFAULT_PATH}",
)


@needs_data
def test_jwave_forward_matches_geometric_tof_in_water():
    """A j-Wave 2D forward sim in homogeneous water reproduces the
    geometric time of flight for a real ICL transducer pair within 1
    µs (≈ 23 samples at dt=44 ns).

    Tolerance: grid quantisation at dx=2 mm caps position error at
    1 mm → 0.67 µs; pseudospectral numerical dispersion at 3.75 ppw
    over ~13 wavelengths is small. We assert <1 µs.
    """
    import jax.numpy as jnp

    from brain_fwi.data.icl_dual_probe import load_icl_dual_probe
    from brain_fwi.simulation.forward import (
        build_domain,
        build_medium,
        build_time_axis,
        simulate_shot_sensors,
    )
    from brain_fwi.utils.wavelets import ricker_wavelet

    sample = load_icl_dual_probe(DEFAULT_PATH, mode="synthetic")
    try:
        positions = sample["transducer_positions_m"]
        pairs = sample["src_recv_pairs"]

        # Pick the pair with baseline closest to 100 mm.
        baselines = np.linalg.norm(
            positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1
        )
        chosen = int(np.argmin(np.abs(baselines - 0.10)))
        src_pos = positions[pairs[chosen, 0]]
        rcv_pos = positions[pairs[chosen, 1]]
        baseline_m = float(np.linalg.norm(src_pos - rcv_pos))
        assert 0.09 < baseline_m < 0.11, baseline_m

        # Working plane is xz (y == 0 confirmed by planar test).
        c_water = 1500.0
        dx = 0.002
        pml_size = 12
        pad_m = 0.03

        x_lo = float(min(src_pos[0], rcv_pos[0])) - pad_m
        x_hi = float(max(src_pos[0], rcv_pos[0])) + pad_m
        z_lo = float(min(src_pos[2], rcv_pos[2])) - pad_m
        z_hi = float(max(src_pos[2], rcv_pos[2])) + pad_m

        nx = int(np.ceil((x_hi - x_lo) / dx)) + 2 * pml_size
        nz = int(np.ceil((z_hi - z_lo) / dx)) + 2 * pml_size
        grid_shape = (nx, nz)

        def to_grid(p):
            ix = int(round((float(p[0]) - x_lo) / dx)) + pml_size
            iz = int(round((float(p[2]) - z_lo) / dx)) + pml_size
            return ix, iz

        src_grid = to_grid(src_pos)
        rcv_grid = to_grid(rcv_pos)

        domain = build_domain(grid_shape, dx)
        medium = build_medium(domain, c_water, 1000.0, pml_size=pml_size)
        time_axis = build_time_axis(medium, cfl=0.3, t_end=180e-6)
        dt_sim = float(time_axis.dt)
        n_samples = int(float(time_axis.t_end) / dt_sim)

        f0 = 200e3
        delay = 1.5 / f0
        signal = ricker_wavelet(f0, dt_sim, n_samples, delay=delay)

        sensor_grid = ([rcv_grid[0]], [rcv_grid[1]])
        rec = np.asarray(
            simulate_shot_sensors(
                medium, time_axis, src_grid, sensor_grid, signal, dt_sim,
            )
        ).squeeze()
        assert rec.ndim == 1, rec.shape
        assert np.isfinite(rec).all()
        assert np.abs(rec).max() > 0.0

        t_predicted = baseline_m / c_water + delay
        t_observed = float(np.argmax(np.abs(rec)) * dt_sim)
        delta_us = (t_observed - t_predicted) * 1e6

        assert abs(delta_us) < 1.0, (
            f"sim peak TOF {t_observed*1e6:.2f} us vs geometric "
            f"{t_predicted*1e6:.2f} us (delta {delta_us:+.2f} us, "
            f"baseline {baseline_m*1000:.1f} mm)"
        )
    finally:
        sample["_h5_file"].close()
