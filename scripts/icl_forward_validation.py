"""ICL dual-probe: end-to-end forward-sim sanity check.

Wires the loader, geometry table, and j-Wave wrapper together on a
single near-diametric src/rcv pair: pull the source wavelet from
``signal_filtered.mat``, resample it to j-Wave's CFL-bound dt, run a
2D homogeneous-water forward sim, resample back to 44 ns, and bandpass.

This is **not** a Phase-5 motivation experiment. The ICL synthetic
traces were generated through the brain phantom + skull, so a water-
only sim is expected to disagree with them on shape — we just confirm
the pipeline runs end-to-end without errors and that the simulated
direct arrival is in the right place.

For the actual synthetic-vs-experimental gap analysis (the ICL-data-
level diagnostic), see ``icl_syn_vs_exp_diagnostic.py``.

Run:
    ICL_DUAL_PROBE_PATH=/path/to/icl-dual-probe-2023 \
        uv run python scripts/icl_forward_validation.py
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.signal import butter, sosfiltfilt

from brain_fwi.data.icl_dual_probe import load_icl_dual_probe, select_shot
from brain_fwi.simulation.forward import (
    build_domain,
    build_medium,
    build_time_axis,
    simulate_shot_sensors,
)


def bandpass(x: np.ndarray, dt: float, lo_hz: float, hi_hz: float) -> np.ndarray:
    fnyq = 0.5 / dt
    sos = butter(4, [lo_hz / fnyq, hi_hz / fnyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def normalised_residual(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (L2 residual / trace energy, peak cross-correlation)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    resid = float(np.linalg.norm(a - b) ** 2 / 2.0)
    xc = np.correlate(a, b, mode="full")
    return resid, float(np.max(xc))


def find_long_baseline_pair(positions: np.ndarray, pairs: np.ndarray):
    """Pick the trace with the longest src-rcv baseline."""
    d = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    idx = int(np.argmax(d))
    return idx, float(d[idx])


def main():
    data_path = Path(
        os.environ.get(
            "ICL_DUAL_PROBE_PATH",
            "/Users/mhough/Workspace/brain-fwi/data/icl-dual-probe-2023",
        )
    )

    print(f"Loading ICL dataset from {data_path}\n")
    syn = load_icl_dual_probe(data_path, mode="synthetic")
    exp = load_icl_dual_probe(data_path, mode="experimental")

    try:
        positions = syn["transducer_positions_m"]
        pairs = syn["src_recv_pairs"]
        dt_data = syn["dt"]
        wavelets = syn["source_signal"]  # (2730, 384)

        trace_idx, baseline = find_long_baseline_pair(positions, pairs)
        src_id = int(pairs[trace_idx, 0])
        rcv_id = int(pairs[trace_idx, 1])
        src_pos = positions[src_id]
        rcv_pos = positions[rcv_id]

        # Identify which of the 384 source positions this is — the
        # wavelet column is per source position, indexed 0..383, in
        # the order they appear in src_recv_pairs.
        unique_src = np.unique(pairs[:, 0])
        source_position_index = int(np.flatnonzero(unique_src == src_id)[0])
        wavelet = wavelets[:, source_position_index].astype(np.float64)

        print(f"Selected pair (trace {trace_idx}):")
        print(f"  src element {src_id} at xz=({src_pos[0]*1e3:6.1f}, "
              f"{src_pos[2]*1e3:6.1f}) mm")
        print(f"  rcv element {rcv_id} at xz=({rcv_pos[0]*1e3:6.1f}, "
              f"{rcv_pos[2]*1e3:6.1f}) mm")
        print(f"  baseline = {baseline*1e3:.2f} mm")
        print(f"  source wavelet = column {source_position_index} of "
              f"signal_filtered (peak at sample "
              f"{int(np.argmax(np.abs(wavelet)))})")

        syn_trace = np.asarray(syn["traces"][trace_idx]).astype(np.float64)
        exp_trace = np.asarray(exp["traces"][trace_idx]).astype(np.float64)
        n_data = syn_trace.size
        assert exp_trace.size == n_data

        # ------- forward simulation in homogeneous water -------
        c_water = 1500.0
        dx = 0.001  # 1 mm — finer than parity test for better wavelet shape
        pml = 16
        pad = 0.03

        x_lo = float(min(src_pos[0], rcv_pos[0])) - pad
        x_hi = float(max(src_pos[0], rcv_pos[0])) + pad
        z_lo = float(min(src_pos[2], rcv_pos[2])) - pad
        z_hi = float(max(src_pos[2], rcv_pos[2])) + pad
        nx = int(np.ceil((x_hi - x_lo) / dx)) + 2 * pml
        nz = int(np.ceil((z_hi - z_lo) / dx)) + 2 * pml

        def to_grid(p):
            ix = int(round((float(p[0]) - x_lo) / dx)) + pml
            iz = int(round((float(p[2]) - z_lo) / dx)) + pml
            return ix, iz

        src_grid = to_grid(src_pos)
        rcv_grid = to_grid(rcv_pos)

        domain = build_domain((nx, nz), dx)
        medium = build_medium(domain, c_water, 1000.0, pml_size=pml)
        # Run long enough to see direct arrival + a tail: data is 146.4 µs;
        # match it.
        t_end = n_data * dt_data
        time_axis = build_time_axis(medium, cfl=0.3, t_end=t_end)
        dt_sim = float(time_axis.dt)
        n_sim = int(float(time_axis.t_end) / dt_sim)

        # Resample the ICL wavelet onto the sim time grid (linear interp).
        t_data = np.arange(wavelet.size) * dt_data
        t_sim = np.arange(n_sim) * dt_sim
        sim_wavelet = np.interp(t_sim, t_data, wavelet, left=0.0, right=0.0)

        print(f"\nForward sim grid: {nx}×{nz} cells at dx={dx*1e3:.1f} mm "
              f"({nx*nz/1e3:.1f} k cells, PML={pml})")
        print(f"  dt_sim = {dt_sim*1e9:.1f} ns ({n_sim} steps), "
              f"t_end = {t_end*1e6:.0f} µs")
        print(f"  src grid {src_grid}, rcv grid {rcv_grid}")
        print("  running j-Wave forward (compile + execute, 1-2 min on CPU)…")

        sensor_grid = ([rcv_grid[0]], [rcv_grid[1]])
        rec = np.asarray(
            simulate_shot_sensors(
                medium, time_axis, src_grid, sensor_grid,
                jnp.asarray(sim_wavelet), dt_sim,
            )
        ).squeeze()
        assert np.isfinite(rec).all()
        print(f"  sim trace shape={rec.shape}, peak={np.abs(rec).max():.3e}")

        # Resample sim trace back to data dt — use actual sim length
        # (j-Wave can return n_sim+1 samples)
        t_rec = np.arange(rec.size) * dt_sim
        sim_on_data_dt = np.interp(
            np.arange(n_data) * dt_data, t_rec, rec, left=0.0, right=0.0
        )

        # ------- alignment + comparison -------
        geom_tof = baseline / c_water
        # Window 50 µs around direct arrival (covers wavelet duration)
        t = np.arange(n_data) * dt_data
        win = (t > geom_tof - 10e-6) & (t < geom_tof + 50e-6)
        if not win.any():
            raise RuntimeError("alignment window empty")

        sim_b = bandpass(sim_on_data_dt[win], dt_data, 1e5, 1.5e6)
        syn_b = bandpass(syn_trace[win], dt_data, 1e5, 1.5e6)
        exp_b = bandpass(exp_trace[win], dt_data, 1e5, 1.5e6)

        resid_syn, xc_syn = normalised_residual(sim_b, syn_b)
        resid_exp, xc_exp = normalised_residual(sim_b, exp_b)

        # Energy ratio (raw amplitude shouldn't be expected to match
        # because of 2D vs 2.5D and source-injection scaling, so we
        # report it but don't penalise)
        e_sim = float(np.linalg.norm(sim_b))
        e_syn = float(np.linalg.norm(syn_b))
        e_exp = float(np.linalg.norm(exp_b))

        # Peak arrival comparison
        def peak_t(x):
            return int(np.argmax(np.abs(x))) * dt_data * 1e6  # µs

        print("\n" + "=" * 70)
        print("RESULT — bandpassed (100 kHz – 1.5 MHz), windowed ±50 µs")
        print("=" * 70)
        print(f"{'metric':40s}{'sim vs syn':>14s}{'sim vs exp':>14s}")
        print("-" * 70)
        print(f"{'normalised L2 residual':40s}"
              f"{resid_syn:>14.4f}{resid_exp:>14.4f}")
        print(f"{'peak cross-correlation (1.0=perfect)':40s}"
              f"{xc_syn:>14.4f}{xc_exp:>14.4f}")
        print(f"{'energy ratio  (data / sim)':40s}"
              f"{e_syn/e_sim:>14.3f}{e_exp/e_sim:>14.3f}")
        print(f"{'peak arrival (µs)':40s}"
              f"{peak_t(sim_b):>14.2f}{'':>0s}")
        print(f"{'  syn={:.2f}  exp={:.2f}  geom={:.2f}'.format(peak_t(syn_b), peak_t(exp_b), geom_tof*1e6):40s}")
        print("=" * 70)

        # Residuals here will be O(1) regardless: ICL synthetic and
        # experimental both contain a brain phantom + skull whose
        # geometry our water-only sim doesn't model. The test passes as
        # long as the pipeline runs and the geometric arrival lines up.
        print(f"\n(residual ratio exp / syn = "
              f"{resid_exp / max(resid_syn, 1e-9):.2f}×; both ≈ 1.0 because "
              f"this is a water-only sim, not a phantom sim)")

    finally:
        syn["_h5_file"].close()
        exp["_h5_file"].close()


if __name__ == "__main__":
    main()
