#!/usr/bin/env python
"""Example 4: Very-High-Density (VHD) FWI on a MIDA axial slice.

Demonstrates transcranial Full Waveform Inversion driven by a 1024-element
receiver ring around a real anatomical head model (IT'IS Foundation MIDA
v1.0, 116 tissue classes, 3-layer skull). Compared to the 128-element
default in `examples/01_2d_axial_fwi.py`, this script exercises the
"VHD" regime: sub-millimetre angular sampling on the receive side,
sparse-source / dense-receiver geometry typical of next-generation USCT
helmets.

Pipeline:
  1. Load MIDA NIfTI volume, extract the most brain-rich axial slice.
  2. Resample to a working grid (256² at ~1 mm by default).
  3. Map 116 tissue labels -> ITRUSST acoustic properties (c, rho, alpha).
  4. Build a 1024-element ring with 128 firing sources (sparse-source).
  5. Generate ground-truth data with the true model.
  6. Multi-frequency FWI starting from homogeneous water.
  7. Save reconstruction figure, run metrics, and HDF5 with all volumes.

Tuned for an Apple-Silicon M5 Max (CPU XLA, 36 GB unified memory, 18 cores).
Estimated wall time: ~80 minutes at the defaults.

Run:
    uv run python examples/04_vhd_mida_fwi.py
    uv run python examples/04_vhd_mida_fwi.py --quick   # smaller, ~10 min
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Force XLA-CPU multi-threading across all M5 Max cores.
os.environ.setdefault("OMP_NUM_THREADS", "18")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=18",
)

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

jax.config.update("jax_enable_x64", False)

from brain_fwi.inversion.fwi import FWIConfig, run_fwi
from brain_fwi.phantoms.mida import (
    load_mida_volume,
    map_mida_labels_to_acoustic,
)
from brain_fwi.simulation.forward import (
    _build_source_signal,
    build_domain,
    build_medium,
    build_time_axis,
    generate_observed_data,
)
from brain_fwi.transducers import ring_array_2d, transducer_positions_to_grid


def find_axial_slice(labels_3d: np.ndarray, axis: int = 2) -> int:
    """Return the index along `axis` with the highest brain GM+WM voxel count."""
    brain = (labels_3d == 10) | (labels_3d == 12)
    counts = brain.sum(axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(counts))


def main():
    parser = argparse.ArgumentParser(description="VHD FWI on MIDA axial slice")
    parser.add_argument("--mida-path", type=str,
                        default="data/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii",
                        help="Path to MIDA volume (.nii / .mat)")
    parser.add_argument("--slice-axis", type=int, default=2,
                        help="Axis along which to extract the 2D slice (0/1/2)")
    parser.add_argument("--slice-idx", type=int, default=-1,
                        help="Slice index. -1 = auto (max brain content).")
    parser.add_argument("--grid-size", type=int, default=256,
                        help="Working grid dimension (NxN). Native MIDA is 480.")
    parser.add_argument("--n-receivers", type=int, default=1024,
                        help="VHD receiver-ring element count.")
    parser.add_argument("--n-sources", type=int, default=128,
                        help="Firing source count (subset of receivers).")
    parser.add_argument("--shots-per-iter", type=int, default=4)
    parser.add_argument("--iters-per-band", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=50.0,
                        help="Max velocity update per FWI iteration (m/s)")
    parser.add_argument("--output-dir", type=str, default="results/vhd_mida")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller config: 192 grid, 256 receivers, 5 iters/band")
    args = parser.parse_args()

    if args.quick:
        args.grid_size = 192
        args.n_receivers = 256
        args.n_sources = 64
        args.iters_per_band = 5

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Brain FWI — VHD-MIDA 2D Axial Demo")
    print(f"  Grid: {args.grid_size}², receivers: {args.n_receivers},"
          f" sources: {args.n_sources}")
    print(f"  Iters/band: {args.iters_per_band}, shots/iter: {args.shots_per_iter}")
    print(f"  Backend: {jax.default_backend()}, devices: {jax.devices()}")
    print("=" * 78)

    metrics: dict = {"args": vars(args), "stages": {}}

    # ============================================================
    # 1. Load MIDA + pick slice
    # ============================================================
    print("\n[1] Loading MIDA volume...")
    t0 = time.time()
    labels_3d = load_mida_volume(Path(args.mida_path))
    if args.slice_idx < 0:
        args.slice_idx = find_axial_slice(labels_3d, axis=args.slice_axis)

    slicer = [slice(None)] * 3
    slicer[args.slice_axis] = args.slice_idx
    labels_2d = labels_3d[tuple(slicer)]
    print(f"    MIDA: {labels_3d.shape}, slice axis={args.slice_axis} idx={args.slice_idx}")
    print(f"    Native slice: {labels_2d.shape}, "
          f"unique labels: {len(np.unique(labels_2d))}")
    metrics["stages"]["load"] = {"sec": time.time() - t0,
                                  "slice_idx": args.slice_idx,
                                  "n_unique_labels": int(len(np.unique(labels_2d)))}

    # ============================================================
    # 2. Resample to working grid
    # ============================================================
    print(f"\n[2] Resampling to {args.grid_size}² ...")
    t0 = time.time()
    scale = args.grid_size / labels_2d.shape[0]
    labels_resampled = zoom(labels_2d, (scale, scale), order=0).astype(np.int32)
    if labels_resampled.shape != (args.grid_size, args.grid_size):
        # Defensive: zoom can off-by-one
        labels_resampled = labels_resampled[: args.grid_size, : args.grid_size]
    native_dx = 0.5e-3
    dx = native_dx / scale
    print(f"    Resampled: {labels_resampled.shape}, dx={dx*1e3:.3f} mm,"
          f" extent={args.grid_size*dx*100:.1f} cm")
    metrics["stages"]["resample"] = {"sec": time.time() - t0,
                                      "dx_m": float(dx)}

    # Map to acoustic properties
    props = map_mida_labels_to_acoustic(jnp.array(labels_resampled))
    c_true = props["sound_speed"].astype(jnp.float32)
    rho = props["density"].astype(jnp.float32)

    # External air -> water coupling medium (otherwise FWI tries to invert air)
    c_true = jnp.where(c_true < 500.0, 1500.0, c_true)
    rho = jnp.where(rho < 100.0, 1000.0, rho)

    print(f"    Sound speed: [{float(jnp.min(c_true)):.0f},"
          f" {float(jnp.max(c_true)):.0f}] m/s")
    print(f"    Density:     [{float(jnp.min(rho)):.0f},"
          f" {float(jnp.max(rho)):.0f}] kg/m³")

    # ============================================================
    # 3. VHD ring array
    # ============================================================
    print(f"\n[3] Building {args.n_receivers}-element receiver ring...")
    cx = args.grid_size * dx / 2
    cy = args.grid_size * dx / 2

    # Slightly elliptical to wrap an adult head + 5 mm coupling water
    semi_a = min(args.grid_size * dx * 0.42, 0.105)  # AP, capped at 10.5 cm
    semi_b = semi_a * 0.85
    receiver_pos = ring_array_2d(
        n_elements=args.n_receivers,
        center=(cx, cy),
        semi_major=semi_a,
        semi_minor=semi_b,
        standoff=0.005,
    )
    sensor_grid = transducer_positions_to_grid(receiver_pos, dx, c_true.shape)

    # Sparse-source: every (n_recv / n_src)-th element fires
    src_step = args.n_receivers // args.n_sources
    src_grid_xy = (sensor_grid[0][::src_step][: args.n_sources],
                    sensor_grid[1][::src_step][: args.n_sources])
    src_positions_list = [(int(src_grid_xy[0][i]), int(src_grid_xy[1][i]))
                           for i in range(args.n_sources)]

    print(f"    Ring semi-axes: {semi_a*100:.1f} × {semi_b*100:.1f} cm")
    print(f"    Receiver spacing: ~{2*np.pi*semi_a*1000/args.n_receivers:.2f} mm")
    print(f"    Sources fire every {src_step} elements")

    # ============================================================
    # 4. Time axis & source wavelet (computed once for consistency)
    # ============================================================
    domain = build_domain(c_true.shape, dx)
    ref_medium = build_medium(domain, c_true, rho, pml_size=20)
    time_axis = build_time_axis(ref_medium, cfl=0.3)
    dt = float(time_axis.dt)
    t_end = float(time_axis.t_end)
    n_samples = int(t_end / dt)
    print(f"\n[4] Time axis: dt={dt*1e6:.3f} µs, t_end={t_end*1e3:.2f} ms,"
          f" {n_samples} samples")

    # Use the highest band frequency for data generation (broadband Ricker)
    freq_data = 250e3
    source_signal = _build_source_signal(freq_data, dt, n_samples)

    # ============================================================
    # 5. Generate observed data
    # ============================================================
    print(f"\n[5] Generating observed data ({args.n_sources} shots)...")
    t0 = time.time()
    observed = generate_observed_data(
        c_true, rho, dx,
        src_positions_list, sensor_grid,
        freq_data,
        pml_size=20,
        cfl=0.3,
        t_end=t_end,
        time_axis=time_axis,
        source_signal=source_signal,
        dt=dt,
        verbose=True,
    )
    obs_sec = time.time() - t0
    print(f"    Observed shape: {observed.shape}, "
          f"{obs_sec:.1f}s ({obs_sec/args.n_sources:.2f}s/shot)")
    metrics["stages"]["observed"] = {
        "sec": obs_sec,
        "shape": list(observed.shape),
        "sec_per_shot": obs_sec / args.n_sources,
    }

    # ============================================================
    # 6. FWI
    # ============================================================
    print("\n[6] Running multi-frequency FWI...")
    t0 = time.time()

    c_init = jnp.full(c_true.shape, 1500.0, dtype=jnp.float32)

    # Inversion mask: only update interior of the ring
    yy, xx = jnp.meshgrid(jnp.arange(args.grid_size),
                          jnp.arange(args.grid_size), indexing="ij")
    rx = (semi_a * 0.95) / dx
    ry = (semi_b * 0.95) / dx
    inv_mask = jnp.where(
        ((xx - args.grid_size // 2) / ry) ** 2
        + ((yy - args.grid_size // 2) / rx) ** 2 <= 1.0,
        1.0, 0.0,
    )

    config = FWIConfig(
        freq_bands=[(50e3, 100e3), (100e3, 200e3), (200e3, 300e3)],
        n_iters_per_band=args.iters_per_band,
        shots_per_iter=args.shots_per_iter,
        learning_rate=args.learning_rate,
        c_min=1400.0,
        c_max=3200.0,
        pml_size=20,
        cfl=0.3,
        gradient_smooth_sigma=3.0,
        loss_fn="multiscale",
        envelope_weight=0.5,
        mask=inv_mask,
        verbose=True,
    )

    result = run_fwi(
        observed_data=observed,
        initial_velocity=c_init,
        density=rho,
        dx=dx,
        src_positions_grid=src_positions_list,
        sensor_positions_grid=sensor_grid,
        source_signal=source_signal,
        dt=dt,
        t_end=t_end,
        config=config,
    )
    fwi_sec = time.time() - t0
    print(f"\n    FWI total: {fwi_sec:.1f}s ({fwi_sec/60:.1f} min)")
    metrics["stages"]["fwi"] = {
        "sec": fwi_sec,
        "n_total_iters": args.iters_per_band * len(config.freq_bands),
        "sec_per_iter": fwi_sec / max(1, args.iters_per_band * len(config.freq_bands)),
        "final_loss": float(result.loss_history[-1]) if result.loss_history else None,
    }

    c_recon = result.velocity

    # ============================================================
    # 7. Metrics + figure + HDF5
    # ============================================================
    err = c_recon - c_true
    mask_arr = np.array(inv_mask)
    n_masked = jnp.maximum(jnp.sum(inv_mask), 1.0)
    rmse = float(jnp.sqrt(jnp.sum((err ** 2) * inv_mask) / n_masked))
    mae = float(jnp.sum(jnp.abs(err) * inv_mask) / n_masked)
    print(f"\n[7] Reconstruction metrics (masked):")
    print(f"    RMSE: {rmse:.1f} m/s")
    print(f"    MAE:  {mae:.1f} m/s")
    metrics["reconstruction"] = {"rmse_m_s": rmse, "mae_m_s": mae}

    # Save HDF5
    h5_path = out_dir / "vhd_mida_fwi.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("c_true", data=np.array(c_true))
        f.create_dataset("c_init", data=np.array(c_init))
        f.create_dataset("c_recon", data=np.array(c_recon))
        f.create_dataset("density", data=np.array(rho))
        f.create_dataset("labels", data=labels_resampled)
        f.create_dataset("inv_mask", data=mask_arr)
        f.create_dataset("loss_history", data=np.array(result.loss_history))
        f.create_dataset("receiver_positions",
                         data=np.array(receiver_pos))
        f.create_dataset("source_indices",
                         data=np.arange(0, args.n_receivers, src_step)[: args.n_sources])
        f.attrs["dx_m"] = dx
        f.attrs["dt_s"] = dt
        f.attrs["n_receivers"] = args.n_receivers
        f.attrs["n_sources"] = args.n_sources
    print(f"    Saved volumes to {h5_path}")

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    extent_cm = args.grid_size * dx * 100
    pos_cm = np.array(receiver_pos) * 100

    def _show(ax, arr, title, vmin=None, vmax=None, cmap="seismic", cbar="m/s"):
        im = ax.imshow(np.array(arr).T, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", extent=[0, extent_cm, 0, extent_cm])
        ax.set_title(title)
        ax.set_xlabel("cm"); ax.set_ylabel("cm")
        plt.colorbar(im, ax=ax, label=cbar)
        return im

    _show(axes[0, 0], c_true, "True sound speed (MIDA)", 1400, 3200)
    axes[0, 0].plot(pos_cm[:, 0], pos_cm[:, 1], "k.", ms=1.5,
                     label=f"{args.n_receivers} receivers")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    _show(axes[0, 1], c_init, "Initial (homogeneous water)", 1400, 3200)
    _show(axes[0, 2], c_recon, "FWI reconstruction", 1400, 3200)

    err_arr = np.array(err) * mask_arr
    _show(axes[1, 0], err_arr, "Error (recon − true), masked",
          -300, 300, cmap="RdBu_r")

    # Snapshot trajectory
    if result.velocity_history:
        last = result.velocity_history[-1]
        _show(axes[1, 1], last, f"Final band snapshot ({len(result.velocity_history)} bands)",
              1400, 3200)

    # Loss curve
    axes[1, 2].semilogy(result.loss_history, lw=1)
    axes[1, 2].set_xlabel("iteration"); axes[1, 2].set_ylabel("loss")
    axes[1, 2].set_title("Convergence (multi-band)")
    axes[1, 2].grid(True, alpha=0.3)
    n_per = config.n_iters_per_band
    for i, (fmin, fmax) in enumerate(config.freq_bands):
        axes[1, 2].axvline(x=i * n_per, color="gray", ls="--", alpha=0.4)
        axes[1, 2].text(i * n_per + 0.5, max(result.loss_history) * 0.8,
                         f"{fmin/1e3:.0f}-{fmax/1e3:.0f} kHz", fontsize=8)

    plt.suptitle(
        f"VHD-FWI on MIDA axial slice — {args.n_receivers} receivers,"
        f" {args.n_sources} sources, {args.grid_size}² @ {dx*1e3:.2f} mm",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig_path = out_dir / "vhd_mida_fwi.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved figure to {fig_path}")

    # JSON metrics
    json_path = out_dir / "vhd_mida_fwi.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"    Saved metrics to {json_path}")

    print("\n" + "=" * 78)
    print(f"Done. Total wall: "
          f"{(metrics['stages']['observed']['sec'] + metrics['stages']['fwi']['sec'])/60:.1f} min")
    print("=" * 78)


if __name__ == "__main__":
    main()
