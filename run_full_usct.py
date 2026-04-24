#!/usr/bin/env python
"""Full-scale brain USCT simulation with complete results output.

Single script that runs the entire pipeline and produces:
  1. HDF5 with all volumes, metrics, and helmet geometry
  2. Multi-panel comparison figure
  3. Printed summary table

Designed for DGX Spark / A100 GPU execution.
"""

import argparse
import time
from pathlib import Path

import os
# Request GPU before importing JAX
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
# DGX Spark (GB10) uses unified memory — don't preallocate
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("TF_FORCE_UNIFIED_MEMORY", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

# Fail loud if no GPU
_backend = jax.default_backend()
if _backend != "gpu":
    import warnings
    warnings.warn(
        f"JAX backend is '{_backend}', not 'gpu'. "
        f"Install CUDA support: uv pip install 'jax[cuda12]'\n"
        f"Falling back to CPU — this will be very slow for 3D.",
        stacklevel=1,
    )

from brain_fwi.phantoms.properties import map_labels_to_all
from brain_fwi.transducers.helmet import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    simulate_shot_sensors, _build_source_signal,
)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi


def create_head_phantom(grid_shape, dx):
    """Anatomical 3D head with three-layer heterogeneous skull.

    Skull structure (ITRUSST BM3 benchmark, Aubry et al. 2022):
      - Outer cortical table: ~2mm, label 7, c=2800 m/s, rho=1850
      - Diploe (trabecular):  ~3mm, label 11, c=2300 m/s, rho=1700
      - Inner cortical table: ~2mm, label 7, c=2800 m/s, rho=1850
    Total skull thickness: ~7mm (realistic adult calvarium).

    Labels come from ``phantoms/synthetic.py:make_three_layer_head``;
    canonical acoustic properties are assigned here via
    ``map_labels_to_all`` with water coupling for the background.
    """
    from brain_fwi.phantoms.synthetic import make_three_layer_head

    labels_np = make_three_layer_head(grid_shape, dx)
    labels = jnp.asarray(labels_np)

    props = map_labels_to_all(labels)
    c = jnp.where(labels == 0, 1500.0, props["sound_speed"])
    rho = jnp.where(labels == 0, 1000.0, props["density"])
    alpha = jnp.where(labels == 0, 0.0, props["attenuation"])

    return labels, c, rho, alpha


def create_helmet(n_elements, grid_shape, dx):
    """Kernel Flow-inspired probe helmet."""
    cx_m = grid_shape[0] * dx / 2
    cy_m = grid_shape[1] * dx / 2
    cz_m = grid_shape[2] * dx / 2

    r_ap = min(0.095, (grid_shape[0] * dx / 2) - 5 * dx)
    r_lr = min(0.075, (grid_shape[1] * dx / 2) - 5 * dx)
    r_si = min(0.090, (grid_shape[2] * dx / 2) - 5 * dx)

    positions = helmet_array_3d(
        n_elements=n_elements,
        center=(cx_m, cy_m, cz_m),
        radius_ap=r_ap, radius_lr=r_lr, radius_si=r_si,
        standoff=0.005, coverage_angle=2.8, exclude_face=True,
    )
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    n = len(pos_grid[0])
    src_list = [(int(pos_grid[0][i]), int(pos_grid[1][i]), int(pos_grid[2][i]))
                for i in range(n)]
    return positions, pos_grid, src_list, n


def main():
    parser = argparse.ArgumentParser(description="Full Brain USCT")
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--n-elements", type=int, default=128)
    parser.add_argument("--dx", type=float, default=0.002)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--output", type=str, default="/data/datasets/brain-fwi/brain_usct_results.h5")
    parser.add_argument("--figures", type=str, default="/data/datasets/brain-fwi/brain_usct_figures.png")
    parser.add_argument(
        "--phantom", choices=("synthetic", "mida"), default="synthetic",
        help="synthetic = parametric three-layer ellipsoid; mida = MIDA v1.0 NIfTI",
    )
    parser.add_argument(
        "--mida-path", type=str,
        default="/data/datasets/MIDAv1-0/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii",
        help="Path to the MIDA label NIfTI (only used when --phantom mida)",
    )
    parser.add_argument(
        "--parameterization", choices=("voxel", "siren"), default="voxel",
        help="FWI parameterisation: voxel grid (default) or SIREN MLP.",
    )
    parser.add_argument(
        "--siren-pretrain-steps", type=int, default=600,
        help="SIREN Adam-pretrain steps (ignored for voxel path).",
    )
    parser.add_argument(
        "--siren-lr", type=float, default=1e-3,
        help="Adam learning rate for SIREN-path FWI.",
    )
    args = parser.parse_args()

    N = args.grid_size
    grid_shape = (N, N, N)
    dx = args.dx
    domain_cm = N * dx * 100
    t_total_start = time.time()

    print("=" * 70)
    print("  Transcranial Ultrasound Computed Tomography")
    print("  Full Waveform Inversion — Brain Sound Speed Recovery")
    print("=" * 70)
    print(f"  Device:      {jax.devices()[0]}")
    print(f"  Backend:     {jax.default_backend()}")
    print(f"  Grid:        {N}^3 = {N**3:,} voxels")
    print(f"  Spacing:     {dx*1e3:.1f} mm")
    print(f"  Domain:      {domain_cm:.1f} cm^3")
    print(f"  Elements:    {args.n_elements}")
    print(f"  FWI iters:   {args.iters}/band")
    print(f"  Shots/iter:  {args.shots}")

    # ---- Phantom ----
    print(f"\n{'='*70}")
    print("  STEP 1: Head Phantom")
    print(f"{'='*70}")
    t0 = time.time()
    if args.phantom == "synthetic":
        labels, c_true, rho, alpha = create_head_phantom(grid_shape, dx)
        tissue_counts = {
            "Water (coupling)": int(jnp.sum(labels == 0)),
            "CSF + ventricles": int(jnp.sum(labels == 1)),
            "Grey matter": int(jnp.sum(labels == 2)),
            "White matter": int(jnp.sum(labels == 3)),
            "Scalp": int(jnp.sum(labels == 6)),
            "Skull (cortical)": int(jnp.sum(labels == 7)),
            "Skull (trabecular)": int(jnp.sum(labels == 11)),
            "Lesion (haemorrhage)": int(jnp.sum(labels == 8)),
        }
    elif args.phantom == "mida":
        from brain_fwi.phantoms.mida import make_mida_phantom, MIDA_LABEL_NAMES
        print(f"  Source:      MIDA v1.0 NIfTI ({args.mida_path})")
        labels, c_true, rho, alpha = make_mida_phantom(
            args.mida_path, grid_shape, dx, add_lesion=True,
        )
        # MIDA label inventory — show the most populous + the lesion
        import numpy as np_
        uniq, counts = np_.unique(np_.asarray(labels), return_counts=True)
        sorted_pairs = sorted(zip(counts.tolist(), uniq.tolist()), reverse=True)
        tissue_counts = {}
        for count, lab in sorted_pairs[:10]:
            name = (
                "Lesion (haemorrhage)" if lab == 8
                else MIDA_LABEL_NAMES.get(int(lab), f"Label {lab}")
            )
            tissue_counts[name] = int(count)
        lesion_count = int(jnp.sum(labels == 8))
        if lesion_count and "Lesion (haemorrhage)" not in tissue_counts:
            tissue_counts["Lesion (haemorrhage)"] = lesion_count
    else:
        raise ValueError(f"Unknown --phantom {args.phantom!r}")

    for tissue, count in tissue_counts.items():
        if count > 0:
            print(f"  {tissue:25s}: {count:>8,} voxels")
    print(f"  Speed range: [{float(jnp.min(c_true)):.0f}, {float(jnp.max(c_true)):.0f}] m/s")
    print(f"  Time: {time.time()-t0:.1f} s")

    # ---- Helmet ----
    print(f"\n{'='*70}")
    print("  STEP 2: Probe Helmet")
    print(f"{'='*70}")
    t0 = time.time()
    positions_m, pos_grid, src_list, n_actual = create_helmet(
        args.n_elements, grid_shape, dx
    )
    print(f"  Elements placed: {n_actual}")
    print(f"  Coverage: ~160 deg (face excluded)")
    print(f"  Standoff: 5 mm water coupling")
    print(f"  Time: {time.time()-t0:.1f} s")

    # ---- Forward data ----
    print(f"\n{'='*70}")
    print("  STEP 3: Forward Data Generation")
    print(f"{'='*70}")
    t0 = time.time()

    c_max_fwi = 3200.0
    # Frequency bands
    if N >= 192:
        freq_bands = [(50e3, 100e3), (100e3, 200e3), (200e3, 300e3)]
    elif N >= 96:
        freq_bands = [(40e3, 80e3), (80e3, 150e3)]
    else:
        freq_bands = [(30e3, 60e3)]

    max_freq = max(f for _, f in freq_bands)
    wl = 1500.0 / max_freq
    ppw = wl / dx

    print(f"  Frequency bands: {len(freq_bands)}")
    for i, (fmin, fmax) in enumerate(freq_bands):
        print(f"    Band {i+1}: {fmin/1e3:.0f} – {fmax/1e3:.0f} kHz")
    print(f"  Min wavelength: {wl*1e3:.1f} mm ({ppw:.0f} PPW)")

    # Reference time axis from c_max (used for everything)
    domain = build_domain(grid_shape, dx)
    ref_medium = build_medium(domain, c_max_fwi, 1000.0, pml_size=10)
    time_axis = build_time_axis(ref_medium, cfl=0.3)
    dt = float(time_axis.dt)
    t_end = float(time_axis.t_end)
    n_samples = int(t_end / dt)

    print(f"  dt = {dt*1e6:.3f} us, t_end = {t_end*1e3:.2f} ms, {n_samples} steps")

    source_signal = _build_source_signal(max_freq, dt, n_samples)

    # Generate observed data with true medium (including attenuation)
    medium_true = build_medium(domain, c_true, rho, pml_size=10, attenuation=alpha)
    n_data_src = min(n_actual, args.shots * len(freq_bands) * 3)

    print(f"  Simulating {n_data_src} shots...")
    observed = []
    for i in range(n_data_src):
        if (i + 1) % max(1, n_data_src // 20) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_data_src - i - 1) / rate
            print(f"    Shot {i+1}/{n_data_src} ({rate:.1f} shots/s, ETA {eta:.0f}s)")
        d = simulate_shot_sensors(
            medium_true, time_axis, src_list[i], pos_grid, source_signal, dt
        )
        observed.append(d)
    observed = jnp.stack(observed, axis=0)

    data_time = time.time() - t0
    print(f"  Observed data: {observed.shape}")
    print(f"  Time: {data_time:.1f} s ({n_data_src/data_time:.2f} shots/s)")

    # ---- FWI ----
    print(f"\n{'='*70}")
    print("  STEP 4: Full Waveform Inversion")
    print(f"{'='*70}")
    t0 = time.time()

    c_init = jnp.full(grid_shape, 1500.0, dtype=jnp.float32)

    # Checkpoint dir sits next to the output file for resume after preemption.
    # Must be unique per configuration (grid size, phantom, parameterisation)
    # to avoid stale-checkpoint leakage between unrelated runs — cause of the
    # 917 shape-broadcast crash and 919 NaN failure (2026-04-23 overnight).
    ckpt_dir = str(Path(args.output).parent / "checkpoints" / Path(args.output).stem)

    # Water mask: only update voxels inside the head (labels > 0)
    head_mask = (labels > 0).astype(jnp.float32)

    config = FWIConfig(
        freq_bands=freq_bands,
        n_iters_per_band=args.iters,
        shots_per_iter=args.shots,
        learning_rate=50.0,  # Max velocity update per iteration (m/s, voxel path)
        c_min=1400.0,
        c_max=c_max_fwi,
        pml_size=10,
        gradient_smooth_sigma=3.0,
        loss_fn="l2",
        skip_bandpass=True,
        mask=head_mask,
        checkpoint_dir=ckpt_dir,
        verbose=True,
        parameterization=args.parameterization,
        siren_learning_rate=args.siren_lr,
        siren_pretrain_steps=args.siren_pretrain_steps,
    )

    result = run_fwi(
        observed_data=observed,
        initial_velocity=c_init,
        density=rho,
        dx=dx,
        src_positions_grid=src_list[:n_data_src],
        sensor_positions_grid=pos_grid,
        source_signal=source_signal,
        dt=dt,
        t_end=t_end,
        config=config,
    )

    fwi_time = time.time() - t0

    # ---- Results ----
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")

    c_recon = result.velocity

    masks = {
        "Head (all tissue)": labels > 0,
        "Brain (GM + WM)": (labels == 2) | (labels == 3),
        "Grey matter": labels == 2,
        "White matter": labels == 3,
        "CSF": labels == 1,
        "Skull (cortical)": labels == 7,
        "Skull (trabecular)": labels == 11,
        "Skull (all bone)": (labels == 7) | (labels == 11),
        "Scalp": labels == 6,
    }

    if int(jnp.sum(labels == 8)) > 0:
        masks["Lesion"] = labels == 8

    print(f"\n  {'Region':25s} | {'True c':>8s} | {'Recon c':>8s} | {'RMSE':>8s} | {'Improve':>8s}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    metrics = {}
    for name, mask in masks.items():
        n_vox = float(jnp.sum(mask))
        if n_vox < 1:
            continue
        c_true_mean = float(jnp.sum(c_true * mask) / n_vox)
        c_recon_mean = float(jnp.sum(c_recon * mask) / n_vox)
        rmse = float(jnp.sqrt(jnp.sum((c_recon - c_true) ** 2 * mask) / n_vox))
        rmse_init = float(jnp.sqrt(jnp.sum((c_init - c_true) ** 2 * mask) / n_vox))
        improve = (1 - rmse / rmse_init) * 100 if rmse_init > 0 else 0

        metrics[name] = {
            "c_true": c_true_mean, "c_recon": c_recon_mean,
            "rmse": rmse, "rmse_init": rmse_init, "improvement": improve,
        }
        print(f"  {name:25s} | {c_true_mean:8.1f} | {c_recon_mean:8.1f} | {rmse:8.1f} | {improve:+7.1f}%")

    total_time = time.time() - t_total_start
    print(f"\n  Timing:")
    print(f"    Data generation: {data_time:.1f} s")
    print(f"    FWI inversion:   {fwi_time:.1f} s")
    print(f"    Total:           {total_time:.1f} s ({total_time/60:.1f} min)")
    print(f"  Final loss: {result.loss_history[-1]:.8f}")

    # ---- Save HDF5 ----
    print(f"\n  Saving to {args.output}...")
    import h5py
    with h5py.File(args.output, "w") as f:
        f.create_dataset("velocity_true", data=np.array(c_true), compression="gzip")
        f.create_dataset("velocity_recon", data=np.array(c_recon), compression="gzip")
        f.create_dataset("velocity_init", data=np.array(c_init), compression="gzip")
        f.create_dataset("density", data=np.array(rho), compression="gzip")
        f.create_dataset("labels", data=np.array(labels), compression="gzip")
        f.create_dataset("helmet_positions", data=np.array(positions_m))
        f.create_dataset("loss_history", data=np.array(result.loss_history))
        for i, v in enumerate(result.velocity_history):
            f.create_dataset(f"velocity_band_{i}", data=np.array(v), compression="gzip")

        f.attrs["grid_shape"] = list(grid_shape)
        f.attrs["dx_m"] = dx
        f.attrs["n_elements"] = n_actual
        f.attrs["n_data_sources"] = n_data_src
        f.attrs["freq_bands_hz"] = str(freq_bands)
        f.attrs["n_iters_per_band"] = args.iters
        f.attrs["shots_per_iter"] = args.shots
        f.attrs["learning_rate"] = config.learning_rate
        f.attrs["data_gen_time_s"] = data_time
        f.attrs["fwi_time_s"] = fwi_time
        f.attrs["total_time_s"] = total_time
        f.attrs["device"] = str(jax.devices()[0])
        f.attrs["jax_backend"] = jax.default_backend()

        for name, m in metrics.items():
            for k, v in m.items():
                f.attrs[f"metric_{name}_{k}"] = v

    print(f"  Saved ({Path(args.output).stat().st_size / 1e6:.1f} MB)")

    # ---- Figures ----
    print(f"  Generating figures...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid = N // 2
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    slices = [
        ("Axial (z={})".format(mid), c_true[:, :, mid], c_recon[:, :, mid], labels[:, :, mid]),
        ("Coronal (y={})".format(mid), c_true[:, mid, :], c_recon[:, mid, :], labels[:, mid, :]),
        ("Sagittal (x={})".format(mid), c_true[mid, :, :], c_recon[mid, :, :], labels[mid, :, :]),
    ]

    for row, (title, c_t, c_r, lab) in enumerate(slices):
        c_t, c_r, lab = np.array(c_t), np.array(c_r), np.array(lab)
        diff = c_r - c_t
        ext = [0, N * dx * 100, 0, N * dx * 100]

        im0 = axes[row, 0].imshow(c_t.T, cmap="seismic", vmin=1400, vmax=3200,
                                   origin="lower", extent=ext)
        axes[row, 0].set_title(f"True — {title}")

        im1 = axes[row, 1].imshow(c_r.T, cmap="seismic", vmin=1400, vmax=3200,
                                   origin="lower", extent=ext)
        axes[row, 1].set_title(f"Recon — {title}")

        im2 = axes[row, 2].imshow(diff.T, cmap="RdBu_r", vmin=-200, vmax=200,
                                   origin="lower", extent=ext)
        axes[row, 2].set_title(f"Error — {title}")

        im3 = axes[row, 3].imshow(lab.T, cmap="tab10", vmin=0, vmax=10,
                                   origin="lower", extent=ext)
        axes[row, 3].set_title(f"Labels — {title}")

        for ax in axes[row, :]:
            ax.set_xlabel("cm")
            ax.set_ylabel("cm")

    plt.colorbar(im0, ax=axes[0, 0], label="m/s", shrink=0.8)
    plt.colorbar(im1, ax=axes[0, 1], label="m/s", shrink=0.8)
    plt.colorbar(im2, ax=axes[0, 2], label="m/s", shrink=0.8)

    brain_rmse = metrics.get("Brain (GM + WM)", {}).get("rmse", 0)
    skull_rmse = metrics.get("Skull", {}).get("rmse", 0)
    plt.suptitle(
        f"Brain USCT — {N}^3 grid, {n_actual} elements, dx={dx*1e3:.1f}mm\n"
        f"Brain RMSE: {brain_rmse:.1f} m/s | Skull RMSE: {skull_rmse:.1f} m/s | "
        f"Time: {total_time/60:.1f} min on {jax.default_backend().upper()}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(args.figures, dpi=150, bbox_inches="tight")
    print(f"  Saved {args.figures}")

    print(f"\n{'='*70}")
    print(f"  COMPLETE — {total_time/60:.1f} min total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
