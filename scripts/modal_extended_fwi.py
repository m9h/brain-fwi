"""
Modal.com: Extended 3D FWI with multi-frequency banding + ARFI chain.

Uses the synthetic 3D head phantom (96^3), runs multi-frequency FWI
(50→200→500 kHz), then connects to sbi4dwi's radiation force + ARFI chain.

    modal run scripts/modal_extended_fwi.py
"""

import modal

app = modal.App("brain-fwi-extended")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]==0.4.38",
        "jaxlib==0.4.38",
        "jwave==0.2.1",
        "optax==0.2.5",
        "equinox==0.11.12",
        "scipy",
        "numpy<2",
        "xarray",
        "h5py",
        "nibabel",
    )
    .add_local_dir(
        "/Users/mhough/dev/brain-fwi/src",
        remote_path="/opt/brain-fwi-src",
    )
    .add_local_dir(
        "/Users/mhough/dev/sbi4dwi/dmipy_jax/biophysics",
        remote_path="/opt/sbi4dwi-biophysics",
    )
)


@app.function(image=image, gpu="A100", timeout=3600, memory=32768)
def run_extended_fwi():
    import time, sys, os
    import numpy as np
    import jax
    import jax.numpy as jnp
    import importlib.util

    sys.path.insert(0, "/opt/brain-fwi-src")

    # Load sbi4dwi biophysics modules directly
    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    BP = "/opt/sbi4dwi-biophysics"
    acoustic = _load("acoustic", f"{BP}/acoustic.py")
    rf = _load("radiation_force", f"{BP}/radiation_force.py")
    arfi = _load("mr_arfi", f"{BP}/mr_arfi.py")

    print(f"JAX: {jax.__version__}, devices: {jax.devices()}")
    print()

    results = {}

    # ==================================================================
    # Step 1: Build 3D head phantom
    # ==================================================================
    print("=" * 70)
    print("STEP 1: Build 3D Head Phantom (96^3, 2mm)")
    print("=" * 70)

    N = 96
    dx = 2e-3
    cx, cy, cz = N//2, N//2, N//2
    xx, yy, zz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
    r = np.sqrt(((xx-cx)/42)**2 + ((yy-cy)/36)**2 + ((zz-cz)/40)**2)

    labels = np.zeros((N,N,N), dtype=int)
    labels[r < 1.0] = 1   # scalp
    labels[r < 0.93] = 2  # skull
    labels[r < 0.83] = 3  # CSF
    labels[r < 0.80] = 4  # grey matter
    labels[r < 0.65] = 5  # white matter

    # Map to acoustic properties using sbi4dwi
    props = acoustic.map_labels_to_properties(jnp.array(labels))
    c_true = np.array(props["sound_speed"])
    rho = np.array(props["density"])

    for lab in np.unique(labels):
        count = np.sum(labels == lab)
        c_val = c_true[labels == lab][0] if count > 0 else 0
        print(f"  Label {lab}: {count:,} vox, c={c_val:.0f} m/s")

    results["phantom"] = {"shape": list(c_true.shape), "dx_mm": dx*1e3}
    print()

    # ==================================================================
    # Step 2: 256-element helmet array
    # ==================================================================
    print("=" * 70)
    print("STEP 2: Generate 256-Element Helmet Array")
    print("=" * 70)

    from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
    center = (N*dx/2, N*dx/2, N*dx/2)
    positions = helmet_array_3d(n_elements=256, center=center)
    print(f"  {positions.shape[0]} elements placed")

    # Pick sources and sensors
    n_src = 8
    n_sen = 32
    src_grid = transducer_positions_to_grid(positions[:n_src], dx, (N,N,N))
    sen_grid = transducer_positions_to_grid(positions[n_src:n_src+n_sen], dx, (N,N,N))
    src_list = [tuple(int(src_grid[d][i]) for d in range(3)) for i in range(n_src)]
    print(f"  Sources: {n_src}, Sensors: {n_sen}")
    print()

    # ==================================================================
    # Step 3: Generate observed data
    # ==================================================================
    print("=" * 70)
    print("STEP 3: Generate Observed Data ({n_src} shots)")
    print("=" * 70)

    from brain_fwi.simulation.forward import (
        build_domain, build_medium, build_time_axis, generate_observed_data,
    )
    from brain_fwi.utils.wavelets import ricker_wavelet

    t0 = time.time()
    obs = generate_observed_data(
        jnp.array(c_true), jnp.array(rho), dx,
        src_positions_grid=src_list,
        sensor_positions_grid=sen_grid,
        freq=500e3, t_end=5e-5,
    )
    t_obs = time.time() - t0
    print(f"  Observed data: {obs.shape} in {t_obs:.1f}s")
    results["observed"] = {"shape": list(obs.shape), "time_s": t_obs}
    print()

    # ==================================================================
    # Step 4: Multi-frequency FWI (3 bands, 20 iters total)
    # ==================================================================
    print("=" * 70)
    print("STEP 4: Multi-Frequency FWI (3 bands × 10 iters)")
    print("=" * 70)

    from brain_fwi.inversion.fwi import FWIConfig, run_fwi

    # Build source signal and time params
    domain = build_domain((N,N,N), dx)
    medium = build_medium(domain, jnp.array(c_true), jnp.array(rho), pml_size=10)
    ta = build_time_axis(medium, cfl=0.3, t_end=5e-5)
    dt = float(ta.dt)
    n_t = int(5e-5 / dt)
    source_sig = ricker_wavelet(500e3, dt, n_t)

    config = FWIConfig(
        freq_bands=[(50e3, 150e3), (150e3, 350e3), (350e3, 600e3)],
        n_iters_per_band=10,
        shots_per_iter=2,
        learning_rate=0.3,
        c_min=1400.0,
        c_max=4500.0,
        gradient_smooth_sigma=2.0,
    )

    t0 = time.time()
    result = run_fwi(
        observed_data=obs,
        initial_velocity=jnp.ones((N,N,N)) * 1500.0,
        density=jnp.array(rho),
        dx=dx,
        src_positions_grid=src_list,
        sensor_positions_grid=sen_grid,
        source_signal=source_sig,
        dt=dt,
        t_end=5e-5,
        config=config,
    )
    t_fwi = time.time() - t0

    c_rec = np.array(result.velocity)
    print(f"  FWI time: {t_fwi:.1f}s ({t_fwi/30:.1f}s/iter)")
    print(f"  Loss: {result.loss_history[0]:.4f} → {result.loss_history[-1]:.4f}")
    print(f"  Recon c: [{c_rec.min():.0f}, {c_rec.max():.0f}] m/s")

    # Error analysis
    skull_mask = labels == 2
    brain_mask = (labels == 4) | (labels == 5)
    mse_total = float(np.mean((c_rec - c_true)**2))
    mse_skull = float(np.mean((c_rec[skull_mask] - c_true[skull_mask])**2)) if skull_mask.sum() > 0 else 0
    mse_brain = float(np.mean((c_rec[brain_mask] - c_true[brain_mask])**2)) if brain_mask.sum() > 0 else 0
    print(f"  MSE total: {mse_total:.0f}")
    print(f"  MSE skull: {mse_skull:.0f}")
    print(f"  MSE brain: {mse_brain:.0f}")

    results["fwi"] = {
        "n_bands": 3, "n_iters_per_band": 10,
        "time_s": t_fwi,
        "loss_start": float(result.loss_history[0]),
        "loss_end": float(result.loss_history[-1]),
        "c_range": [float(c_rec.min()), float(c_rec.max())],
        "mse_total": mse_total, "mse_skull": mse_skull, "mse_brain": mse_brain,
    }
    print()

    # ==================================================================
    # Step 5: Radiation Force + Displacement from FWI result
    # ==================================================================
    print("=" * 70)
    print("STEP 5: Radiation Force → Displacement → ARFI")
    print("=" * 70)

    # ARFI on 2D slice to avoid 3D OOM (96^3 time-series = 113GB)
    mid_z = N // 2
    c_slice = c_rec[:, :, mid_z]
    rho_slice = np.array(rho)[:, :, mid_z]
    labels_slice = labels[:, :, mid_z]

    # Run 2D simulation on the axial slice for ARFI
    import importlib.util as ilu
    jwa_spec = ilu.spec_from_file_location("jwa", f"{BP}/jwave_adapter.py")
    jwa = ilu.module_from_spec(jwa_spec)
    jwa_spec.loader.exec_module(jwa)

    from jwave.geometry import TimeAxis as TA
    domain_2d = jwa.create_domain(c_slice.shape, dx)
    med_2d = jwa.create_medium(domain_2d, jnp.array(c_slice, dtype=jnp.float32),
                                jnp.array(rho_slice, dtype=jnp.float32), pml_size=8)
    ta_2d = TA.from_medium(med_2d, cfl=0.3, t_end=5e-5)
    dt_2d = float(ta_2d.dt)
    src_2d = (N//2, 5)
    pos_2d = jnp.array([[src_2d[0]*dx, src_2d[1]*dx]])
    sources_2d = jwa.create_sources(domain_2d, pos_2d, 500e3, jnp.zeros(1), jnp.ones(1), dt=dt_2d, n_cycles=5)
    p_2d = jwa.run_simulation_jax(med_2d, src_2d, 500e3, 5e-5, time_axis=ta_2d, sources=sources_2d)
    p_slice = np.array(p_2d)

    # Intensity
    Z = rho_slice * c_slice
    intensity = p_slice**2 / (2 * Z + 1e-10)
    print(f"  Pressure slice max: {np.max(np.abs(p_slice)):.6f} Pa")
    print(f"  Intensity max: {np.max(intensity):.6e} W/m²")

    # Radiation force
    attn = np.array(props["attenuation"][:, :, mid_z])
    alpha_np_m = attn * 100.0 / 8.686  # dB/cm → Np/m
    force = rf.compute_radiation_force(
        jnp.array(intensity), jnp.array(c_slice), jnp.array(alpha_np_m)
    )

    # Shear modulus from labels
    mu_map = rf.map_labels_to_shear_modulus(jnp.array(labels_slice))

    # Displacement
    displacement = rf.solve_displacement_quasistatic(force, mu_map, dx)

    # ARFI phase
    phase = arfi.predict_arfi_phase(np.array(displacement), 40e-3, 5e-3)

    max_force = float(jnp.max(jnp.abs(force)))
    max_disp_um = float(jnp.max(jnp.abs(displacement))) * 1e6
    max_phase = float(np.max(np.abs(phase)))

    print(f"  Max force: {max_force:.6e} N/m³")
    print(f"  Max displacement: {max_disp_um:.6f} µm")
    print(f"  Max ARFI phase: {max_phase:.6f} rad")

    # Regional comparison
    gm_mask = labels_slice == 4
    wm_mask = labels_slice == 5
    if gm_mask.sum() > 0 and wm_mask.sum() > 0:
        gm_d = float(jnp.mean(jnp.abs(displacement[jnp.array(gm_mask)]))) * 1e6
        wm_d = float(jnp.mean(jnp.abs(displacement[jnp.array(wm_mask)]))) * 1e6
        ratio = wm_d / gm_d if gm_d > 0 else 0
        print(f"  GM mean disp: {gm_d:.6f} µm")
        print(f"  WM mean disp: {wm_d:.6f} µm")
        print(f"  WM/GM ratio: {ratio:.2f}x (expect ~1.8x from Kuhl CANN)")

    results["arfi"] = {
        "max_force": max_force,
        "max_displacement_um": max_disp_um,
        "max_phase_rad": max_phase,
    }
    print()

    # ==================================================================
    # Step 6: Save results
    # ==================================================================
    print("=" * 70)
    print("STEP 6: Save Results")
    print("=" * 70)

    import h5py
    with h5py.File("/tmp/fwi_extended_results.h5", "w") as f:
        f.create_dataset("velocity_true", data=c_true)
        f.create_dataset("velocity_recon", data=c_rec)
        f.create_dataset("density", data=rho)
        f.create_dataset("labels", data=labels)
        f.create_dataset("loss_history", data=result.loss_history)
        f.create_dataset("helmet_positions", data=np.array(positions))
        f.create_dataset("displacement_slice", data=np.array(displacement))
        f.create_dataset("arfi_phase_slice", data=np.array(phase))
        f.create_dataset("force_slice", data=np.array(force))
    print("  Saved to /tmp/fwi_extended_results.h5")
    print()

    # ==================================================================
    # Summary
    # ==================================================================
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Phantom: {N}^3 at {dx*1e3}mm, 5 tissues")
    print(f"  Array: 256 elements, {n_src} src + {n_sen} sen")
    print(f"  FWI: 3 bands × 10 iters = 30 total, {t_fwi:.0f}s")
    print(f"  Velocity: {c_rec.min():.0f}-{c_rec.max():.0f} m/s (true: {c_true.min():.0f}-{c_true.max():.0f})")
    print(f"  ARFI: force {max_force:.2e}, disp {max_disp_um:.4f} µm, phase {max_phase:.4f} rad")
    print()

    return results


@app.local_entrypoint()
def main():
    import json
    results = run_extended_fwi.remote()
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2, default=str))
