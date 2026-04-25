"""
Modal.com: 256-element FWI on MIDA head model.

Uses brain-fwi's existing infrastructure:
  - phantoms/mida.py: MIDA tissue loading + acoustic mapping
  - transducers/helmet.py: 256-element Fibonacci helmet
  - simulation/forward.py: j-Wave PSTD solver
  - inversion/fwi.py: Multi-frequency FWI with reparameterized velocity

The MIDA model (MIDAv1-0.zip) must be uploaded to a Modal volume:
    modal volume create mida-data
    modal volume put mida-data /path/to/MIDAv1-0.zip /MIDAv1-0.zip

    modal run scripts/modal_mida_256.py
"""

import modal

app = modal.App("brain-fwi-mida-256")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unzip", "wget", "git")
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
    .env({"BRAIN_FWI_COMMIT": "ec801eb"})
    .run_commands(
        "rm -rf /opt/brain-fwi"
        " && git clone --depth 1 https://github.com/m9h/brain-fwi.git /opt/brain-fwi"
        " && cd /opt/brain-fwi && git log --oneline -1"
        " || echo 'No remote yet — will use mounted code'",
        "echo '/opt/brain-fwi/src' > /usr/local/lib/python3.12/site-packages/brain-fwi.pth || true",
    )
)

# Volume for MIDA data (upload once)
mida_vol = modal.Volume.from_name("mida-data", create_if_missing=True)


# Install brain-fwi source into the image directly
image = image.add_local_dir(
    "/Users/mhough/dev/brain-fwi/src",
    remote_path="/opt/brain-fwi-src",
)


@app.function(image=image, gpu="A100", timeout=4 * 3600, memory=32768,
              volumes={"/mida": mida_vol})
def run_mida_256():
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    import sys, os, glob

    # Add brain-fwi to path (mounted local source)
    for p in ["/opt/brain-fwi-src", "/opt/brain-fwi/src"]:
        if os.path.exists(p):
            sys.path.insert(0, p)
            print(f"  Added to path: {p}")

    print(f"JAX: {jax.__version__}, devices: {jax.devices()}")
    print()

    all_results = {}

    # ==================================================================
    # Step 1: Load MIDA head model
    # ==================================================================
    print("=" * 70)
    print("STEP 1: Load MIDA Head Model")
    print("=" * 70)

    # Check for MIDA data
    mida_files = glob.glob("/mida/**/*.mat", recursive=True) + \
                 glob.glob("/mida/**/*.nii*", recursive=True) + \
                 glob.glob("/mida/**/*.h5", recursive=True) + \
                 glob.glob("/mida/**/*.zip", recursive=True)
    print(f"  MIDA files found: {len(mida_files)}")
    for f in mida_files[:10]:
        sz = os.path.getsize(f) if os.path.isfile(f) else 0
        print(f"    {f} ({sz/1e6:.1f} MB)")

    # Unzip if needed
    zips = [f for f in mida_files if f.endswith('.zip')]
    if zips:
        import subprocess
        for z in zips:
            print(f"  Extracting {z}...")
            subprocess.run(["unzip", "-qo", z, "-d", "/mida/extracted"], check=True)
        mida_files = glob.glob("/mida/extracted/**/*.*", recursive=True)
        print(f"  Extracted files: {len(mida_files)}")
        for f in mida_files[:20]:
            sz = os.path.getsize(f) if os.path.isfile(f) else 0
            print(f"    {f} ({sz/1e6:.1f} MB)")

    # Try loading MIDA
    try:
        from brain_fwi.phantoms.mida import load_mida_acoustic
        mat_files = [f for f in mida_files if f.endswith('.mat')]
        nii_files = [f for f in mida_files if f.endswith(('.nii', '.nii.gz'))]
        h5_files = [f for f in mida_files if f.endswith(('.h5', '.hdf5'))]

        mida_path = None
        for candidates in [mat_files, nii_files, h5_files]:
            if candidates:
                mida_path = candidates[0]
                break

        if mida_path:
            print(f"  Loading MIDA from: {mida_path}")
            t0 = time.time()
            # Downsample to 2mm for tractable simulation
            props = load_mida_acoustic(mida_path, target_dx=2e-3)
            t_load = time.time() - t0
            c = np.array(props["sound_speed"])
            rho = np.array(props["density"])
            print(f"  Shape: {c.shape}")
            print(f"  Sound speed: [{c.min():.0f}, {c.max():.0f}] m/s")
            print(f"  Density: [{rho.min():.0f}, {rho.max():.0f}] kg/m³")
            print(f"  Loaded in {t_load:.1f}s")
            use_mida = True
        else:
            use_mida = False
            print("  No MIDA data files found")
    except Exception as e:
        use_mida = False
        print(f"  MIDA load failed: {e}")

    if not use_mida:
        print("  Falling back to synthetic 3D head phantom...")
        try:
            from brain_fwi.phantoms import make_synthetic_head
            from brain_fwi.phantoms.properties import map_labels_to_all
            # 3D synthetic head
            N = 96
            cx, cy, cz = N//2, N//2, N//2
            xx, yy, zz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
            r = np.sqrt(((xx-cx)/42)**2 + ((yy-cy)/36)**2 + ((zz-cz)/40)**2)
            labels = np.zeros((N,N,N), dtype=int)
            labels[r < 1.0] = 1   # scalp
            labels[r < 0.93] = 2  # skull
            labels[r < 0.83] = 3  # CSF
            labels[r < 0.80] = 4  # grey matter
            labels[r < 0.65] = 5  # white matter
            # Map to acoustic properties
            from brain_fwi.phantoms.properties import TISSUE_SPEEDS, TISSUE_DENSITIES
            c = np.full((N,N,N), 1500.0, dtype=np.float32)
            rho = np.full((N,N,N), 1000.0, dtype=np.float32)
            for lab, speed in TISSUE_SPEEDS.items():
                c[labels == lab] = speed
            for lab, dens in TISSUE_DENSITIES.items():
                rho[labels == lab] = dens
        except Exception as e2:
            print(f"  Synthetic also failed: {e2}")
            # Absolute fallback: manual properties
            N = 96
            c = np.ones((N,N,N), dtype=np.float32) * 1500.0
            rho = np.ones((N,N,N), dtype=np.float32) * 1000.0
            # Skull shell
            cx, cy, cz = N//2, N//2, N//2
            xx, yy, zz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
            r = np.sqrt(((xx-cx)/42)**2 + ((yy-cy)/36)**2 + ((zz-cz)/40)**2)
            skull_mask = (r > 0.83) & (r < 0.93)
            c[skull_mask] = 2800.0
            rho[skull_mask] = 1850.0
            brain_mask = r < 0.80
            c[brain_mask] = 1560.0
            rho[brain_mask] = 1040.0

        print(f"  Synthetic phantom: {c.shape}")
        print(f"  Sound speed: [{c.min():.0f}, {c.max():.0f}] m/s")

    grid_shape = c.shape
    dx = 2e-3  # 2mm
    print()

    # ==================================================================
    # Step 2: Generate 256-element helmet array
    # ==================================================================
    print("=" * 70)
    print("STEP 2: Generate 256-Element Helmet Array")
    print("=" * 70)

    try:
        from brain_fwi.transducers import helmet_array_3d
        center = np.array([grid_shape[i] * dx / 2 for i in range(3)])
        positions = helmet_array_3d(
            n_elements=256,
            center=tuple(center),
            radius_ap=0.095,
            radius_lr=0.075,
            radius_si=0.090,
            standoff=0.005,
        )
    except Exception as e:
        print(f"  helmet_array_3d failed: {e}, using Fibonacci fallback")
        # Fibonacci hemisphere fallback
        n = 256
        golden = (1 + np.sqrt(5)) / 2
        center = np.array([grid_shape[i] * dx / 2 for i in range(3)])
        positions = []
        for i in range(n):
            theta = np.arccos(1 - (i + 0.5) / n)
            phi = 2 * np.pi * i / golden
            if theta > np.pi * 0.8:
                theta = np.pi * 0.8
            positions.append([
                center[0] + 0.100 * np.sin(theta) * np.cos(phi),
                center[1] + 0.080 * np.sin(theta) * np.sin(phi),
                center[2] + 0.095 * np.cos(theta),
            ])
        positions = jnp.array(positions)

    print(f"  Elements: {positions.shape[0]}")
    print(f"  Center: {center*1e3} mm")
    print()

    # ==================================================================
    # Step 3: 3D Forward simulation (single shot)
    # ==================================================================
    print("=" * 70)
    print("STEP 3: 3D Forward Simulation")
    print("=" * 70)

    try:
        from brain_fwi.simulation.forward import (
            build_domain, build_medium, build_time_axis, simulate_shot_sensors,
        )

        domain = build_domain(grid_shape, dx)
        medium = build_medium(domain, jnp.array(c), jnp.array(rho), pml_size=10)
        ta = build_time_axis(medium, cfl=0.3, t_end=5e-5)

        # Use one element as source
        src_pos = np.array(positions[0])
        src_idx = tuple(np.round(src_pos / dx).astype(int))
        print(f"  Source element 0: {src_pos*1e3} mm → grid {src_idx}")
        print(f"  Grid: {grid_shape}, dx={dx*1e3}mm")

        t0 = time.time()
        from brain_fwi.simulation.forward import simulate_shot
        p_field = simulate_shot(medium, ta, src_idx, 500e3)
        t_sim = time.time() - t0

        print(f"  Simulation: {t_sim:.1f}s")
        # Extract raw array from FourierSeries if needed
        if hasattr(p_field, 'on_grid'):
            p_arr = p_field.on_grid
            if p_arr.shape[-1] == 1:
                p_arr = p_arr[..., 0]
        else:
            p_arr = jnp.asarray(p_field)
        print(f"  p_max: {float(jnp.max(jnp.abs(p_arr))):.6f}")

        all_results["forward_3d"] = {
            "grid_shape": list(grid_shape),
            "dx_mm": dx * 1e3,
            "n_elements": 256,
            "sim_time_s": t_sim,
            "p_max": float(jnp.max(p_field)),
        }
    except Exception as e:
        print(f"  Forward sim error: {e}")
        import traceback
        traceback.print_exc()
        all_results["forward_3d"] = {"error": str(e)}
    print()

    # ==================================================================
    # Step 4: FWI (quick test: 5 iterations, single frequency)
    # ==================================================================
    print("=" * 70)
    print("STEP 4: FWI (5 iterations)")
    print("=" * 70)

    try:
        from brain_fwi.inversion.fwi import FWIConfig, run_fwi
        from brain_fwi.simulation.forward import (
            build_domain as bd, build_medium as bm, build_time_axis as bta,
            generate_observed_data, _build_source_signal,
        )
        from brain_fwi.transducers import transducer_positions_to_grid

        config = FWIConfig(
            freq_bands=[(400e3, 600e3)],
            n_iters_per_band=5,
            shots_per_iter=1,
            learning_rate=0.5,
            c_min=1400.0,
            c_max=3500.0,
        )

        # Convert positions to grid indices
        src_grid = transducer_positions_to_grid(positions[:4], dx, grid_shape)
        sen_grid = transducer_positions_to_grid(positions[4:20], dx, grid_shape)

        # Generate synthetic observed data
        print("  Generating observed data (4 shots)...")
        t0 = time.time()
        obs_data = generate_observed_data(
            jnp.array(c), jnp.array(rho), dx,
            src_positions_grid=[tuple(int(src_grid[d][i]) for d in range(3)) for i in range(4)],
            sensor_positions_grid=sen_grid,
            freq=500e3, t_end=5e-5,
        )
        print(f"  Observed data: {obs_data.shape}, generated in {time.time()-t0:.1f}s")

        # Build source signal for FWI
        from brain_fwi.utils.wavelets import ricker_wavelet
        ta_fwi = bta(bm(bd(grid_shape, dx), jnp.array(c), jnp.array(rho), pml_size=10),
                      cfl=0.3, t_end=5e-5)
        dt_fwi = float(ta_fwi.dt)
        n_t = int(5e-5 / dt_fwi)
        source_sig = ricker_wavelet(500e3, dt_fwi, n_t)

        # Run FWI
        print("  Running FWI (5 iterations)...")
        t0 = time.time()
        result = run_fwi(
            observed_data=obs_data,
            initial_velocity=jnp.ones(grid_shape) * 1500.0,
            density=jnp.array(rho),
            dx=dx,
            src_positions_grid=[tuple(int(src_grid[d][i]) for d in range(3)) for i in range(4)],
            sensor_positions_grid=sen_grid,
            source_signal=source_sig,
            dt=dt_fwi,
            t_end=5e-5,
            config=config,
        )
        t_fwi = time.time() - t0

        print(f"  FWI time: {t_fwi:.1f}s")
        print(f"  Final loss: {result.loss_history[-1]:.6f}")
        c_rec = np.array(result.velocity)
        print(f"  Recon c: [{c_rec.min():.0f}, {c_rec.max():.0f}] m/s")

        all_results["fwi"] = {
            "time_s": t_fwi,
            "loss_history": [float(h) for h in result.loss_history],
            "c_range": [float(c_rec.min()), float(c_rec.max())],
        }
    except Exception as e:
        print(f"  FWI error: {e}")
        import traceback
        traceback.print_exc()
        all_results["fwi"] = {"error": str(e)}
    print()

    # ==================================================================
    # Summary
    # ==================================================================
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    for key, val in all_results.items():
        if "error" in val:
            print(f"  {key}: FAILED — {val['error'][:80]}")
        else:
            print(f"  {key}: OK")

    return all_results


@app.local_entrypoint()
def main():
    import json
    results = run_mida_256.remote()
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2, default=str))
