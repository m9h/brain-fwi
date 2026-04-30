#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

# Request GPU before importing JAX
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Add src to path if needed (though it should be installed in the container)
sys.path.append(str(Path(__file__).parent.parent / "src"))

from brain_fwi.phantoms.simnibs import load_simnibs_acoustic
from brain_fwi.transducers.helmet import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    simulate_shot_sensors,
)

def main():
    # 1. Load config
    with open("config.json") as f:
        config = json.load(f)

    m2m_dir = Path(config["m2m"])
    grid_size = config.get("grid_size", 96)
    dx = config.get("dx", 0.002)
    n_elements = config.get("n_elements", 128)
    freq = config.get("frequency", 500000.0)
    shots = config.get("shots", 1)

    grid_shape = (grid_size, grid_size, grid_size)
    print(f"--- Running USCT Simulation ---")
    print(f"  m2m dir:    {m2m_dir}")
    print(f"  Grid size:  {grid_size}^3")
    print(f"  Spacing:    {dx*1000:.1f} mm")
    print(f"  Frequency:  {freq/1e3:.0f} kHz")
    print(f"  GPU available: {jax.default_backend() == 'gpu'}")

    # 2. Load Phantom from SimNIBS
    print("Loading SimNIBS phantom...")
    props = load_simnibs_acoustic(m2m_dir)
    
    # Resample to target grid
    from brain_fwi.phantoms.mida import resample_volume
    c = resample_volume(np.asarray(props["sound_speed"]), grid_shape, order=1)
    rho = resample_volume(np.asarray(props["density"]), grid_shape, order=1)
    alpha = resample_volume(np.asarray(props["attenuation"]), grid_shape, order=1)
    labels = resample_volume(np.asarray(props["labels"]), grid_shape, order=0)

    # 3. Setup Helmet
    print("Setting up helmet array...")
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
    )
    
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    
    src_list = [(int(pos_grid[0][i]), int(pos_grid[1][i]), int(pos_grid[2][i]))
                for i in range(len(pos_grid[0]))]

    # 4. Simulation
    print("Building domain and medium...")
    domain = build_domain(grid_shape, dx)
    medium = build_medium(domain, c, rho, attenuation=alpha)
    time_axis = build_time_axis(medium, cfl=0.3)

    os.makedirs("usct", exist_ok=True)
    
    # Simulate shots
    print(f"Simulating {shots} shots...")
    # Just take the first N sources for the requested number of shots
    results = []
    for i in range(min(shots, len(src_list))):
        print(f"  Shot {i+1}/{shots}...")
        obs = simulate_shot_sensors(
            domain, medium, time_axis,
            src_pos_grid=src_list[i],
            sensor_pos_grid=pos_grid,
            freq=freq
        )
        results.append(obs)

    # 5. Save Results
    print("Saving results...")
    with h5py.File("usct/simulation.h5", "w") as f:
        f.create_dataset("data", data=np.stack(results))
        f.create_dataset("c", data=c)
        f.create_dataset("rho", data=rho)
        f.create_dataset("labels", data=labels)
        f.create_dataset("sensor_positions", data=positions)
        f.attrs["dx"] = dx
        f.attrs["freq"] = freq

    # 6. Figures
    print("Generating figures...")
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(c[:, :, grid_shape[2]//2].T, origin="lower", cmap="viridis")
    plt.title("Sound Speed (Axial)")
    plt.colorbar(label="m/s")
    
    plt.subplot(132)
    plt.imshow(labels[:, :, grid_shape[2]//2].T, origin="lower", cmap="tab20")
    plt.title("Labels (Axial)")
    
    plt.subplot(133)
    # Wavefield plot (take last shot, middle time)
    # Note: simulate_shot_sensors returns (n_sensors, n_time)
    # For wavefield we would need the full pressure volume, 
    # but for a summary we can just show the recorded traces.
    plt.plot(results[0][:, :].T)
    plt.title("Sensor Traces (Shot 0)")
    
    plt.tight_layout()
    plt.savefig("usct/summary.png")
    
    print("--- Done ---")

if __name__ == "__main__":
    main()
