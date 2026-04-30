#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

# Ensure GPU is used
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

def main():
    # 1. Load config
    with open("config.json") as f:
        config = json.load(f)

    m2m_dir = config["m2m"]
    transducer_config_path = config["transducer"]
    target_mm = [config["target_x"], config["target_y"], config["target_z"]]
    freq = config.get("frequency", 500000)

    print(f"--- OpenLIFU Brainlife App ---")
    print(f"  m2m:      {m2m_dir}")
    print(f"  Target:   {target_mm} mm")
    print(f"  Freq:     {freq/1e3} kHz")

    # 2. OpenLIFU Setup
    try:
        import openlifu
        from openlifu.sim import jwave_engine
        from openlifu.transducers import load_transducer
    except ImportError:
        print("Error: openlifu-python not found in the environment.")
        sys.exit(1)

    # 3. Load Transducer
    print("Loading transducer configuration...")
    transducer = load_transducer(transducer_config_path)

    # 4. Prepare Head Model (from SimNIBS m2m)
    print("Preparing head model from SimNIBS...")
    head_model = openlifu.HeadModel.from_simnibs(m2m_dir)

    # 5. Run Simulation
    print("Starting j-Wave simulation...")
    results = jwave_engine.run_simulation(
        head_model=head_model,
        transducer=transducer,
        target_mm=target_mm,
        frequency=freq,
        grid_spacing=0.0005 # 0.5mm
    )

    # 6. Save Outputs
    print("Saving results...")
    os.makedirs("pressure", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # Save Peak Pressure as NIfTI
    pressure_img = nib.Nifti1Image(results.peak_pressure, head_model.affine)
    nib.save(pressure_img, "pressure/peak_pressure.nii.gz")

    # Save metrics
    metrics = {
        "peak_pressure_pa": float(np.max(results.peak_pressure)),
        "focal_size_mm": results.focal_size_mm,
        "target_error_mm": results.target_error_mm
    }
    with open("metrics/focal_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("--- Simulation Complete ---")

if __name__ == "__main__":
    main()
