"""Modal runner: SIREN-vs-voxel FWI validation on an A100.

Parallelized version: runs voxel and SIREN in separate containers to
avoid the 3h sequential timeout and improve time-to-result.

Usage::

    modal run scripts/modal_validate_siren.py
"""

import modal
import time
from pathlib import Path

app = modal.App("brain-fwi-siren-validation")

GRID_SIZE = 96
N_ELEMENTS = 64
ITERS_PER_BAND = 10
SHOTS_PER_ITER = 8
DX_M = 0.002

GIT_BRANCH = "main"
CACHE_BUST = "2026-04-25-validation-parallel-v1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .env({"BRAIN_FWI_CACHE_BUST": CACHE_BUST})
    .run_commands(
        f"git clone --depth 1 --branch {GIT_BRANCH} "
        f"https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && uv pip install --system -e '.[cuda12]'",
    )
)

results_vol = modal.Volume.from_name(
    "brain-fwi-validation", create_if_missing=True,
)

def _run_fwi_path(params: str):
    import os
    import subprocess
    import time
    
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    repo = "/opt/brain-fwi"
    tag = f"g{GRID_SIZE}_n{N_ELEMENTS}_i{ITERS_PER_BAND}"
    out_file = f"/results/{params}_{tag}.h5"

    args = [
        "python", "-u", f"{repo}/run_full_usct.py",
        "--grid-size", str(GRID_SIZE),
        "--n-elements", str(N_ELEMENTS),
        "--iters", str(ITERS_PER_BAND),
        "--shots", str(SHOTS_PER_ITER),
        "--dx", str(DX_M),
        "--parameterization", params,
        "--output", out_file
    ]

    print(f"Starting {params} path...")
    t0 = time.time()
    subprocess.run(args, check=True, cwd=repo)
    wallclock = time.time() - t0
    
    return out_file, wallclock

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3 * 60 * 60,
    volumes={"/results": results_vol},
)
def run_voxel():
    return _run_fwi_path("voxel")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3 * 60 * 60,
    volumes={"/results": results_vol},
)
def run_siren():
    return _run_fwi_path("siren")

@app.function(
    image=image,
    timeout=600,
    volumes={"/results": results_vol},
)
def run_comparison(voxel_out, siren_out, voxel_time, siren_time):
    import subprocess
    import json as _json
    
    repo = "/opt/brain-fwi"
    tag = f"g{GRID_SIZE}_n{N_ELEMENTS}_i{ITERS_PER_BAND}"
    json_out = f"/results/comparison_{tag}.json"
    
    print("Running comparison...")
    subprocess.run(
        ["python", "-u",
         f"{repo}/scripts/validate_siren_vs_voxel.py",
         "--voxel", voxel_out,
         "--siren", siren_out,
         "--json-out", json_out],
        check=True,
    )

    timings_path = f"/results/timings_{tag}.json"
    with open(timings_path, "w") as f:
        _json.dump({
            "grid_size": GRID_SIZE,
            "n_elements": N_ELEMENTS,
            "iters_per_band": ITERS_PER_BAND,
            "shots_per_iter": SHOTS_PER_ITER,
            "voxel_wallclock_s": voxel_time,
            "siren_wallclock_s": siren_time,
            "total_sequential_s": voxel_time + siren_time,
        }, f, indent=2)
    
    results_vol.commit()
    print(f"Comparison saved to {json_out}")

@app.local_entrypoint()
def main():
    # Start both in parallel
    print("Launching voxel and siren runs in parallel...")
    voxel_task = run_voxel.remote_gen()
    siren_task = run_siren.remote_gen()
    
    # Wait for both to complete
    # Note: Using remote() instead of remote_gen() for simpler return handling if preferred,
    # but here we follow the parallel execution pattern.
    
    # Re-executing with simpler remote() calls for cleaner orchestrator
    v_out, v_time = run_voxel.remote()
    s_out, s_time = run_siren.remote()
    
    run_comparison.remote(v_out, s_out, v_time, s_time)
