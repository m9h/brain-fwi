"""Modal runner: SIREN-vs-voxel FWI validation on A100.

Runs ``run_full_usct.py`` twice (voxel, siren) on the synthetic phantom,
then invokes ``validate_siren_vs_voxel.py`` to print a regional-RMSE
comparison table. All outputs land in a persistent Modal volume.

The first 96^3 run hit the 1 h function timeout, so this script now
defaults to a **48^3 timing probe** that finishes comfortably inside
1 h and gives real Modal-A100 forward-sim numbers. Edit the constants
below to scale up once the probe's timings are in.

Key hardening beyond the original version:

- Per-path timing printed to stdout (voxel wall-clock, siren
  wall-clock, JIT cost visible from first-iter gap).
- ``results_vol.commit()`` after EACH path — so a timeout during the
  second path no longer wipes the first path's results.
- Output paths are grid-tagged so probe runs and full runs don't
  stomp each other.

Usage::

    modal run --detach scripts/modal_validate_siren.py
"""

import modal

app = modal.App("brain-fwi-siren-validation")

# --- Timing probe configuration --------------------------------------
# 48^3 with 32 elements runs in an estimated 10-15 min voxel + ~15-20 min
# SIREN at j-Wave's typical A100 throughput. Comfortably inside 1 h.
GRID_SIZE = 48
N_ELEMENTS = 32
ITERS_PER_BAND = 10
SHOTS_PER_ITER = 8
DX_M = 0.004

GIT_BRANCH = "feature/siren-validation-probe"

# Bump to force image rebuild after pushing a fix to GIT_BRANCH (Modal
# fingerprints the build spec, not the git clone output).
CACHE_BUST = "2026-04-24-probe"

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


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60,  # 1 hour — probe is designed to fit
    volumes={"/results": results_vol},
)
def run_validation():
    import os
    import subprocess
    import time

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    subprocess.run(["nvidia-smi", "-L"], check=True)

    repo = "/opt/brain-fwi"
    tag = f"g{GRID_SIZE}_n{N_ELEMENTS}_i{ITERS_PER_BAND}"
    voxel_out = f"/results/voxel_{tag}.h5"
    siren_out = f"/results/siren_{tag}.h5"
    json_out = f"/results/comparison_{tag}.json"
    timings_path = f"/results/timings_{tag}.json"
    timings = {}

    base_args = [
        "python", "-u", f"{repo}/run_full_usct.py",
        "--grid-size", str(GRID_SIZE),
        "--n-elements", str(N_ELEMENTS),
        "--iters", str(ITERS_PER_BAND),
        "--shots", str(SHOTS_PER_ITER),
        "--dx", str(DX_M),
    ]

    # --- Voxel -------------------------------------------------------
    print("=" * 70)
    print(f"  Running voxel-path FWI  ({GRID_SIZE}^3, {N_ELEMENTS} elements)")
    print("=" * 70)
    t0 = time.time()
    subprocess.run(
        base_args + ["--parameterization", "voxel", "--output", voxel_out],
        check=True, cwd=repo,
    )
    timings["voxel_wallclock_s"] = round(time.time() - t0, 1)
    print(f"\n  voxel wall-clock: {timings['voxel_wallclock_s']:.1f}s")

    # Commit partial results so a subsequent timeout doesn't erase them.
    results_vol.commit()

    # --- SIREN -------------------------------------------------------
    print("=" * 70)
    print(f"  Running SIREN-path FWI  ({GRID_SIZE}^3, {N_ELEMENTS} elements)")
    print("=" * 70)
    t0 = time.time()
    subprocess.run(
        base_args + ["--parameterization", "siren", "--output", siren_out],
        check=True, cwd=repo,
    )
    timings["siren_wallclock_s"] = round(time.time() - t0, 1)
    print(f"\n  siren wall-clock: {timings['siren_wallclock_s']:.1f}s")

    results_vol.commit()

    # --- Compare -----------------------------------------------------
    print("=" * 70)
    print("  Comparison (regional RMSE)")
    print("=" * 70)
    subprocess.run(
        ["python", "-u",
         f"{repo}/scripts/validate_siren_vs_voxel.py",
         "--voxel", voxel_out,
         "--siren", siren_out,
         "--json-out", json_out],
        check=True,
    )

    # --- Persist timings --------------------------------------------
    import json as _json
    with open(timings_path, "w") as f:
        _json.dump({
            "grid_size": GRID_SIZE,
            "n_elements": N_ELEMENTS,
            "iters_per_band": ITERS_PER_BAND,
            "shots_per_iter": SHOTS_PER_ITER,
            **timings,
            "sims_per_path": N_ELEMENTS + 3 * ITERS_PER_BAND * SHOTS_PER_ITER,
        }, f, indent=2)
    results_vol.commit()

    print(f"\nTimings saved: {timings_path}")
    print(f"  voxel: {timings.get('voxel_wallclock_s')}s")
    print(f"  siren: {timings.get('siren_wallclock_s')}s")
    print(f"  total: {sum(timings.values()):.0f}s")


@app.local_entrypoint()
def main():
    run_validation.remote()
