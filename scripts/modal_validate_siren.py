"""Modal runner: SIREN-vs-voxel FWI validation on an A100.

Runs ``run_full_usct.py`` twice (voxel, siren) on the synthetic phantom,
then invokes ``validate_siren_vs_voxel.py`` to print a regional-RMSE
comparison table. All outputs land in a persistent Modal volume so
they can be inspected after the run.

Usage::

    modal run scripts/modal_validate_siren.py

Deliberately minimal compared to ``scripts/modal_mida_256.py``:

- No MIDA (synthetic phantom only) — keeps the dep surface small.
- No hardcoded package pins — image installs from the live ``pyproject.toml``
  via ``uv pip install -e .[cuda12]``, tracking current main.
- Small grid (96³) to keep the validation cheap; scale up by editing
  ``GRID_SIZE`` / ``N_ELEMENTS`` once the pipeline is verified.
"""

import modal

app = modal.App("brain-fwi-siren-validation")

GRID_SIZE = 96
N_ELEMENTS = 64
ITERS_PER_BAND = 15
SHOTS_PER_ITER = 8
DX_M = 0.002

# SIREN reconciliation + validation harness are both on main now.
GIT_BRANCH = "main"

# Bump on each push that should force the image to rebuild (Modal
# fingerprints the build spec, not what `git clone` fetches at runtime).
CACHE_BUST = "2026-04-25-validation-fullscale"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .env({"BRAIN_FWI_CACHE_BUST": CACHE_BUST})
    .run_commands(
        f"git clone --depth 1 --branch {GIT_BRANCH} "
        f"https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        # System-wide install — ephemeral container, no need for a venv.
        "cd /opt/brain-fwi && uv pip install --system -e '.[cuda12]'",
    )
)

# Persistent output volume — inspect results with `modal volume ls` /
# `modal volume get brain-fwi-validation <path>`.
results_vol = modal.Volume.from_name(
    "brain-fwi-validation", create_if_missing=True,
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3 * 60 * 60,  # 3h — combined voxel+siren is ~60-90 min per #16
    volumes={"/results": results_vol},
)
def run_validation():
    import os
    import subprocess
    import time

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    # Sanity check: GPU is visible.
    subprocess.run(["nvidia-smi", "-L"], check=True)

    repo = "/opt/brain-fwi"
    tag = f"g{GRID_SIZE}_n{N_ELEMENTS}_i{ITERS_PER_BAND}"
    voxel_out = f"/results/voxel_{tag}.h5"
    siren_out = f"/results/siren_{tag}.h5"
    json_out = f"/results/comparison_{tag}.json"
    timings = {}

    base_args = [
        "python", "-u", f"{repo}/run_full_usct.py",
        "--grid-size", str(GRID_SIZE),
        "--n-elements", str(N_ELEMENTS),
        "--iters", str(ITERS_PER_BAND),
        "--shots", str(SHOTS_PER_ITER),
        "--dx", str(DX_M),
    ]

    # --- Voxel ----------------------------------------------------------
    print("=" * 70)
    print(f"  Voxel-path FWI ({GRID_SIZE}^3, {N_ELEMENTS} elements)")
    print("=" * 70)
    t0 = time.time()
    subprocess.run(
        base_args + ["--parameterization", "voxel", "--output", voxel_out],
        check=True, cwd=repo,
    )
    timings["voxel_wallclock_s"] = round(time.time() - t0, 1)
    print(f"\n  voxel wall-clock: {timings['voxel_wallclock_s']:.1f}s")
    # Commit partial — a subsequent timeout doesn't erase voxel results.
    results_vol.commit()

    # --- SIREN ----------------------------------------------------------
    print("=" * 70)
    print(f"  SIREN-path FWI ({GRID_SIZE}^3, {N_ELEMENTS} elements)")
    print("=" * 70)
    t0 = time.time()
    subprocess.run(
        base_args + ["--parameterization", "siren", "--output", siren_out],
        check=True, cwd=repo,
    )
    timings["siren_wallclock_s"] = round(time.time() - t0, 1)
    print(f"\n  siren wall-clock: {timings['siren_wallclock_s']:.1f}s")
    results_vol.commit()

    # --- Compare --------------------------------------------------------
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

    # --- Timings --------------------------------------------------------
    import json as _json
    timings_path = f"/results/timings_{tag}.json"
    with open(timings_path, "w") as f:
        _json.dump({
            "grid_size": GRID_SIZE,
            "n_elements": N_ELEMENTS,
            "iters_per_band": ITERS_PER_BAND,
            "shots_per_iter": SHOTS_PER_ITER,
            **timings,
            "total_s": sum(timings.values()),
        }, f, indent=2)
    results_vol.commit()

    print(f"\nResults on Modal volume 'brain-fwi-validation':")
    print(f"  {voxel_out}")
    print(f"  {siren_out}")
    print(f"  {json_out}")
    print(f"  {timings_path}")
    print(f"  voxel: {timings['voxel_wallclock_s']:.0f}s, "
          f"siren: {timings['siren_wallclock_s']:.0f}s, "
          f"total: {sum(timings.values()):.0f}s")


@app.local_entrypoint()
def main():
    run_validation.remote()
