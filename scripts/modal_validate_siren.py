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

# Clone the branch that contains BOTH the SIREN reconciliation (already
# on main via PR #3) AND this validation harness. Once this branch
# merges, callers can switch back to main.
GIT_BRANCH = "feature/siren-validation"

# Build the container image from the live repo — matches current main
# including the SIREN reconciliation, avoiding the pin-drift trap that
# modal_mida_256.py fell into.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        f"git clone --depth 1 --branch {GIT_BRANCH} "
        f"https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && uv venv --system-site-packages /opt/venv",
        "cd /opt/brain-fwi && /opt/venv/bin/uv pip install -e '.[cuda12]'",
    )
    .env({"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"})
)

# Persistent output volume — inspect results with `modal volume ls` /
# `modal volume get brain-fwi-validation <path>`.
results_vol = modal.Volume.from_name(
    "brain-fwi-validation", create_if_missing=True,
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60,  # 1 hour ceiling
    volumes={"/results": results_vol},
)
def run_validation():
    import subprocess
    import os

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    # Sanity check: GPU is visible.
    subprocess.run(["nvidia-smi", "-L"], check=True)

    repo = "/opt/brain-fwi"
    voxel_out = "/results/voxel.h5"
    siren_out = "/results/siren.h5"

    base_args = [
        "/opt/venv/bin/python", "-u", f"{repo}/run_full_usct.py",
        "--grid-size", str(GRID_SIZE),
        "--n-elements", str(N_ELEMENTS),
        "--iters", str(ITERS_PER_BAND),
        "--shots", str(SHOTS_PER_ITER),
        "--dx", str(DX_M),
    ]

    print("=" * 70)
    print("  Running voxel-path FWI")
    print("=" * 70)
    subprocess.run(
        base_args + ["--parameterization", "voxel", "--output", voxel_out],
        check=True, cwd=repo,
    )

    print("=" * 70)
    print("  Running SIREN-path FWI")
    print("=" * 70)
    subprocess.run(
        base_args + ["--parameterization", "siren", "--output", siren_out],
        check=True, cwd=repo,
    )

    print("=" * 70)
    print("  Comparison (regional RMSE)")
    print("=" * 70)
    subprocess.run(
        ["/opt/venv/bin/python", "-u",
         f"{repo}/scripts/validate_siren_vs_voxel.py",
         "--voxel", voxel_out,
         "--siren", siren_out,
         "--json-out", "/results/comparison.json"],
        check=True,
    )

    results_vol.commit()
    print(f"\nResults saved to Modal volume 'brain-fwi-validation'")
    print(f"  /results/voxel.h5, /results/siren.h5, /results/comparison.json")


@app.local_entrypoint()
def main():
    run_validation.remote()
