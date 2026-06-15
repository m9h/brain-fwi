"""Modal runner: MOFI Stage 1 — FD gradient check through j-Wave.

Runs tests/test_mofi.py::test_pose_gradient_matches_finite_difference on a
GPU. The DGX-Spark Slurm node was saturated by an unrelated 20-core CPU
job, so this ports the tiny (64², 4-shot) verification to Modal.

The image is built from GitHub `main` only to install the dependency
stack (jax[cuda12] + j-Wave + jaxdf + equinox/optax). The actual code
under test — the uncommitted mofi.py / test_mofi.py on the local feature
branch — is overlaid from the local working tree via add_local_dir, so
nothing has to be pushed.

Usage::

    modal run scripts/modal_mofi_stage1.py
"""

from pathlib import Path

import modal

app = modal.App("brain-fwi-mofi-stage1")

GIT_BRANCH = "main"
CACHE_BUST = "2026-06-14-mofi-stage1-v1"

REPO_ROOT = Path(__file__).resolve().parent.parent  # ~/dev/brain-fwi

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
    # pytest is a dev-only dep (not in the cuda12 extra). Separate layer so the
    # expensive jax/j-Wave install above stays cached across re-runs.
    .pip_install("pytest")
    # Overlay the LOCAL working tree over the cloned package + tests so the
    # container runs the uncommitted MOFI code. Must come after run_commands.
    .add_local_dir(
        str(REPO_ROOT / "src" / "brain_fwi"),
        "/opt/brain-fwi/src/brain_fwi",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        str(REPO_ROOT / "tests"),
        "/opt/brain-fwi/tests",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)


@app.function(image=image, gpu="A10G", timeout=45 * 60)
def run_stage1() -> int:
    import os
    import subprocess

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    repo = "/opt/brain-fwi"
    print(f"GPU check:")
    subprocess.run(
        ["python", "-c",
         "import jax; print(' devices:', jax.devices()); "
         "print(' backend:', jax.default_backend())"],
        check=True, cwd=repo,
    )

    cmd = [
        "python", "-m", "pytest",
        "tests/test_mofi.py",      # full slow MOFI suite: Stage 1 (FD) + 2 (pose recovery) + 3 (unblock FWI)
        "-m", "slow", "-v", "-s",
    ]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd, cwd=repo)
    print(f"\npytest exit code: {proc.returncode}")
    return proc.returncode


@app.local_entrypoint()
def main():
    code = run_stage1.remote()
    if code == 0:
        print("\n✅ MOFI Stage 1 PASSED on Modal.")
    else:
        print(f"\n❌ MOFI Stage 1 FAILED (pytest exit {code}).")
        raise SystemExit(code)
