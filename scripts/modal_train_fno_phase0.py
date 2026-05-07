"""Modal runner: FNO surrogate training on the Phase-0 v2a dataset.

Reads the v2a Phase-0 sharded dataset off the same ``brain-fwi-phase0``
volume that the generation pipeline writes to — no extra upload step.
H100-80GB / H200-141GB has the headroom to train the default
``(hidden_channels=32, num_modes=12, depth=2)`` architecture, which the
DGX Spark GB10 OOM'd on (peak 150 GiB).

Usage::

    # Smoke (200 steps, ~$5):
    modal run --detach scripts/modal_train_fno_phase0.py \\
        --gpu H100 --version phase0_v2a --n-steps 200 \\
        --hidden-channels 16 --num-modes 8 --depth 1

    # Production (1000 steps, ~$30-60 depending on GPU + arch):
    modal run --detach scripts/modal_train_fno_phase0.py \\
        --gpu H100 --version phase0_v2a --n-steps 1000 \\
        --hidden-channels 32 --num-modes 12 --depth 2 \\
        --n-timesteps 1100

Pull the trained surrogate back::

    modal volume get brain-fwi-fno-output /output ./fno_phase4_v2a

Why H100 vs H200/B200:

  - H100-80GB: enough for hidden=16/modes=12/depth=2 OR hidden=32/modes=8/depth=1.
  - H200-141GB: comfortably fits hidden=32/modes=12/depth=2 (the full default).
  - B200-192GB: overkill, useful if we go to 192^3 input.

Pass ``--n-timesteps`` to fix the FNO output time-axis. Phase-0 samples
have variable n_t (CFL-derived dt depends on jittered c_max), so the
trainer crops or pads each sample to ``model.n_timesteps``. Set to the
dataset-wide min (~1100 for v2a at 96^3) so it always crops, never pads.
"""

from __future__ import annotations

import time

import modal

app = modal.App("brain-fwi-fno-phase4")

GIT_BRANCH = "feature/parallel-modal-phase0"
CACHE_BUST = "2026-05-06-fno-skip-validation"

# v2a lives on the same volume that gen_phase0 writes to.
DATASET_VOL = "brain-fwi-phase0"
OUTPUT_VOL = "brain-fwi-fno-output"

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

dataset_vol = modal.Volume.from_name(DATASET_VOL, create_if_missing=False)
output_vol = modal.Volume.from_name(OUTPUT_VOL, create_if_missing=True)


def _train_body(
    version: str,
    phantom: str,
    grid_size: int,
    hidden_channels: int,
    num_modes: int,
    depth: int,
    n_steps: int,
    learning_rate: float,
    lambda_spec: float,
    held_out_fraction: float,
    n_timesteps: int,
    skip_validation: bool,
    n_grad_samples: int,
):
    """Body of the training run, identical regardless of which GPU it runs on."""
    import os
    import subprocess

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    subprocess.run(["nvidia-smi", "-L"], check=True)
    subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free",
         "--format=csv,noheader"], check=True,
    )

    data_path = f"/dataset/output/{version}_{phantom}_{grid_size}/merged"
    out_path = f"/output/{version}_{phantom}_{grid_size}/fno_surrogate"

    args = [
        "python", "-u", "/opt/brain-fwi/scripts/train_fno_on_phase0.py",
        "--data", data_path,
        "--out", out_path,
        "--n-steps", str(n_steps),
        "--learning-rate", str(learning_rate),
        "--lambda-spec", str(lambda_spec),
        "--hidden-channels", str(hidden_channels),
        "--num-modes", str(num_modes),
        "--depth", str(depth),
        "--held-out-fraction", str(held_out_fraction),
    ]
    if n_timesteps > 0:
        args += ["--n-timesteps", str(n_timesteps)]
    if skip_validation:
        args += ["--skip-validation"]
    args += ["--n-grad-samples", str(n_grad_samples)]
    print(f"\nLaunching: {' '.join(args)}\n")

    t0 = time.time()
    subprocess.run(args, check=True, cwd="/opt/brain-fwi")
    wall = time.time() - t0

    output_vol.commit()
    print(f"\nTraining wall: {wall/60:.1f} min")
    return {"wall_s": wall}


@app.function(
    image=image,
    gpu="H100",
    timeout=3 * 60 * 60,            # 3h short-time budget
    memory=32 * 1024,
    volumes={
        "/dataset": dataset_vol,
        "/output": output_vol,
    },
    retries=2,
)
def train_h100(**kwargs):
    return _train_body(**kwargs)


@app.function(
    image=image,
    gpu="H200",
    timeout=3 * 60 * 60,
    memory=32 * 1024,
    volumes={
        "/dataset": dataset_vol,
        "/output": output_vol,
    },
    retries=2,
)
def train_h200(**kwargs):
    return _train_body(**kwargs)


@app.local_entrypoint()
def main(
    gpu: str = "H100",
    version: str = "phase0_v2a",
    phantom: str = "mida",
    grid_size: int = 96,
    hidden_channels: int = 32,
    num_modes: int = 12,
    depth: int = 2,
    n_steps: int = 1000,
    learning_rate: float = 1e-3,
    lambda_spec: float = 0.3,
    held_out_fraction: float = 0.2,
    n_timesteps: int = 0,           # 0 = infer from first sample
    skip_validation: bool = False,  # smoke: bypass §7.2/§7.3 gates
    n_grad_samples: int = 20,       # cap gradient-accuracy sample count
):
    print("=" * 64)
    print(f"  FNO surrogate training on Modal {gpu}")
    print(f"  Dataset: {version}_{phantom}_{grid_size}")
    print(f"  Arch: hidden={hidden_channels}, modes={num_modes}, depth={depth}")
    print(f"  Steps: {n_steps}, LR: {learning_rate}, λ_spec: {lambda_spec}")
    print(f"  Held out: {held_out_fraction*100:.0f}%, n_timesteps: "
          f"{n_timesteps if n_timesteps > 0 else 'auto'}")
    print("=" * 64)

    runner = {"H100": train_h100, "H200": train_h200}.get(gpu.upper())
    if runner is None:
        raise SystemExit(f"Unknown gpu={gpu!r}; use H100 or H200")

    result = runner.remote(
        version=version, phantom=phantom, grid_size=grid_size,
        hidden_channels=hidden_channels,
        num_modes=num_modes,
        depth=depth,
        n_steps=n_steps,
        learning_rate=learning_rate,
        lambda_spec=lambda_spec,
        held_out_fraction=held_out_fraction,
        n_timesteps=n_timesteps,
        skip_validation=skip_validation,
        n_grad_samples=n_grad_samples,
    )
    print(f"\nDone in {result['wall_s']/60:.1f} min")
    print(f"Pull results: modal volume get {OUTPUT_VOL} "
          f"/output/{version}_{phantom}_{grid_size} ./fno_phase4_{version}")
