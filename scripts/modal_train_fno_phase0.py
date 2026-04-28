"""Modal runner: FNO surrogate training on the Phase-0 dataset.

The DGX Spark's unified-memory GB10 OOM'd at the default
``(hidden_channels=32, num_modes=12, depth=2)`` architecture (job 1020,
peak allocation 150 GiB). H100-80GB / H200-141GB on Modal have the
headroom to actually train this size.

Two-step setup (run from a machine with the modal CLI configured)::

    # 1. Upload the Phase-0 dataset (one-time, ~6.3 GB).
    modal volume create brain-fwi-phase0-dataset
    modal volume put brain-fwi-phase0-dataset \\
        /data/datasets/brain-fwi/phase0_v1_mida_96 \\
        /datasets/phase0_v1_mida_96

    # 2. Volume for trained-surrogate output.
    modal volume create brain-fwi-fno-output

    # 3. Run.
    modal run scripts/modal_train_fno_phase0.py \\
        --gpu H100 --hidden-channels 32 --num-modes 12 --depth 2 \\
        --n-steps 1000

Pull the trained surrogate back::

    modal volume get brain-fwi-fno-output /output ./fno_phase4_v1

Why H100 vs H200/B200:

  - H100-80GB: enough headroom for hidden=16 / modes=12 / depth=2 OR
    hidden=32 / modes=8 / depth=1. ~$3-4 / hr.
  - H200-141GB: comfortably fits hidden=32 / modes=12 / depth=2 (the
    full default). May need account-level access. ~$5 / hr.
  - B200-192GB: overkill but available; useful if we go to 192^3 input.

Defaults are sized for H100; pass ``--gpu H200`` and/or larger
hyperparams if you want the production-scale architecture.
"""

from __future__ import annotations

import time

import modal

app = modal.App("brain-fwi-fno-phase4")

GIT_BRANCH = "main"
CACHE_BUST = "2026-04-28-fno-train-v2-scan-remat"

DATASET_VOL = "brain-fwi-phase0-dataset"
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
    hidden_channels: int,
    num_modes: int,
    depth: int,
    n_steps: int,
    learning_rate: float,
    lambda_spec: float,
    held_out_fraction: float,
):
    """Body of the training run, identical regardless of which GPU it runs on."""
    import os
    import subprocess
    import sys

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    subprocess.run(["nvidia-smi", "-L"], check=True)
    subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.free",
         "--format=csv,noheader"], check=True,
    )

    args = [
        "python", "-u", "/opt/brain-fwi/scripts/train_fno_on_phase0.py",
        "--data", "/dataset/datasets/phase0_v1_mida_96",
        "--out", "/output/fno_surrogate",
        "--n-steps", str(n_steps),
        "--learning-rate", str(learning_rate),
        "--lambda-spec", str(lambda_spec),
        "--hidden-channels", str(hidden_channels),
        "--num-modes", str(num_modes),
        "--depth", str(depth),
        "--held-out-fraction", str(held_out_fraction),
    ]
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
)
def train_h200(**kwargs):
    return _train_body(**kwargs)


@app.local_entrypoint()
def main(
    gpu: str = "H100",
    hidden_channels: int = 32,
    num_modes: int = 12,
    depth: int = 2,
    n_steps: int = 1000,
    learning_rate: float = 1e-3,
    lambda_spec: float = 0.3,
    held_out_fraction: float = 0.2,
):
    print("=" * 64)
    print(f"  FNO surrogate training on Modal {gpu}")
    print(f"  Arch: hidden={hidden_channels}, modes={num_modes}, depth={depth}")
    print(f"  Steps: {n_steps}, LR: {learning_rate}, λ_spec: {lambda_spec}")
    print(f"  Held out: {held_out_fraction*100:.0f}%")
    print("=" * 64)

    runner = {"H100": train_h100, "H200": train_h200}.get(gpu.upper())
    if runner is None:
        raise SystemExit(f"Unknown gpu={gpu!r}; use H100 or H200")

    result = runner.remote(
        hidden_channels=hidden_channels,
        num_modes=num_modes,
        depth=depth,
        n_steps=n_steps,
        learning_rate=learning_rate,
        lambda_spec=lambda_spec,
        held_out_fraction=held_out_fraction,
    )
    print(f"\nDone in {result['wall_s']/60:.1f} min")
    print(f"Pull results: modal volume get {OUTPUT_VOL} /output ./fno_phase4_v1")
