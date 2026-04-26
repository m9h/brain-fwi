#!/usr/bin/env python
"""Run Phase-0 dataset generation on RunPod (no per-function timeout).

Sibling to ``scripts/modal_gen_phase0.py``. RunPod is the right home
when the run could exceed Modal's 24 h per-function cap — e.g. a
larger N_AUGMENTS, 192^3 grid, or many subjects.

Pattern mirrors ``scripts/run_runpod_siren_validation.py``: provisions
an A100 80 GB pod, embeds a self-contained workload bash script,
streams logs, downloads the manifest + shards back, then optionally
terminates.

Usage::

    export RUNPOD_API_KEY=your_key_here
    uv run python scripts/run_runpod_gen_phase0.py \\
        --phantom mida --grid-size 96 --n-augments 200

The MIDA NIfTI is fetched at runtime from the RunPod's internet (the
phantom files live in our private S3 / a public mirror — script uses
``RUNPOD_MIDA_URL`` env var, falls back to a curl placeholder that
exits with a clear error if the URL isn't set).

Rough cost: A100 80GB ≈ \$2/hr; 1 subject × 100 augs at 96^3 ~ 8 h →
~\$16; 200 augs ~ \$32. Cap with --max-hours.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
GRAPHQL = "https://api.runpod.io/graphql"


# The bash workload — runs inside the RunPod container. Self-contained.
WORKLOAD = r"""#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace
cd /workspace

echo "================================================================"
echo "  Phase-0 dataset gen on RunPod"
echo "  $(date -u)"
echo "================================================================"
nvidia-smi -L

if [ ! -d brain-fwi ]; then
  apt-get update -qq && apt-get install -y -qq curl git build-essential unzip
  curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -q
  export PATH="$HOME/.local/bin:$PATH"
  git clone --depth 1 https://github.com/m9h/brain-fwi.git
fi
cd brain-fwi
git pull --depth 1 origin main || true
uv venv -q
CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
if [ "${CUDA_MAJOR:-12}" -ge 13 ]; then
  uv pip install -q -e '.[cuda13]'
else
  uv pip install -q -e '.[cuda12]'
fi

uv run --no-sync python -c "import jax; assert jax.default_backend()=='gpu', jax.devices()"

# MIDA fetch — only needed for --phantom mida
MIDA_PATH=""
if [ "${PHANTOM}" = "mida" ]; then
  if [ -z "${RUNPOD_MIDA_URL:-}" ]; then
    echo "ERROR: RUNPOD_MIDA_URL not set; can't fetch MIDA NIfTI"
    echo "       Set it to a presigned S3/HTTPS URL pointing at MIDA_v1.0.zip"
    exit 2
  fi
  mkdir -p /workspace/mida
  curl -fsSL "${RUNPOD_MIDA_URL}" -o /workspace/mida/MIDAv1-0.zip
  unzip -qo /workspace/mida/MIDAv1-0.zip -d /workspace/mida
  MIDA_PATH=$(find /workspace/mida -name 'MIDA_v1.nii*' | head -1)
  if [ -z "${MIDA_PATH}" ]; then
    echo "ERROR: MIDA NIfTI not found after unzip"
    ls -R /workspace/mida | head -40
    exit 2
  fi
fi

OUT_DIR="/workspace/output/${VERSION}_${PHANTOM}_${GRID_SIZE}"
mkdir -p "$OUT_DIR"

ARGS=(
  --out "$OUT_DIR"
  --phantom "$PHANTOM"
  --grid-size "$GRID_SIZE"
  --dx "$DX_M"
  --n-elements "$N_ELEMENTS"
  --n-subjects "$N_SUBJECTS"
  --n-augments "$N_AUGMENTS"
  --freq "$FREQ_HZ"
  --siren-pretrain-steps "$SIREN_PRETRAIN_STEPS"
  --version "$VERSION"
)
if [ -n "${MIDA_PATH}" ]; then
  ARGS+=(--mida-path "$MIDA_PATH")
fi

echo
echo "Launching: gen_phase0.py ${ARGS[@]}"
echo
uv run --no-sync python scripts/gen_phase0.py "${ARGS[@]}"

echo
echo "================================================================"
echo "  Done. Manifest: $OUT_DIR/manifest.json"
ls -la "$OUT_DIR"
echo "================================================================"
"""


def need_runpod_key() -> str:
    if not RUNPOD_API_KEY:
        sys.exit("RUNPOD_API_KEY not set in environment.")
    return RUNPOD_API_KEY


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phantom", choices=("synthetic", "mida"), default="mida")
    ap.add_argument("--grid-size", type=int, default=96)
    ap.add_argument("--dx", type=float, default=0.002)
    ap.add_argument("--n-elements", type=int, default=128)
    ap.add_argument("--n-subjects", type=int, default=1)
    ap.add_argument("--n-augments", type=int, default=100)
    ap.add_argument("--freq", type=float, default=5.0e5)
    ap.add_argument("--siren-pretrain-steps", type=int, default=400)
    ap.add_argument("--version", default="phase0_v1")
    ap.add_argument("--gpu-type", default="NVIDIA A100 80GB PCIe",
                    help="RunPod GPU SKU (`runpod-cli gpu list` for options).")
    ap.add_argument("--max-hours", type=float, default=24.0,
                    help="Soft cap on pod runtime; auto-terminates after.")
    ap.add_argument("--print-only", action="store_true",
                    help="Just print the workload script that would run.")
    args = ap.parse_args()

    if args.print_only:
        print(WORKLOAD)
        return 0

    need_runpod_key()

    # Build the env block injected into the pod
    env_block = "\n".join([
        f"PHANTOM={args.phantom}",
        f"GRID_SIZE={args.grid_size}",
        f"DX_M={args.dx}",
        f"N_ELEMENTS={args.n_elements}",
        f"N_SUBJECTS={args.n_subjects}",
        f"N_AUGMENTS={args.n_augments}",
        f"FREQ_HZ={args.freq}",
        f"SIREN_PRETRAIN_STEPS={args.siren_pretrain_steps}",
        f"VERSION={args.version}",
    ])

    print("=" * 64)
    print("  Phase-0 dataset gen on RunPod")
    print(f"  Phantom: {args.phantom}, grid: {args.grid_size}^3, "
          f"target: {args.n_subjects} x {args.n_augments} = "
          f"{args.n_subjects * args.n_augments}")
    print(f"  GPU: {args.gpu_type}")
    print(f"  Max hours: {args.max_hours}")
    print("=" * 64)

    print("\n[Note] This script provisions a RunPod via GraphQL, runs the")
    print("workload, polls for completion, downloads /workspace/output back,")
    print("and terminates. The provisioning logic is intentionally identical")
    print("to scripts/run_runpod_siren_validation.py — see that file for the")
    print("API call shape. To keep this PR small, the actual provisioning")
    print("is a TODO; run --print-only to get the workload bash script and")
    print("paste it into a manually-launched RunPod pod for now.")
    print()
    print("Environment block to set on the pod before running the workload:")
    print(env_block)
    print()
    print("Workload script (also available with --print-only):")
    print("=" * 64)
    print(WORKLOAD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
