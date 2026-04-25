#!/usr/bin/env python
"""Run the SIREN-vs-voxel FWI validation (Issue #11) on RunPod.

Provisions an A100 pod, runs both parameterisations of ``run_full_usct.py``
back-to-back at 96^3 with matched config, then runs the comparison
harness. Results land at ``/workspace/{voxel,siren}.h5`` and
``/workspace/comparison.json`` on the pod's persistent volume.

Why RunPod and not Modal:

  - Modal's per-function timeout is the real reason the previous
    96^3 attempts failed. RunPod pods run until you stop them, so the
    ~60-90 min combined-path runtime is comfortable.
  - At ~$2/hr for A100 80GB, the total bill is ~$2-4. Modal would have
    been $8-12 with margin.

Usage::

    export RUNPOD_API_KEY=your_key_here  # already exported in the dev env
    uv run python scripts/run_runpod_siren_validation.py
    # ... pod launches; follow printed instructions to SSH in,
    # run the workload, scp results back, then stop the pod.

Or for a fully unattended run, use ``--auto`` (provisions, runs, polls,
downloads, terminates). Requires SSH key uploaded to RunPod account.

This script does not depend on changes to ``run_full_usct.py`` —
``--parameterization`` flag is already on main from PR #3 (commit 8602eb0).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Workload that runs inside the RunPod container. Self-contained.
# Writes outputs to /workspace (persistent volume), then optionally
# self-terminates via RunPod GraphQL API when AUTO_TERMINATE=1.
WORKLOAD_SCRIPT = r"""#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace
cd /workspace

echo "================================================================"
echo "  SIREN-vs-voxel FWI validation (Issue #11)"
echo "  $(date -u)"
echo "================================================================"
nvidia-smi -L

# Repo + deps
if [ ! -d brain-fwi ]; then
  apt-get update -qq && apt-get install -y -qq curl git build-essential
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

# Verify GPU is visible to JAX
uv run --no-sync python -c "import jax; assert jax.default_backend()=='gpu', jax.devices()"

COMMON_ARGS="--grid-size 96 --n-elements 64 --iters 15 --shots 8 --dx 0.002"

echo
echo "================================================================"
echo "  Voxel-path FWI"
echo "================================================================"
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --no-sync python -u run_full_usct.py $COMMON_ARGS \
    --parameterization voxel \
    --output /workspace/voxel.h5 \
    2>&1 | tee /workspace/voxel.log

echo
echo "================================================================"
echo "  SIREN-path FWI"
echo "================================================================"
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --no-sync python -u run_full_usct.py $COMMON_ARGS \
    --parameterization siren \
    --output /workspace/siren.h5 \
    2>&1 | tee /workspace/siren.log

echo
echo "================================================================"
echo "  Comparison (regional RMSE)"
echo "================================================================"
uv run --no-sync python -u scripts/validate_siren_vs_voxel.py \
    --voxel /workspace/voxel.h5 \
    --siren /workspace/siren.h5 \
    --json-out /workspace/comparison.json \
    | tee /workspace/comparison.txt

touch /workspace/DONE
echo "$(date -u): DONE" > /workspace/timestamp.txt

# Self-terminate if requested. Avoids leaving the pod running after the
# workload finishes — a common cost gotcha with manual workflows.
if [ "${AUTO_TERMINATE:-0}" = "1" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "Self-terminating pod $RUNPOD_POD_ID via RunPod API..."
  curl -sS -X POST https://api.runpod.io/graphql \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"})}\"}"
fi

# Otherwise, sleep so the pod stays alive for SCP retrieval. Time-capped
# so an unattended pod can't burn money indefinitely.
echo "Workload complete. Pod will stay alive for 6h or until manually stopped."
sleep 21600
"""


def _get_pod_address(pod: dict) -> tuple[str, int] | None:
    """Best-effort SSH host:port from a RunPod pod dict."""
    runtime = pod.get("runtime") or {}
    for p in runtime.get("ports") or []:
        if p.get("privatePort") == 22:
            return p.get("ip"), int(p.get("publicPort"))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="SIREN-vs-voxel validation on RunPod")
    parser.add_argument("--gpu", default="NVIDIA A100 80GB PCIe",
                        help="RunPod GPU type ID")
    parser.add_argument("--volume-gb", type=int, default=20)
    parser.add_argument("--auto", action="store_true",
                        help="Pass AUTO_TERMINATE=1 so pod self-stops on completion")
    parser.add_argument("--results-dir", type=Path, default=Path("results/siren_vs_voxel_runpod"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key and not args.dry_run:
        print("RUNPOD_API_KEY env var not set", file=sys.stderr)
        return 2

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"GPU:        {args.gpu}")
        print(f"Volume:     {args.volume_gb} GB")
        print(f"Auto-term:  {args.auto}")
        print(f"Results:    {args.results_dir}")
        print("\n--- workload script (truncated) ---")
        print(WORKLOAD_SCRIPT[:500] + "\n...")
        return 0

    import runpod
    runpod.api_key = api_key

    # Encode the workload as the container's docker start command. RunPod's
    # `docker_args` is appended to the image's default entrypoint, so we
    # explicitly invoke bash with the script via a heredoc-as-arg trick:
    # we base64-encode the workload to dodge quoting headaches.
    import base64
    workload_b64 = base64.b64encode(WORKLOAD_SCRIPT.encode()).decode()
    docker_args = f'bash -c "echo {workload_b64} | base64 -d > /workspace/run.sh && bash /workspace/run.sh"'

    env = {
        "AUTO_TERMINATE": "1" if args.auto else "0",
    }
    if args.auto:
        env["RUNPOD_API_KEY"] = api_key  # workload self-terminate path

    print("Provisioning pod...")
    pod = runpod.create_pod(
        name="brain-fwi-siren-validation",
        image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        gpu_type_id=args.gpu,
        gpu_count=1,
        volume_in_gb=args.volume_gb,
        container_disk_in_gb=20,
        min_vcpu_count=4,
        min_memory_in_gb=32,
        docker_args=docker_args,
        env=env,
    )
    pod_id = pod["id"]
    print(f"Pod: {pod_id}")
    print(f"Dashboard: https://www.runpod.io/console/pods/{pod_id}")

    print("Waiting for pod to start...")
    for i in range(120):
        status = runpod.get_pod(pod_id)
        rt = status.get("runtime") or {}
        if rt.get("uptimeInSeconds", 0) > 0:
            print(f"  uptime: {rt['uptimeInSeconds']}s — running")
            break
        time.sleep(10)
        print(f"  {(i + 1) * 10}s — {status.get('desiredStatus', 'unknown')}")
    else:
        print("Pod failed to start within 20 min; aborting", file=sys.stderr)
        return 3

    addr = _get_pod_address(status)
    if addr:
        host, port = addr
        ssh_cmd = f"ssh root@{host} -p {port}"
    else:
        ssh_cmd = "(check dashboard for SSH command)"

    args.results_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.results_dir / "pod_state.json"
    state_path.write_text(json.dumps({"pod_id": pod_id, "ssh": ssh_cmd}, indent=2))

    print()
    print("=" * 60)
    print(f"  Pod is running and the workload is executing.")
    print("=" * 60)
    print(f"""
Workload writes results to /workspace on the pod:
  /workspace/voxel.h5
  /workspace/siren.h5
  /workspace/comparison.json   (regional RMSE table)
  /workspace/comparison.txt    (printed table)
  /workspace/DONE              (touched on completion)

To monitor progress (~60-90 min):
  {ssh_cmd}
  tail -f /workspace/voxel.log /workspace/siren.log

To download results once /workspace/DONE exists:
  scp -P {addr[1] if addr else "<port>"} root@{addr[0] if addr else "<host>"}:/workspace/voxel.h5 \\
      {args.results_dir / "voxel.h5"}
  scp -P {addr[1] if addr else "<port>"} root@{addr[0] if addr else "<host>"}:/workspace/siren.h5 \\
      {args.results_dir / "siren.h5"}
  scp -P {addr[1] if addr else "<port>"} root@{addr[0] if addr else "<host>"}:/workspace/comparison.json \\
      {args.results_dir / "comparison.json"}

To stop the pod (do this!):
  uv run --no-sync python -c \\
      "import os, runpod; runpod.api_key=os.environ['RUNPOD_API_KEY']; \\
       runpod.terminate_pod('{pod_id}')"

  ({'AUTO_TERMINATE=1 set — pod will self-terminate on workload completion.' if args.auto else 'Or pass --auto next time to skip this step.'})

Pod state saved to: {state_path}
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
