#!/usr/bin/env python
"""Run the 3D absorption-aware FWI demo on a RunPod B200 pod.

Provisions a single top-end Blackwell (B200) pod and runs
``examples/06_absorption_aware_fwi_3d.py --full`` (192^3, helmet array,
multi-band) twice — lossless vs known-alpha forward — on a self-contained
synthetic anatomical head. Results (figure + metrics + npz) land on the pod's
``/workspace`` and are pulled back via scp.

Why RunPod B200 (not the local DGX Spark, not Modal/beam):
  - The Spark (GB10) is a Blackwell GPU but ~120 GB; a B200 is ~180 GB HBM3e at
    ~8 TB/s, which is what a 192^3 multi-band A/B actually wants. A *persistent*
    RunPod pod has no per-function timeout (Modal's timeout killed FWI before),
    so the multi-hour run is comfortable — "super stable".
  - No DGX Station (GB300/ARM) is available; a single cloud B200 is the
    accessible top-end. It is x86 + jax[cuda12], so the workload installs the
    ``.[cuda12]`` extra (the same repackage the beam/Modal runners use).

CRITICAL — push first. The pod ``pip install``s from GitHub:
  - brain-fwi is cloned at ``--branch`` (default feature/diffusion-prior-fwi),
    which must contain examples/06 + FWIConfig.attenuation.
  - pyproject pins ``jwave @ ...@feature/time-domain-absorption``; that fork
    branch on GitHub must carry the *canonical Treeby-Cox EoS* fix. If it is
    behind, the pod silently runs the buggy per-step-decay absorption and the
    A/B is meaningless. Push both before launching (see --print-only to dry-run).

Data: uses ``--phantom synthetic`` (a built-in anatomical head). No patient data
is shipped to the cloud. (Real Birnbaum runs stay on the local Spark.)

Usage::

    export RUNPOD_API_KEY=...        # already exported in the dev env
    uv run python scripts/run_runpod_absorption_fwi_3d.py            # provision + run
    uv run python scripts/run_runpod_absorption_fwi_3d.py --print-only   # just the bash
    uv run python scripts/run_runpod_absorption_fwi_3d.py --auto     # self-terminate when done

Cost: B200 ~$4-6/hr on RunPod; a 192^3 3-band A/B is roughly 2-5 h => ~$10-30.
Cap with --auto (terminates on completion) and watch the dashboard.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path


# Bash workload — runs inside the pod. Self-contained; reads config from env.
WORKLOAD = r"""#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace && cd /workspace
echo "=== 3D absorption-aware FWI on RunPod $(date -u) ==="
nvidia-smi -L

if [ ! -d brain-fwi ]; then
  apt-get update -qq && apt-get install -y -qq curl git build-essential
  curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -q
  export PATH="$HOME/.local/bin:$PATH"
  git clone --depth 1 --branch "${BRANCH}" https://github.com/m9h/brain-fwi.git
fi
cd brain-fwi
git fetch -q origin "${BRANCH}" && git checkout -q "${BRANCH}" && git pull -q origin "${BRANCH}" || true
echo "brain-fwi @ $(git rev-parse --short HEAD)"

uv venv -q
CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
if [ "${CUDA_MAJOR:-12}" -ge 13 ]; then uv pip install -q -e '.[cuda13]'; else uv pip install -q -e '.[cuda12]'; fi

# Sanity: GPU visible AND the canonical absorbing EoS is in the installed jwave.
uv run --no-sync python -c "import jax; assert jax.default_backend()=='gpu', jax.devices(); \
import jwave.acoustics.time_varying as tv; assert hasattr(tv,'absorbing_pressure_from_density'), \
'jwave fork lacks absorbing_pressure_from_density — push m9h/jwave@feature/time-domain-absorption'; \
print('OK: GPU + absorbing EoS present')"

ARGS=(--full --phantom "${PHANTOM:-synthetic}")
[ -n "${GRID_OVERRIDE:-}" ] && ARGS+=(--n "${GRID_OVERRIDE}")
echo "Launching: examples/06_absorption_aware_fwi_3d.py ${ARGS[*]}"
uv run --no-sync python examples/06_absorption_aware_fwi_3d.py "${ARGS[@]}" 2>&1 | tee /workspace/run.log

cp -r results/absorption_aware_fwi_3d/* /workspace/ 2>/dev/null || true
touch /workspace/DONE
echo "=== done; results in /workspace ==="
ls -la /workspace

if [ "${AUTO_TERMINATE:-0}" = "1" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -s https://api.runpod.io/graphql -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"})}\"}" || true
fi
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="NVIDIA B200",
                    help="RunPod GPU SKU (override if the catalogue name differs).")
    ap.add_argument("--image",
                    default="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04",
                    help="Base image (needs a Blackwell-capable CUDA userspace).")
    ap.add_argument("--branch", default="feature/diffusion-prior-fwi",
                    help="brain-fwi branch the pod clones (must hold examples/06).")
    ap.add_argument("--phantom", choices=("synthetic", "birnbaum"), default="synthetic")
    ap.add_argument("--grid", type=int, default=None, help="override the 192^3 --full grid")
    ap.add_argument("--volume-gb", type=int, default=40)
    ap.add_argument("--auto", action="store_true", help="self-terminate the pod on completion")
    ap.add_argument("--print-only", action="store_true",
                    help="print the workload bash + env (paste into a manual pod).")
    args = ap.parse_args()

    env_block = {
        "BRANCH": args.branch,
        "PHANTOM": args.phantom,
        "AUTO_TERMINATE": "1" if args.auto else "0",
    }
    if args.grid:
        env_block["GRID_OVERRIDE"] = str(args.grid)

    if args.print_only:
        print("# --- env to set on the pod ---")
        for k, v in env_block.items():
            print(f"export {k}={v}")
        print("\n# --- workload ---")
        print(WORKLOAD)
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY not set. Use --print-only to get the bash to paste "
              "into a manually-launched pod.", file=sys.stderr)
        return 2

    try:
        import runpod
    except ImportError:
        print("`runpod` SDK not installed (uv pip install runpod), or use --print-only.",
              file=sys.stderr)
        return 2
    runpod.api_key = api_key

    workload_b64 = base64.b64encode(WORKLOAD.encode()).decode()
    docker_args = ('bash -c "echo %s | base64 -d > /workspace/run.sh && '
                   'bash /workspace/run.sh"' % workload_b64)
    env = dict(env_block)
    if args.auto:
        env["RUNPOD_API_KEY"] = api_key   # self-terminate path inside the workload

    print("=" * 64)
    print(f"  3D absorption-aware FWI on RunPod")
    print(f"  GPU={args.gpu}  branch={args.branch}  phantom={args.phantom}"
          f"  grid={args.grid or 192}^3")
    print("=" * 64)
    print("Provisioning pod ...")
    pod = runpod.create_pod(
        name="brain-fwi-absorption-3d",
        image_name=args.image,
        gpu_type_id=args.gpu,
        gpu_count=1,
        volume_in_gb=args.volume_gb,
        container_disk_in_gb=30,
        min_vcpu_count=8,
        min_memory_in_gb=64,
        docker_args=docker_args,
        env=env,
    )
    pod_id = pod["id"]
    print(f"Pod: {pod_id}\nDashboard: https://www.runpod.io/console/pods/{pod_id}")

    print("Waiting for pod to start ...")
    status = {}
    for i in range(120):
        status = runpod.get_pod(pod_id)
        rt = status.get("runtime") or {}
        if rt.get("uptimeInSeconds", 0) > 0:
            print(f"  running ({rt['uptimeInSeconds']}s)")
            break
        time.sleep(10)
        print(f"  {(i + 1) * 10}s — {status.get('desiredStatus', 'unknown')}")
    else:
        print("Pod did not start within 20 min; check the dashboard.", file=sys.stderr)
        return 3

    out = Path("results/absorption_aware_fwi_3d"); out.mkdir(parents=True, exist_ok=True)
    (out / "runpod_pod.json").write_text(json.dumps({"pod_id": pod_id}, indent=2))
    print(f"""
{'=' * 64}
  Pod is running the workload. It writes to /workspace on the pod.
  Watch progress:   ssh into the pod, then: tail -f /workspace/run.log
  When /workspace/DONE exists, pull results back (replace HOST/PORT from the
  dashboard SSH line):
    scp -P PORT root@HOST:/workspace/absorption_aware_fwi_3d.png ./results/absorption_aware_fwi_3d/
    scp -P PORT root@HOST:/workspace/absorption_aware_fwi_3d.npz ./results/absorption_aware_fwi_3d/
    scp -P PORT root@HOST:/workspace/run.log ./results/absorption_aware_fwi_3d/
  {'Pod self-terminates on completion (--auto).' if args.auto
     else 'Terminate when done: runpod.terminate_pod(%r) or the dashboard.' % pod_id}
{'=' * 64}""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
