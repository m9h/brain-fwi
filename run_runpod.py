#!/usr/bin/env python
"""Deploy and run full brain USCT on RunPod.

Usage:
    # Set your RunPod API key
    export RUNPOD_API_KEY=your_key_here

    # Launch full-scale run on A100
    uv run python run_runpod.py

    # Or specify GPU type
    uv run python run_runpod.py --gpu "NVIDIA A100 80GB PCIe"
    uv run python run_runpod.py --gpu "NVIDIA H100 80GB HBM3"
    uv run python run_runpod.py --gpu "NVIDIA RTX 4090"
"""

import argparse
import os
import time
import json

SETUP_SCRIPT = r"""#!/bin/bash
set -euo pipefail

echo "========================================"
echo "  Brain USCT — RunPod Setup"
echo "========================================"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone and install
cd /workspace
git clone https://github.com/m9h/brain-fwi.git
cd brain-fwi

# Detect CUDA version and install matching JAX
CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
echo "Detected CUDA $CUDA_MAJOR"

uv venv
if [ "$CUDA_MAJOR" -ge 13 ]; then
    uv pip install -e '.[cuda13]'
else
    uv pip install -e '.[cuda12]'
fi

# Verify GPU
uv run python -c "
import jax
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
assert jax.default_backend() == 'gpu', 'GPU not detected!'
print('GPU verified!')
"

echo "Setup complete!"
"""

RUN_SCRIPT = r"""#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/brain-fwi

echo "========================================"
echo "  Brain USCT — Full Scale Run"
echo "========================================"

nvidia-smi

# Full preset: 192^3, 256 elements, 3 bands, 20 iters
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -u run_full_usct.py \
    --grid-size 192 \
    --n-elements 256 \
    --iters 20 \
    --shots 8 \
    --dx 0.001 \
    --output /workspace/brain_usct_full.h5 \
    --figures /workspace/brain_usct_full.png \
    2>&1 | tee /workspace/brain_usct_full.log

echo ""
echo "Results saved to /workspace/"
echo "  brain_usct_full.h5   — reconstruction data"
echo "  brain_usct_full.png  — comparison figures"
echo "  brain_usct_full.log  — full log"
"""


def main():
    parser = argparse.ArgumentParser(description="Run brain USCT on RunPod")
    parser.add_argument("--gpu", type=str, default="NVIDIA A100 80GB PCIe",
                        help="GPU type")
    parser.add_argument("--disk", type=int, default=20,
                        help="Disk size in GB")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without launching")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key and not args.dry_run:
        print("Set RUNPOD_API_KEY environment variable")
        print("Get your key at: https://www.runpod.io/console/user/settings")
        return

    config = {
        "name": "brain-usct-full",
        "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "gpuTypeId": args.gpu,
        "gpuCount": 1,
        "volumeInGb": args.disk,
        "containerDiskInGb": 10,
        "minVcpuCount": 4,
        "minMemoryInGb": 32,
        "dockerArgs": "",
        "env": {},
    }

    if args.dry_run:
        print("RunPod configuration:")
        print(json.dumps(config, indent=2))
        print("\nSetup script:")
        print(SETUP_SCRIPT[:200] + "...")
        print("\nRun script:")
        print(RUN_SCRIPT[:200] + "...")
        return

    try:
        import runpod
    except ImportError:
        print("Installing runpod SDK...")
        os.system("uv pip install runpod")
        import runpod

    runpod.api_key = api_key

    # Create pod
    print(f"Launching {args.gpu}...")
    pod = runpod.create_pod(
        name=config["name"],
        image_name=config["imageName"],
        gpu_type_id=config["gpuTypeId"],
        gpu_count=config["gpuCount"],
        volume_in_gb=config["volumeInGb"],
        container_disk_in_gb=config["containerDiskInGb"],
        min_vcpu_count=config["minVcpuCount"],
        min_memory_in_gb=config["minMemoryInGb"],
    )

    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    print(f"Dashboard: https://www.runpod.io/console/pods/{pod_id}")

    # Wait for pod to be ready
    print("Waiting for pod to start...")
    for i in range(60):
        status = runpod.get_pod(pod_id)
        state = status.get("desiredStatus", "unknown")
        runtime = status.get("runtime", {})
        if runtime and runtime.get("uptimeInSeconds", 0) > 0:
            print(f"Pod running! Uptime: {runtime['uptimeInSeconds']}s")
            break
        print(f"  Status: {state}... ({i*10}s)")
        time.sleep(10)

    print("\n" + "=" * 50)
    print("Pod is ready. Connect and run:")
    print("=" * 50)
    print(f"""
1. SSH into the pod:
   ssh root@{pod_id}-ssh.runpod.io  (check dashboard for exact command)

2. Run setup (one time):
   {SETUP_SCRIPT.strip().split(chr(10))[0]}
   # ... (paste the full setup script, or run:)
   curl -sSL https://raw.githubusercontent.com/m9h/brain-fwi/main/run_runpod.py | python3 - --setup

3. Run the simulation:
   cd /workspace/brain-fwi
   bash -c '{RUN_SCRIPT.strip().split(chr(10))[-4].strip()}'

4. Download results:
   scp root@{pod_id}:/workspace/brain_usct_full.* .

5. IMPORTANT: Stop the pod when done!
   runpod stop_pod {pod_id}
""")


if __name__ == "__main__":
    main()
