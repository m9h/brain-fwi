#!/bin/bash
# run_cloud.sh — One-command brain USCT on any cloud GPU
#
# Paste this into a RunPod / Lambda / Vast.ai terminal:
#   curl -sSL https://raw.githubusercontent.com/m9h/brain-fwi/main/run_cloud.sh | bash
#
# Or clone and run:
#   git clone https://github.com/m9h/brain-fwi.git && cd brain-fwi && bash run_cloud.sh
#
# Requirements: NVIDIA GPU with CUDA 12+, ~20GB disk

set -euo pipefail

echo "=============================================="
echo "  Brain USCT — Cloud GPU Runner"
echo "=============================================="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone if not in repo
if [ ! -f "pyproject.toml" ]; then
    echo "[2/5] Cloning brain-fwi..."
    cd /workspace 2>/dev/null || cd ~
    git clone https://github.com/m9h/brain-fwi.git
    cd brain-fwi
else
    echo "[2/5] Already in brain-fwi repo"
    git pull
fi

# Detect CUDA and install
echo "[3/5] Installing dependencies..."
CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' || echo "12")
echo "  CUDA version: ${CUDA_MAJOR}"

uv venv --quiet 2>/dev/null || true

if [ "$CUDA_MAJOR" -ge 13 ]; then
    uv pip install --quiet -e '.[cuda13]'
else
    uv pip install --quiet -e '.[cuda12]'
fi

# Verify GPU
echo "[4/5] Verifying GPU..."
uv run python -c "
import jax
devs = jax.devices()
backend = jax.default_backend()
print(f'  JAX backend: {backend}')
print(f'  Devices: {devs}')
if backend != 'gpu':
    print('  WARNING: No GPU detected, will run on CPU (very slow)')
else:
    print('  GPU ready!')
"

# Run full simulation
echo "[5/5] Running full-scale brain USCT..."
echo "  Grid: 192^3 (7M voxels)"
echo "  Elements: 256 (Kernel Flow-inspired helmet)"
echo "  Frequency: 3 bands (50-300 kHz)"
echo "  Estimated time: ~7 hours on A100"
echo ""

OUTDIR="${1:-/workspace}"
mkdir -p "$OUTDIR"

JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -u run_full_usct.py \
    --grid-size 192 \
    --n-elements 256 \
    --iters 20 \
    --shots 8 \
    --dx 0.001 \
    --output "$OUTDIR/brain_usct_full.h5" \
    --figures "$OUTDIR/brain_usct_full.png" \
    2>&1 | tee "$OUTDIR/brain_usct_full.log"

echo ""
echo "=============================================="
echo "  COMPLETE"
echo "  Results: $OUTDIR/brain_usct_full.*"
echo "=============================================="
