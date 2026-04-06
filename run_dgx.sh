#!/usr/bin/env bash
# run_dgx.sh — Full brain USCT on DGX Spark (GPU)
#
# Usage:
#   ./run_dgx.sh              # medium (96^3, ~17 min on GPU)
#   ./run_dgx.sh --full       # full scale (192^3, ~7 hr on GPU)
#
# Outputs:
#   brain_usct_*.h5           — reconstruction volumes + metrics
#   brain_usct_*.log          — full console log
#   brain_usct_*_figures.png  — comparison figure

set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================="
echo "  Brain USCT — DGX Spark Runner"
echo "=============================================="

# Install deps with CUDA 13 support (DGX Spark GB10)
echo "[1/4] Installing dependencies (with CUDA 13)..."
uv sync --quiet
uv pip install --quiet "jax[cuda13]>=0.5"

# Verify GPU
echo "[2/4] Checking GPU..."
uv run python -c "
import jax
devs = jax.devices()
print(f'  Devices: {devs}')
print(f'  Backend: {jax.default_backend()}')
backend = jax.default_backend()
if backend != 'gpu':
    print()
    print('  WARNING: JAX is using CPU, not GPU!')
    print('  Try: uv pip install \"jax[cuda12]\"')
    print('  Or set: export JAX_PLATFORMS=cuda')
    print()
else:
    for d in devs:
        print(f'  GPU: {d}')
"

# Force JAX to use GPU
export JAX_PLATFORMS=cuda

# Determine preset
PRESET="${1:---default}"
if [ "$PRESET" = "--full" ]; then
    GRID=192
    ELEMENTS=256
    ITERS=20
    SHOTS=8
    DX=0.001
    OUTFILE="brain_usct_full.h5"
    LOGFILE="brain_usct_full.log"
    FIGFILE="brain_usct_full_figures.png"
    echo "[3/4] Running FULL preset: ${GRID}^3, ${ELEMENTS} elements"
    echo "       Estimated: ~7 hours on A100"
else
    GRID=96
    ELEMENTS=128
    ITERS=15
    SHOTS=8
    DX=0.002
    OUTFILE="brain_usct_medium.h5"
    LOGFILE="brain_usct_medium.log"
    FIGFILE="brain_usct_medium_figures.png"
    echo "[3/4] Running MEDIUM preset: ${GRID}^3, ${ELEMENTS} elements"
    echo "       Estimated: ~17 minutes on A100"
fi

# Run
JAX_PLATFORMS=cuda uv run python -u run_full_usct.py \
    --grid-size "$GRID" \
    --n-elements "$ELEMENTS" \
    --iters "$ITERS" \
    --shots "$SHOTS" \
    --dx "$DX" \
    --output "$OUTFILE" \
    --figures "$FIGFILE" \
    2>&1 | tee "$LOGFILE"

echo ""
echo "[4/4] Done!"
echo "  Results:  $OUTFILE"
echo "  Figures:  $FIGFILE"
echo "  Log:      $LOGFILE"
