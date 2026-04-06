#!/usr/bin/env bash
# run_dgx.sh — Full brain USCT on DGX Spark
#
# Usage:
#   ./run_dgx.sh              # medium (96^3, ~17 min on GPU)
#   ./run_dgx.sh --full       # full scale (192^3, ~7 hr on GPU)
#
# Outputs:
#   brain_usct_results.h5     — reconstruction volumes + metrics
#   brain_usct.log            — full console log
#   brain_usct_figures.png    — comparison figure (true vs recon)

set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================="
echo "  Brain USCT — DGX Spark Runner"
echo "=============================================="

# Install deps
echo "[1/4] Installing dependencies..."
uv sync --quiet

# Verify GPU
echo "[2/4] Checking GPU..."
uv run python -c "
import jax
devs = jax.devices()
print(f'  JAX devices: {devs}')
for d in devs:
    if 'gpu' in str(d).lower() or 'cuda' in str(d).lower():
        print(f'  GPU detected: {d}')
print(f'  JAX version: {jax.__version__}')
print(f'  Platform: {jax.default_backend()}')
"

# Determine preset
PRESET="${1:---default}"
if [ "$PRESET" = "--full" ]; then
    GRID=192
    ELEMENTS=256
    ITERS=20
    SHOTS=8
    DX=0.001
    BANDS="3"
    OUTFILE="brain_usct_full.h5"
    LOGFILE="brain_usct_full.log"
    FIGFILE="brain_usct_full_figures.png"
    echo "[3/4] Running FULL preset: ${GRID}^3, ${ELEMENTS} elements, ${BANDS} bands"
    echo "       Estimated time: ~7 hours on A100-class GPU"
else
    GRID=96
    ELEMENTS=128
    ITERS=15
    SHOTS=8
    DX=0.002
    BANDS="2"
    OUTFILE="brain_usct_medium.h5"
    LOGFILE="brain_usct_medium.log"
    FIGFILE="brain_usct_medium_figures.png"
    echo "[3/4] Running MEDIUM preset: ${GRID}^3, ${ELEMENTS} elements, ${BANDS} bands"
    echo "       Estimated time: ~17 minutes on A100-class GPU"
fi

# Run the simulation
uv run python -u run_full_usct.py \
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
echo "  Log:      $LOGFILE"
echo "  Figures:  $FIGFILE"
echo ""
echo "  View results:"
echo "    uv run python -c \"import h5py; f=h5py.File('$OUTFILE'); print(dict(f.attrs))\""
