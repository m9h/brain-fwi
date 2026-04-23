#!/bin/bash
#SBATCH --job-name=brain-usct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/data/datasets/brain-fwi/slurm_%j_usct.log
#SBATCH --error=/data/datasets/brain-fwi/slurm_%j_usct.log
#
# Brain USCT via Slurm on DGX Spark
#
# Usage:
#   sbatch slurm_usct.sh                    # medium (96^3, ~45 min)
#   sbatch slurm_usct.sh --export=GRID=192  # full (192^3, hours)
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm_*_usct.log

set -euo pipefail
cd ~/dev/brain-fwi

echo "=============================================="
echo "  Brain USCT — Slurm Job ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Started: $(date)"
echo "=============================================="

# Defaults (override via --export=ALL,GRID=192,ELEM=256,ITERS=20,PHANTOM=mida,PARAM=siren)
GRID=${GRID:-96}
ELEM=${ELEM:-128}
ITERS=${ITERS:-30}
SHOTS=${SHOTS:-8}
DX=${DX:-0.002}
PHANTOM=${PHANTOM:-synthetic}
PARAM=${PARAM:-voxel}

if [ "$GRID" -ge 192 ]; then
    DX=0.001
    ELEM=${ELEM:-256}
fi

OUTDIR="/data/datasets/brain-fwi"
mkdir -p "$OUTDIR"
TAG="${GRID}_${PHANTOM}_${PARAM}"
OUTFILE="${OUTDIR}/brain_usct_${TAG}_${SLURM_JOB_ID}.h5"
FIGFILE="${OUTDIR}/brain_usct_${TAG}_${SLURM_JOB_ID}.png"

echo "  Grid:    ${GRID}^3, Elements: ${ELEM}, Iters: ${ITERS}/band"
echo "  Phantom: ${PHANTOM}, Parameterisation: ${PARAM}"
echo "  Output:  ${OUTFILE}"
echo ""

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python -u run_full_usct.py \
    --grid-size "$GRID" \
    --n-elements "$ELEM" \
    --iters "$ITERS" \
    --shots "$SHOTS" \
    --dx "$DX" \
    --phantom "$PHANTOM" \
    --parameterization "$PARAM" \
    --output "$OUTFILE" \
    --figures "$FIGFILE"

echo ""
echo "=============================================="
echo "  Completed: $(date)"
echo "  Results: ${OUTFILE}"
echo "=============================================="
