#!/bin/bash
#SBATCH --job-name=phase0-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/datasets/brain-fwi/slurm_%j_phase0.log
#SBATCH --error=/data/datasets/brain-fwi/slurm_%j_phase0.log
#
# Phase-0 dataset generation driver for brain-fwi.
#
# Usage (defaults: MIDA at 96^3, 1 subject x 50 augmentations):
#   sbatch slurm_gen_phase0.sh
#
# Override via env:
#   sbatch --export=ALL,PHANTOM=synthetic,GRID=64,AUG=20 slurm_gen_phase0.sh
#   sbatch --export=ALL,GRID=128,AUG=200,ELEM=256       slurm_gen_phase0.sh
#
# Nightly resumable run (handoff doc workflow). Manifest commits per
# sample, so killing the job mid-run loses at most one in-flight sample.
# Re-submit nightly with the same AUG target to continue:
#   sbatch --time=14:00:00 \
#     --export=ALL,AUG=1024,SIREN_HIDDEN=64,SIREN_LAYERS=3,MEM_FRACTION=0.6 \
#     slurm_gen_phase0.sh

set -euo pipefail
cd ~/dev/brain-fwi

echo "=============================================="
echo "  Phase-0 dataset gen — Slurm Job ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Started: $(date)"
echo "=============================================="

PHANTOM=${PHANTOM:-mida}
GRID=${GRID:-96}
DX=${DX:-0.002}
ELEM=${ELEM:-128}
AUG=${AUG:-50}
SUBJECTS=${SUBJECTS:-1}
FREQ=${FREQ:-500000}
PRETRAIN=${PRETRAIN:-400}
VERSION=${VERSION:-phase0_v1}
SIREN_HIDDEN=${SIREN_HIDDEN:-128}
SIREN_LAYERS=${SIREN_LAYERS:-3}
MEM_FRACTION=${MEM_FRACTION:-0.8}

if [ "$GRID" -ge 192 ]; then
    DX=0.001
    ELEM=${ELEM:-256}
fi

OUTDIR="/data/datasets/brain-fwi/${VERSION}_${PHANTOM}_${GRID}"
mkdir -p "$OUTDIR"

echo "  Phantom:  ${PHANTOM}"
echo "  Grid:     ${GRID}^3, dx=${DX}"
echo "  Subjects: ${SUBJECTS} x ${AUG} augmentations = $((SUBJECTS * AUG)) samples"
echo "  Elements: ${ELEM}, Freq: ${FREQ} Hz"
echo "  SIREN:    hidden=${SIREN_HIDDEN}, layers=${SIREN_LAYERS}"
echo "  Memory:   XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION}"
echo "  Output:   ${OUTDIR}"
echo ""

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION"

.venv/bin/python -u scripts/gen_phase0.py \
    --out "$OUTDIR" \
    --phantom "$PHANTOM" \
    --grid-size "$GRID" \
    --dx "$DX" \
    --n-elements "$ELEM" \
    --n-subjects "$SUBJECTS" \
    --n-augments "$AUG" \
    --freq "$FREQ" \
    --siren-hidden "$SIREN_HIDDEN" \
    --siren-layers "$SIREN_LAYERS" \
    --siren-pretrain-steps "$PRETRAIN" \
    --version "$VERSION"

echo ""
echo "=============================================="
echo "  Completed: $(date)"
echo "  Manifest: ${OUTDIR}/manifest.json"
echo "=============================================="
