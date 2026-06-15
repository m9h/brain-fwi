#!/bin/bash
#SBATCH --job-name=mofi-stage1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/datasets/brain-fwi/slurm_%j_mofi_stage1.log
#SBATCH --error=/data/datasets/brain-fwi/slurm_%j_mofi_stage1.log
#
# MOFI Stage 1 — finite-difference check of ∂f/∂φ through the j-Wave forward.
# Verifies the warp→j-Wave VJP (sign + scale) for the SE(2) pose loss in
# src/brain_fwi/inversion/mofi.py. Tiny (64², 4 shots) — seconds of GPU.
#
# Usage:   sbatch slurm_mofi_stage1.sh
# Monitor: squeue -u $USER ; tail -f /data/datasets/brain-fwi/slurm_*_mofi_stage1.log

set -euo pipefail
cd ~/dev/brain-fwi

echo "=============================================="
echo "  MOFI Stage 1 (FD gradient check) — Job ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Started: $(date)"
echo "=============================================="

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

.venv/bin/python -m pytest \
    tests/test_mofi.py::test_pose_gradient_matches_finite_difference \
    -m slow -v -s

echo ""
echo "=============================================="
echo "  Completed: $(date)"
echo "=============================================="
