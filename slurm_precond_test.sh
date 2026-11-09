#!/bin/bash
#SBATCH --job-name=precond-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/datasets/brain-fwi/slurm_%j_precond.log
#SBATCH --error=/data/datasets/brain-fwi/slurm_%j_precond.log
#
# Does pseudo-Hessian preconditioning fix 3D skull recovery from water?
# Runs run_full_usct.py twice (synthetic 64^3, from water): precondition OFF
# (the job-984 setting, 0% skull) vs ON, and prints % skull contrast recovered.
#
# OOM-safe by design: 64^3 with j-Wave's internal checkpointed backward peaks
# at a few GB << 120 GB; XLA_PYTHON_CLIENT_PREALLOCATE=false (on-demand alloc).
# Few elements (32) bound the per-source FWI backward cost (the thing that
# made 96^3/64^3 exceed Modal's 2 h cap — here Slurm has no wall cap anyway).
#
# Usage:   sbatch slurm_precond_test.sh
# Monitor: tail -f /data/datasets/brain-fwi/slurm_*_precond.log

set -euo pipefail
cd ~/dev/brain-fwi

echo "=============================================="
echo "  Preconditioning skull-recovery test — Job ${SLURM_JOB_ID}"
echo "  Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Started: $(date)"
echo "=============================================="

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

OUTDIR=/data/datasets/brain-fwi/precond_test
mkdir -p "$OUTDIR"

COMMON="--phantom synthetic --grid-size 64 --dx 0.0025 --n-elements 32 --iters 12 --shots 4"

echo ">>> RUN 1/2: precondition OFF (984 setting)"
.venv/bin/python -u run_full_usct.py $COMMON \
    --output "$OUTDIR/synth64_precond_off.h5" \
    --figures "$OUTDIR/synth64_precond_off.png"

echo ">>> RUN 2/2: precondition ON"
.venv/bin/python -u run_full_usct.py $COMMON --precondition \
    --output "$OUTDIR/synth64_precond_on.h5" \
    --figures "$OUTDIR/synth64_precond_on.png"

echo ">>> COMPARE: % skull contrast recovered (from water)"
.venv/bin/python - <<'PY'
import h5py, numpy as np
def skull(p):
    f = h5py.File(p, "r")
    ct = np.array(f["velocity_true"]); cr = np.array(f["velocity_recon"]); f.close()
    m = ct > 2200
    ts, rs = float(ct[m].mean()), float(cr[m].mean())
    return 100.0 * (rs - 1500.0) / (ts - 1500.0), ts, rs
base = "/data/datasets/brain-fwi/precond_test"
o = skull(f"{base}/synth64_precond_off.h5")
n = skull(f"{base}/synth64_precond_on.h5")
print(f"\n=== synthetic 64^3, from water — % skull contrast recovered ===")
print(f"  precond OFF (984 setting): {o[0]:6.1f}%   (true {o[1]:.0f} -> recon {o[2]:.0f})")
print(f"  precond ON               : {n[0]:6.1f}%   (true {n[1]:.0f} -> recon {n[2]:.0f})")
print(f"  delta = {n[0]-o[0]:+.1f} pts")
PY

echo "=============================================="
echo "  Completed: $(date)"
echo "=============================================="
