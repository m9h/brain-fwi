#!/bin/bash
#SBATCH --job-name=gmwm-real-iface
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gb10:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/run5_%j.log
# Run 5: the MIDA trilayer rerun in the configuration that fixed the synthetic ribbon
# (scope_3d_gmwm_run.md). Run-4 control (prop_base: frozen skull, sigma=1, 8 it x 3 shots) is
# the baseline; three arms at the same budget.
set -u
cd /home/mhough/dev/brain-fwi
PY=.venv/bin/python
IT=8; SH=3
run () {  # $1=interface $2=smooth $3=tag
  if [ -f "results/gmwm_real_anatomy/summary_$3.json" ]; then echo "[skip] $3"; return; fi
  echo "[run] $3  interface=$1 smooth=$2"
  $PY scripts/gmwm_real_anatomy.py --iters $IT --shots-per-iter $SH \
      --interface "$1" --smooth "$2" --tag "$3" > "results/run5/$3.log" 2>&1
  grep -E "interface=|residual|AUC|WM median|FWI done" "results/run5/$3.log" | tail -7
}
run free   2.0 real_free_sm2
run soft   2.0 real_soft_sm2
run frozen 2.0 real_frozen_sm2
echo "run5 done"
