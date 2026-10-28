#!/usr/bin/env bash
# Run 3: how accurate must the skull pose be for GM/WM to survive?
# Sweeps inversion-skull pose error from MOFI-achieved accuracy to catastrophic.
# The decisive config is pose007 (0.07 vox / 0.10 deg) -- MOFI's MEASURED accuracy.
set -u
cd /home/mhough/dev/brain-fwi
PY=.venv/bin/python
mkdir -p results/run3

run () {  # $1=vox $2=deg $3=tag
  if [ -f "results/gmwm_unknown_skull/summary_$3.json" ]; then echo "[skip] $3"; return; fi
  echo "[run] $3  pose ${1} vox / ${2} deg"
  $PY scripts/gmwm_unknown_skull.py --pose-vox "$1" --pose-deg "$2" --tag "$3" \
      > "results/run3/$3.log" 2>&1
  grep -E "mismatch|residual|AUC|WM median" "results/run3/$3.log" | tail -6
}

run 0.00 0.00 pose000     # baseline: skull at truth (ellipsoidal)
run 0.07 0.10 pose007     # <- MOFI MEASURED accuracy (transmission-TT staged)
run 0.25 0.30 pose025
run 0.50 0.60 pose050
run 1.00 1.20 pose100
run 2.00 2.50 pose200
run 4.00 5.00 pose400     # campaign-measured catastrophic misalignment

echo "===== aggregating ====="
$PY scripts/run3_aggregate.py
