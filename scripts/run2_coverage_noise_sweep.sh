#!/usr/bin/env bash
# Run 2: how FEW elements, and how much noise, can transcranial 3-D GM/WM take?
#
# Run 1 showed 3-D full-encirclement recovers GM/WM at 96% / AUC 0.998 through the
# skull at only 300 kHz -- so frequency is NOT the transcranial bottleneck and the
# question inverts: find the coverage KNEE, and test the risks that actually bite
# (noise; unknown skull is the follow-on).
#
# All runs: stage 1 (300 kHz, N=112), transcranial (phase B), 15 iters/band.
# Sequential -- one GPU.
set -u
cd /home/mhough/dev/brain-fwi
PY=.venv/bin/python
IT=15
mkdir -p results/run2

run () {   # $1=n_src $2=n_rec $3=noise_db("none"|dB) $4=tag
  local nsrc=$1 nrec=$2 nz=$3 tag=$4
  local log="results/run2/${tag}.log"
  if [ -f "results/gmwm_velocity_3d/summary_s1B_${tag}.json" ]; then
    echo "[skip] $tag already done"; return
  fi
  local extra=""
  [ "$nz" != "none" ] && extra="--noise-db $nz"
  echo "[run] $tag  src=$nsrc rec=$nrec noise=$nz"
  $PY scripts/gmwm_velocity_3d.py --stage 1 --phase B --iters $IT \
      --n-src "$nsrc" --n-rec "$nrec" --tag "$tag" $extra > "$log" 2>&1
  grep -E "AUC|VERDICT|FWI done|residual" "$log" | tail -6
}

echo "===== Run 2A: coverage knee (noiseless) ====="
run   8   8 none cov008
run  16  16 none cov016
run  16  32 none cov032
run  16  64 none cov064
run  16 128 none cov128

echo "===== Run 2B: noise robustness ====="
run  16  32 20 cov032_snr20
run  16  32 10 cov032_snr10
run  16 128 20 cov128_snr20
run  16 128 10 cov128_snr10

echo "===== aggregating ====="
$PY scripts/run2_aggregate.py
