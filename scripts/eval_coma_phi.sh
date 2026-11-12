#!/usr/bin/env bash
set -euo pipefail

cd /workspace/MAIC_repro
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for env in MTE16 PushBox16 PassSparse16; do
  for seed in {42..51}; do
    for mode in greedy stochastic; do

      run="runs_phi_coma/$env/COMA/seed$seed"
      out="reports/coma_phi_eval/${env}_seed${seed}_${mode}.json"

      if [[ -s "$out" ]] &&
         python3 -m json.tool "$out" >/dev/null 2>&1 &&
         grep -q '"success_rate"' "$out"; then
        echo "SKIP $env seed$seed $mode"
        continue
      fi

      echo "START $env seed$seed $mode"

      python3 tools/evaluate_rngcheck.py \
        --maic-root /workspace/MAIC_repro/third_party/MAIC/refactor \
        --run-dir "$run" \
        --checkpoint model_final.pth \
        --episodes 100 \
        --mode "$mode" \
        --eval-seed 20260 \
        --out "$out"

      echo "DONE $env seed$seed $mode"
    done
  done
done

echo "ALL COMA PHI EVALUATIONS COMPLETED"
