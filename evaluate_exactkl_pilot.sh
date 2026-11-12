#!/bin/bash
set -euo pipefail

cd /workspace/MAIC_repro

SRC=/workspace/MAIC_repro/runs_diagnostic/source_exactkl

for algo in MAIC MASAC; do
    for seed in 42 43 44; do

        RUN="/workspace/MAIC_repro/runs_diagnostic/pilot_exactkl/PushBox16/$algo/seed$seed"

        # MAIC seed43 đã huấn luyện riêng
        if [[ "$algo" == "MAIC" && "$seed" == "43" ]]; then
            RUN="/workspace/MAIC_repro/runs_diagnostic/PushBox16/MAIC/seed43_exactKL_5000"
        fi

        for mode in greedy stochastic; do
            echo "START $algo seed$seed $mode"

            python3 tools/evaluate_rngcheck.py \
                --maic-root "$SRC" \
                --config "$RUN/config.json" \
                --run-dir "$RUN" \
                --checkpoint model_final.pth \
                --episodes 200 \
                --mode "$mode" \
                --eval-seed 30300 \
                --out "$RUN/eval_pilot_${mode}.json"

            echo "DONE $algo seed$seed $mode"
        done
    done
done

echo "ALL EVALUATIONS COMPLETED"
