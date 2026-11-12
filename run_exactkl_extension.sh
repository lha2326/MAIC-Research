#!/usr/bin/env bash
set -euo pipefail

cd /workspace/MAIC_repro

SRC="runs_diagnostic/source_exactkl/main.py"
BASE="runs_diagnostic/pilot_exactkl/PushBox16"

for algo in MAIC MASAC; do
    for seed in 45 46 47 48 49 50 51; do
        RUN="$BASE/$algo/seed$seed"

        if [[ -f "$RUN/train.log" ]]; then
            if grep -q 'Training finished' "$RUN/train.log" &&
               [[ -f "$RUN/model_final.pth" ]]; then
                echo "SKIP COMPLETED $algo seed$seed"
                continue
            else
                echo "INCOMPLETE RUN: $RUN - stopping to avoid overwrite"
                exit 1
            fi
        fi

        echo "START $algo seed$seed"

        python3 -u "$SRC" \
            --config "/workspace/MAIC_repro/$RUN/config.json" \
            > "$RUN/train.log" 2>&1

        test -f "$RUN/model_final.pth"
        grep -q 'Training finished' "$RUN/train.log"

        echo "DONE $algo seed$seed"
    done
done

echo "ALL 14 RUNS COMPLETED"
