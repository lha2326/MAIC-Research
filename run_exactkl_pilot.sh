#!/usr/bin/env bash
set -euo pipefail

cd /workspace/MAIC_repro

SOURCE="runs_diagnostic/source_exactkl/main.py"
BASE="runs_diagnostic/pilot_exactkl/PushBox16"

for algo in MAIC MASAC; do
    for seed in 42 43 44; do

        if [[ "$algo" == "MAIC" && "$seed" == "43" ]]; then
            echo "SKIP MAIC seed43: da co ket qua"
            continue
        fi

        RUN="$BASE/$algo/seed$seed"

        echo "START $algo seed$seed"

        python3 -u "$SOURCE" \
            --config "/workspace/MAIC_repro/$RUN/config.json" \
            > "$RUN/train.log" 2>&1

        echo "DONE $algo seed$seed"
    done
done

echo "PILOT COMPLETED"
