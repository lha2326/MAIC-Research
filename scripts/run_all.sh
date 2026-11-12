#!/usr/bin/env bash
# Full pipeline: train every (env, algo, seed) in experiments.json -> evaluate -> tables/figures.
# Usage: bash scripts/run_all.sh [parallel_jobs]       e.g. bash scripts/run_all.sh 6
set -euo pipefail
cd "$(dirname "$0")/.."
J=${1:-4}
python3 tools/run_matrix.py -j "$J" 2>&1 | tee -a run_matrix.log
python3 tools/evaluate_all.py -j "$J" 2>&1 | tee -a evaluate.log
python3 tools/aggregate.py --runs runs --out reports
