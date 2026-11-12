#!/usr/bin/env bash
# ~10-20 min check that every (env, algo) pair runs end to end (30 episodes, 1 seed). Writes to runs_smoke/.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tools/run_matrix.py --episodes 30 --seeds 42 --out runs_smoke -j "${1:-4}"
python3 tools/evaluate_all.py --runs runs_smoke -j "${1:-4}"
python3 tools/aggregate.py --runs runs_smoke --out reports_smoke --window 10
