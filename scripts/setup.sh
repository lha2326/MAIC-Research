#!/usr/bin/env bash
# Clone the original MAIC repo at the pinned commit and install it exactly as its README says.
# Usage: bash scripts/setup.sh            (run from the MAIC_repro folder)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(python3 -c "import json;print(json.load(open('experiments.json'))['maic_repo'])")
COMMIT=$(python3 -c "import json;print(json.load(open('experiments.json'))['maic_commit'])")
mkdir -p third_party
if [ ! -d third_party/MAIC/.git ]; then
  git clone "$REPO" third_party/MAIC
fi
git -C third_party/MAIC fetch -q origin || true
git -C third_party/MAIC checkout -q "$COMMIT"
echo "MAIC at $(git -C third_party/MAIC rev-parse HEAD)"
python3 -c "import torch" 2>/dev/null || { echo "PyTorch missing: install it first (https://pytorch.org/get-started/locally/)"; exit 1; }
pip install -q -r requirements.txt
pip install -q -e third_party/MAIC/refactor/multigrid
python3 - <<'PY'
import torch, gymnasium, multigrid.envs
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gymnasium', gymnasium.__version__)
PY
