"""Train MADDPG through the MAIC repo's own training loop (refactor/train.py::run_training).

Nothing in the repo is edited: we import its `train` module and replace the agent class
that run_training instantiates. Environment creation, observation/state extraction,
seeding, replay buffer, update schedule, metrics JSON and plots are therefore identical
to the MAIC / MASAC / COMA runs.
"""
import argparse, json, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--maic-root', required=True, help='path to MAIC/refactor')
p.add_argument('--config', required=True)
a = p.parse_args()

sys.path.insert(0, str(Path(a.maic_root).resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train  # noqa: E402  (the repo's module)
from baselines.maddpg import MADDPG  # noqa: E402

train.MAIC = MADDPG
cfg = json.loads(Path(a.config).read_text())
train.run_training(**cfg)
