# HUONG DAN VAST MOI - MAIC RESEARCH 16x16

## Scope
- Environments: MTE16, PushBox16, PassSparse16
- Methods: MAIC, MASAC, MADDPG, COMA_Phi (COMA-configured MAIC; NOT standard on-policy COMA), MAIC_ExactKL, MASAC_ExactKL
- Seeds: 42, 43, 44, 45, 46
- 10,000 episodes per run; 40-step horizon
- 90 runs = 900,000 training episodes
- 180 final evaluations: 90 models x 2 modes x 100 episodes
- Each trained model has 500-episode checkpoints, plus final model.

## Step 1: Create Vast instance
Use a recent PyTorch + CUDA image with Jupyter/SSH, enough persistent storage (recommend at least 100GB if keeping all checkpoints) and preferably a separate off-host backup.

## Step 2: Clone code and pinned original source
```bash
cd /workspace
git clone https://github.com/lha2326/MAIC-Research.git
cd MAIC-Research

git clone https://github.com/hoangvanphi2004/MAIC.git source_maic
git -C source_maic checkout ded6164f99742db4769a7f05b45f6e6007ff8b1a

python3 -m pip install -r requirements.txt
python3 -m pip install -e source_maic/refactor/multigrid
python3 - <<'PY'
import torch
import gymnasium as gym
import multigrid.envs
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA UNAVAILABLE')
for env_id in ('MultiGrid-MultiTargetEmpty-16x16-v0', 'MultiGrid-PushBox-16x16-v0', 'MultiGrid-PassSparse-16x16-v0'):
    env = gym.make(env_id, num_agents=2, max_steps=40)
    env.reset(seed=42)
    env.close()
    print('PASS', env_id)
PY
```
**Important:** Do not run this repository's older `scripts/setup.sh` on a fresh clone for this study; `third_party/MAIC` is present as a copied directory without its own Git history, so that helper may try to clone into a nonempty folder. Use `source_maic` as above.

## Step 3: Upload script
Upload the file `study16_pipeline.py` to `/workspace/MAIC-Research/` via the Jupyter Upload UI or SCP.

```bash
cd /workspace/MAIC-Research
python3 study16_pipeline.py prepare
python3 study16_pipeline.py status
```

## Step 4: External backup BEFORE training (recommended)
Create a supported rclone remote (Google Drive / S3 / other) and test its access:

```bash
# Example only: AFTER configuring a remote named mydrive
rclone lsd mydrive:
export STUDY16_BACKUP_REMOTE='mydrive:MAIC-Research/study16'
```

For persistent tmux sessions, pass the variable in the session command (do not assume exporting it in a separate terminal carries over):

```bash
cd /workspace/MAIC-Research
tmux new-session -d -s study16 'cd /workspace/MAIC-Research && export STUDY16_BACKUP_REMOTE="mydrive:MAIC-Research/study16" && python3 -u study16_pipeline.py train --jobs 2 > study16.log 2>&1'
```

The pipeline copies each **completed** run to the remote. It does **not** back up partial runs until they complete. To protect partial results during a lengthy training run, sync periodically from a second session:

```bash
# Run this from the terminal whenever you need to preserve partial results.
# If your rclone remote is configured:
rclone copy /workspace/MAIC-Research/runs_study16 mydrive:MAIC-Research/study16 --transfers 4
```

If you have no rclone setup yet, use the non-backup tmux launch below, but understand that loss of the instance could lose everything created there.

## Step 5: Train all 90 runs

```bash
cd /workspace/MAIC-Research
# Skip this if a tmux session called study16 is already active!
tmux new-session -d -s study16 'cd /workspace/MAIC-Research && python3 -u study16_pipeline.py train --jobs 2 > study16.log 2>&1'
tmux ls
```

**Run only one of the launch commands**, not both. GPU concurrency can be adjusted from `--jobs 2` after inspecting utilization.

## Step 6: Monitor progress

```bash
cd /workspace/MAIC-Research
watch -n 10 'python3 study16_pipeline.py status'
```

Live controller log:

```bash
tail -f /workspace/MAIC-Research/study16.log
```

Raw run outputs are in `runs_study16/<ENV>/<METHOD>/seed<SEED>/`.

**Important:** The original model checkpoints only store actor/critic weights, not replay buffer or optimizer state. A crashed/interrupted run cannot be resumed as the exact same stochastic training trajectory. Use `--retry-failed` only after the prior process is certainly stopped; it archives the run and starts the entire seed again.

## Step 7: Evaluation and reports

After training finishes:

```bash
cd /workspace/MAIC-Research
# Evaluate final checkpoint for all finished runs, greedy + stochastic, 100 episodes each.
tmux new-session -d -s study16_eval 'cd /workspace/MAIC-Research && python3 -u study16_pipeline.py eval --jobs 2 > study16_eval.log 2>&1'
```

Optionally evaluate checkpoints 2000/4000/6000/8000 from MAIC, MASAC and Exact KL variants:

```bash
python3 study16_pipeline.py eval --jobs 2 --intermediate
```

Summarize using the repository's existing aggregation tool:

```bash
python3 study16_pipeline.py summarize
```

Output: `reports_study16/runs.csv`, `summary.csv`, `summary.md`, and `curves_MTE16.png`, `curves_PushBox16.png`, `curves_PassSparse16.png`.

## Checkpoint and methodology notes
- Keep 10,000 episode results separate from previous 5,000 episode experiments.
- For MAIC: IG coefficient 1, entropy coefficient 1.
- MASAC: IG 0, entropy 1.
- COMA_Phi: IG 0, entropy 0 (not standard COMA).
- MAIC_ExactKL: IG 1, entropy 1, direct Exact KL policy loss.
- MASAC_ExactKL: IG 0, entropy 1, direct Exact KL policy loss.
- MADDPG: separate baseline, no MAIC KL machinery.
- Check scalar return, success rate, per-agent positive reward, entropy, KL, unique state count, and learning curves. Positive reward on PassSparse does not imply full task success.
- One 16x16 environment with 40-step horizon might not be sufficiently solvable in sparse-reward mode; treat such failures as measured results, not proof that an algorithm is intrinsically bad.
- The script does not start any GPU process in this chat. You must run it on your own VAST instance.
