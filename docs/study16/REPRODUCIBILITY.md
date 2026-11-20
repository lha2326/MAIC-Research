# Study16 — Reproducibility

## Fixed experiment matrix

- MTE16: `MultiGrid-MultiTargetEmpty-16x16-v0`
- PushBox16: `MultiGrid-PushBox-16x16-v0`
- PassSparse16: `MultiGrid-PassSparse-16x16-v0`
- Algorithms: `MAIC`, `MAIC_ExactKL`, `MASAC`, `MASAC_ExactKL`, `MADDPG`, `COMA_Phi`.
- Training seeds: 42, 43, 44, 45, 46.
- 10,000 episodes, 40 environment steps per episode, 2 agents.
- Final evaluation: 100 episodes per seed per mode (greedy and stochastic), with evaluation seed base 20260.
- Original MAIC pinned source revision: `ded6164f99742db4769a7f05b45f6e6007ff8b1a`.

## Environment setup

Use the existing `requirements.txt` and the original pinned MAIC code. The pipeline expects the original source repository at `source_maic/` and the Exact-KL source at `variants/exactkl/`; this intentionally separates the two implementations.

```bash
git clone https://github.com/hoangvanphi2004/MAIC source_maic
git -C source_maic checkout ded6164f99742db4769a7f05b45f6e6007ff8b1a
python3 -m pip install -r requirements.txt
python3 -m pip install -e source_maic/refactor/multigrid
python3 study16_pipeline.py prepare
python3 study16_pipeline.py status
```

Verify the version and dependencies for your environment before beginning any repeat training. Saved per-run `config.json` and `run_meta.json` are the source of truth for hyperparameters; do not infer hyperparameters from a thesis or an earlier trial.

## Training and evaluation commands

```bash
python3 study16_pipeline.py train --jobs 2
python3 study16_pipeline.py status
python3 study16_pipeline.py eval --jobs 2
python3 study16_pipeline.py summarize
```

**Warning:** Partial checkpoints contain weights only, not all replay and optimizer state. `train --retry-failed` moves incomplete runs into `backups_study16_partial` and restarts them from episode 0 rather than continuing the exact training state.

## Figure regeneration

```bash
python3 -m pip install pandas numpy matplotlib
python3 scripts/study16/redraw_study16_journal.py \
  --csv results/study16/per_seed_values.csv \
  --runs runs_study16 \
  --out assets/study16/figures
```

Figures at final checkpoints regenerate from 90-row CSV data. Learning curves and checkpoint-wise comparisons require original run-level history/evaluation JSON; if unavailable, the script marks those figures as missing instead of inventing intermediate results.

## Artifacts, archiving and provenance

The Git repository contains code, summary tables, figures and selected small rollout demos. Large raw checkpoints and training logs should remain outside ordinary Git history and may be stored as verified GitHub Release assets. Never state that the complete raw archive is available until a Release asset listing has been checked.

The Git repo's existing `.gitignore` excludes `.pth`, `.pt`, `.ckpt`, raw `runs_*`, and `.mp4` files. The update script preserves these exclusions, and therefore does not silently push multi-gigabyte checkpoints. A separate publisher is required for full archive upload.

A selected policy-demo GIF is only an example from one checkpoint and evaluation episode, not a statistically representative measure of average performance. Do not represent scripted control trajectories as learned-policy rollouts.

## Known research limitations

- Results are based on five training seeds, so uncertainty is substantial.
- Success metrics are based on the implementation's termination logic; environment-level success semantics should be audited against task definitions.
- Recorded unique-state counts depend on logging code and may not reflect actual useful exploration.
- Intermediate checkpoint statistics must use actual evaluation records for all compared seeds.
- Expanded 24×24 and 32×32 claims belong to separate experiments and are not included in this completed Study16 summary.
