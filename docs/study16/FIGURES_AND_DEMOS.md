# Study16 — Figures and policy demonstrations

This document separates visual summaries from underlying experimental evidence.

## Figures in the main manuscript

| Figure | Interpretation |
|:--|:--|
| [`MTE16_success_greedy_stochastic`](../../assets/study16/figures/MTE16_success_greedy_stochastic.png) | Final checkpoint success; seed mean with individual seeds |
| [`MTE16_paired_seed_response`](../../assets/study16/figures/MTE16_paired_seed_response.png) | Paired seed-level changes after Exact KL |
| [`PushBox16_success_per_seed`](../../assets/study16/figures/PushBox16_success_per_seed.png) | 3/5 successful MAIC + Exact KL seeds vs 0/5 MAIC |
| [`PushBox16_train_reward_last10pct`](../../assets/study16/figures/PushBox16_train_reward_last10pct.png) | Late-stage training reward by method |
| [`PassSparse16_unique_states_final`](../../assets/study16/figures/PassSparse16_unique_states_final.png) | Log-derived unique states, not success |

Vector PDF equivalents are saved alongside the PNGs. Other generated figures include final reward, greedy/stochastic comparisons, checkpoint stability and learning curves when intermediate data are available.

## Policy rollout metadata

GIFs and metadata in `assets/study16/demos/` are exported directly from the final trained policy checkpoints on VAST when present. The filename encodes the environment, method, training seed, policy mode, evaluation seed and outcome.

Expected exported examples (not all are necessarily present in this checkout):

| Environment | Method | Training seed | Evaluation mode | Recorded outcome |
|:--|:--|--:|:--|:--|
| MTE16 | MAIC | 44 | Stochastic | Success |
| MTE16 | MASAC + Exact KL | 42 | Stochastic | Success |
| PushBox16 | MADDPG | 45 | Greedy | Success |
| PushBox16 | MAIC + Exact KL | 46 | Greedy | Success |
| PassSparse16 | MAIC + Exact KL | 42 | Greedy | Failure |

Check each companion JSON to verify checkpoint path, evaluation seed and task outcome. A successful example does not replace aggregate success rates. The PassSparse example depicts a **failed learned-policy rollout**, not a successful scripted demonstration.

Large MP4s and raw checkpoints are excluded from normal Git history. If uploaded to GitHub Releases, verify the Release assets and place the actual Release URL in the repository; no Release URL is assumed to exist.
