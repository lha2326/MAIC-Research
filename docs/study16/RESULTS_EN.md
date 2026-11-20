# Study16 — Experimental results

## Protocol and interpretation

The controlled study comprises three 16×16 environments, six method configurations, five training seeds (42–46), and 10,000 episodes per run (90 runs). Each final checkpoint is evaluated in greedy and stochastic modes with 100 evaluation episodes for each mode and seed. All values below are **mean ± population SD across five independent training seeds**. Evaluation episodes are not treated as five hundred independent training replicates.

Training reward is the mean over the final 10% of training episodes. `unique_states_final` is a logging-derived count; it is not a success or coverage measure in a mathematically specified state space.

## Mean success rate (%)

| Method | MTE16 greedy | MTE16 stochastic | PushBox16 stochastic | PassSparse16 stochastic |
|:--|--:|--:|--:|--:|
| MAIC | 9.8% | 41.4% | 0.0% | 0.0% |
| MAIC + Exact KL | 25.6% | 37.8% | 60.0% | 0.0% |
| MASAC | 7.2% | 17.4% | 0.0% | 0.0% |
| MASAC + Exact KL | 37.2% | 44.2% | 0.0% | 0.0% |
| MADDPG | 4.6% | 13.4% | 60.0% | 0.0% |
| COMA-configured MAIC | 0.0% | 0.0% | 0.0% | 0.0% |

## Full results by environment

### MTE16

| Method | Last-10% training reward | Final unique states | Greedy success | Stochastic success |
|:--|--:|--:|--:|--:|
| MAIC | 63.510 ± 18.883 | 106621.200 ± 23360.240 | 9.8 ± 6.2% | 41.4 ± 32.9% |
| MAIC + Exact KL | 69.200 ± 15.261 | 93876.600 ± 8376.675 | 25.6 ± 21.5% | 37.8 ± 29.0% |
| MASAC | 60.390 ± 19.752 | 103797.400 ± 8794.028 | 7.2 ± 6.6% | 17.4 ± 16.0% |
| MASAC + Exact KL | 71.800 ± 19.684 | 88002.600 ± 8953.441 | 37.2 ± 35.6% | 44.2 ± 41.2% |
| MADDPG | 31.340 ± 17.863 | 133188.200 ± 9237.409 | 4.6 ± 9.2% | 13.4 ± 25.8% |
| COMA-configured MAIC | 1.630 ± 1.517 | 42649.000 ± 28562.875 | 0.0 ± 0.0% | 0.0 ± 0.0% |

### PushBox16

| Method | Last-10% training reward | Final unique states | Greedy success | Stochastic success |
|:--|--:|--:|--:|--:|
| MAIC | 0.000 ± 0.000 | 559.800 ± 102.234 | 0.0 ± 0.0% | 0.0 ± 0.0% |
| MAIC + Exact KL | 59.980 ± 48.973 | 7569.200 ± 12859.883 | 60.0 ± 49.0% | 60.0 ± 49.0% |
| MASAC | 0.000 ± 0.000 | 538.000 ± 80.352 | 0.0 ± 0.0% | 0.0 ± 0.0% |
| MASAC + Exact KL | 0.000 ± 0.000 | 727.600 ± 152.745 | 0.0 ± 0.0% | 0.0 ± 0.0% |
| MADDPG | 59.720 ± 48.763 | 12652.000 ± 15315.821 | 60.0 ± 49.0% | 60.0 ± 49.0% |
| COMA-configured MAIC | 0.000 ± 0.000 | 537.800 ± 80.829 | 0.0 ± 0.0% | 0.0 ± 0.0% |

### PassSparse16

| Method | Last-10% training reward | Final unique states | Greedy success | Stochastic success |
|:--|--:|--:|--:|--:|
| MAIC | 0.000 ± 0.000 | 581.800 ± 105.874 | 0.0 ± 0.0% | 0.0 ± 0.0% |
| MAIC + Exact KL | 0.000 ± 0.000 | 20669.000 ± 18515.580 | 0.0 ± 0.0% | 0.0 ± 0.0% |
| MASAC | 0.000 ± 0.000 | 685.600 ± 232.421 | 0.0 ± 0.0% | 0.0 ± 0.0% |
| MASAC + Exact KL | 0.000 ± 0.000 | 749.200 ± 139.779 | 0.0 ± 0.0% | 0.0 ± 0.0% |
| MADDPG | 0.060 ± 0.120 | 34505.800 ± 1977.236 | 0.0 ± 0.0% | 0.0 ± 0.0% |
| COMA-configured MAIC | 0.000 ± 0.000 | 533.800 ± 63.537 | 0.0 ± 0.0% | 0.0 ± 0.0% |

## Main observations

1. **PushBox16.** MAIC + Exact KL obtains 60% mean final task success (three out of five seeds reach 100%; two reach 0%), compared with 0% for MAIC. MADDPG also obtains 60%, but its successful seeds differ from those of MAIC + Exact KL. The paired two-sided exact sign-flip test for MAIC vs. MAIC + Exact KL yields p = 0.25, and therefore does not reject the null at alpha = 0.05.
2. **MTE16.** MAIC + Exact KL achieves 25.6% greedy success vs. 9.8% for MAIC, while stochastic success changes from 41.4% to 37.8%. MASAC + Exact KL improves the average of both modes relative to MASAC. Between-seed variability is substantial.
3. **PassSparse16.** Every method has 0% final success. MAIC + Exact KL records a much larger unique-state count than MAIC, but the logging definition and lack of task success prohibit equating this difference with beneficial coordinated exploration.
4. **Generality.** The results suggest an implementation- and environment-dependent effect of actor KL regularization, not a general superiority guarantee.

## Caveats and validation priorities

- Checkpoint intermediate evaluation can reveal policy collapse that a single final checkpoint would conceal. Include only genuinely available, correctly parsed intermediate evaluation JSON files.
- Small n=5 makes uncertainty large; prioritize effect sizes, paired per-seed comparisons, and descriptive plots over unsupported significance claims.
- `COMA_Phi` is a configured ablation of the MAIC implementation and not the original on-policy COMA algorithm.
- Reward may differ from strict whole-team completion: report both metrics.
- Confirm environment equivalence and identical configuration across original MAIC and modified Exact-KL implementation before publication.
- Validate `unique_states_final` encoding and comparability before claiming additional exploration coverage.

## Source files

- [`results/study16/per_seed_values.csv`](../../results/study16/per_seed_values.csv): individual run-level final metrics (90 records).
- [`results/study16/summary.csv`](../../results/study16/summary.csv): aggregate metric values (18 records).
- [`assets/study16/figures/`](../../assets/study16/figures/): machine-generated figures with vector PDF counterparts.

All summary numbers shown here were transcribed from the supplied 90-run analysis; this document does not substitute for the raw evaluation records and original checkpoints.
