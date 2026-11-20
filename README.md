# Policy KL Regularization in Cooperative Multi-Agent Reinforcement Learning

**MAIC replication and controlled Exact-KL ablation on 16 × 16 environments**

This repository studies policy KL regularization in information-gain-based cooperative multi-agent reinforcement learning (MARL). The main experimental comparison retains the MAIC architecture and its information-gain estimator, changing only how the discrete-action KL penalty contributes to the actor gradient. Related baselines distinguish the effect of information gain from the effect of KL regularization.

The research is based on [Hoàng Văn Phi's MAIC implementation](https://github.com/hoangvanphi2004/MAIC) (source revision: `ded6164f99742db4769a7f05b45f6e6007ff8b1a`). The Exact-KL treatment should be understood as a controlled modification, **not** a new information-gain estimator, a new MARL architecture, or a guarantee of policy improvement.

## Experimental protocol

| Property | Configuration |
|:--|:--|
| Environments | Multi-Target (MTE16), PushBox16, PassSparse16 |
| Grid size | 16 × 16 |
| Agents / episode horizon | 2 / 40 steps |
| Methods | MAIC, MAIC + Exact KL, MASAC, MASAC + Exact KL, MADDPG, COMA-configured MAIC |
| Training seeds | 42–46 (5 seeds per configuration) |
| Training budget | 10,000 episodes per run |
| Total runs | **90 (3 × 6 × 5)** |
| Evaluation | Greedy and stochastic; 100 episodes per checkpoint/seed/mode |
| Final-checkpoint evaluations | **180 JSON records (90 × 2)** |

Each row in the following table is the **mean of five per-seed final-checkpoint success rates**. It is not the success rate of a single policy. Detailed per-seed values and standard deviations are reported in [Results](docs/study16/RESULTS_EN.md).

## Final-checkpoint task success

| Method | MTE16 greedy | MTE16 stochastic | PushBox16 stochastic | PassSparse16 stochastic |
|:--|--:|--:|--:|--:|
| MAIC | 9.8% | 41.4% | 0.0% | 0.0% |
| MAIC + Exact KL | 25.6% | 37.8% | 60.0% | 0.0% |
| MASAC | 7.2% | 17.4% | 0.0% | 0.0% |
| MASAC + Exact KL | 37.2% | 44.2% | 0.0% | 0.0% |
| MADDPG | 4.6% | 13.4% | 60.0% | 0.0% |
| COMA-configured MAIC | 0.0% | 0.0% | 0.0% | 0.0% |

The strongest contrast on PushBox16 is **MAIC: 0/5 successful seeds versus MAIC + Exact KL: 3/5 successful seeds**, with the same 60% seed-average success as MADDPG (3/5). On MTE16, Exact KL improves greedy MAIC success (9.8% → 25.6%), while stochastic MAIC success decreases slightly (41.4% → 37.8%). MASAC + Exact KL yields higher mean success than MASAC on both MTE16 evaluation modes. All methods obtain zero final evaluation success on PassSparse16.

**Statistical qualification.** Five seeds are insufficient to establish robust superiority. The paired, two-sided exact sign-flip test for the PushBox16 MAIC contrast yields `p = 0.25`; observed differences are descriptive, not a statistically significant demonstration of improvement at the 5% level. The definition of `unique_states_final` is implementation-specific and must not be interpreted as task success.

## Main figures

Figures use restrained grayscale styling. Dots and thin lines represent seeds; means are shown separately. Raw evaluation data, rather than inferred values from old figures, underlie these plots.

**MTE16 — final task success**

![MTE16 task success](assets/study16/figures/MTE16_success_greedy_stochastic.png)

**MTE16 — paired seed comparison**

![MTE16 paired seed comparison](assets/study16/figures/MTE16_paired_seed_response.png)

**PushBox16 — success by training seed**

![PushBox16 seed-wise success](assets/study16/figures/PushBox16_success_per_seed.png)

**PassSparse16 — recorded state diversity**

![PassSparse16 recorded unique states](assets/study16/figures/PassSparse16_unique_states_final.png)

All available PNG and vector PDF plots are located in [`assets/study16/figures/`](assets/study16/figures/). See [figure notes](docs/study16/FIGURES_AND_DEMOS.md) for data definitions and intended interpretation.

## Policy rollouts

<!-- STUDY16_DEMOS_START -->
The following examples were recorded from trained-policy checkpoints; each represents a **single episode**, not the average success rate.

| Run and evaluation metadata | Outcome | Media |
|:--|:--|:--|
| `MTE16_MAIC_trainseed44_stochastic_evalseed20261_SUCCESS` | Successful episode | [GIF](assets/study16/demos/MTE16_MAIC_trainseed44_stochastic_evalseed20261_SUCCESS.gif)  [MP4](assets/study16/demos/MTE16_MAIC_trainseed44_stochastic_evalseed20261_SUCCESS.mp4) |
| `MTE16_MASAC_ExactKL_trainseed42_stochastic_evalseed20261_SUCCESS` | Successful episode | [GIF](assets/study16/demos/MTE16_MASAC_ExactKL_trainseed42_stochastic_evalseed20261_SUCCESS.gif)  [MP4](assets/study16/demos/MTE16_MASAC_ExactKL_trainseed42_stochastic_evalseed20261_SUCCESS.mp4) |
| `PassSparse16_MAIC_ExactKL_trainseed42_greedy_evalseed20261_FAILURE` | Unsuccessful episode | [GIF](assets/study16/demos/PassSparse16_MAIC_ExactKL_trainseed42_greedy_evalseed20261_FAILURE.gif)  [MP4](assets/study16/demos/PassSparse16_MAIC_ExactKL_trainseed42_greedy_evalseed20261_FAILURE.mp4) |
| `PushBox16_MADDPG_trainseed45_greedy_evalseed20261_SUCCESS` | Successful episode | [GIF](assets/study16/demos/PushBox16_MADDPG_trainseed45_greedy_evalseed20261_SUCCESS.gif)  [MP4](assets/study16/demos/PushBox16_MADDPG_trainseed45_greedy_evalseed20261_SUCCESS.mp4) |
| `PushBox16_MAIC_ExactKL_trainseed46_greedy_evalseed20261_SUCCESS` | Successful episode | [GIF](assets/study16/demos/PushBox16_MAIC_ExactKL_trainseed46_greedy_evalseed20261_SUCCESS.gif)  [MP4](assets/study16/demos/PushBox16_MAIC_ExactKL_trainseed46_greedy_evalseed20261_SUCCESS.mp4) |

Companion `.json` records and further descriptions are in [Figures and demonstrations](docs/study16/FIGURES_AND_DEMOS.md).
<!-- STUDY16_DEMOS_END -->

## Reproducing the experiments

The original MAIC source revision and modified Exact-KL variant are kept distinct. All comparison conditions use the same environment implementations, observation/state mode, training horizon, and evaluation protocol. `COMA_Phi` means MAIC configured with IG = 0 and entropy = 0; **it is not a standalone on-policy COMA implementation**.

```bash
# From repository root, after installing dependencies and the pinned source:
python3 study16_pipeline.py prepare
python3 study16_pipeline.py train --jobs 2
python3 study16_pipeline.py status
python3 study16_pipeline.py eval --jobs 2
python3 study16_pipeline.py summarize
```

**Do not rerun `train --retry-failed` on a directory with partial runs unless you intend to archive and restart those runs.** The checkpoint files do not contain enough optimizer and replay-buffer state for an exact resume. See [Reproducibility](docs/study16/REPRODUCIBILITY.md) and [Study16 setup](SETUP_STUDY16.md).

## Repository organization

```text
baselines/                        Reference baselines, including MADDPG
third_party/MAIC/                 Original source snapshot (existing research material)
variants/exactkl/                 Exact-KL actor-loss modification
study16_pipeline.py               Reproduction and evaluation runner
scripts/study16/                  Research-quality figure generation
results/study16/                  90-row per-seed data and 18-row summary
assets/study16/figures/           Journal-style PNG and vector PDF figures
assets/study16/demos/             Selected rollout GIFs and JSON metadata
docs/study16/                     Study design, analysis and limitations
```

Previous exploratory experiments remain in their original directories. **Do not combine their 5,000-episode/10-seed statistics with this 10,000-episode/5-seed study.** The complete checkpoint collection and raw logs should be archived outside standard Git history (for example, as verified GitHub Release assets); no external archive is claimed as published until its upload is confirmed.

## Documentation

- [Results and limitations (English)](docs/study16/RESULTS_EN.md)
- [Kết quả và diễn giải (Tiếng Việt)](docs/study16/KET_QUA_TV.md)
- [Reproducibility and checkpoint provenance](docs/study16/REPRODUCIBILITY.md)
- [Figures and policy demonstrations](docs/study16/FIGURES_AND_DEMOS.md)
- [Original MAIC source](https://github.com/hoangvanphi2004/MAIC)

## Attribution and scope

The upstream MAIC code and associated concepts belong to their respective authors. Baseline names in this repository refer to the documented implementation configurations. This repository is a research artifact and does not claim a new general theoretical guarantee for exact KL regularization. Please consult the upstream licensing and attribution terms before redistributing inherited code.
