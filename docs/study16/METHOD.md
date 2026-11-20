# Study16 — Methodological scope and Exact-KL modification

The controlled intervention changes only the actor KL regularization implementation in the existing MAIC network. The information-gain estimator and underlying actor–critic architecture remain unchanged.

For a factorized decentralized policy,

$$
\pi_\theta(\mathbf a\mid\mathbf o)
= \prod_{i=1}^{n}\pi_{\theta_i}(a_i\mid o_i).
$$

For each discrete-action agent, the directly differentiable KL term at a replay observation is

$$
D_i(o_i;\theta_i)
=\sum_{a_i\in\mathcal A_i}
\pi_{\theta_i}(a_i\mid o_i)
\log\frac{\pi_{\theta_i}(a_i\mid o_i)}{\pi_{\mathrm{old},i}(a_i\mid o_i)}.
$$

The original implementation uses an action-sampled surrogate in the actor objective, with its sampled coefficient detached from the actor-gradient path. The modified implementation removes that surrogate contribution and adds the exact discrete KL term to the actor loss, preserving the baseline advantage, entropy and information-gain terms. The old policy is a gradient-free snapshot.

Exact action-distribution KL is a known mathematical quantity. Its direct application here is a controlled implementation choice, not a claimed invention of KL divergence or a new information-gain estimator. Off-policy replay observations do not automatically justify monotonic policy improvement or trust-region guarantees.

Comparison conditions:

| Method | Information gain | Entropy | KL actor term |
|:--|--:|--:|:--|
| MAIC | 1 | 1 | Sampled surrogate |
| MAIC + Exact KL | 1 | 1 | Direct discrete KL |
| MASAC | 0 | 1 | Sampled surrogate |
| MASAC + Exact KL | 0 | 1 | Direct discrete KL |
| COMA-configured MAIC | 0 | 0 | Sampled surrogate |
| MADDPG | Separate baseline | Separate baseline | Separate baseline |

Ablation names reflect the actual code configuration, not necessarily a complete independent implementation of a named published algorithm. Methodological conclusions are restricted to these tested implementations and three 16 × 16 tasks.
