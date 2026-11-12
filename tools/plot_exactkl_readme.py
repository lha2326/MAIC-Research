import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/workspace/MAIC_repro")
OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

methods = {
    "MAIC": ("MAIC", False),
    "MASAC": ("MASAC", False),
    "MAIC + Exact KL": ("MAIC", True),
    "MASAC + Exact KL": ("MASAC", True),
}

window = 100
curves = {}
seed_success = {}

for name, (algo, exactkl) in methods.items():
    runs = []
    successes = 0

    for seed in range(42, 52):
        if exactkl:
            path = ROOT / (
                f"runs_diagnostic/pilot_exactkl/"
                f"PushBox16/{algo}/seed{seed}"
            )
            if algo == "MAIC" and seed == 43:
                path = ROOT / (
                    "runs_diagnostic/PushBox16/"
                    "MAIC/seed43_exactKL_5000"
                )
        else:
            path = ROOT / f"runs/PushBox16/{algo}/seed{seed}"

        metric_file = path / "metrics_latest.json"

        if not metric_file.is_file():
            raise FileNotFoundError(metric_file)

        data = json.loads(metric_file.read_text())
        raw = data["episode_rewards"]

        rewards = np.asarray([
            sum(float(v) for v in r)
            if isinstance(r, list)
            else float(r)
            for r in raw
        ])

        if len(rewards) != 5000:
            raise ValueError(
                f"{name} seed{seed}: "
                f"expected 5000, found {len(rewards)}"
            )

        if rewards[-500:].mean() > 0:
            successes += 1

        smoothed = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )
        runs.append(smoothed)

    curves[name] = np.stack(runs)
    seed_success[name] = successes

# 1. Learning curves
fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(window, 5001)

for name, values in curves.items():
    mean = values.mean(axis=0)
    std = values.std(axis=0)

    line, = ax.plot(x, mean, label=name, linewidth=2)
    ax.fill_between(
        x,
        np.maximum(0, mean - std),
        mean + std,
        alpha=0.15,
        color=line.get_color()
    )

ax.set_title("PushBox16 - Training reward comparison (10 seeds)")
ax.set_xlabel("Training episode")
ax.set_ylabel("Reward, moving average (100 episodes)")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "exactkl_pushbox16_learning.png",
    dpi=180
)
plt.close(fig)

# 2. Number of successful training seeds
fig, ax = plt.subplots(figsize=(10, 5))

names = list(seed_success)
counts = [seed_success[k] for k in names]

bars = ax.bar(names, counts)

for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        count + 0.15,
        f"{count}/10",
        ha="center"
    )

ax.set_title("PushBox16 - Positive final-training reward")
ax.set_ylabel("Seeds with mean reward > 0 in last 500 episodes")
ax.set_ylim(0, 10.5)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUT / "exactkl_pushbox16_seeds.png",
    dpi=180
)
plt.close(fig)

print("GENERATED CHARTS")
for name, n in seed_success.items():
    print(f"  {name}: {n}/10 seeds")

for path in sorted(OUT.glob("*.png")):
    print(path)
