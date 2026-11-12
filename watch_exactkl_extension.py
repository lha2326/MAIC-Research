import json
from pathlib import Path

root = Path("/workspace/MAIC_repro")
base = root / "runs_diagnostic/pilot_exactkl/PushBox16"

jobs = [(algo, seed)
        for algo in ["MAIC", "MASAC"]
        for seed in range(45, 52)]

log_path = root / "exactkl_extension.log"
log = log_path.read_text(errors="replace") if log_path.exists() else ""

def bar(pct, width=20):
    filled = round(width * pct / 100)
    return "[" + "█" * filled + "░" * (width - filled) + "]"

total_ep = 0
finished = 0

print("===== EXACT KL - PUSHBOX16 (14 RUNS) =====\n")

for algo, seed in jobs:
    run = base / algo / f"seed{seed}"
    metrics = run / "metrics_latest.json"
    train_log = run / "train.log"

    ep = 0
    try:
        m = json.loads(metrics.read_text())
        ep = min(5000, len(m["episode_rewards"]))
    except (OSError, ValueError, KeyError, TypeError):
        pass

    completed = False
    if (run / "model_final.pth").is_file() and train_log.exists():
        completed = "Training finished" in train_log.read_text(
            errors="replace"
        )

    if completed:
        ep = 5000
        status = "DONE"
        finished += 1
    elif f"START {algo} seed{seed}" in log:
        status = "RUNNING"
    else:
        status = "PENDING"

    total_ep += ep
    pct = ep / 5000 * 100

    print(
        f"{algo:5} seed{seed} "
        f"{bar(pct)} {pct:6.1f}% "
        f"{ep:4}/5000  {status}"
    )

overall = total_ep / (14 * 5000) * 100

print("\n" + "=" * 65)
print(f"TOTAL        {bar(overall)} {overall:6.1f}%")
print(f"Completed: {total_ep:,} / 70,000 episodes")
print(f"Finished: {finished} / 14 runs")

if "ALL 14 RUNS COMPLETED" in log:
    print("\nALL RUNS COMPLETED")
