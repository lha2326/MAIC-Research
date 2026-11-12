import json
from pathlib import Path

root = Path("/workspace/MAIC_repro")
base = root / "runs_diagnostic/pilot_exactkl/PushBox16"

jobs = [
    ("MAIC", 42),
    ("MAIC", 44),
    ("MASAC", 42),
    ("MASAC", 43),
    ("MASAC", 44),
]

log_file = root / "exactkl_pilot.log"
log = log_file.read_text() if log_file.exists() else ""

total = 0
completed = 0

def bar(pct):
    n = int(pct / 5)
    return "[" + "#" * n + "-" * (20 - n) + "]"

print("===== EXACT KL PILOT - PUSHBOX16 =====\n")

for algo, seed in jobs:
    run = base / algo / f"seed{seed}"
    tag = f"{algo} seed{seed}"

    if f"DONE {tag}" in log:
        ep = 5000
        completed += 1
        status = "DONE"
    elif f"START {tag}" in log:
        metrics = run / "metrics_latest.json"
        try:
            data = json.loads(metrics.read_text())
            ep = len(data["episode_rewards"])
        except (OSError, ValueError, KeyError):
            ep = 0
        status = "RUNNING"
    else:
        ep = 0
        status = "PENDING"

    ep = min(ep, 5000)
    total += ep
    pct = ep / 5000 * 100

    print(
        f"{algo:5} seed{seed} "
        f"{bar(pct)} {pct:6.1f}% "
        f"({ep:4}/5000) {status}"
    )

overall = total / 25000 * 100

print("\n" + "=" * 58)
print(f"TOTAL      {bar(overall)} {overall:.1f}%")
print(f"Completed: {total}/25000 episodes")
print(f"Finished runs: {completed}/5")
