import json
import subprocess
from pathlib import Path

ROOT = Path("/workspace/MAIC_repro/runs_scaled_pilot_24")
ENVS = ["MTE24", "PushBox24", "PassSparse24"]
ALGOS = ["MAIC", "MASAC"]
SEEDS = [42]

try:
    processes = subprocess.check_output(
        ["ps", "-ww", "-eo", "args="],
        text=True
    ).splitlines()
except Exception:
    processes = []

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def bar(pct, width=20):
    n = min(width, max(0, round(pct * width / 100)))
    return "[" + "#" * n + "-" * (width - n) + "]"

def check_run(path):
    cfg = read_json(path / "config.json")
    total = int(cfg.get("episodes", 5000))

    metrics = read_json(path / "metrics_latest.json")
    rewards = metrics.get("episode_rewards", [])

    ep = len(rewards) if isinstance(rewards, list) else 0

    if isinstance(metrics.get("episode"), (int, float)):
        ep = max(ep, int(metrics["episode"]))

    done_path = path / "DONE.json"

    if done_path.exists():
        done = read_json(done_path)
        if done.get("returncode") == 0:
            return total, total, "DONE"
        return min(ep, total), total, "FAILED"

    config_path = str((path / "config.json").resolve())

    running = any(
        "main.py" in cmd
        and "--config" in cmd
        and config_path in cmd
        for cmd in processes
    )

    if running:
        status = "RUNNING"
    elif (path / "train.log").exists():
        status = "INCOMPLETE"
    else:
        status = "PENDING"

    return min(ep, total), total, status

grand_ep = 0
grand_total = 0
finished = 0

print("=" * 68)
print("      MAIC / MASAC - 24x24 PILOT PROGRESS")
print("=" * 68)

for env in ENVS:
    print(f"\n=== {env} ===")
    env_ep = 0
    env_total = 0

    for algo in ALGOS:
        for seed in SEEDS:
            run = ROOT / env / algo / f"seed{seed}"
            ep, total, status = check_run(run)
            pct = 100 * ep / total

            print(
                f"{algo:5} seed{seed} "
                f"{bar(pct)} "
                f"{pct:6.2f}% "
                f"{ep:4d}/{total} "
                f"{status}"
            )

            env_ep += ep
            env_total += total
            finished += (status == "DONE")

    group_pct = 100 * env_ep / env_total
    print(f"{env} TOTAL: {group_pct:.2f}%")

    grand_ep += env_ep
    grand_total += env_total

overall = 100 * grand_ep / grand_total

print("\n" + "=" * 68)
print(f"TOTAL: {bar(overall)} {overall:.2f}%")
print(f"Episodes: {grand_ep:,}/{grand_total:,}")
print(f"Completed: {finished}/6 runs")
print("=" * 68)
