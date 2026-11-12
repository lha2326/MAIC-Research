import json
import re
import sys
import subprocess
from pathlib import Path

ROOT = Path("/workspace/MAIC_repro")
MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "coma"
PROCS = subprocess.check_output(
    ["ps", "-ww", "-eo", "args="], text=True
).splitlines()

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def tail(path, size=131072):
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - size))
            return f.read().decode(errors="replace")
    except OSError:
        return ""

def is_running(run):
    config = str((run / "config.json").resolve())
    return any(
        "main.py" in cmd
        and "--config" in cmd
        and config in cmd
        for cmd in PROCS
    )

def get_episode(run):
    m = read_json(run / "metrics_latest.json")
    ep = 0

    if isinstance(m.get("episode_rewards"), list):
        ep = len(m["episode_rewards"])

    for key in ("episode", "episodes_completed"):
        val = m.get(key)
        if isinstance(val, (int, float)):
            ep = max(ep, int(val))

    txt = tail(run / "train.log")
    matches = re.findall(
        r"(?m)^\s*\d+(?:\.\d+)?s\s+(\d+)\s+", txt
    )
    if matches:
        ep = max(ep, int(matches[-1]))

    return max(0, min(ep, 5000))

def inspect_run(run, kind):
    ep = get_episode(run)
    log = run / "train.log"

    if kind == "coma":
        done_path = run / "DONE.json"
        if done_path.exists():
            data = read_json(done_path)
            if data.get("returncode") == 0:
                return 5000, "DONE"
            return ep, "FAILED"
    else:
        if (run / "model_final.pth").exists():
            if "Training finished" in tail(log):
                return 5000, "DONE"

    if is_running(run):
        return ep, "RUNNING"

    if log.exists() or (run / "metrics_latest.json").exists():
        return ep, "INCOMPLETE"

    return 0, "PENDING"

def bar(percent):
    n = min(20, round(percent / 5))
    return "[" + "#" * n + "-" * (20 - n) + "]"

if MODE == "exact":
    jobs = [
        (algo, seed,
         ROOT / "runs_diagnostic/pilot_exactkl/PushBox16"
         / algo / f"seed{seed}")
        for algo in ("MAIC", "MASAC")
        for seed in range(45, 52)
    ]
    kind = "exact"
    title = "EXACT KL - PUSHBOX16"
else:
    jobs = [
        (env, seed,
         ROOT / "runs_phi_coma"
         / env / "COMA" / f"seed{seed}")
        for env in ("MTE16", "PushBox16", "PassSparse16")
        for seed in range(42, 52)
    ]
    kind = "coma"
    title = "COMA PHI - TRAINING"

print(title)
print("=" * 72)

total_ep = 0
finished = 0
last_group = None
group_ep = 0
group_count = 0

def show_group(name, episodes, count):
    if name is not None:
        pct = episodes / (count * 5000) * 100
        print(f"  {name} TOTAL: {bar(pct)} {pct:.2f}%\n")

for group, seed, run in jobs:
    if group != last_group:
        show_group(last_group, group_ep, group_count)
        print(f"\n=== {group} ===")
        last_group = group
        group_ep = 0
        group_count = 0

    ep, status = inspect_run(run, kind)
    pct = ep / 5000 * 100

    print(
        f"seed{seed}: {bar(pct)} "
        f"{pct:6.2f}% {ep:4d}/5000 {status}"
    )

    group_ep += ep
    group_count += 1
    total_ep += ep
    finished += (status == "DONE")

show_group(last_group, group_ep, group_count)

total = len(jobs) * 5000
pct = total_ep / total * 100

print("=" * 72)
print(f"TOTAL: {bar(pct)} {pct:.2f}%")
print(f"Episodes: {total_ep:,}/{total:,}")
print(f"Completed: {finished}/{len(jobs)} runs")
