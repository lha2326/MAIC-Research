import json
import re
from pathlib import Path

ROOT = Path("/workspace/MAIC_repro/runs_phi_coma")
ENVS = ["MTE16", "PushBox16", "PassSparse16"]
SEEDS = range(42, 52)

def read_json(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}

def progress(p):
    cfg = read_json(p / "config.json")
    total = int(cfg.get("episodes", 5000))

    done = read_json(p / "DONE.json")
    if done.get("returncode") == 0:
        return total, total, "DONE"

    metrics = read_json(p / "metrics_latest.json")
    ep = int(metrics.get("episode", 0))

    log = p / "train.log"
    if log.exists():
        try:
            with log.open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 65536))
                txt = f.read().decode(errors="replace")
            matches = re.findall(
                r"(?m)^\s*\d+(?:\.\d+)?s\s+(\d+)\s+", txt
            )
            if matches:
                ep = max(ep, int(matches[-1]))
        except OSError:
            pass

    status = ("FAIL" if done else
              "STARTED" if log.exists() else
              "WAIT")
    return min(ep, total), total, status

def bar(p):
    n = min(20, round(p / 5))
    return "[" + "#" * n + "-" * (20 - n) + "]"

grand_ep = grand_total = finished = failed = 0

print("COMA PHI - TRAINING PROGRESS")
print("=" * 75)

for env in ENVS:
    env_ep = env_total = env_done = 0
    print(f"\n=== {env} ===")

    for seed in SEEDS:
        p = ROOT / env / "COMA" / f"seed{seed}"
        ep, total, status = progress(p)
        pct = ep / total * 100

        env_ep += ep
        env_total += total
        env_done += status == "DONE"
        failed += status == "FAIL"

        print(
            f"Seed {seed}: {ep:4d}/{total} "
            f"{bar(pct)} {pct:6.2f}%  {status}"
        )

    pct = env_ep / env_total * 100
    print(f"{env}: {bar(pct)} {pct:.2f}% ({env_done}/10 done)")

    grand_ep += env_ep
    grand_total += env_total
    finished += env_done

pct = grand_ep / grand_total * 100
print("\n" + "=" * 75)
print(f"TOTAL: {bar(pct)} {pct:.2f}%")
print(f"Episodes: {grand_ep:,}/{grand_total:,}")
print(f"Runs completed: {finished}/30 | Failed: {failed}")
