import json
import time
from pathlib import Path

root = Path("/workspace/MAIC_repro/runs")
envs = ["PushBox16", "PassSparse16"]
algos = ["MAIC", "MASAC", "MADDPG"]

total_done = 0
recent = []

print("========== STAGE C - 60 RUNS ==========")

for env in envs:
    print(f"\n{env}:")
    for algo in algos:
        done = 0
        for seed in range(42, 52):
            p = root / env / algo / f"seed{seed}"
            f = p / "metrics_latest.json"
            d = p / "DONE.json"

            if d.exists():
                try:
                    status = json.loads(d.read_text())
                    if status.get("returncode") == 0:
                        done += 1
                        continue
                except (OSError, ValueError):
                    pass

            if f.exists():
                try:
                    m = json.loads(f.read_text())
                    ep = len(m.get("episode_rewards", []))
                    last_update = time.time() - f.stat().st_mtime

                    if last_update < 600:
                        recent.append(
                            (last_update, env, algo, seed, ep)
                        )
                except (OSError, ValueError):
                    pass

        total_done += done
        print(f"  {algo}: {done}/10 completed")

print(f"\nTONG HOAN THANH: {total_done}/60")

print("\n====== RUN CAP NHAT GAN DAY ======")
for age, env, algo, seed, ep in sorted(recent)[:6]:
    print(
        f"{env} | {algo} | seed {seed} | "
        f"episode {ep}/5000 | {ep/50:.1f}% | "
        f"cap nhat {age:.0f}s truoc"
    )

if not recent:
    print("Chua co metrics moi trong 10 phut.")
    print("Kiem tra tmux va train.log.")
