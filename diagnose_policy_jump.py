import sys
import json
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, "third_party/MAIC/refactor")

import multigrid
import multigrid.envs
from algorithms.network import Actor

torch.set_num_threads(1)

root = "runs_diagnostic/PushBox16/MAIC/seed43"
cfg = json.load(open(f"{root}/config.json"))

env = gym.make(
    cfg["env_id"],
    num_agents=cfg["num_agents"],
    max_steps=cfg["steps_per_episode"]
)

# Fixed observations, used for every checkpoint
observations = []

for j in range(20):
    env.reset(seed=44+j)
    s = env.unwrapped.agent_states
    features = []

    for i in range(cfg["num_agents"]):
        x, y = s.pos[i]
        direction = s.dir[i]
        carrying = float(
            env.unwrapped.agents[i].state.carrying is not None
        )
        features.append([x, y, direction, carrying])

    observations.append(features)

obs = torch.tensor(np.array(observations), dtype=torch.float32)

steps = [25, 26, 27, 28, 29, 30, 35, 40]
previous = None

for ep in steps:
    ckpt = torch.load(
        f"{root}/model_ep{ep}.pth",
        map_location="cpu",
        weights_only=True
    )

    current = []

    for i, weights in enumerate(ckpt["actors"]):
        actor = Actor(
            (4,),
            env.action_space[i].n,
            cfg["model_config"]["hidden_dim"]
        )
        actor.load_state_dict(weights)
        actor.eval()

        with torch.no_grad():
            p = actor(obs[:, i, :])
            current.append(p.clamp_min(1e-12))

    print(f"\n===== EP {ep} =====")

    for i, p in enumerate(current):
        H = -(p * p.log()).sum(-1).mean().item()
        maxp = p.max(-1).values.mean().item()

        kl = None
        if previous is not None:
            old = previous[i]
            kl = (old * (old.log()-p.log())).sum(-1).mean().item()

        print(
            f"Agent {i+1}: "
            f"Entropy={H:.8f} | "
            f"MaxProb={maxp:.6f} | "
            f"KL_from_previous={kl}"
        )

    previous = current

env.close()
