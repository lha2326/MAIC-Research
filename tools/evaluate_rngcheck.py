"""Post-hoc evaluation of MAIC-repo checkpoints (does not modify the original code).

The rollout loop mirrors refactor/inference.py of hoangvanphi2004/MAIC
(same observation/state extraction, same select_action), and adds:
  * --mode greedy (argmax, identical to inference.py) or stochastic (sample from pi)
  * a fixed evaluation seed, so every method is evaluated on the same episodes
  * success = episode ends by environment termination (task solved) before the step limit
Usage:
  python tools/evaluate.py --maic-root third_party/MAIC/refactor --run-dir runs/<env>/<algo>/seed42 \
      --checkpoint model_final.pth --episodes 100 --mode greedy
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--maic-root', required=True, help='path to MAIC/refactor')
    p.add_argument('--run-dir', default=None, help='run folder containing config.json and checkpoints')
    p.add_argument('--config', default=None, help='training config (default: <run-dir>/config.json)')
    p.add_argument('--checkpoint', default='model_final.pth')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--mode', choices=['greedy', 'stochastic'], default='greedy')
    p.add_argument('--eval-seed', type=int, default=20260)
    p.add_argument('--algo', default=None, help='maddpg -> load the MADDPG class instead of MAIC')
    p.add_argument('--out', default=None)
    a = p.parse_args()

    root = Path(a.maic_root).resolve()
    sys.path.insert(0, str(root))
    import gymnasium as gym
    import multigrid, multigrid.envs  # noqa: F401
    from inference import _extract_obs_state  # original helper from the repo
    if a.algo == 'maddpg':
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from baselines.maddpg import MADDPG as Agent
    else:
        from algorithms.MAIC import MAIC as Agent

    run_dir = Path(a.run_dir) if a.run_dir else None
    cfg_path = Path(a.config) if a.config else run_dir / 'config.json'
    cfg = json.loads(cfg_path.read_text())
    ckpt = Path(a.checkpoint)
    if not ckpt.is_absolute() and run_dir is not None and not ckpt.exists():
        ckpt = run_dir / ckpt
    env_id, n, mode, T = cfg['env_id'], cfg['num_agents'], cfg['obs_state_mode'], cfg['steps_per_episode']
    mc = dict(hidden_dim=128, lr=3e-4, gamma=0.99, tau=0.02, alpha1=0.02, alpha2=0.02, alpha_kl=0.1,
              policy_update_steps=3, auto_entropy_tuning=True, target_entropy_scale=0.2)
    mc.update(cfg.get('model_config') or {})

    torch.manual_seed(a.eval_seed); np.random.seed(a.eval_seed)
    env = gym.make(env_id, num_agents=n, max_steps=T)
    o, info = env.reset(seed=a.eval_seed)
    o_arr, s_arr = _extract_obs_state(env, o, info, n, mode)
    agent = Agent(o_arr.shape[1:], s_arr.shape, env.action_space[0].n, num_agents=n, **mc)
    agent.load(str(ckpt))

    # Reseed after model and ensemble initialization.
    import random
    random.seed(a.eval_seed)
    np.random.seed(a.eval_seed)
    torch.manual_seed(a.eval_seed)


    rewards, lengths, success, per_agent = [], [], [], []
    for ep in range(a.episodes):
        o, info = env.reset(seed=a.eval_seed + 1 + ep)
        obs, _ = _extract_obs_state(env, o, info, n, mode)
        R, L, solved, ind = 0.0, 0, False, np.zeros(n)
        for t in range(T):
            acts, _ = agent.select_action(obs, evaluate=(a.mode == 'greedy'))
            o, r, term, trunc, info = env.step({i: int(acts[i]) for i in range(n)})
            ind += np.array([r[i] for i in range(n)], dtype=float)
            R += float(sum(r[i] for i in range(n))); L = t + 1
            obs, _ = _extract_obs_state(env, o, info, n, mode)
            if all(term[i] for i in term) and not any(trunc[i] for i in trunc):
                solved = True
            if all(term[i] or trunc[i] for i in term):
                break
        rewards.append(R); lengths.append(L); success.append(solved); per_agent.append(ind.tolist())
    pa = np.array(per_agent)
    res = dict(env_id=env_id, checkpoint=str(ckpt), mode=a.mode, eval_seed=a.eval_seed, episodes=a.episodes,
               reward_mean=float(np.mean(rewards)), reward_std=float(np.std(rewards)),
               success_rate=float(np.mean(success)), length_mean=float(np.mean(lengths)),
               any_reward_rate=float(np.mean(np.array(rewards) > 0)),
               per_agent_positive_rate=[float((pa[:, i] > 0).mean()) for i in range(n)])
    print(json.dumps(res))
    out = Path(a.out) if a.out else (run_dir / f'eval_{Path(ckpt).stem}_{a.mode}.json' if run_dir else None)
    if out:
        out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
