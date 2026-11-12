from pathlib import Path
import sys, json, random
import numpy as np
import torch
import gymnasium as gym
from PIL import Image

BASE = Path("/workspace/MAIC_repro")
DEST = Path.cwd() / "assets/demos"
DEST.mkdir(parents=True, exist_ok=True)

REF = BASE / "third_party/MAIC/refactor"
sys.path.insert(0, str(REF))

import multigrid.envs
from inference import _extract_obs_state
from algorithms.MAIC import MAIC


def save_gif(frames, filename):
    if not frames:
        raise RuntimeError("Không có khung hình RGB")

    images = []
    for frame in frames:
        im = Image.fromarray(
            np.asarray(frame, dtype=np.uint8)
        ).convert("RGB")

        if im.width > 520:
            im = im.resize(
                (520, round(im.height * 520 / im.width)),
                Image.Resampling.LANCZOS
            )
        images.append(im)

    path = DEST / filename
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=220,
        loop=0,
        optimize=True
    )
    print("GIF ĐÃ TẠO:", path)


def model_demo(name, run_dir, checkpoint, greedy,
               trials, filename):

    cfg = json.loads(
        (run_dir / "config.json").read_text()
    )

    n = cfg["num_agents"]
    horizon = cfg["steps_per_episode"]

    env = gym.make(
        cfg["env_id"],
        num_agents=n,
        max_steps=horizon,
        render_mode="rgb_array",
        disable_env_checker=True
    )

    try:
        o, info = env.reset(seed=20260)

        obs0, state0 = _extract_obs_state(
            env, o, info, n, cfg["obs_state_mode"]
        )

        mc = dict(
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            tau=0.02,
            alpha1=0.02,
            alpha2=0.02,
            alpha_kl=0.1,
            policy_update_steps=3,
            auto_entropy_tuning=True,
            target_entropy_scale=0.2
        )
        mc.update(cfg.get("model_config") or {})

        torch.manual_seed(20260)

        agent = MAIC(
            obs0.shape[1:],
            state0.shape,
            env.action_space[0].n,
            num_agents=n,
            **mc
        )
        agent.load(str(checkpoint))

        for actor in agent.actors:
            actor.eval()

        for trial in range(trials):
            seed = 20261 + trial

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            o, info = env.reset(seed=seed)

            obs, _ = _extract_obs_state(
                env, o, info, n, cfg["obs_state_mode"]
            )

            frames = [np.array(env.render(), copy=True)]
            total_reward = 0
            success = False

            for step in range(horizon):
                actions, _ = agent.select_action(
                    obs, evaluate=greedy
                )

                o, rewards, terminated, truncated, info = (
                    env.step({
                        i: int(actions[i])
                        for i in range(n)
                    })
                )

                total_reward += sum(
                    float(rewards[i]) for i in range(n)
                )

                frames.append(
                    np.array(env.render(), copy=True)
                )

                obs, _ = _extract_obs_state(
                    env, o, info, n,
                    cfg["obs_state_mode"]
                )

                if (all(terminated.values())
                        and not any(truncated.values())):
                    success = True
                    break

                if all(
                    terminated[i] or truncated[i]
                    for i in terminated
                ):
                    break

            if success:
                save_gif(frames, filename)

                print(
                    f"THÀNH CÔNG {name}: "
                    f"seed_eval={seed}, "
                    f"steps={step+1}, "
                    f"reward={total_reward:.1f}"
                )
                return True

        print(
            f"CHƯA TÌM THẤY LẦN THÀNH CÔNG: {name}"
        )
        return False

    finally:
        env.close()


def passsparse_reference():
    # Bộ điều khiển tham chiếu, KHÔNG phải MAIC.
    env = gym.make(
        "MultiGrid-PassSparse-16x16-v0",
        num_agents=2,
        max_steps=40,
        render_mode="rgb_array",
        disable_env_checker=True
    )

    try:
        env.reset(seed=20260)
        frames = [np.array(env.render(), copy=True)]

        # Agent 0 giữ công tắc bên trái.
        # Agent 1 sang phòng phải và giữ công tắc.
        # Agent 0 sau đó đi qua cửa.
        # 0: quay trái, 1: quay phải,
        # 2: đi thẳng, 6: chờ.
        actions0 = (
            [2, 2, 2, 1, 2]
            + [6] * 9
            + [0]
            + [2] * 5
        )

        actions1 = (
            [2] * 4
            + [0, 2, 1]
            + [2] * 7
            + [6] * 6
        )

        assert len(actions0) == len(actions1) == 20

        reward = 0
        success = False

        for step, (a0, a1) in enumerate(
            zip(actions0, actions1)
        ):
            _, rewards, terminated, truncated, _ = (
                env.step({0: a0, 1: a1})
            )

            reward += sum(float(v) for v in rewards.values())

            frames.append(
                np.array(env.render(), copy=True)
            )

            if (all(terminated.values())
                    and not any(truncated.values())):
                success = True
                break

        if not success:
            raise RuntimeError(
                "Bộ điều khiển tham chiếu chưa giải được PassSparse"
            )

        save_gif(
            frames,
            "passsparse16_reference_scripted.gif"
        )

        print(
            f"PASS SPARSE THÀNH CÔNG: "
            f"steps={step+1}, reward={reward:.1f}"
        )

    finally:
        env.close()


results = {}

try:
    run = BASE / "runs/MTE16/MAIC/seed42"
    results["MTE16"] = model_demo(
        "MAIC Multi-Target",
        run,
        run / "model_final.pth",
        False,
        120,
        "mte16_maic_seed42_success.gif"
    )
except Exception as e:
    results["MTE16"] = False
    print("MTE16 ERROR:", e)

try:
    run = BASE / (
        "runs_diagnostic/pilot_exactkl/"
        "PushBox16/MAIC/seed51"
    )

    results["PushBox16"] = model_demo(
        "MAIC Exact KL PushBox",
        run,
        run / "model_ep2000.pth",
        True,
        10,
        "pushbox16_exactkl_success.gif"
    )
except Exception as e:
    results["PushBox16"] = False
    print("PUSHBOX16 ERROR:", e)

try:
    passsparse_reference()
    results["PassSparse16_reference"] = True
except Exception as e:
    results["PassSparse16_reference"] = False
    print("PASSSPARSE16 ERROR:", e)

print("\nKẾT QUẢ:", results)

if not all(results.values()):
    sys.exit(1)
