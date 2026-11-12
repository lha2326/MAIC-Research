import gymnasium as gym
from collections import deque
from statistics import mean

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

def bfs(start_pos, start_dir, target, size):
    start = (
        int(start_pos[0]),
        int(start_pos[1]),
        int(start_dir)
    )

    queue = deque([(start, [])])
    visited = {start}

    while queue:
        (x, y, direction), path = queue.popleft()

        for action in (0, 1, 2):
            nx, ny, nd = x, y, direction

            if action == 0:
                nd = (direction - 1) % 4
            elif action == 1:
                nd = (direction + 1) % 4
            else:
                dx, dy = DIRECTIONS[direction]
                nx, ny = x + dx, y + dy

                if not (1 <= nx < size-1 and
                        1 <= ny < size-1):
                    continue

                # Goal is collected on forward entry.
                if (nx, ny) == target:
                    return path + [action]

            nxt = (nx, ny, nd)

            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [action]))

    return None


for size, horizon in [(16, 40), (24, 60), (32, 80)]:
    env_id = f"MultiGrid-MultiTargetEmpty-{size}x{size}-v0"

    env = gym.make(
        env_id,
        num_agents=2,
        max_steps=horizon,
        disable_env_checker=True
    )

    successes = 0
    steps_list = []
    rewards_list = []
    failures = []

    for seed in range(1000, 1050):
        env.reset(seed=seed)
        raw = env.unwrapped

        targets = [
            (size-2, size-2),
            (1, 1)
        ]

        paths = []

        for i in range(2):
            path = bfs(
                raw.agent_states.pos[i],
                raw.agent_states.dir[i],
                targets[i],
                size
            )
            paths.append(path)

        if any(p is None for p in paths):
            failures.append(seed)
            continue

        total_reward = 0
        solved = False

        for t in range(horizon):
            actions = {
                i: paths[i][t] if t < len(paths[i]) else 6
                for i in range(2)
            }

            _, rewards, terminated, truncated, _ = env.step(actions)

            total_reward += sum(rewards.values())

            if (
                raw.remaining_goals == 0
                and all(terminated.values())
                and not any(truncated.values())
            ):
                solved = True
                steps_list.append(t + 1)
                break

            if all(
                terminated[i] or truncated[i]
                for i in terminated
            ):
                break

        rewards_list.append(total_reward)

        if solved:
            successes += 1
        else:
            failures.append(seed)

    print(f"\n=== MTE {size}x{size} ===")
    print(f"Horizon: {horizon}")
    print(f"Success: {successes}/50 ({100*successes/50:.1f}%)")
    print(
        f"Mean steps (successes): "
        f"{mean(steps_list):.2f}" if steps_list else
        "Mean steps: N/A"
    )
    print(f"Mean reward: {mean(rewards_list):.2f}")
    print(f"Failed seeds: {failures}")

    env.close()
