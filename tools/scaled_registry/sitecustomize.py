from gymnasium.envs.registration import register, registry

from multigrid.envs.multitargetempty import MultiTargetEmptyEnv
from multigrid.envs.push_box import PushBoxEnv
from multigrid.envs.pass_spare import PassSparseEnv

ENV_CLASSES = {
    "MultiTargetEmpty": MultiTargetEmptyEnv,
    "PushBox": PushBoxEnv,
    "PassSparse": PassSparseEnv,
}

for size in (24, 32):
    for name, cls in ENV_CLASSES.items():
        env_id = f"MultiGrid-{name}-{size}x{size}-v0"

        if env_id not in registry:
            register(
                id=env_id,
                entry_point=cls,
                kwargs={"size": size, "num_agents": 2},
                disable_env_checker=True,
            )
