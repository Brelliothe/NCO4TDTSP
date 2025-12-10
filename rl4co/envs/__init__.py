# Base environment
from rl4co.envs.common.base import RL4COEnvBase

# Routing
from rl4co.envs.routing import (
    TDTSPEnv,
    TDTSPEnvForPomo,
)

# Scheduling
from rl4co.envs.scheduling import FFSPEnv, FJSPEnv, JSSPEnv, SMTWTPEnv

# Register environments
ENV_REGISTRY = {
    "ffsp": FFSPEnv,
    "jssp": JSSPEnv,
    "fjsp": FJSPEnv,
    "tdtsp": TDTSPEnv,
    "tdtsp_pomo": TDTSPEnvForPomo,
    "smtwtp": SMTWTPEnv,
}


def get_env(env_name: str, *args, **kwargs) -> RL4COEnvBase:
    """Get environment by name.

    Args:
        env_name: Environment name
        *args: Positional arguments for environment
        **kwargs: Keyword arguments for environment

    Returns:
        Environment
    """
    env_cls = ENV_REGISTRY.get(env_name, None)
    if env_cls is None:
        raise ValueError(
            f"Unknown environment {env_name}. Available environments: {ENV_REGISTRY.keys()}"
        )
    return env_cls(*args, **kwargs)
