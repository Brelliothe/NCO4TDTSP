from typing import Union
import copy
import torch
import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.matnet.policy import MatNetPolicy, MultiStageFFSPPolicy
from rl4co.models.zoo.matnet.critic import MatNetCriticNetwork
from rl4co.models.rl.reinforce.reinforce import REINFORCE, REINFORCEBaseline
from rl4co.models.rl.ppo.ppo import PPO
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def select_matnet_policy(env, **policy_params):
    if env.name == "ffsp":
        if env.flatten_stages:
            return MatNetPolicy(env_name=env.name, **policy_params)
        else:
            return MultiStageFFSPPolicy(stage_cnt=env.num_stage, **policy_params)
    else:
        return MatNetPolicy(env_name=env.name, **policy_params)


class MatNet(POMO):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Union[nn.Module, MatNetPolicy] = None,
        num_starts: int = None,
        policy_params: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = select_matnet_policy(env=env, **policy_params)
            policy.tanh_clipping = 10

        # Check if using augmentation and the validation of augmentation function
        if kwargs.get("num_augment", 0) != 0:
            log.warning("MatNet is using augmentation.")
            if (
                kwargs.get("augment_fn") in ["symmetric", "dihedral8"]
                or kwargs.get("augment_fn") is None
            ):
                log.error(
                    "MatNet does not use symmetric or dihedral augmentation. Seeting no augmentation function."
                )
                kwargs["num_augment"] = 0
        else:
            kwargs["num_augment"] = 0

        super(MatNet, self).__init__(
            env=env,
            policy=policy,
            num_starts=num_starts,
            **kwargs,
        )


class MatNetREINFORCE(REINFORCE):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Union[nn.Module, MatNetPolicy] = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_params: dict = {},
        baseline_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = select_matnet_policy(env=env, **policy_params)
            policy.tanh_clipping = 10

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)


class MatNetPPO(PPO):
    def __init__(
            self,
            env: RL4COEnvBase,
            policy: nn.Module = None,
            critic: MatNetCriticNetwork = None,
            policy_kwargs: dict = {},
            critic_kwargs: dict = {},
            **kwargs,
    ):
        if policy is None:
            policy = MatNetPolicy(env_name=env.name, **policy_kwargs)

        if critic is None:
            log.info("Creating critic network for {}".format(env.name))
            # we reuse the parameters of the model
            encoder = getattr(policy, "encoder", None)
            if encoder is None:
                raise ValueError("Critic network requires an encoder")
            critic = MatNetCriticNetwork(
                copy.deepcopy(encoder).to(next(encoder.parameters()).device),
                **critic_kwargs,
            )

        super().__init__(env, policy, critic, **kwargs)
