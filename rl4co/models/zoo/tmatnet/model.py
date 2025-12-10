import copy
from typing import Union, Any
import torch
import torch.nn as nn

from rl4co.envs import TDTSPEnvForPomo
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.baselines.optimal import FastSubOptimalBaseline
from rl4co.models.zoo.tmatnet.policy import TimeMatNetPolicy
from rl4co.models.zoo.matnet.model import MatNet
from rl4co.models.rl.reinforce.reinforce import REINFORCE, REINFORCEBaseline
from rl4co.models.rl.reinforce.baselines import RolloutBaseline, WarmupBaseline
from rl4co.models.rl.ppo.ppo import PPO
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger
from torchrl.data import TensorDictReplayBuffer, TensorDictPrioritizedReplayBuffer, LazyMemmapStorage, ListStorage
from rl4co.data.dataset import TensorDictDataset
from tensordict.tensordict import TensorDict
from rl4co.utils.lightning import get_lightning_device


log = get_pylogger(__name__)


class CheckpointBaseline(REINFORCEBaseline):
    def __init__(self, ckpt, **kw):
        super().__init__(**kw)
        self.ckpt = ckpt
        self.policy = None

    def eval(self, td, reward, env):
        if self.policy is None:
            pretrained = MatNet.load_from_checkpoint(self.ckpt)
            self.policy = copy.deepcopy(pretrained.policy).to(td.device)
        with torch.inference_mode():
            reward = self.policy(td, env)["reward"]
        return reward, 0


class WarmStartBaseline(WarmupBaseline):
    def __init__(self, baseline, ckpt, n_epochs=1, warmup_exp_beta=0.8, **kw):
        super(REINFORCEBaseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = CheckpointBaseline(ckpt=ckpt)
        self.alpha = 0
        self.n_epochs = n_epochs

    def epoch_callback(self, *args, **kw):
        # Need to call epoch callback of inner policy (also after first epoch if we have not used it)
        self.baseline.epoch_callback(*args, **kw)
        if kw["epoch"] < self.n_epochs:
            self.alpha = (kw["epoch"] + 1) / float(self.n_epochs)
            self.alpha = 0
            log.info("Set warmup alpha = {}".format(self.alpha))


class OptimalPolicy:
    def __init__(self, env):
        self.env = env
        self.method = FastSubOptimalBaseline(env)

    def forward(self, td, env, **kwargs):
        # Get the optimal solution
        reward = - self.method.solve(td.clone())["tour_lengths"]
        return TensorDict({'reward': reward}, batch_size=td.batch_size, device=td.device)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MatNetPomoReplayPolicy:
    def __init__(self, env):
        self.policy = MatNet.load_from_checkpoint(
            "~/rl4co/logs/train/runs/tdtsp_pomo/2025.05.10-16.56.51/beijing_constant_10/checkpoints/last.ckpt"
        ).policy

    def __call__(self, td, env, **kwargs):
        env.pomo = True
        env.matrix.static = True
        init_td = env.reset(td.clone())
        rl_solution = self.policy(init_td, env, phase="test", num_starts=td['action_mask'].shape[-1])
        env.pomo = False
        env.matrix.static = False
        reward = unbatchify(rl_solution["reward"], (0, td['action_mask'].shape[-1]))
        actions = unbatchify(rl_solution["actions"], (0, td['action_mask'].shape[-1]))
        max_reward, max_idxs = reward.max(dim=-1)
        actions = gather_by_index(actions, max_idxs, dim=max_idxs.dim())
        idx = torch.argmax((actions == 0).float(), dim=1)
        positions = torch.arange(actions.shape[-1], device=actions.device).unsqueeze(0).expand(actions.shape[0], -1)
        positions = (positions + idx.unsqueeze(1)) % actions.shape[1]
        actions = torch.gather(actions, 1, positions)
        return TensorDict({'reward': env.get_tour_length(td, actions)}, batch_size=td.batch_size, device=td.device)


class MatNetREINFORCEReplayPolicy:
    def __init__(self, env):
        self.replay_policy = MatNet.load_from_checkpoint(
            "~/rl4co/logs/train/runs/tdtsp-mat/2025.05.06-13.24.11/beijing_constant_20/checkpoints/epoch_099.ckpt" # 20 nodes pretrained model
            #     # "~/rl4co/logs/train/runs/tdtsp-mat/2025.05.02-22.50.20/beijing_constant_5/checkpoints/last.ckpt"  # 5 nodes pretrained model
            #     "~/rl4co/logs/train/runs/tdtsp-mat/2025.05.03-09.24.45/beijing_constant_10/checkpoints/epoch_029.ckpt"  # 10 nodes pretrained model
        ).policy
        self.replay_policy.env = env

    def __call__(self, *args, **kwargs):
        return self.replay_policy(*args, **kwargs)


class TimeMatNet(REINFORCE):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Union[nn.Module, TimeMatNetPolicy] = None,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        policy_params: dict = {},
        baseline_kwargs: dict = {},
        **kwargs,
    ):
        if policy is None:
            # baseline = WarmStartBaseline(RolloutBaseline(bl_alpha=0.05), ckpt=ckpt, n_epochs=500)
            policy = TimeMatNetPolicy(env_name=env.name, **policy_params)
            policy.tanh_clipping = 10

        super(TimeMatNet, self).__init__(
            env=env,
            policy=policy,
            baseline=baseline,
            baseline_kwargs=baseline_kwargs,
            **kwargs,
        )
