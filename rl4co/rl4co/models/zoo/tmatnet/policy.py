import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.models.common.constructive.autoregressive import AutoregressivePolicy
from rl4co.models.zoo.tmatnet.encoder import TimeMatNetEncoder
from rl4co.models.zoo.tmatnet.decoder import TimeMatNetDecoder
from rl4co.utils.ops import batchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class TimeMatNetPolicy(AutoregressivePolicy):
    def __init__(
        self,
        env_name: str = "tdtsp",
        embed_dim: int = 256,
        num_encoder_layers: int = 5,
        num_heads: int = 16,
        normalization: str = "instance",
        **kwargs,
    ):
        encoder = TimeMatNetEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            normalization=normalization,
        )
        decoder = TimeMatNetDecoder(
            env_name=env_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_graph_context=False,
        )

        super(TimeMatNetPolicy, self).__init__(
            env_name=env_name,
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )
