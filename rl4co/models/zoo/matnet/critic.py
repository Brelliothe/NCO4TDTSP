import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Tuple, Union
from tensordict import TensorDict
from torch import Tensor
from rl4co.utils.pylogger import get_pylogger
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding


log = get_pylogger(__name__)


class MatNetCriticNetwork(nn.Module):
    """Create a critic network given an encoder (e.g. as the one in the policy network)
    with a value head to transform the embeddings to a scalar value.

    Args:
        encoder: Encoder module to encode the input
        value_head: Value head to transform the embeddings to a scalar value
        embed_dim: Dimension of the embeddings of the value head
        hidden_dim: Dimension of the hidden layer of the value head
    """

    def __init__(
        self,
        encoder: nn.Module,
        value_head: Optional[nn.Module] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        hidden_dim: int = 512,
        customized: bool = False,
    ):
        super(MatNetCriticNetwork, self).__init__()

        self.num_heads = num_heads
        self.mask_inner = True

        self.sdpa_fn = torch.nn.functional.scaled_dot_product_attention

        self.encoder = encoder
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=False
        )

        self.context_embedding = (
            env_context_embedding("tdtsp-mat", {"embed_dim": embed_dim})
        )

        self.project_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Union[Tensor, TensorDict], hidden=None) -> Tensor:
        """Forward pass of the critic network: encode the imput in embedding space and return the value

        Args:
            x: Input containing the environment state. Can be a Tensor or a TensorDict

        Returns:
            Value of the input state
        """
        (row_emb, col_emb), _ = self.encoder(x)  # [batch_size, N, embed_dim] -> [batch_size, N]
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(
            col_emb
        ).chunk(3, dim=-1)

        glimpse_q = self.context_embedding(row_emb, x)
        # add seq_len dim if not present
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q

        # Compute the value of the input state
        mask = x["action_mask"]
        heads = self._inner_mha(glimpse_q, glimpse_key_fixed, glimpse_val_fixed, mask)
        glimpse = self._project_out(heads, mask)

        value = glimpse.mean(dim=1)
        return value

    def _inner_mha(self, query, key, value, attn_mask):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)
        if self.mask_inner:
            # make mask the same number of dimensions as q
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )
        else:
            attn_mask = None
        heads = self.sdpa_fn(q, k, v, attn_mask=attn_mask)
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)

    def _project_out(self, out, *kwargs):
        return self.project_out(out)