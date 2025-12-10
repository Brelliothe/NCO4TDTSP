from dataclasses import dataclass
from typing import Tuple, Union
import torch
import torch.nn as nn
from rl4co.envs import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.attention import PointerAttention, PointerAttnMoE
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.nn.env_embeddings.utils import SinusoidalPosEmb
from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache
from tensordict import TensorDict
from torch import Tensor


@dataclass
class PrecomputedCache:
    node_embeddings: Union[Tensor, TensorDict]
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


class TimeMatNetDecoder(AttentionModelDecoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = False,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        pointer: nn.Module = None,
        moe_kwargs: dict = None,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )
        self.dynamic_embedding = (
            env_dynamic_embedding(self.env_name, {"embed_dim": embed_dim})
            if dynamic_embedding is None
            else dynamic_embedding
        )
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        if pointer is None:
            # MHA with Pointer mechanism (https://arxiv.org/abs/1506.03134)
            pointer_attn_class = (
                PointerAttention if moe_kwargs is None else PointerAttnMoE
            )
            pointer = pointer_attn_class(
                embed_dim,
                num_heads,
                mask_inner=mask_inner,
                out_bias=out_bias_pointer_attn,
                check_nan=check_nan,
                sdpa_fn=sdpa_fn,
                moe_kwargs=moe_kwargs,
            )

        self.pointer = pointer

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_time_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context
        self.temporal_encoder = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def interpolate(self, embedding, td: TensorDict):
        # embedding shape: (batch_size, num_nodes, horizon, dim)
        # Attention Interpolation
        # batch_size, num_nodes, horizon, embed_dim = embedding.shape
        # shaped_embedding = embedding.reshape(batch_size * num_nodes, horizon, embed_dim)  # shape: (batch_size * num_nodes, horizon, dim)
        # time_query = self.temporal_encoder(td["time"])[:, None, :].expand(batch_size, num_nodes, embed_dim) # shape: (batch_size, num_nodes, embed_dim)
        # shaped_query = time_query.reshape(batch_size * num_nodes, 1, embed_dim)  # shape: (batch_size * num_nodes, 1, dim)
        # selected = self.attn(shaped_query, shaped_embedding, shaped_embedding)[0]  # shape: (batch_size * num_nodes, 1, dim)
        # selected = selected.reshape(batch_size, num_nodes, embed_dim)
        # return selected
        return embedding

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict):
        node_embeds_cache = self.interpolate(cached.node_embeddings, td)
        # node_embeds_cache = cached.node_embeddings
        graph_context_cache = cached.graph_context

        if td.dim() == 2 and isinstance(graph_context_cache, Tensor):
            graph_context_cache = graph_context_cache.unsqueeze(1)

        step_context = self.context_embedding(node_embeds_cache, td)
        time_context = self.temporal_encoder(td["time"])
        glimpse_q = step_context + graph_context_cache + time_context
        # add seq_len dim if not present
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q

        return glimpse_q

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
            self.interpolate(cached.glimpse_key, td),
            self.interpolate(cached.glimpse_val, td),
            self.interpolate(cached.logit_key, td),
        )

        # Compute dynamic embeddings and add to static embeddings
        # glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td)
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.project_time_embeddings(self.temporal_encoder(td["time"]).unsqueeze(1)).chunk(3, dim=-1)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn

        return glimpse_k, glimpse_v, logit_k

    def _precompute_cache(self, embeddings: Tuple[Tensor, Tensor], *args, **kwargs):
        row_emb, col_emb = embeddings
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(
            col_emb
        ).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(col_emb.mean(1))
        else:
            graph_context = 0

        # time_emb = self.temporal_encoder(torch.arange(row_emb.shape[-2], device=row_emb.device))
        # time_emb = time_emb[None, None, ...].expand(*row_emb.shape)
        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=row_emb,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key,
        )