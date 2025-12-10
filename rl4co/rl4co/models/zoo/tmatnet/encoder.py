from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict.tensordict import TensorDict
from rl4co.models.nn.env_embeddings.utils import SinusoidalPosEmb
from rl4co.models.zoo.matnet.encoder import MixedScoresSDPA, MatNetLayer, MatNetEncoder
from rl4co.models.nn.attention import MultiHeadTripleAttention, MultiHeadCrossAttention, MultiHeadAttention
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.ops import TransformerFFN


# class MixedScoresSDPA(nn.Module):
#     def __init__(
#         self,
#         num_heads: int,
#         num_scores: int = 1,
#         mixer_hidden_dim: int = 16,
#         mix1_init: float = (1 / 2) ** (1 / 2),
#         mix2_init: float = (1 / 16) ** (1 / 2),
#     ):
#         super().__init__()
#         self.num_heads = num_heads
#         self.num_scores = num_scores
#         mix_W1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
#             (num_heads, self.num_scores + 1, mixer_hidden_dim)
#         )
#         mix_b1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
#             (num_heads, mixer_hidden_dim)
#         )
#         self.mix_W1 = nn.Parameter(mix_W1)
#         self.mix_b1 = nn.Parameter(mix_b1)
#
#         mix_W2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
#             (num_heads, mixer_hidden_dim, 1)
#         )
#         mix_b2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
#             (num_heads, 1)
#         )
#         self.mix_W2 = nn.Parameter(mix_W2)
#         self.mix_b2 = nn.Parameter(mix_b2)
#
#     @staticmethod
#     def approximate_trilinear(a, b, c):
#         assert a.dim() == b.dim() == c.dim(), "a, b, c must have the same dimension"
#         assert a.size(0) == b.size(0) == c.size(0), "a, b, c must have the same batch size"
#         assert a.size(1) == b.size(1) == c.size(1), "a, b, c must have the same number of heads"
#         a = a.unsqueeze(3).unsqeeuze(4)  # [b, h, m, 1, 1, d]
#         b = b.unsqueeze(2).unsqueeze(4)  # [b, h, 1, n, 1, d]
#         c = c.unsqueeze(2).unsqueeze(3)  # [b, h, 1, 1, t, d]
#         output = torch.sum(a * b * c, dim=-1)  # [b, h, m, n, t]
#         return output
#
#     def trilinear(self, a, b, c):
#         assert a.dim() == b.dim() == c.dim(), "a, b, c must have the same dimension"
#         assert a.size(0) == b.size(0) == c.size(0), "a, b, c must have the same batch size"
#         assert a.size(1) == b.size(1) == c.size(1), "a, b, c must have the same number of heads"
#         # output = torch.einsum("bhmi,bhnj,bhtk,ijk->bhmnt", a, b, c, self.T)  # [b, h, m, n, t]
#         output = torch.einsum("bhmi,bhnj,bhtk->bhmnt", a, b, c)  # [b, h, m, n, t]
#         return output
#
#     def forward(self, q, k_1, v_1, k_2, v_2, attn_mask=None, dmat=None, dropout_p=0.0):
#         """Scaled Dot-Product Attention with MatNet Scores Mixer"""
#         assert dmat is not None
#         b, m, n, t = dmat.shape[:4]
#         dmat = dmat.reshape(b, m, n, t, self.num_scores)
#
#         # Calculate scaled dot product
#         # q: [b, h, m, d], k: [b, h, n, d], v: [b, h, t, d]
#         attn_scores = self.trilinear(q, k_1, k_2) / (k_1.size(-1) ** 0.3)  # TODO: test different normalizer
#         # [b, h, m, n, t, num_scores+1]
#         mix_attn_scores = torch.cat(
#             [
#                 attn_scores.unsqueeze(-1),
#                 dmat[:, None, ...].expand(b, self.num_heads, m, n, t, self.num_scores),
#             ],
#             dim=-1,
#         )
#         # [b, h, m, n, t]
#         attn_scores = (
#             (
#                 torch.matmul(
#                     F.relu(
#                         torch.matmul(mix_attn_scores.transpose(1, 3), self.mix_W1)
#                         + self.mix_b1[None, None, None, :, None, :]
#                     ),
#                     self.mix_W2,
#                 )
#                 + self.mix_b2[None, None, None, :, None, :]
#             )
#             .transpose(1, 3)
#             .squeeze(-1)
#         )
#
#         # Apply the provided attention mask
#         if attn_mask is not None:
#             if attn_mask.dtype == torch.bool:
#                 attn_mask[~attn_mask.any(-1)] = True
#                 attn_scores.masked_fill_(~attn_mask, float("-inf"))
#             else:
#                 attn_scores += attn_mask
#
#         # Softmax to get attention weights
#         attn_weights = F.softmax(attn_scores.reshape(b, self.num_heads, m, n * t), dim=-1).reshape(b, self.num_heads, m, n, t)
#
#         # Apply dropout
#         if dropout_p > 0.0:
#             attn_weights = F.dropout(attn_weights, p=dropout_p)
#
#         # Compute the weighted sum of values
#         # results should have the same shape as q
#         v = v_1.unsqueeze(3) * v_2.unsqueeze(2)  # [b, h, n, t, d]
#         output = torch.einsum("bhmnt,bhntd->bhmd", attn_weights, v)
#         return output


# class TimeMixedScoresSDPA(MixedScoresSDPA):
#     def forward(self, q, k, v, attn_mask=None, dmat=None, dropout_p=0.0):
#         """Scaled Dot-Product Attention with MatNet Scores Mixer"""
#         assert dmat is not None
#         b, m, n = dmat.shape[:3]
#         dmat = dmat.reshape(b, m, n, self.num_scores)
#
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
#         mix_attn_scores = torch.cat(
#             [
#                 attn_scores.unsqueeze(-1),
#                 dmat[:, None, ...].expand(b, self.num_heads, m, n, self.num_scores),
#             ],
#             dim=-1,
#         )
#         attn_scores = (
#             (
#                     torch.matmul(
#                         F.relu(
#                             torch.matmul(mix_attn_scores.transpose(1, 2), self.mix_W1)
#                             + self.mix_b1[None, None, :, None, :]
#                         ),
#                         self.mix_W2,
#                     )
#                     + self.mix_b2[None, None, :, None, :]
#             )
#             .transpose(1, 2)
#             .squeeze(-1)
#         )
#         # Softmax to get attention weights
#         attn_weights = F.softmax(attn_scores, dim=-1)
#
#         # Apply dropout
#         if dropout_p > 0.0:
#             attn_weights = F.dropout(attn_weights, p=dropout_p)
#
#         # Compute the weighted sum of values
#         return torch.matmul(attn_weights, v)
#
#
# class TimeMatNetCrossMHA(MultiHeadCrossAttention):
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         bias: bool = False,
#         mixer_hidden_dim: int = 16,
#         mix1_init: float = (1 / 2) ** (1 / 2),
#         mix2_init: float = (1 / 16) ** (1 / 2),
#     ):
#         attn_fn = TimeMixedScoresSDPA(
#             num_heads=num_heads,
#             mixer_hidden_dim=mixer_hidden_dim,
#             mix1_init=mix1_init,
#             mix2_init=mix2_init,
#         )
#
#         super().__init__(
#             embed_dim=embed_dim, num_heads=num_heads, bias=bias, sdpa_fn=attn_fn
#         )
#
#
# class TimeMatNetMHA(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, bias: bool = False):
#         super().__init__()
#         self.row_encoding_block = TimeMatNetCrossMHA(embed_dim, num_heads, bias)
#         self.col_encoding_block = TimeMatNetCrossMHA(embed_dim, num_heads, bias)
#
#     def forward(self, row_time_emb, col_emb, adj, attn_mask=None):
#         update_row_time_emb = self.row_encoding_block(row_time_emb, col_emb, dmat=adj)
#         update_col_emb = self.col_encoding_block(col_emb, row_time_emb, dmat=adj.transpose(1, 2))
#         return update_row_time_emb, update_col_emb

# class TimeMatNetMHA(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, bias: bool = False):
#         super().__init__()
#         self.row_encoding_block = TimeMatNetCrossMHA(embed_dim, num_heads, bias)
#         self.col_encoding_block = TimeMatNetCrossMHA(embed_dim, num_heads, bias)
#         self.time_encoding_block = TimeMatNetCrossMHA(embed_dim, num_heads, bias)
#
#     def forward(self, row_emb, col_emb, time_emb, adj, attn_mask=None):
#         # TODO: check the shape of attn_mask and how it should be transposed
#         # rotational attention between row, col, and time embeddings, set the updated one from the value side
#         update_row_emb = self.row_encoding_block(row_emb, col_emb, time_emb, dmat=adj)
#         update_col_emb = self.col_encoding_block(col_emb, row_emb, time_emb, dmat=adj.transpose(1, 2))
#         update_time_emb = self.time_encoding_block(time_emb, row_emb, col_emb, dmat=adj.transpose(1, 3).transpose(2, 3))
#         return update_row_emb, update_col_emb, update_time_emb


# class TimeMatNetLayer(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         bias: bool = False,
#         feedforward_hidden: int = 512,
#         normalization: Optional[str] = "instance",
#     ):
#         super().__init__()
#         self.MHA = TimeMatNetMHA(embed_dim, num_heads, bias)
#         self.F_a = TransformerFFN(embed_dim, feedforward_hidden, normalization)
#         self.F_b = TransformerFFN(embed_dim, feedforward_hidden, normalization)
#         self.F_t = TransformerFFN(embed_dim, feedforward_hidden, normalization)
#
#     def forward(self, row_emb, col_emb, time_emb, adj, attn_mask=None):
#         row_emb_out, col_emb_out, time_emb_out = self.MHA(row_emb, col_emb, time_emb, adj=adj, attn_mask=attn_mask)
#
#         row_emb_out = self.F_a(row_emb_out, row_emb)
#         col_emb_out = self.F_b(col_emb_out, col_emb)
#         time_emb_out = self.F_t(time_emb_out, time_emb)
#
#         return row_emb_out, col_emb_out, time_emb_out

# class TimeMatNetLayer(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, bias: bool = False, feedforward_hidden: int = 512, normalization: Optional[str] = "instance"):
#         super().__init__()
#         self.MHA = TimeMatNetMHA(embed_dim, num_heads, bias)
#         self.F_a = TransformerFFN(embed_dim, feedforward_hidden, normalization)
#         self.F_b = TransformerFFN(embed_dim, feedforward_hidden, normalization)
#
#     def forward(self, row_time_emb, col_emb, adj, attn_mask=None):
#         row_time_emb_out, col_emb_out = self.MHA(row_time_emb, col_emb, adj=adj, attn_mask=attn_mask)
#         row_time_emb_out = self.F_a(row_time_emb_out, row_time_emb)
#         col_emb_out = self.F_b(col_emb_out, col_emb)
#
#         return row_time_emb_out, col_emb_out
#
#
# class TimeMatNetEncoder(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int = 256,
#         num_heads: int = 16,
#         num_layers: int = 3,
#         normalization: str = "batch",
#         feedforward_hidden: int = 512,
#         init_embedding: nn.Module = None,
#         init_embedding_kwargs: dict = {},
#         bias: bool = False,
#     ):
#         super().__init__()
#
#         self.embed_dim = embed_dim
#
#         if init_embedding is None:
#             init_embedding = env_init_embedding(
#                 "tdtsp-mat", {"embed_dim": embed_dim, **init_embedding_kwargs}
#             )
#         self.init_embedding = init_embedding
#         self.layers = nn.ModuleList(
#             [
#                 TimeMatNetLayer(
#                     embed_dim=embed_dim,
#                     num_heads=num_heads,
#                     bias=bias,
#                     feedforward_hidden=feedforward_hidden,
#                     normalization=normalization,
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#
#     def forward(self, td, attn_mask: Optional[torch.Tensor] = None):
        # row_emb, col_emb, time_emb, _ = self.init_embedding(td)
        #
        # for layer in self.layers:
        #     row_emb, col_emb, time_emb = layer(row_emb, col_emb, time_emb, td['adj'], attn_mask)
        #
        # return (row_emb, col_emb, time_emb), None

        # row_time_emb, col_emb, dmat = self.init_embedding(td)
        # for layer in self.layers:
        #     row_time_emb, col_emb = layer(row_time_emb, col_emb, dmat, attn_mask=attn_mask)
        # return (row_time_emb, col_emb), None


# class TimeMatNetEncoder(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int = 256,
#         num_heads: int = 16,
#         num_layers: int = 3,
#         normalization: str = "batch",
#         feedforward_hidden: int = 512,
#         init_embedding: nn.Module = None,
#         init_embedding_kwargs: dict = {},
#         bias: bool = False,
#     ):
#         super().__init__()
#
#         self.embed_dim = embed_dim
#
#         if init_embedding is None:
#             init_embedding = env_init_embedding(
#                 "tdtsp-mat", {"embed_dim": embed_dim, **init_embedding_kwargs}
#             )
#         self.init_embedding = init_embedding
#         self.layers = nn.ModuleList(
#             [
#                 TimeMatNetLayer(
#                     embed_dim=embed_dim,
#                     num_heads=num_heads,
#                     bias=bias,
#                     feedforward_hidden=feedforward_hidden,
#                     normalization=normalization,
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#
#     def forward(self, td, attn_mask: Optional[torch.Tensor] = None):
#         row_emb, col_emb, time_emb, dmat = self.init_embedding(td)
#         for layer in self.layers:
#             row_emb, col_emb, time_emb = layer(row_emb, col_emb, time_emb, dmat, attn_mask=attn_mask)
#
#         return (row_emb, col_emb, time_emb), None
#
#
# class TimeMatNetLayer(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         bias: bool = False,
#         feedforward_hidden: int = 512,
#         normalization: Optional[str] = "instance",
#     ):
#         super().__init__()
#         self.MHA = TimeMatNetMHA(embed_dim, num_heads, bias)
#         self.F_a = TransformerFFN(embed_dim, feedforward_hidden, normalization)
#         self.F_b = TransformerFFN(embed_dim, feedforward_hidden, normalization)
#
#     def forward(self, row_emb, col_emb, time_emb, adj, attn_mask=None):
#         row_emb_out, col_emb_out, time_emb_out = self.MHA(row_emb, col_emb, time_emb, adj=adj, attn_mask=attn_mask)
#
#         row_emb_out = self.F_a(row_emb_out, row_emb)
#         col_emb_out = self.F_b(col_emb_out, col_emb)
#
#         return row_emb_out, col_emb_out, time_emb_out
#
#
# class TimeMatNetMHA(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, bias: bool = False):
#         super().__init__()
#         self.row_encoding_block = TimeMatNetCrossMHA(embed_dim, num_heads, bias)
#         self.col_encoding_block = TimeMatNetCrossMHA(embed_dim, num_heads, bias)
#
#     def forward(self, row_emb, col_emb, time_emb, adj, attn_mask=None):
#         updated_row_emb = self.row_encoding_block(row_emb, col_emb, time_emb, adj=adj)
#         updated_col_emb = self.col_encoding_block(col_emb, row_emb, time_emb, adj=adj.transpose(1, 2))
#         return updated_row_emb, updated_col_emb, time_emb


# class TimeMatNetCrossMHA(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         bias: bool = False,
#         attention_dropout: float = 0.0,
#         device: str = None,
#         dtype: torch.dtype = None,
#         sdpa_fn: Optional[Union[Callable, nn.Module]] = None,
#         mixer_hidden_dim: int = 16,
#         mix1_init: float = (1 / 2) ** (1 / 2),
#         mix2_init: float = (1 / 16) ** (1 / 2),
#     ):
#         super().__init__()
#
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.bias = bias
#         self.attention_dropout = attention_dropout
#         self.device = device
#         self.dtype = dtype
#         self.sdpa_fn = TimeMixedScoresSDPA(
#             num_heads=num_heads,
#             mixer_hidden_dim=mixer_hidden_dim,
#             mix1_init=mix1_init,
#             mix2_init=mix2_init,
#         )
#
#         self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#
#     def forward(self, node_emb, help_emb, time_emb, adj, attn_mask=None):
#         q = rearrange(self.Wq(node_emb), "b m (h d) -> b h m d", h=self.num_heads)
#         k, v = rearrange(self.Wkv(help_emb), "b n (two h d) -> two b h n d", two=2, h=self.num_heads).unbind(dim=0)
#         out = self.sdpa_fn(q, k, v, attn_mask=attn_mask, dmat=adj)
#         out = rearrange(out, "b h m d -> b m (h d)")
#         out = self.out_proj(out)
#         return out
#
#
# class TimeMixedScoresSDPA(MixedScoresSDPA):
#     def forward(self, q, k, v, attn_mask=None, dmat=None, dropout_p=0.0):
#         """Scaled Dot-Product Attention with MatNet Scores Mixer"""
#         assert dmat is not None
#         b, m, n, t = dmat.shape[:4]
#         dmat = dmat.mean(dim=-1)
#
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)  # [b, h, m, n]
#         mix_attn_scores = torch.cat(
#             [
#                 attn_scores[...].unsqueeze(-1), # [b, h, m, n, t, 1]
#                 dmat[:, None, ..., None].expand(b, self.num_heads, m, n, self.num_scores),  # [b, h, n, m, num_scores]
#             ],
#             dim=-1,
#         )  # [b, h, m, n, num_scores+1]
#         attn_scores = (
#             (
#                 torch.matmul(
#                     F.relu(
#                         torch.matmul(mix_attn_scores.transpose(1, 2), self.mix_W1)
#                         + self.mix_b1[None, None, :, None, :]
#                     ),
#                     self.mix_W2,
#                 )
#                 + self.mix_b2[None, None, :, None, :]
#             )
#             .transpose(1, 2)
#             .squeeze(-1)
#         )  # [b, h, m, n]
#         # Softmax to get attention weights
#         attn_weights = F.softmax(attn_scores, dim=-1)
#
#         # Apply dropout
#         if dropout_p > 0.0:
#             attn_weights = F.dropout(attn_weights, p=dropout_p)
#
#         # Compute the weighted sum of values
#         return torch.matmul(attn_weights, v)  # [b, h, m, d]


# class TimeMatNetEncoder(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int = 256,
#         num_heads: int = 16,
#         num_layers: int = 3,
#         normalization: str = "batch",
#         feedforward_hidden: int = 512,
#         init_embedding: nn.Module = None,
#         init_embedding_kwargs: dict = {},
#         bias: bool = False,
#     ):
#         super().__init__()
#
#         self.embed_dim = embed_dim
#
#         if init_embedding is None:
#             init_embedding = env_init_embedding(
#                 "tdtsp-mat", {"embed_dim": embed_dim, **init_embedding_kwargs}
#             )
#         self.init_embedding = init_embedding
#         # Consider a very simple case:
#         self.encoder = MatNetEncoder(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             num_layers=num_layers,
#             normalization=normalization,
#             feedforward_hidden=feedforward_hidden,
#         )
#         self.attn = MultiHeadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             bias=bias,
#         )
#         self.temporal_encoder = nn.Sequential(
#             SinusoidalPosEmb(embed_dim),
#             nn.Linear(embed_dim, 4 * embed_dim),
#             nn.Mish(),
#             nn.Linear(4 * embed_dim, embed_dim),
#         )
#         self.temporal_proj = nn.Sequential(
#             nn.Linear(2 * embed_dim, 4 * embed_dim),
#             nn.ReLU(),
#             nn.Linear(4 * embed_dim, embed_dim),
#             nn.LayerNorm(embed_dim),
#         )
#
#     def forward(self, td, attn_mask: Optional[torch.Tensor] = None):
#         # return self.encoder(td)
#         matrix = td['adj']
#         batch_size, num_nodes, _, horizon = matrix.shape[:4]
#         matrix = matrix.permute(0, 3, 1, 2).reshape(batch_size * horizon, num_nodes, num_nodes)
#         (row_emb, col_emb), _ = self.encoder(td=TensorDict({'adj': matrix}), attn_mask=attn_mask)
#         row_emb = row_emb.view(batch_size, horizon, num_nodes, self.embed_dim).permute(0, 2, 1, 3)
#         col_emb = col_emb.view(batch_size, horizon, num_nodes, self.embed_dim).permute(0, 2, 1, 3)
#         time_emb = self.temporal_encoder(torch.arange(horizon, device=row_emb.device))
#         time_emb = time_emb[None, None, ...].expand(batch_size, num_nodes, horizon, self.embed_dim)
#         row_emb = row_emb + time_emb
#         col_emb = col_emb + time_emb
#         return (row_emb, col_emb), None


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.attention import MultiHeadCrossAttention
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.ops import TransformerFFN


class MixedScoresSDPA(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_scores: int = 12,
        mixer_hidden_dim: int = 16,
        mix1_init: float = (1 / 2) ** (1 / 2),
        mix2_init: float = (1 / 16) ** (1 / 2),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_scores = num_scores
        mix_W1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (num_heads, self.num_scores + 1, mixer_hidden_dim)
        )
        mix_b1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (num_heads, mixer_hidden_dim)
        )
        self.mix_W1 = nn.Parameter(mix_W1)
        self.mix_b1 = nn.Parameter(mix_b1)

        mix_W2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (num_heads, mixer_hidden_dim, 1)
        )
        mix_b2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (num_heads, 1)
        )
        self.mix_W2 = nn.Parameter(mix_W2)
        self.mix_b2 = nn.Parameter(mix_b2)

    def forward(self, q, k, v, attn_mask=None, dmat=None, dropout_p=0.0):
        """Scaled Dot-Product Attention with MatNet Scores Mixer"""
        assert dmat is not None
        b, m, n, T = dmat.shape[:4]
        dmat = dmat[:, :, :, :self.num_scores]

        # Calculate scaled dot product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        # [b, h, m, n, num_scores+1]
        mix_attn_scores = torch.cat(
            [
                attn_scores.unsqueeze(-1).expand(b, self.num_heads, m, n, 1),
                dmat[:, None, ...].expand(b, self.num_heads, m, n, self.num_scores),
            ],
            dim=-1,
        )
        # [b, h, m, n]
        attn_scores = (
            (
                torch.matmul(
                    F.relu(
                        torch.matmul(mix_attn_scores.transpose(1, 2), self.mix_W1)
                        + self.mix_b1[None, None, :, None, :]
                    ),
                    self.mix_W2,
                )
                + self.mix_b2[None, None, :, None, :]
            )
            .transpose(1, 2)
            .squeeze(-1)
        )

        # Apply the provided attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask[~attn_mask.any(-1)] = True
                attn_scores.masked_fill_(~attn_mask, float("-inf"))
            else:
                attn_scores += attn_mask

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # Compute the weighted sum of values
        return torch.matmul(attn_weights, v)


class MatNetCrossMHA(MultiHeadCrossAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        mixer_hidden_dim: int = 16,
        mix1_init: float = (1 / 2) ** (1 / 2),
        mix2_init: float = (1 / 16) ** (1 / 2),
    ):
        attn_fn = MixedScoresSDPA(
            num_heads=num_heads,
            mixer_hidden_dim=mixer_hidden_dim,
            mix1_init=mix1_init,
            mix2_init=mix2_init,
        )

        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, bias=bias, sdpa_fn=attn_fn
        )


class MatNetMHA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.row_encoding_block = MatNetCrossMHA(embed_dim, num_heads, bias)
        self.col_encoding_block = MatNetCrossMHA(embed_dim, num_heads, bias)

    def forward(self, row_emb, col_emb, dmat, attn_mask=None):
        """
        Args:
            row_emb (Tensor): [b, m, d]
            col_emb (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]

        Returns:
            Updated row_emb (Tensor): [b, m, d]
            Updated col_emb (Tensor): [b, n, d]
        """
        updated_row_emb = self.row_encoding_block(
            row_emb, col_emb, dmat=dmat, cross_attn_mask=attn_mask
        )
        attn_mask_t = attn_mask.transpose(-2, -1) if attn_mask is not None else None
        updated_col_emb = self.col_encoding_block(
            col_emb,
            row_emb,
            dmat=dmat.transpose(-2, -3),
            cross_attn_mask=attn_mask_t,
        )
        return updated_row_emb, updated_col_emb


class MatNetLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
    ):
        super().__init__()
        self.MHA = MatNetMHA(embed_dim, num_heads, bias)
        self.F_a = TransformerFFN(embed_dim, feedforward_hidden, normalization)
        self.F_b = TransformerFFN(embed_dim, feedforward_hidden, normalization)

    def forward(self, row_emb, col_emb, dmat, attn_mask=None):
        """
        Args:
            row_emb (Tensor): [b, m, d]
            col_emb (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]

        Returns:
            Updated row_emb (Tensor): [b, m, d]
            Updated col_emb (Tensor): [b, n, d]
        """

        row_emb_out, col_emb_out = self.MHA(row_emb, col_emb, dmat, attn_mask)
        row_emb_out = self.F_a(row_emb_out, row_emb)
        col_emb_out = self.F_b(col_emb_out, col_emb)
        return row_emb_out, col_emb_out


class TimeMatNetEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 16,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        bias: bool = False,
        mask_non_neighbors: bool = False,
    ):
        super().__init__()

        if init_embedding is None:
            init_embedding = env_init_embedding(
                "matnet", {"embed_dim": embed_dim, **init_embedding_kwargs}
            )
            # init_embedding = env_init_embedding(
            #     "tdtsp-mat", {"embed_dim": embed_dim, **init_embedding_kwargs}
            # )

        self.init_embedding = init_embedding
        self.mask_non_neighbors = mask_non_neighbors
        self.layers = nn.ModuleList(
            [
                MatNetLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=bias,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, td, attn_mask: torch.Tensor = None):
        row_emb, col_emb, dmat = self.init_embedding(td)

        if self.mask_non_neighbors and attn_mask is None:
            # attn_mask (keep 1s discard 0s) to only attend on neighborhood
            attn_mask = dmat.ne(0)

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, td['adj'], attn_mask)

        embedding = (row_emb, col_emb)
        init_embedding = None
        return embedding, init_embedding  # match output signature for the AR policy class