import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict

from rl4co.utils.ops import gather_by_index

from rl4co.models.nn.env_embeddings.utils import SinusoidalPosEmb
from rl4co.models.nn.attention import MultiHeadCrossAttention

from einops import rearrange


def env_context_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment context embedding. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Usually consists of a projection of gathered node embeddings and features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tdtsp": TDTSPContext,
        "tdtsp-mat": TDTSPMatContext,
        "tdtsp_pomo": TDTSPMatContext,
        "tsp": TSPContext,
        "atsp": TSPContext,
        "cvrp": VRPContext,
        "cvrptw": VRPTWContext,
        "ffsp": FFSPContext,
        "svrp": SVRPContext,
        "sdvrp": VRPContext,
        "pctsp": PCTSPContext,
        "spctsp": PCTSPContext,
        "op": OPContext,
        "dpp": DPPContext,
        "mdpp": DPPContext,
        "pdp": PDPContext,
        "mtsp": MTSPContext,
        "smtwtp": SMTWTPContext,
        "mdcpdp": MDCPDPContext,
        "mtvrp": MTVRPContext,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available context embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embed_dim, step_context_dim=None, linear_bias=False):
        super(EnvContext, self).__init__()
        self.embed_dim = embed_dim
        step_context_dim = step_context_dim if step_context_dim is not None else embed_dim
        self.project_context = nn.Linear(step_context_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class FFSPContext(EnvContext):
    def __init__(self, embed_dim, stage_cnt=None):
        self.has_stage_emb = stage_cnt is not None
        step_context_dim = (1 + int(self.has_stage_emb)) * embed_dim
        super().__init__(embed_dim=embed_dim, step_context_dim=step_context_dim)
        if self.has_stage_emb:
            self.stage_emb = nn.Parameter(torch.rand(stage_cnt, embed_dim))

    def _cur_node_embedding(self, embeddings: TensorDict, td):
        cur_node_embedding = gather_by_index(
            embeddings["machine_embeddings"], td["stage_machine_idx"]
        )
        return cur_node_embedding

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        if self.has_stage_emb:
            state_embedding = self._state_embedding(embeddings, td)
            context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
            return self.project_context(context_embedding)
        else:
            return self.project_context(cur_node_embedding)

    def _state_embedding(self, _, td):
        cur_stage_emb = self.stage_emb[td["stage_idx"]]
        return cur_stage_emb


class TDTSPContext(EnvContext):
    def __init__(self, embed_dim):
        super(TDTSPContext, self).__init__(embed_dim, 4 * embed_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embed_dim).uniform_(-1, 1)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.position_embedding = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            if len(td.batch_size) < 2:
                context_embedding = self.W_placeholder[None, :].expand(
                    batch_size, self.W_placeholder.size(-1)
                )
            else:
                context_embedding = self.W_placeholder[None, None, :].expand(
                    batch_size, td.batch_size[1], self.W_placeholder.size(-1)
                )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        position_embedding = self.position_embedding(td['i'])
        time_embedding = self.time_embedding(td['time'].unsqueeze(1))
        context_embedding = torch.cat([context_embedding, position_embedding, time_embedding], -1)
        return self.project_context(context_embedding)


class TDTSPMatContext(EnvContext):
    """Context embedding for the Time-Dependent Traveling Salesman Problem (TDTSP).
    Project the following to the embedding space:
        - current node embedding
        - current time embedding
        - first node embedding
    """

    def __init__(self, embed_dim):
        super(TDTSPMatContext, self).__init__(embed_dim, 2 * embed_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embed_dim).uniform_(-1, 1)
        )
        self.time_embedder = self.time_embedder = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        self.node_embedder = MultiHeadCrossAttention(embed_dim, 8, sdpa_fn=NodeTimeAttentionEmbedder(8))

    def get_emb(self, embeddings, node_idx, time, dmat):
        time_emb = self.time_embedder(time).unsqueeze(1)  # [batch_size, 1, embed_dim]
        node_emb = self.node_embedder(time_emb, embeddings, dmat=dmat)
        node_emb = gather_by_index(node_emb, node_idx)
        return node_emb

    def get_mat(self, dmat, node_idx, time):
        # Get the distance matrix for the current node
        # dmat: [batch_size, num_nodes, num_nodes, horizon]
        dmat = gather_by_index(dmat, torch.floor(time).long() % dmat.size(-1), dim=-1)  # [batch_size, num_nodes, num_nodes]
        return dmat

    def forward(self, embeddings, td):
        # embedding size is [batch_size, horizon, num_nodes, embed_dim]
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            if len(td.batch_size) < 2:
                context_embedding = self.W_placeholder[None, :].expand(
                    batch_size, self.W_placeholder.size(-1)
                )
            else:
                context_embedding = self.W_placeholder[None, None, :].expand(
                    batch_size, td.batch_size[1], self.W_placeholder.size(-1)
                )
        else:

            # start_mat = self.get_mat(td['adj'], td['first_node'], td['start_time'])
            # init_emb = self.get_emb(embeddings, td["first_node"], td['start_time'], start_mat)
            # current_mat = self.get_mat(td['adj'], td["current_node"], td['time'])
            # cur_emb = self.get_emb(embeddings, td["current_node"], td['time'], current_mat)
            # context_embedding = torch.cat([init_emb, cur_emb], -1)
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim) # shape [batch_size, horizon, embed_dim]

        return self.project_context(context_embedding)


class NodeTimeAttentionEmbedder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_scores: int = 1,
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
        assert dmat is not None
        b, m, n = dmat.shape[:3]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(2) / (k.size(-1) ** 0.5)  # [b, h, m]
        mix_attn_scores = torch.cat(
            [
                attn_scores[..., None].expand(b, self.num_heads, m, n).unsqueeze(-1),  # [b, h, m, n, 1]
                dmat[:, None, ..., None].expand(b, self.num_heads, m, n, self.num_scores),
                # [b, h, m, n, num_scores]
            ],
            dim=-1,
        )  # [b, h, m, n, num_scores+1]
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
        )  # [b, h, t, n, m]
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # Compute the weighted sum of values
        return torch.matmul(attn_weights, v)  # [b, h, m, d]


class TSPContext(EnvContext):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(TSPContext, self).__init__(embed_dim, 2 * embed_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embed_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            if len(td.batch_size) < 2:
                context_embedding = self.W_placeholder[None, :].expand(
                    batch_size, self.W_placeholder.size(-1)
                )
            else:
                context_embedding = self.W_placeholder[None, None, :].expand(
                    batch_size, td.batch_size[1], self.W_placeholder.size(-1)
                )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)


class VRPContext(EnvContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 1
        )

    def _state_embedding(self, embeddings, td):
        state_embedding = td["vehicle_capacity"] - td["used_capacity"]
        return state_embedding


class VRPTWContext(VRPContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
        - current time
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 2
        )

    def _state_embedding(self, embeddings, td):
        capacity = super()._state_embedding(embeddings, td)
        current_time = td["current_time"]
        return torch.cat([capacity, current_time], -1)


class SVRPContext(EnvContext):
    """Context embedding for the Skill Vehicle Routing Problem (SVRP).
    Project the following to the embedding space:
        - current node embedding
        - current technician
    """

    def __init__(self, embed_dim):
        super(SVRPContext, self).__init__(embed_dim=embed_dim, step_context_dim=embed_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class PCTSPContext(EnvContext):
    """Context embedding for the Prize Collecting TSP (PCTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining prize (prize_required - cur_total_prize)
    """

    def __init__(self, embed_dim):
        super(PCTSPContext, self).__init__(embed_dim, embed_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = torch.clamp(
            td["prize_required"] - td["cur_total_prize"], min=0
        )[..., None]
        return state_embedding


class OPContext(EnvContext):
    """Context embedding for the Orienteering Problem (OP).
    Project the following to the embedding space:
        - current node embedding
        - remaining distance (max_length - tour_length)
    """

    def __init__(self, embed_dim):
        super(OPContext, self).__init__(embed_dim, embed_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["max_length"][..., 0] - td["tour_length"]
        return state_embedding[..., None]


class DPPContext(EnvContext):
    """Context embedding for the Decap Placement Problem (DPP), EDA (electronic design automation).
    Project the following to the embedding space:
        - current cell embedding
    """

    def __init__(self, embed_dim):
        super(DPPContext, self).__init__(embed_dim)

    def forward(self, embeddings, td):
        """Context cannot be defined by a single node embedding for DPP, hence 0.
        We modify the dynamic embedding instead to capture placed items
        """
        return embeddings.new_zeros(embeddings.size(0), self.embed_dim)


class PDPContext(EnvContext):
    """Context embedding for the Pickup and Delivery Problem (PDP).
    Project the following to the embedding space:
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(PDPContext, self).__init__(embed_dim, embed_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class MTSPContext(EnvContext):
    """Context embedding for the Multiple Traveling Salesman Problem (mTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining_agents
        - current_length
        - max_subtour_length
        - distance_from_depot
    """

    def __init__(self, embed_dim, linear_bias=False):
        super(MTSPContext, self).__init__(embed_dim, 2 * embed_dim)
        proj_in_dim = (
            4  # remaining_agents, current_length, max_subtour_length, distance_from_depot
        )
        self.proj_dynamic_feats = nn.Linear(proj_in_dim, embed_dim, bias=linear_bias)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding.squeeze()

    def _state_embedding(self, embeddings, td):
        dynamic_feats = torch.stack(
            [
                (td["num_agents"] - td["agent_idx"]).float(),
                td["current_length"],
                td["max_subtour_length"],
                self._distance_from_depot(td),
            ],
            dim=-1,
        )
        return self.proj_dynamic_feats(dynamic_feats)

    def _distance_from_depot(self, td):
        # Euclidean distance from the depot (loc[..., 0, :])
        cur_loc = gather_by_index(td["locs"], td["current_node"])
        return torch.norm(cur_loc - td["locs"][..., 0, :], dim=-1)


class SMTWTPContext(EnvContext):
    """Context embedding for the Single Machine Total Weighted Tardiness Problem (SMTWTP).
    Project the following to the embedding space:
        - current node embedding
        - current time
    """

    def __init__(self, embed_dim):
        super(SMTWTPContext, self).__init__(embed_dim, embed_dim + 1)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_job"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        state_embedding = td["current_time"]
        return state_embedding


class MDCPDPContext(EnvContext):
    """Context embedding for the MDCPDP.
    Project the following to the embedding space:
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(MDCPDPContext, self).__init__(embed_dim, embed_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class SchedulingContext(nn.Module):
    def __init__(self, embed_dim: int, scaling_factor: int = 1000):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.proj_busy = nn.Linear(1, embed_dim, bias=False)

    def forward(self, h, td):
        busy_for = (td["busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        busy_proj = self.proj_busy(busy_for.unsqueeze(-1))
        # (b m e)
        return h + busy_proj


class MTVRPContext(VRPContext):
    """Context embedding for Multi-Task VRPEnv.
    Project the following to the embedding space:
        - current node embedding
        - remaining_linehaul_capacity (vehicle_capacity - used_capacity_linehaul)
        - remaining_backhaul_capacity (vehicle_capacity - used_capacity_backhaul)
        - current time
        - current_route_length
        - open route indicator
    """

    def __init__(self, embed_dim):
        super(VRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 5
        )

    def _state_embedding(self, embeddings, td):
        remaining_linehaul_capacity = (
            td["vehicle_capacity"] - td["used_capacity_linehaul"]
        )
        remaining_backhaul_capacity = (
            td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        current_time = td["current_time"]
        current_route_length = td["current_route_length"]
        open_route = td["open_route"]
        return torch.cat(
            [
                remaining_linehaul_capacity,
                remaining_backhaul_capacity,
                current_time,
                current_route_length,
                open_route,
            ],
            -1,
        )
