import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict

from rl4co.models.nn.ops import PositionalEncoding
from rl4co.models.nn.env_embeddings.utils import SinusoidalPosEmb
from tensorflow.python.ops.gen_string_ops import string_upper


def env_init_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment initial embedding. The init embedding is used to initialize the
    general embedding of the problem nodes without any solution information.
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tdtsp": TDTSPInitEmbedding,
        "tdtsp-mat": TDTSPMatInitEmbedding,
        "tsp": TSPInitEmbedding,
        "atsp": TSPInitEmbedding,
        "matnet": MatNetInitEmbedding,
        "cvrp": VRPInitEmbedding,
        "cvrptw": VRPTWInitEmbedding,
        "svrp": SVRPInitEmbedding,
        "sdvrp": VRPInitEmbedding,
        "pctsp": PCTSPInitEmbedding,
        "spctsp": PCTSPInitEmbedding,
        "op": OPInitEmbedding,
        "dpp": DPPInitEmbedding,
        "mdpp": MDPPInitEmbedding,
        "pdp": PDPInitEmbedding,
        "pdp_ruin_repair": TSPInitEmbedding,
        "tsp_kopt": TSPInitEmbedding,
        "mtsp": MTSPInitEmbedding,
        "smtwtp": SMTWTPInitEmbedding,
        "mdcpdp": MDCPDPInitEmbedding,
        "fjsp": FJSPInitEmbedding,
        "jssp": FJSPInitEmbedding,
        "mtvrp": MTVRPInitEmbedding,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class TDTSPInitEmbedding(nn.Module):
    """Initial embedding for the Time-Dependent Traveling Salesman Problems (TDTSP).
    Embed the following node features to the embedding space:
        - locs: index i of the nodes (customers)
        - matrix: time-dependent adjacent matrix, shape (n, n, T)
    """
    def __init__(self, embed_dim, linear_bias=True):
        super(TDTSPInitEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.linear_bias = linear_bias
        self.matrix_embed = None

    def forward(self, td):
        # lazy initialization
        if self.matrix_embed is None:
            batch_size, nodes, _, horizon = td["adj"].size()
            # linear model to embed the time-dependent adjacent matrix
            self.matrix_embed = SpatialTemporalEmbedding(self.embed_dim, nodes, horizon).to(td["adj"].device)
            # self.matrix_embed = nn.Linear(nodes * horizon, self.embed_dim, self.linear_bias).to(td["adj"].device)
            # self.matrix_embed = LinearMatrixEmbedding(self.embed_dim, self.linear_bias, nodes, horizon).to(td["adj"].device)
            # gnn model to embed the time-dependent adjacent matrix
            # self.matrix_embed = EdgeGNN(self.embed_dim, nodes, horizon).to(td["adj"].device)
            # self.matrix_embed = TimeConvGNN(self.embed_dim, nodes, horizon).to(td["adj"].device)
            # attention model to embed the time-dependent adjacent matrix
            # self.matrix_embed = STGNNEmbedding(nodes, horizon, hidden_dim=self.embed_dim).to(td["adj"].device)
            # set model to embed the time-dependent adjacent matrix
        matrix_embed = self.matrix_embed(td["adj"])
        out = matrix_embed
        return out


class SpatialTemporalEmbedding(nn.Module):
    def __init__(self, embed_dim, num_nodes, horizon):
        super(SpatialTemporalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.horizon = horizon

        self.temporal_encoder = nn.Linear(horizon, embed_dim // 2)
        self.spatial_encoder = nn.Linear(num_nodes, embed_dim // 2)

        self.projector = nn.Linear(embed_dim, embed_dim)

    def forward(self, matrix):
        spatial_data = torch.mean(matrix, dim=3)
        temporal_data = torch.mean(matrix, dim=2)

        spatial_features = self.spatial_encoder(spatial_data)
        temporal_features = self.temporal_encoder(temporal_data)

        features = torch.cat([spatial_features, temporal_features], dim=-1)
        output = self.projector(features)
        return output


class LinearMatrixEmbedding(nn.Module):
    """
    Linear embedding for matrix embedding in TDTSP
    """
    def __init__(self, embed_dim, linear_bias, num_nodes, horizon):
        super(LinearMatrixEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.matrix_embed = nn.Linear(num_nodes * horizon, embed_dim, linear_bias)

    def forward(self, matrix):
        x = matrix.reshape((-1, self.num_nodes, self.num_nodes * self.horizon))
        output = self.matrix_embed(x)
        return output


class GNNEmbedding(nn.Module):
    def __init__(self, embed_dim, num_nodes, horizon):
        super(GNNEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes
        self.horizon = horizon


class EdgeGNN(GNNEmbedding):
    """
    GNN embedding for matrix embedding in TDTSP
    """
    def __init__(self, embed_dim, num_nodes, horizon):
        super(EdgeGNN, self).__init__(embed_dim, num_nodes, horizon)

        self.temporal_encoder = nn.Sequential(
            nn.Linear(horizon, 2 * horizon),
            nn.ReLU(),
            nn.Linear(2 * horizon, embed_dim)
        )

        self.message_passing = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.projector = nn.Linear(embed_dim, embed_dim)

    def forward(self, matrix):
        temporal_features = self.temporal_encoder(matrix)  # shape [batch, num_nodes, num_nodes, embed_dim]
        node_features = torch.sum(temporal_features, dim=2)  # shape [batch, num_nodes, embed_dim]
        node_features = self.message_passing(node_features)  # shape [batch, num_nodes, embed_dim]
        output = self.projector(node_features)  # shape [batch, num_nodes, embed_dim]
        return output


class TimeConvGNN(GNNEmbedding):
    def __init__(self, embed_dim, num_nodes, horizon):
        super(TimeConvGNN, self).__init__(embed_dim, num_nodes, horizon)
        self.edge_embed = nn.Linear(1, embed_dim // 4)

        # Temporal relationship processor
        self.temporal_conv = nn.Conv2d(
            in_channels=self.horizon,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1
        )

        # Node aggregation
        self.node_aggregation = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, matrix):
        # Initial edge embedding
        edge_features = self.edge_embed(matrix.unsqueeze(-1))  # [batch, n, n, T, embed//4]

        # Reshape for temporal convolution
        # View time as channels for convolutional processing
        edge_features = edge_features.permute(0, 3, 1, 2, 4).reshape(
            -1, self.horizon, self.num_nodes, self.num_nodes * (self.embed_dim // 4)
        )

        # Apply temporal convolution to capture inter-time relationships
        temporal_features = self.temporal_conv(edge_features)  # [batch, embed, n, n*embed//4]

        # Reshape and aggregate
        temporal_features = temporal_features.reshape(
            -1, self.embed_dim, self.num_nodes, self.num_nodes, self.embed_dim // 4
        ).sum(dim=-1)  # [batch, embed, n, n]

        # Aggregate over neighbors
        node_features = temporal_features.sum(dim=3).permute(0, 2, 1)  # [batch, n, embed]

        # Final node projection
        output = self.node_aggregation(node_features)

        return output


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important time steps
    """

    def __init__(self, hidden_dim, time_steps):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps

        # Attention layers
        self.W_1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, time_steps, hidden_dim]
        u = torch.tanh(self.W_1(x))  # [batch_size, time_steps, hidden_dim]
        att = self.W_2(u).squeeze(-1)  # [batch_size, time_steps]
        att_score = F.softmax(att, dim=1)  # [batch_size, time_steps]

        # Apply attention weights
        scored_x = x * att_score.unsqueeze(-1)  # [batch_size, time_steps, hidden_dim]
        context = torch.sum(scored_x, dim=1)  # [batch_size, hidden_dim]

        return context, att_score


class SpatialAttention(nn.Module):
    """
    Spatial attention for focusing on important nodes/edges
    """

    def __init__(self, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # Spatial attention layers
        self.W_s1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_s2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape for nodes: [batch_size, num_nodes, hidden_dim]
        # x shape for edges: [batch_size, num_nodes, num_nodes, hidden_dim]

        input_shape = x.shape
        if len(input_shape) == 4:  # Edge features
            batch_size, n, m, hidden = input_shape
            x_flat = x.view(batch_size, n * m, hidden)
        else:  # Node features
            x_flat = x  # [batch_size, num_nodes, hidden_dim]

        u = torch.tanh(self.W_s1(x_flat))  # [batch_size, num_nodes(*num_nodes), hidden_dim]
        att = self.W_s2(u).squeeze(-1)  # [batch_size, num_nodes(*num_nodes)]
        att_score = F.softmax(att, dim=1)  # [batch_size, num_nodes(*num_nodes)]

        # Reshape attention scores back to original dimensions if needed
        if len(input_shape) == 4:
            att_score = att_score.view(batch_size, n, m)

        return att_score


class STGNNEmbedding(nn.Module):
    """
    Spatio-Temporal Graph Neural Network embedding for dynamic routing problems
    with edge weights that change over time.
    """

    def __init__(self, num_nodes, time_horizon, hidden_dim=64, output_dim=128,
                 dropout=0.1, use_edge_features=True, use_node_positions=False):
        super(STGNNEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.time_horizon = time_horizon
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_edge_features = use_edge_features
        self.use_node_positions = use_node_positions

        # Initial feature embedding for edge weights
        self.edge_embedding = nn.Linear(1, hidden_dim)

        # Optional node position embedding
        if use_node_positions:
            self.node_pos_embedding = nn.Linear(2, hidden_dim)

        # Temporal convolution to process time dimension
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim, time_horizon)

        # Spatial attention for focusing on important edges
        self.spatial_attention = SpatialAttention(hidden_dim)

        # GNN layers for message passing
        self.gnn_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.gnn_layer2 = nn.Linear(hidden_dim, hidden_dim)

        # Final projection layer
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def temporal_conv_block(self, x):
        """Process features along the time dimension using temporal convolutions"""
        # x shape: [batch_size, num_nodes, num_nodes, time_horizon, hidden_dim]
        batch_size, n, m, t, h = x.shape

        # Reshape for temporal convolution
        x_reshaped = x.view(batch_size * n * m, t, h)
        x_reshaped = x_reshaped.transpose(1, 2)  # [batch_size*n*m, hidden_dim, time_horizon]

        # Apply temporal convolution
        x_conv = self.temporal_conv(x_reshaped)  # [batch_size*n*m, hidden_dim, time_horizon]

        # Reshape back
        x_conv = x_conv.transpose(1, 2)  # [batch_size*n*m, time_horizon, hidden_dim]
        x_conv = x_conv.view(batch_size, n, m, t, h)

        return x_conv

    def forward(self, edge_weights, node_positions=None):
        """
        Forward pass of the STGNN embedding

        Args:
            edge_weights: Tensor of shape [batch_size, num_nodes, num_nodes, time_horizon]
                          representing the edge weights over time
            node_positions: Optional tensor of shape [batch_size, num_nodes, 2]
                           representing 2D coordinates of nodes

        Returns:
            node_embeddings: Tensor of shape [batch_size, num_nodes, output_dim]
            edge_embeddings: Tensor of shape [batch_size, num_nodes, num_nodes, output_dim]
            graph_embedding: Tensor of shape [batch_size, output_dim] representing the entire graph
        """
        batch_size = edge_weights.shape[0]

        # Add channel dimension for edge weights and embed them
        edge_weights_expanded = edge_weights.unsqueeze(-1)  # [batch, n, n, t, 1]
        edge_features = self.edge_embedding(edge_weights_expanded)  # [batch, n, n, t, hidden]

        # Apply temporal convolution to process time dimension
        edge_features = self.temporal_conv_block(edge_features)  # [batch, n, n, t, hidden]

        # Apply temporal attention to aggregate time dimension
        # Reshape for temporal attention
        n = self.num_nodes
        edge_features_flat = edge_features.view(batch_size * n * n, self.time_horizon, self.hidden_dim)
        edge_features_temp, _ = self.temporal_attention(edge_features_flat)  # [batch*n*n, hidden]
        edge_features_agg = edge_features_temp.view(batch_size, n, n, self.hidden_dim)  # [batch, n, n, hidden]

        # Apply spatial attention to focus on important edges
        edge_attention = self.spatial_attention(edge_features_agg)  # [batch, n, n]
        edge_features_weighted = edge_features_agg * edge_attention.unsqueeze(-1)  # [batch, n, n, hidden]

        # Aggregate edge features to get node embeddings
        # Sum incoming edges for each node
        node_features = torch.sum(edge_features_weighted, dim=2)  # [batch, n, hidden]

        # Add optional node position embeddings
        if self.use_node_positions and node_positions is not None:
            node_pos_features = self.node_pos_embedding(node_positions)  # [batch, n, hidden]
            node_features = node_features + node_pos_features

        # Apply GNN layers (message passing)
        node_features = F.relu(self.gnn_layer1(node_features))  # [batch, n, hidden]
        node_features = F.relu(self.gnn_layer2(node_features))  # [batch, n, hidden]

        # Project to output dimension
        node_embeddings = self.output_projection(node_features)  # [batch, n, output_dim]

        # Create edge embeddings by combining source and target node embeddings
        # Expand node embeddings to create edge features
        source_nodes = node_embeddings.unsqueeze(2).expand(batch_size, n, n, self.output_dim)
        target_nodes = node_embeddings.unsqueeze(1).expand(batch_size, n, n, self.output_dim)

        # Combine with the existing edge features (project them first)
        edge_features_proj = self.output_projection(edge_features_agg)  # [batch, n, n, output_dim]
        edge_embeddings = source_nodes + target_nodes + edge_features_proj  # [batch, n, n, output_dim]

        # Create a graph-level embedding by pooling node embeddings
        graph_embedding = torch.mean(node_embeddings, dim=1)  # [batch, output_dim]

        # return node_embeddings, edge_embeddings, graph_embedding
        return node_embeddings


class TSPInitEmbedding(nn.Module):
    """Initial embedding for the Traveling Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(TSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        out = self.init_embed(td["locs"])
        return out


class MatNetInitEmbedding(nn.Module):
    """
    Preparing the initial row and column embeddings for MatNet.

    Reference:
    https://github.com/yd-kwon/MatNet/blob/782698b60979effe2e7b61283cca155b7cdb727f/ATSP/ATSP_MatNet/ATSPModel.py#L51


    """

    def __init__(self, embed_dim: int, mode: str = "RandomOneHot") -> None:
        super().__init__()

        self.embed_dim = embed_dim
        assert mode in {
            "RandomOneHot",
            "Random",
        }, "mode must be one of ['RandomOneHot', 'Random']"
        self.mode = mode

    def forward(self, td: TensorDict):
        # dmat = td["cost_matrix"]
        dmat = td["adj"][..., 0] if td["adj"].dim() == 4 else td["adj"]  # [b, r, c]
        b, r, c = dmat.shape

        row_emb = torch.zeros(b, r, self.embed_dim, device=dmat.device)

        if self.mode == "RandomOneHot":
            # MatNet uses one-hot encoding for column embeddings
            # https://github.com/yd-kwon/MatNet/blob/782698b60979effe2e7b61283cca155b7cdb727f/ATSP/ATSP_MatNet/ATSPModel.py#L60
            col_emb = torch.zeros(b, c, self.embed_dim, device=dmat.device)
            rand = torch.rand(b, c)
            rand_idx = rand.argsort(dim=1)
            b_idx = torch.arange(b)[:, None].expand(b, c)
            n_idx = torch.arange(c)[None, :].expand(b, c)
            col_emb[b_idx, n_idx, rand_idx] = 1.0

        elif self.mode == "Random":
            col_emb = torch.rand(b, c, self.embed_dim, device=dmat.device)
        else:
            raise NotImplementedError

        return row_emb, col_emb, dmat


class TDTSPMatInitEmbedding(MatNetInitEmbedding):
    # TODO: explore better embedding methods
    def __init__(self, embed_dim: int, mode: str = "RandomOneHot"):
        super(TDTSPMatInitEmbedding, self).__init__(embed_dim, mode)
        self.time_embedder = SinusoidalPosEmb(embed_dim)

    def forward(self, td: TensorDict):
        dmat = td["adj"]
        b, r, c, t = dmat.shape

        row_emb = torch.zeros(b * t, r, self.embed_dim, device=dmat.device)
        rand = torch.rand(b * t, r)
        rand_idx = rand.argsort(dim=1)
        b_idx = torch.arange(b * t)[:, None].expand(b * t, r)
        n_idx = torch.arange(r)[None, :].expand(b * t, r)
        row_emb[b_idx, n_idx, rand_idx] = 1.0
        row_emb = row_emb.reshape(b, t, r, self.embed_dim)

        col_emb = torch.zeros(b * t, c, self.embed_dim, device=dmat.device)
        rand = torch.rand(b * t, c)
        rand_idx = rand.argsort(dim=1)
        b_idx = torch.arange(b * t)[:, None].expand(b * t, c)
        n_idx = torch.arange(c)[None, :].expand(b * t, c)
        col_emb[b_idx, n_idx, rand_idx] = 1.0
        col_emb = col_emb.reshape(b, t, c, self.embed_dim)

        time_emb = self.time_embedder(torch.arange(t, device=dmat.device)).unsqueeze(0).expand(b, t, self.embed_dim)
        return row_emb, col_emb, time_emb, dmat

        # row_time_emb = torch.zeros(b, r * t, self.embed_dim, device=dmat.device)
        # if self.mode == "RandomOneHot":
        #     col_emb = torch.zeros(b, c, self.embed_dim, device=dmat.device)
        #     rand = torch.rand(b, c)
        #     rand_idx = rand.argsort(dim=1)
        #     b_idx = torch.arange(b)[:, None].expand(b, c)
        #     n_idx = torch.arange(c)[None, :].expand(b, c)
        #     col_emb[b_idx, n_idx, rand_idx] = 1.0
        # elif self.mode == "Random":
        #     col_emb = torch.rand(b, c, self.embed_dim, device=dmat.device)
        # else:
        #     raise NotImplementedError
        # return row_time_emb, col_emb, dmat.permute(0, 1, 3, 2).reshape(b, r * t, c)


class VRPInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - demand: demand of the customers
    """

    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 3):
        super(VRPInitEmbedding, self).__init__()
        node_dim = node_dim  # 3: x, y, demand
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        # [batch, 1, 2]-> [batch, 1, embed_dim]
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embed_dim]
        node_embeddings = self.init_embed(
            torch.cat((cities, td["demand"][..., None]), -1)
        )
        # [batch, n_city+1, embed_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class VRPTWInitEmbedding(VRPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 6):
        # node_dim = 6: x, y, demand, tw start, tw end, service time
        super(VRPTWInitEmbedding, self).__init__(embed_dim, linear_bias, node_dim)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        durations = td["durations"][..., 1:]
        time_windows = td["time_windows"][..., 1:, :]
        # embeddings
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (cities, td["demand"][..., None], time_windows, durations[..., None]), -1
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)


class SVRPInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 3):
        super(SVRPInitEmbedding, self).__init__()
        node_dim = node_dim  # 3: x, y, skill
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        # [batch, 1, 2]-> [batch, 1, embed_dim]
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embed_dim]
        node_embeddings = self.init_embed(torch.cat((cities, td["skills"]), -1))
        # [batch, n_city+1, embed_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class PCTSPInitEmbedding(nn.Module):
    """Initial embedding for the Prize Collecting Traveling Salesman Problems (PCTSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - expected_prize: expected prize for visiting the customers.
            In PCTSP, this is the actual prize. In SPCTSP, this is the expected prize.
        - penalty: penalty for not visiting the customers
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(PCTSPInitEmbedding, self).__init__()
        node_dim = 4  # x, y, prize, penalty
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    cities,
                    td["expected_prize"][..., None],
                    td["penalty"][..., 1:, None],
                ),
                -1,
            )
        )
        # batch, n_city+1, embed_dim
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class OPInitEmbedding(nn.Module):
    """Initial embedding for the Orienteering Problems (OP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - prize: prize for visiting the customers
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(OPInitEmbedding, self).__init__()
        node_dim = 3  # x, y, prize
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    cities,
                    td["prize"][..., 1:, None],  # exclude depot
                ),
                -1,
            )
        )
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out


class DPPInitEmbedding(nn.Module):
    """Initial embedding for the Decap Placement Problem (DPP), EDA (electronic design automation).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (cells)
        - probe: index of the (single) probe cell. We embed the euclidean distance from the probe to all cells.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(DPPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim // 2, linear_bias)  # locs
        self.init_embed_probe = nn.Linear(1, embed_dim // 2, linear_bias)  # probe

    def forward(self, td):
        node_embeddings = self.init_embed(td["locs"])
        probe_embedding = self.init_embed_probe(
            self._distance_probe(td["locs"], td["probe"])
        )
        return torch.cat([node_embeddings, probe_embedding], -1)

    def _distance_probe(self, locs, probe):
        # Euclidean distance from probe to all locations
        probe_loc = torch.gather(locs, 1, probe.unsqueeze(-1).expand(-1, -1, 2))
        return torch.norm(locs - probe_loc, dim=-1).unsqueeze(-1)


class MDPPInitEmbedding(nn.Module):
    """Initial embedding for the Multi-port Placement Problem (MDPP), EDA (electronic design automation).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (cells)
        - probe: indexes of the probe cells (multiple). We embed the euclidean distance of each cell to the closest probe.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(MDPPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)  # locs
        self.init_embed_probe_distance = nn.Linear(
            1, embed_dim, linear_bias
        )  # probe_distance
        self.project_out = nn.Linear(embed_dim * 2, embed_dim, linear_bias)

    def forward(self, td):
        probes = td["probe"]
        locs = td["locs"]
        node_embeddings = self.init_embed(locs)

        # Get the shortest distance from any probe
        dist = torch.cdist(locs, locs, p=2)
        dist[~probes] = float("inf")
        min_dist, _ = torch.min(dist, dim=1)
        min_probe_dist_embedding = self.init_embed_probe_distance(min_dist[..., None])

        return self.project_out(
            torch.cat([node_embeddings, min_probe_dist_embedding], -1)
        )


class PDPInitEmbedding(nn.Module):
    """Initial embedding for the Pickup and Delivery Problem (PDP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, pickups and deliveries separately)
           Note that pickups and deliveries are interleaved in the input.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(PDPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed_pick = nn.Linear(node_dim * 2, embed_dim, linear_bias)
        self.init_embed_delivery = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        depot, locs = td["locs"][..., 0:1, :], td["locs"][..., 1:, :]
        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        depot_embeddings = self.init_embed_depot(depot)
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)
        # concatenate on graph size dimension
        return torch.cat([depot_embeddings, pick_embeddings, delivery_embeddings], -2)


class MTSPInitEmbedding(nn.Module):
    """Initial embedding for the Multiple Traveling Salesman Problem (mTSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, cities)
    """

    def __init__(self, embed_dim, linear_bias=True):
        """NOTE: new made by Fede. May need to be checked"""
        super(MTSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)  # depot embedding

    def forward(self, td):
        depot_embedding = self.init_embed_depot(td["locs"][..., 0:1, :])
        node_embedding = self.init_embed(td["locs"][..., 1:, :])
        return torch.cat([depot_embedding, node_embedding], -2)


class SMTWTPInitEmbedding(nn.Module):
    """Initial embedding for the Single Machine Total Weighted Tardiness Problem (SMTWTP).
    Embed the following node features to the embedding space:
        - job_due_time: due time of the jobs
        - job_weight: weights of the jobs
        - job_process_time: the processing time of jobs
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(SMTWTPInitEmbedding, self).__init__()
        node_dim = 3  # job_due_time, job_weight, job_process_time
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        job_due_time = td["job_due_time"]
        job_weight = td["job_weight"]
        job_process_time = td["job_process_time"]
        feat = torch.stack((job_due_time, job_weight, job_process_time), dim=-1)
        out = self.init_embed(feat)
        return out


class MDCPDPInitEmbedding(nn.Module):
    """Initial embedding for the MDCPDP environment
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, pickups and deliveries separately)
           Note that pickups and deliveries are interleaved in the input.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(MDCPDPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed_pick = nn.Linear(node_dim * 2, embed_dim, linear_bias)
        self.init_embed_delivery = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        num_depots = td["capacity"].size(-1)
        depot, locs = td["locs"][..., 0:num_depots, :], td["locs"][..., num_depots:, :]
        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        depot_embeddings = self.init_embed_depot(depot)
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)
        # concatenate on graph size dimension
        return torch.cat([depot_embeddings, pick_embeddings, delivery_embeddings], -2)


class JSSPInitEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        linear_bias: bool = True,
        scaling_factor: int = 1000,
        num_op_feats=5,
    ):
        super(JSSPInitEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.scaling_factor = scaling_factor
        self.init_ops_embed = nn.Linear(num_op_feats, embed_dim, linear_bias)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.0)

    def _op_features(self, td):
        proc_times = td["proc_times"]
        mean_durations = proc_times.sum(1) / (proc_times.gt(0).sum(1) + 1e-9)
        feats = [
            mean_durations / self.scaling_factor,
            # td["lbs"] / self.scaling_factor,
            td["is_ready"],
            td["num_eligible"],
            td["ops_job_map"],
            td["op_scheduled"],
        ]
        return torch.stack(feats, dim=-1)

    def _init_ops_embed(self, td: TensorDict):
        ops_feat = self._op_features(td)
        ops_emb = self.init_ops_embed(ops_feat)
        ops_emb = self.pos_encoder(ops_emb, td["ops_sequence_order"])

        # zero out padded and finished ops
        mask = td["pad_mask"]  # NOTE dont mask scheduled - leads to instable training
        ops_emb[mask.unsqueeze(-1).expand_as(ops_emb)] = 0
        return ops_emb

    def forward(self, td):
        return self._init_ops_embed(td)


class FJSPInitEmbedding(JSSPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=False, scaling_factor: int = 100):
        super().__init__(embed_dim, linear_bias, scaling_factor)
        self.init_ma_embed = nn.Linear(1, self.embed_dim, bias=linear_bias)
        self.edge_embed = nn.Linear(1, embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict):
        ops_emb = self._init_ops_embed(td)
        ma_emb = self._init_machine_embed(td)
        edge_emb = self._init_edge_embed(td)
        # get edges between operations and machines
        # (bs, ops, ma)
        edges = td["ops_ma_adj"].transpose(1, 2)
        return ops_emb, ma_emb, edge_emb, edges

    def _init_edge_embed(self, td: TensorDict):
        proc_times = td["proc_times"].transpose(1, 2) / self.scaling_factor
        edge_embed = self.edge_embed(proc_times.unsqueeze(-1))
        return edge_embed

    def _init_machine_embed(self, td: TensorDict):
        busy_for = (td["busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        ma_embeddings = self.init_ma_embed(busy_for.unsqueeze(2))
        return ma_embeddings


class FJSPMatNetInitEmbedding(JSSPInitEmbedding):
    def __init__(
        self,
        embed_dim,
        linear_bias: bool = False,
        scaling_factor: int = 1000,
    ):
        super().__init__(embed_dim, linear_bias, scaling_factor)
        self.init_ma_embed = nn.Linear(1, self.embed_dim, bias=linear_bias)

    def _init_machine_embed(self, td: TensorDict):
        busy_for = (td["busy_until"] - td["time"].unsqueeze(1)) / self.scaling_factor
        ma_embeddings = self.init_ma_embed(busy_for.unsqueeze(2))
        return ma_embeddings

    def forward(self, td: TensorDict):
        proc_times = td["proc_times"]
        ops_emb = self._init_ops_embed(td)
        # encoding machines
        ma_emb = self._init_machine_embed(td)
        # edgeweights for matnet
        matnet_edge_weights = proc_times.transpose(1, 2) / self.scaling_factor
        return ops_emb, ma_emb, matnet_edge_weights


class MTVRPInitEmbedding(VRPInitEmbedding):
    def __init__(self, embed_dim, linear_bias=True, node_dim: int = 7):
        # node_dim = 7: x, y, demand_linehaul, demand_backhaul, tw start, tw end, service time
        super(MTVRPInitEmbedding, self).__init__(embed_dim, linear_bias, node_dim)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        demand_linehaul, demand_backhaul = (
            td["demand_linehaul"][..., 1:],
            td["demand_backhaul"][..., 1:],
        )
        service_time = td["service_time"][..., 1:]
        time_windows = td["time_windows"][..., 1:, :]
        # [!] convert [0, inf] -> [0, 0] if a problem does not include the time window constraint, do not modify in-place
        time_windows = torch.nan_to_num(time_windows, posinf=0.0)
        # embeddings
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    cities,
                    demand_linehaul[..., None],
                    demand_backhaul[..., None],
                    time_windows,
                    service_time[..., None],
                ),
                -1,
            )
        )
        return torch.cat((depot_embedding, node_embeddings), -2)
