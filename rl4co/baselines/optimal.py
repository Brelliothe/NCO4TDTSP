from typing import Optional, Dict, Any
import torch
from rl4co.envs import TDTSPEnv
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from pytorch_lightning.profilers import SimpleProfiler
import gurobipy as gp
from gurobipy import GRB
import numpy as np


logger = get_pylogger(__name__)


class OptimalBaseline:
    """Optimal baseline for TDTSP Env, solving with dynamic programming."""

    def __init__(
            self,
            env: TDTSPEnv,
            dynamic=True
    ):
        """Initialize greedy baseline.

        Args:
            env: RL4CO environment
            dynamic: whether to solve the problem as dynamic TSP
        """
        self.env = env
        self.dynamic = dynamic

    def reset(self):
        """Reset the baseline state."""
        pass

    def _construct_solutions(
            self,
            batch: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct solutions for all ants.

        Args:
            batch: Dictionary containing problem instance data

        Returns:
            Tuple of (tours, tour_lengths)
        """

        batch_size = batch['adj'].size(0)
        n_nodes = batch['adj'].size(1)
        device = batch['adj'].device

        td = self.env.reset(td=batch, batch_size=batch_size)

        n_subsets = 1 << n_nodes

        # Initialize dp table [batch_size, 2^n_nodes, n_nodes]
        dp = torch.full((batch_size, n_subsets, n_nodes), float('inf'), device=device)

        # Initialize parent pointers to reconstruct the path
        parent = torch.zeros(batch_size, n_subsets, n_nodes, dtype=torch.int, device=device)

        # Construct tours
        start_node = 0
        dp[:, 0, start_node] = batch['start_time']

        # precompute the prefix sets
        prefix_sets = [[] for _ in range(1, n_nodes + 1)]
        visited = [[] for _ in range(n_subsets)]
        to_visit = [[] for _ in range(n_subsets)]
        for subset in range(1, n_subsets):
            subset_size = bin(subset).count('1')
            if subset_size > 1:
                visited[subset] = [i for i in range(n_nodes) if subset & (1 << i) > 0 and (i != start_node)]
            else:
                visited[subset] = [i for i in range(n_nodes) if subset & (1 << i) > 0]
            to_visit[subset] = [i for i in range(n_nodes) if subset & (1 << i) == 0]
            if subset & (1 << start_node) > 0:
                prefix_sets[subset_size - 1].append(subset)

        for size in range(1, n_nodes):
            for prefix in prefix_sets[size - 1]:
                for node in to_visit[prefix]:
                    for last_node in visited[prefix]:
                        new_prefix = prefix & ~(1 << last_node)
                        # print(prefix, new_prefix, last_node, node)
                        time = (dp[:, new_prefix, last_node] +
                                self.env.get_distance(td['locs'][:, last_node], td['locs'][:, node],
                                                      dp[:, new_prefix, last_node].clone())
                                if self.dynamic else
                                dp[:, new_prefix, last_node] +
                                self.env.get_distance(td['locs'][:, last_node], td['locs'][:, node],
                                                      batch['start_time']))  # [batch_size, ]
                        mask = time < dp[:, prefix, node]
                        dp[mask, prefix, node] = time[mask].clone()
                        parent[mask, prefix, node] = last_node

        # get the optimal tour length
        final_subset = (1 << n_nodes) - 1
        for node in range(1, n_nodes):
            prefix = final_subset & ~(1 << node)
            time = (dp[:, prefix, node] +
                    self.env.get_distance(td['locs'][:, node], td['locs'][:, start_node], dp[:, prefix, node].clone())
                    if self.dynamic else dp[:, prefix, node] +
                    self.env.get_distance(td['locs'][:, node], td['locs'][:, start_node], batch['start_time']))
            mask = time < dp[:, final_subset, start_node]
            parent[mask, final_subset, start_node] = node
            dp[mask, final_subset, start_node] = time[mask].clone()
        best_lengths = dp[:, final_subset, start_node] - batch['start_time']

        # reconstruct the optimal path by parent pointers
        best_tours = torch.zeros(batch_size, n_nodes, dtype=torch.int, device=device)
        prefixes = torch.full((batch_size,), final_subset, dtype=torch.int, device=device)
        best_tours[:, -1] = parent[:, final_subset, start_node]
        prefixes = prefixes & ~(1 << best_tours[:, -1])
        batch_indices = torch.arange(batch_size, device=device)
        for i in range(n_nodes - 2, 0, -1):
            best_tours[:, i] = parent[batch_indices, prefixes, best_tours[:, i + 1]]
            prefixes = prefixes & ~(1 << best_tours[:, i])

        if not self.dynamic:
            best_lengths = self.env.get_tour_length(td, best_tours)
        else:
            # assert (best_lengths == self.env.get_tour_length(td, best_tours)).all()
            pass

        return best_tours.clone(), best_lengths.clone()

    def solve(self, batch: TensorDict) -> Dict[str, Any]:
        """Solve a batch of problem instances.

        Args:
            batch: Dictionary containing problem instance data

        Returns:
            Dictionary containing solution information
        """
        tours, lengths = self._construct_solutions(batch)

        return {
            'tours': tours,  # (batch_size, n_nodes)
            'tour_lengths': lengths,  # (batch_size,)
        }


class SubOptimalBaseline(OptimalBaseline):
    def __init__(self, env: TDTSPEnv, dynamic=True):
        """Initialize sub-optimal baseline.

        Args:
            env: RL4CO environment
            dynamic: whether to solve the problem as dynamic TSP
        """
        super().__init__(env, False)


class FastSubOptimalBaseline(OptimalBaseline):
    def __init__(self, env: TDTSPEnv, dynamic=True):
        """Initialize fast sub-optimal baseline.

        Args:
            env: RL4CO environment
            dynamic: whether to solve the problem as dynamic TSP
        """
        super().__init__(env, False)
        self.profiler = SimpleProfiler()

    def _construct_solutions(
            self,
            batch: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct solutions for all ants.

        Args:
            batch: Dictionary containing problem instance data

        Returns:
            Tuple of (tours, tour_lengths)
        """

        batch_size = batch['adj'].size(0)
        n_nodes = batch['adj'].size(1)
        device = batch['adj'].device

        td = self.env.reset(td=batch, batch_size=batch_size)
        distances = td['adj'][..., 0]  # [batch_size, n_nodes, n_nodes]
        ones = torch.ones(batch_size, dtype=torch.long, device=device)
        batch_indices = torch.arange(batch_size, device=device)

        n_subsets = 1 << n_nodes

        # Initialize dp table [batch_size, 2^n_nodes, n_nodes]
        dp = torch.full((batch_size, n_subsets, n_nodes), float('inf'), device=device)

        # Initialize parent pointers to reconstruct the path
        parent = torch.zeros(batch_size, n_subsets, n_nodes, dtype=torch.int32, device=device)

        # Construct tours
        start_node = 0
        dp[:, 0, start_node] = batch['start_time']

        # precompute the prefix sets
        with self.profiler.profile("Init Subset"):
            prefix_sets = [[] for _ in range(1, n_nodes + 1)]
            visited = [[] for _ in range(n_subsets)]
            to_visit = [[] for _ in range(n_subsets)]
            for subset in range(1, n_subsets):
                subset_size = bin(subset).count('1')
                if subset_size > 1:
                    visited[subset] = [i for i in range(n_nodes) if subset & (1 << i) > 0 and (i != start_node)]
                else:
                    visited[subset] = [i for i in range(n_nodes) if subset & (1 << i) > 0]
                to_visit[subset] = [i for i in range(n_nodes) if subset & (1 << i) == 0]
                if subset & (1 << start_node) > 0:
                    prefix_sets[subset_size - 1].append(subset)


        with self.profiler.profile("DP"):
            with self.profiler.profile("DP Loop"):
                for size in range(1, n_nodes):
                    for prefix in prefix_sets[size - 1]:
                        for node in to_visit[prefix]:
                            for last_node in visited[prefix]:
                                new_prefix = prefix & ~(1 << last_node)
                                time = dp[:, new_prefix, last_node] + distances[batch_indices, ones * last_node, ones * node]  # [batch_size, ]
                                mask = time < dp[:, prefix, node]
                                dp[mask, prefix, node] = time[mask].clone()
                                parent[mask, prefix, node] = last_node

            # get the optimal tour length
            final_subset = (1 << n_nodes) - 1
            for node in range(1, n_nodes):
                prefix = final_subset & ~(1 << node)
                time = dp[:, prefix, node] + distances[batch_indices, ones * node, ones * start_node]
                mask = time < dp[:, final_subset, start_node]
                parent[mask, final_subset, start_node] = node
                dp[mask, final_subset, start_node] = time[mask].clone()

            # reconstruct the optimal path by parent pointers
            best_tours = torch.zeros(batch_size, n_nodes, dtype=torch.long, device=device)
            prefixes = torch.full((batch_size,), final_subset, dtype=torch.long, device=device)
            best_tours[:, -1] = parent[:, final_subset, start_node]
            prefixes = prefixes & ~(1 << best_tours[:, -1])
            batch_indices = torch.arange(batch_size, device=device)
            for i in range(n_nodes - 2, 0, -1):
                best_tours[:, i] = parent[batch_indices, prefixes, best_tours[:, i + 1]]
                prefixes = prefixes & ~(1 << best_tours[:, i])

            best_lengths = self.env.get_tour_length(td, best_tours)

        return best_tours.clone(), best_lengths.clone()


class ATSPBaseline:
    def __init__(self, env: TDTSPEnv):
        """Initialize fast sub-optimal baseline.

        Args:
            env: RL4CO environment
            dynamic: whether to solve the problem as dynamic TSP
        """
        self.env = env

    def solve(self, batch: TensorDict) -> Dict[str, Any]:
        "solve the problems with MILP one by one"
        batch_size = batch['adj'].size(0)
        n_nodes = batch['adj'].size(1)
        device = batch['adj'].device
        init_td = self.env.reset(td=batch.clone(), batch_size=batch_size)
        tours = []
        for i in range(batch_size):
            tour = self._construct_solution(init_td[i])
            tours.append(tour)
        tours = torch.stack(tours, dim=0)
        lengths = self.env.get_tour_length(init_td, tours)
        return {
            'tours': tours,  # (batch_size, n_nodes)
            'tour_lengths': lengths,  # (batch_size,)
        }

    def _construct_solution(self, td: TensorDict) -> torch.Tensor:
        "Construct solution for single instance, note that there is no batch_size dimension"
        n_nodes = td['adj'].size(0)
        model = gp.Model("atsp")
        model.setParam('OutputFlag', 0)
        edges = model.addMVar(shape=(n_nodes, n_nodes), vtype=GRB.BINARY, name="edges")
        model.addConstrs(edges[i, :] @ np.ones((n_nodes,)) == 1 for i in range(n_nodes))
        model.addConstrs(edges[:, i] @ np.ones((n_nodes,)) == 1 for i in range(n_nodes))
        flow = model.addMVar(shape=(n_nodes, n_nodes), vtype=GRB.CONTINUOUS, name="flow")
        model.addConstr(flow <= n_nodes * edges)
        model.addConstr(flow[0, :] @ np.ones((n_nodes,)) == n_nodes - 1)
        model.addConstrs(flow[:, i] @ np.ones((n_nodes,)) - flow[i, :] @ np.ones((n_nodes,)) >= 1 for i in range(1, n_nodes))
        model.setObjective((edges * td['adj'][..., 0].numpy()).sum(), GRB.MINIMIZE)
        model.optimize()
        tour = torch.zeros(n_nodes, dtype=torch.long, device=td['adj'].device)
        for i in range(1, n_nodes):
            for j in range(n_nodes):
                if edges[tour[i - 1], j].X > 0.5:
                    tour[i] = j
                    continue
        return tour
