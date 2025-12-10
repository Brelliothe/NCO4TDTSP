from typing import Optional, Dict, Any
import torch
from rl4co.envs import TDTSPEnv
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from pytorch_lightning.profilers import SimpleProfiler


logger = get_pylogger(__name__)


class ACOBaseline:
    """Ant Colony Optimization baseline for TDTSP Env."""

    def __init__(
            self,
            env: TDTSPEnv,
            n_ants: int = 20,
            n_iterations: int = 100,
            alpha: float = 1.0,
            beta: float = 2.0,
            rho: float = 0.1,
            q0: float = 0.9,
            initial_pheromone: float = 1.0,
            **kwargs
    ):
        """Initialize ACO baseline.

        Args:
            env: RL4CO environment
            n_ants: Number of ants
            n_iterations: Number of iterations
            alpha: Pheromone importance
            beta: Heuristic importance
            rho: Evaporation rate
            q0: Exploitation vs exploration parameter
            initial_pheromone: Initial pheromone value
        """
        self.env = env
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.initial_pheromone = initial_pheromone
        self.profiler = SimpleProfiler()

    def reset(self):
        """Reset the baseline state."""
        pass

    def _init_pheromone(self, batch_size: int, n_nodes: int) -> torch.Tensor:
        """Initialize pheromone matrix for each instance in batch."""
        return torch.ones(batch_size, n_nodes, n_nodes) * self.initial_pheromone

    def _get_heuristic(self, distances: torch.Tensor) -> torch.Tensor:
        """Calculate heuristic information (inverse of distance)."""
        return 1.0 / (distances + 1e-10)

    def _get_distances(self, adj: torch.Tensor, current: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Get distances between nodes."""
        # Assuming adj is a distance matrix
        time_idx = torch.floor(time)
        surplus = (time - time_idx).unsqueeze(-1)
        time_idx = time_idx.long() % adj.size(-1)
        next_idx = (time_idx + 1) % adj.size(-1)
        batch_idx = torch.arange(adj.size(0), device=adj.device)
        distances = adj[batch_idx, current, :, time_idx] * (1 - surplus) + adj[batch_idx, current, :, next_idx] * surplus
        return distances  # (batch_size, n_nodes)

    def _select_next_node(
            self,
            pheromone: torch.Tensor,
            heuristic: torch.Tensor,
            mask: torch.Tensor,
            current: torch.Tensor
    ) -> torch.Tensor:
        """Select next node using ACO transition rule.

        Args:
            pheromone: (batch_size, n_nodes, n_nodes) Pheromone matrix
            heuristic: (batch_size, n_nodes, n_nodes) Heuristic information
            mask: (batch_size, n_nodes) Available nodes mask
            current: (batch_size,) Current nodes

        Returns:
            (batch_size,) Selected next nodes
        """
        batch_size, n_nodes = mask.size()

        # Gather pheromone and heuristic values for current nodes
        current_pheromone = gather_by_index(pheromone, current)  # (batch_size, n_nodes)
        # current_heuristic = gather_by_index(heuristic, current)  # (batch_size, n_nodes)
        current_heuristic = heuristic  # (batch_size, n_nodes)

        # Calculate transition probabilities
        scores = (current_pheromone ** self.alpha) * (current_heuristic ** self.beta)
        scores = scores.masked_fill(~mask, -1e9)  # Mask visited nodes

        # Exploitation (choose best) vs Exploration (sample from distribution)
        if torch.rand(1) < self.q0:
            next_node = scores.argmax(dim=-1)
        else:
            probs = torch.softmax(scores, dim=-1)
            next_node = torch.multinomial(probs, 1).squeeze(-1)

        return next_node

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

        # Initialize pheromone matrix
        pheromone = self._init_pheromone(batch_size, n_nodes).to(device)
        with self.profiler.profile("env_reset"):
            # Reset environment for each ant
            init_td = self.env.reset(td=batch.clone(), batch_size=batch_size)

        best_tours = None
        best_lengths = torch.full((batch_size,), float('inf'), device=device)

        for iteration in range(self.n_iterations):
            # Run all ants
            for _ in range(self.n_ants):
                td = init_td.clone()
                # Initialize solution construction
                tours = torch.zeros(batch_size, n_nodes - 1, dtype=torch.long, device=device)

                # Construct tours
                for step in range(n_nodes - 1):
                    with self.profiler.profile("get_distances"):
                        # Get pheromone and heuristic information
                        distances = self._get_distances(td['adj'], td['current_node'], td['time'])  # (batch_size, n_nodes, n_nodes)
                    with self.profiler.profile("get_heuristic"):
                        heuristic = self._get_heuristic(distances)
                    with self.profiler.profile("select next node"):
                        next_node = self._select_next_node(pheromone, heuristic, td['action_mask'], td['current_node'])
                    # tours.append(next_node)
                    tours[:, step] = next_node.clone()
                    td.set('current_node', next_node.clone())
                    td.set('action_mask', td['action_mask'].scatter(-1, next_node.unsqueeze(-1).expand_as(td['action_mask']), 0))
                    td.set('time', td['time'] + distances[torch.arange(batch_size, device=device), next_node])
                    # td.set('action', next_node)
                    # with self.profiler.profile("env_step"):
                    #     td = self.env.step(td)['next']

                # Calculate tour lengths
                with self.profiler.profile('get lengths'):
                    distances = self._get_distances(td['adj'], td['current_node'], td['time'])  # (batch_size, n_nodes, n_nodes)
                    next_node = td['first_node'].clone()
                    lengths = td['time'] - td['start_time'] + distances[torch.arange(batch_size, device=device), next_node]
                    tour_tensor = torch.cat([td['first_node'].unsqueeze(1).clone(), tours.clone()], dim=1)  # (batch_size, n_nodes)
                # tour_tensor = torch.stack(tours, dim=1)  # (batch_size, n_nodes - 1)
                # with self.profiler.profile("get_reward"):
                #     lengths = -self.env.get_reward(batch, tour_tensor)
                # tour_tensor = torch.cat([td['first_node'].unsqueeze(1).clone(), tour_tensor.clone()], dim=1)  # (batch_size, n_nodes)

                # Update best solutions
                improve_mask = lengths < best_lengths
                if improve_mask.any():
                    if best_tours is None:
                        best_tours = tour_tensor.clone()
                    else:
                        best_tours[improve_mask] = tour_tensor[improve_mask].clone()
                    best_lengths[improve_mask] = lengths[improve_mask]

                # Update pheromone matrix
                with self.profiler.profile("update_pheromone"):
                    self._update_pheromone(pheromone, tour_tensor, lengths)

            # Evaporation
            pheromone *= (1 - self.rho)

        return best_tours, best_lengths

    def _update_pheromone(
            self,
            pheromone: torch.Tensor,
            tours: torch.Tensor,
            lengths: torch.Tensor
    ):
        """Update pheromone matrix based on constructed tours."""
        batch_size, tour_len = tours.size()

        # Calculate pheromone deposit
        deposit = 1.0 / lengths.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)

        # Update pheromone for each tour
        for i in range(tour_len - 1):
            current = tours[:, i]
            next_node = tours[:, i + 1]
            pheromone[torch.arange(batch_size), current, next_node] += deposit.squeeze()  # Asymmetric pheromone
            pheromone[torch.arange(batch_size), next_node, current] += deposit.squeeze()  # Asymmetric pheromone


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


class TACOBaseline(ACOBaseline):
    def __init__(
            self,
            env: TDTSPEnv,
            n_ants: int = 20,
            n_iterations: int = 100,
            alpha: float = 1.0,
            beta: float = 2.0,
            rho: float = 0.1,
            q0: float = 0.9,
            initial_pheromone: float = 1.0,
            **kwargs
    ):
        super().__init__(env, n_ants, n_iterations, alpha, beta, rho, q0, initial_pheromone)
        self.pheromone = None

    def _init_pheromone(self, batch_size: int, n_nodes: int, horizon: int = 1) -> torch.Tensor:
        return torch.ones(batch_size, n_nodes, n_nodes, horizon) * self.initial_pheromone

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
        horizon = batch['adj'].size(-1)
        device = batch['adj'].device

        # Initialize pheromone matrix
        self.pheromone = self._init_pheromone(batch_size, n_nodes, horizon=horizon).to(device)

        best_tours = None
        best_lengths = torch.full((batch_size,), float('inf'), device=device)

        for iteration in range(self.n_iterations):
            # Run all ants
            for _ in range(self.n_ants):
                td = self.env.reset(td=batch.clone(), batch_size=batch_size)
                # Initialize solution construction
                tours = []

                # Construct tours
                for step in range(n_nodes - 1):
                    distances = self.env.get_distances(td['locs'], td['time'])
                    heuristic = self._get_heuristic(distances)
                    pheromone = self._get_pheromone(td['time'])
                    next_node = self._select_next_node(pheromone, heuristic, td['action_mask'], td['current_node'])
                    tours.append(next_node)
                    td.set('action', next_node)
                    td = self.env.step(td)['next']

                # Calculate tour lengths
                tour_tensor = torch.stack(tours, dim=1)  # (batch_size, n_nodes - 1)
                lengths = -self.env.get_reward(batch, tour_tensor)
                tour_tensor = torch.cat([td['first_node'].unsqueeze(1).clone(), tour_tensor.clone()],
                                        dim=1)  # (batch_size, n_nodes)

                # Update best solutions
                improve_mask = lengths < best_lengths
                if improve_mask.any():
                    if best_tours is None:
                        best_tours = tour_tensor.clone()
                    else:
                        best_tours[improve_mask] = tour_tensor[improve_mask]
                    best_lengths[improve_mask] = lengths[improve_mask]

                # Update pheromone matrix
                self._update_pheromone(self.pheromone, tour_tensor, lengths)

                # Evaporation
            self.pheromone *= (1 - self.rho)

        return best_tours, best_lengths

    def _get_pheromone(self, time: torch.Tensor) -> torch.Tensor:
        batch_size = time.size(0)
        horizon = self.pheromone.size(-1)
        batch_indices = torch.arange(batch_size, device=time.device)
        return self.pheromone[batch_indices, :, :, torch.floor(time).long() % horizon]  # shape (batch_size, n_nodes, n_nodes)

    def _update_pheromone(
            self,
            pheromone: torch.Tensor,
            tours: torch.Tensor,
            lengths: torch.Tensor
    ):
        """Update pheromone matrix based on constructed tours."""
        batch_size, tour_len = tours.size()
        horizon = pheromone.size(-1)

        # Calculate pheromone deposit
        deposit = 1.0 / lengths.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
        time = torch.zeros(batch_size, device=tours.device)

        # Update pheromone for each tour
        for i in range(tour_len - 1):
            current = tours[:, i]
            next_node = tours[:, i + 1]
            time += self.env.get_distance(current, next_node, time)
            pheromone[torch.arange(batch_size), current, next_node, torch.floor(time).long() % horizon] += deposit.squeeze()