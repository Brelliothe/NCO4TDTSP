from typing import Optional, Dict, Any
import torch
from rl4co.envs import TDTSPEnv
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict


logger = get_pylogger(__name__)


class RandomBaseline:
    """Random baseline for TDTSP Env."""

    def __init__(
            self,
            env: TDTSPEnv,
            **kwargs
    ):
        """Initialize random baseline.

        Args:
            env: RL4CO environment
        """
        self.env = env

    def reset(self):
        """Reset the baseline state."""
        pass

    def _select_next_node(
            self,
            mask: torch.Tensor
    ) -> torch.Tensor:
        """Select next node randomly.

        Args:
            mask: (batch_size, n_nodes) Available nodes mask
            current: (batch_size,) Current nodes

        Returns:
            (batch_size,) Selected next nodes
        """
        batch_size, n_nodes = mask.size()

        # Calculate transition probabilities
        scores = torch.ones(batch_size, n_nodes).to(mask.device)
        scores = scores.masked_fill(~mask, -1e9)  # Mask visited nodes
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

        best_tours = []
        best_lengths = torch.full((batch_size,), float('inf'), device=device)


        td = self.env.reset(td=batch, batch_size=batch_size)

        # Construct tours
        for step in range(n_nodes - 1):
            next_node = self._select_next_node(td['action_mask'])
            best_tours.append(next_node)
            td.set('action', next_node)
            td = self.env.step(td)['next']

        # Calculate tour lengths
        tour_tensor = torch.stack(best_tours, dim=1)  # (batch_size, n_nodes - 1)
        best_lengths = -self.env.get_reward(td, tour_tensor)
        best_tours = torch.cat([td['first_node'].unsqueeze(1).clone(), tour_tensor.clone()], dim=1)  # (batch_size, n_nodes)

        return best_tours, best_lengths

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
