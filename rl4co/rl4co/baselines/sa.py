from typing import Optional, Dict, Any
import torch
from rl4co.envs import TDTSPEnv
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from pytorch_lightning.profilers import SimpleProfiler


logger = get_pylogger(__name__)


class SimulatedAnnealingBaseline:
    def __init__(self, env: TDTSPEnv, max_iter: int = 1000, alpha: float = 0.7, initial_temp: float = 100):
        self.env = env
        self.max_iter = max_iter
        self.alpha = alpha
        self.initial_temp = initial_temp
        self.profiler = SimpleProfiler()

    def solve(self, batch: TensorDict) -> Dict[str, Any]:
        tours, lengths = self._construct_solutions(batch)
        return {
            "tours": tours,
            "tour_lengths": lengths
        }

    def _construct_solutions(self, batch: TensorDict) -> (torch.Tensor, torch.Tensor):
        # Initialize the tours and lengths
        td = self.env.reset(td=batch.clone(), batch_size=batch.shape[0])
        tours, lengths = [], []
        from tqdm import tqdm
        for i in tqdm(range(batch.shape[0])):
            tour, length = self._simulated_annealing(td[i])
            tours.append(tour)
            lengths.append(length)
        tours = torch.stack(tours, dim=0)
        lengths = torch.stack(lengths, dim=0)
        print(self.profiler.summary())
        return tours, lengths

    def _get_tour_length(self, batch: TensorDict, tour: torch.Tensor) -> torch.Tensor:
        # Gather the distances from the adjacency matrix
        horizon = batch["adj"].shape[-1]
        time = batch['start_time']
        for i in range(len(tour) - 1):
            idx = torch.floor(time).long()
            surplus = time - idx.float()
            time = time + batch['adj'][tour[i], tour[i + 1], idx % horizon] * (1 - surplus) + batch['adj'][tour[i], tour[i + 1], (idx + 1) % horizon] * surplus
        # Add the distance from the last node to the first node
        idx = torch.floor(time).long()
        surplus = time - idx.float()
        time = time + batch['adj'][tour[-1], tour[0], idx % horizon] * (1 - surplus) + batch['adj'][tour[-1], tour[0], (idx + 1) % horizon] * surplus
        return time - batch['start_time']

    def _simulated_annealing(self, batch: TensorDict) -> (torch.Tensor, torch.Tensor):
        num_nodes = batch["adj"].shape[0]
        # Initialize the tour
        tour = torch.arange(num_nodes, device=batch.device)
        torch.manual_seed(0)
        tour = tour[torch.randperm(num_nodes)]
        best_tour = tour.clone()
        with self.profiler.profile("get_tour_length"):
            best_length = self._get_tour_length(batch, tour)

        # Initialize the temperature
        temperature = self.initial_temp

        for _ in range(self.max_iter):
            for i in range(5):
                tour = best_tour.clone()
                tour_length = best_length.clone()
                for _ in range(num_nodes * num_nodes):
                    with self.profiler.profile("neighbor_solution"):
                        new_tour = self.neighbor_solution(tour.clone(), i)

                    # Calculate the length of the new solution
                    with self.profiler.profile("get_tour_length"):
                        new_length = self._get_tour_length(batch, new_tour)

                    # Accept or reject the new solution
                    with self.profiler.profile("accept"):
                        if self._accept(tour_length, new_length, temperature):
                            tour = new_tour.clone()
                            tour_length = new_length.clone()

                    if new_length < best_length:
                        best_tour = new_tour.clone()
                        best_length = new_length.clone()

            # Decrease the temperature
            temperature *= self.alpha

        return best_tour, best_length

    def _accept(self, old_length: torch.Tensor, new_length: torch.Tensor, temperature: float) -> bool:
        if new_length < old_length:
            return True
        else:
            delta = new_length - old_length
            acceptance_probability = torch.exp(-delta / temperature)
            return torch.rand(1).item() < acceptance_probability.item()

    def neighbor_solution(self, tour: torch.Tensor, method: int) -> torch.Tensor:
        if method == 0:
            new_tour = self.two_opt(tour.clone())
        elif method == 1:
            new_tour = self.three_opt(tour.clone())
        elif method == 2:
            new_tour = self.exchange(tour.clone())
        elif method == 3:
            new_tour = self.relocate(tour.clone())
        elif method == 4:
            new_tour = self.or_opt(tour.clone())
        else:
            raise ValueError("Invalid method")
        assert torch.unique(new_tour).shape[0] == new_tour.shape[0], f"The tour {tour} should be unique after operation {method}, which is {new_tour}"
        idx = (new_tour == 0).nonzero(as_tuple=True)[0]
        assert idx.shape[0] == 1, "There should be only one 0 in the tour {}".format(new_tour)
        return torch.cat([new_tour[idx:], new_tour[:idx[0]]], dim=0)

    def two_opt(self, tour):
        with self.profiler.profile("two_opt"):
            idx = torch.sort(torch.randperm(len(tour))[:2], 0)[0]
            tour[idx[0]: idx[1]] = torch.flip(tour[idx[0]: idx[1]], dims=[0])
            return tour

    def three_opt(self, tour):
        with self.profiler.profile("three_opt"):
            idx = torch.sort(torch.randperm(len(tour))[:3], 0)[0]
            if torch.rand(1).item() < 0.5:
                tour[idx[0]: idx[1]] = torch.flip(tour[idx[0]: idx[1]], dims=[0])
            if torch.rand(1).item() < 0.5:
                tour[idx[1]: idx[2]] = torch.flip(tour[idx[1]: idx[2]], dims=[0])
            if torch.rand(1).item() < 0.5:
                tour = torch.cat([tour[:idx[0]], tour[idx[1]: idx[2]], tour[idx[0]: idx[1]], tour[idx[2]:]])
            return tour

    def exchange(self, tour):
        with self.profiler.profile("exchange"):
            # Randomly select two indices and swap their values
            idx1, idx2 = torch.randint(0, len(tour), (2,))
            value1, value2 = tour[idx1].clone(), tour[idx2].clone()
            tour[idx1], tour[idx2] = value2, value1
            return tour

    def relocate(self, tour):
        with self.profiler.profile("relocate"):
            idx = torch.sort(torch.randperm(len(tour))[:2], 0)[0]
            tour[idx[0]: idx[1]] = torch.roll(tour[idx[0]: idx[1]], shifts=-1, dims=0)
            return tour

    def or_opt(self, tour):
        with self.profiler.profile("or_opt"):
            idx = torch.sort(torch.randperm(len(tour))[:2], 0)[0]
            tour[idx[0]: idx[1]] = torch.roll(tour[idx[0]: idx[1]], shifts=-2, dims=0)
            return tour


class BatchSABaseline(SimulatedAnnealingBaseline):
    def __init__(self, env: TDTSPEnv, max_iter: int = 50, alpha: float = 0.7, initial_temp: float = 100):
        super().__init__(env=env, max_iter=max_iter, alpha=alpha, initial_temp=initial_temp)

    def solve(self, batch: TensorDict) -> Dict[str, Any]:
        # Initialize the tours and lengths
        td = self.env.reset(td=batch.clone(), batch_size=batch.shape[0])
        tours, lengths = self._simulated_annealing(td)
        return {
            "tours": tours,
            "tour_lengths": lengths
        }

    def _simulated_annealing(self, batch: TensorDict) -> (torch.Tensor, torch.Tensor):
        batch_size = batch.shape[0]
        num_nodes = batch["adj"].shape[1]
        # Initialize the tour
        tour = torch.arange(num_nodes, device=batch.device).unsqueeze(0).expand(batch_size, -1)
        torch.manual_seed(0)
        best_tour = tour.clone()
        with self.profiler.profile("get_tour_length"):
            best_length = self.env.get_tour_length(batch, tour)

        # Initialize the temperature
        temperature = self.initial_temp
        history = []

        count = 0
        from tqdm import tqdm
        for _ in tqdm(range(self.max_iter)):
            update = False
            for i in range(5):
                tour = best_tour.clone()
                tour_length = best_length.clone()
                for _ in range(num_nodes * num_nodes):
                    with self.profiler.profile("neighbor_solution"):
                        new_tour = self.neighbor_solution(tour.clone(), i)

                    # Calculate the length of the new solution
                    with self.profiler.profile("get_tour_length"):
                        new_length = self.env.get_tour_length(batch, new_tour)

                    # Accept or reject the new solution
                    with self.profiler.profile("accept"):
                        mask = self._accept(tour_length, new_length, temperature)
                        tour[mask] = new_tour[mask].clone()
                        tour_length[mask] = new_length[mask].clone()

                    mask = new_length < best_length
                    best_tour[mask] = new_tour[mask].clone()
                    best_length[mask] = new_length[mask].clone()
                    if torch.max(mask.float()) > 0:
                        update = True

            history.append(best_length.mean().item())
            # Decrease the temperature
            temperature *= self.alpha
            if not update:
                count += 1
            if count > 2:
                break
        return best_tour, best_length

    def neighborhood_search(self, start_tour: torch.Tensor, start_length: torch.Tensor, batch: TensorDict, iters: int) -> (torch.Tensor, torch.Tensor):
        best_tour = start_tour.clone()
        best_length = start_length.clone()
        for _ in range(iters):
            for i in range(5):
                tour = best_tour.clone()
                tour_length = best_length.clone()
                for _ in range(tour.shape[-1] ** 2):
                    with self.profiler.profile("neighbor_solution"):
                        new_tour = self.neighbor_solution(tour.clone(), i)

                    # Calculate the length of the new solution
                    with self.profiler.profile("get_tour_length"):
                        new_length = self.env.get_tour_length(batch, new_tour)

                    # Accept or reject the new solution
                    with self.profiler.profile("accept"):
                        mask = new_length < tour_length
                        tour[mask] = new_tour[mask].clone()
                        tour_length[mask] = new_length[mask].clone()

                    mask = new_length < best_length
                    best_tour[mask] = new_tour[mask].clone()
                    best_length[mask] = new_length[mask].clone()
        return best_tour, best_length

    def _accept(self, old_length: torch.Tensor, new_length: torch.Tensor, temperature: float) -> torch.Tensor:
        mask1 = new_length < old_length
        mask2 = torch.rand_like(old_length) < torch.exp(-(new_length - old_length) / temperature)
        return mask1 | mask2

    def two_opt(self, tour):
        with self.profiler.profile("two_opt"):
            idx = torch.sort(torch.randperm(tour.shape[1])[:2], 0)[0]
            tour[:, idx[0]: idx[1]] = torch.flip(tour[:, idx[0]: idx[1]], dims=[1])
            return tour

    def three_opt(self, tour):
        with self.profiler.profile("three_opt"):
            idx = torch.sort(torch.randperm(tour.shape[1])[:3], 0)[0]
            if torch.rand(1).item() < 0.5:
                tour[:, idx[0]: idx[1]] = torch.flip(tour[:, idx[0]: idx[1]], dims=[1])
            if torch.rand(1).item() < 0.5:
                tour[:, idx[1]: idx[2]] = torch.flip(tour[:, idx[1]: idx[2]], dims=[1])
            if torch.rand(1).item() < 0.5:
                tour = torch.cat([tour[:, :idx[0]], tour[:, idx[1]: idx[2]], tour[:, idx[0]: idx[1]], tour[:, idx[2]:]], dim=1)
            return tour

    def exchange(self, tour):
        with self.profiler.profile("exchange"):
            # Randomly select two indices and swap their values
            idx1, idx2 = torch.randint(0, tour.shape[1], (2,))
            value1, value2 = tour[:, idx1].clone(), tour[:, idx2].clone()
            tour[:, idx1], tour[:, idx2] = value2, value1
            return tour

    def relocate(self, tour):
        with self.profiler.profile("relocate"):
            idx = torch.sort(torch.randperm(tour.shape[1])[:2], 0)[0]
            tour[:, idx[0]: idx[1]] = torch.roll(tour[:, idx[0]: idx[1]], shifts=-1, dims=1)
            return tour

    def or_opt(self, tour):
        with self.profiler.profile("or_opt"):
            idx = torch.sort(torch.randperm(tour.shape[1])[:2], 0)[0]
            tour[:, idx[0]: idx[1]] = torch.roll(tour[:, idx[0]: idx[1]], shifts=-2, dims=1)
            return tour

    def neighbor_solution(self, tour: torch.Tensor, method: int) -> torch.Tensor:
        if method == 0:
            new_tour = self.two_opt(tour.clone())
        elif method == 1:
            new_tour = self.three_opt(tour.clone())
        elif method == 2:
            new_tour = self.exchange(tour.clone())
        elif method == 3:
            new_tour = self.relocate(tour.clone())
        elif method == 4:
            new_tour = self.or_opt(tour.clone())
        else:
            raise ValueError("Invalid method")
        idx = torch.argmax((new_tour == 0).float(), dim=1)
        positions = torch.arange(tour.shape[-1], device=tour.device).unsqueeze(0).expand(tour.shape[0], -1)
        positions = (positions + idx.unsqueeze(1)) % tour.shape[1]
        new_tour = torch.gather(new_tour, 1, positions)
        return new_tour