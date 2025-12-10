from rl4co.envs import TDTSPEnv
from rl4co.baselines.aco import ACOBaseline, TACOBaseline
from rl4co.baselines.random import RandomBaseline
from rl4co.baselines.greedy import GreedyBaseline
from rl4co.baselines.sa import SimulatedAnnealingBaseline, BatchSABaseline
from rl4co.baselines.optimal import OptimalBaseline, SubOptimalBaseline, FastSubOptimalBaseline, ATSPBaseline
from rl4co.utils.ops import gather_by_index, unbatchify
import torch
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader
from tensordict.tensordict import TensorDict
from tqdm import tqdm
import numpy as np
import math
import os
import pickle

import numba as nb
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from argparse import ArgumentParser


def prepare_data(loc, num, prob_instance_num):
    device = torch.device("cpu")
    torch.random.manual_seed(0)
    env = TDTSPEnv(time_matrix_params={'data': loc, 'scale': 1, 'interpolate': "linear", 'downsample': 1000},
                   generator_params={"num_loc": num}).to(device)
    test_set = env.dataset(prob_instance_num)
    torch.save(test_set, f'{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}.pt')


def prepare_opt(loc, num, prob_instance_num, test_idx, total_idx):
    print(f"Starting {test_idx}/ {total_idx}.")
    device = torch.device("cpu")
    torch.random.manual_seed(0)
    env = TDTSPEnv(time_matrix_params={'data': loc, 'scale': 1, 'interpolate': "linear", 'downsample': 1000},
                   generator_params={"num_loc": num}).to(device)
    torch.serialization.add_safe_globals([TensorDictDataset])
    test_set = torch.load(f'{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}.pt')
    test_set_loader = DataLoader(test_set, batch_size=1, collate_fn=test_set.collate_fn, shuffle=False)
    matrix = env.matrix.matrix.numpy()
    horizon = env.matrix.matrix.shape[-1]
    n_nodes = num

    n_subsets = 1 << n_nodes
    start_node = 0

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

    @nb.njit
    def query_distance_numba(loc1, loc2, time, matrix, horizon):
        time_idx = math.floor(time)
        surplus = time - time_idx
        time_mod = time_idx % horizon
        time_next = (time_idx + 1) % horizon
        start_cost = matrix[loc1][loc2][time_mod]  # Use separate brackets for each dimension
        end_cost = matrix[loc1][loc2][time_next]  # Use separate brackets for each dimension
        return start_cost + (end_cost - start_cost) * surplus

    lengths = []

    start_prob_instance_num = max(test_idx * np.ceil(len(test_set_loader) / total_idx), 0)
    end_prob_instance_num = min((test_idx + 1) * np.ceil(len(test_set_loader) / total_idx), len(test_set_loader))
    for batch_idx, batch in tqdm(enumerate(test_set_loader)):
        if batch_idx < start_prob_instance_num or batch_idx >= end_prob_instance_num:
            continue
        td = env.reset(batch)
        locs = td["locs"][0].numpy().reshape(-1)
        dp = torch.full((n_subsets, n_nodes), float('inf'), device=device).numpy()
        dp[0][start_node] = 0
        for size in range(1, n_nodes):
            for prefix in prefix_sets[size - 1]:
                for node in to_visit[prefix]:
                    for last_node in visited[prefix]:
                        new_prefix = prefix & ~(1 << last_node)
                        arrival_time = dp[new_prefix, last_node] + query_distance_numba(locs[last_node], locs[node],
                                                                                        dp[new_prefix, last_node],
                                                                                        matrix, horizon)

                        if arrival_time < dp[prefix, node]:
                            dp[prefix, node] = arrival_time
        final_subset = (1 << n_nodes) - 1
        for node in range(1, n_nodes):
            prefix = final_subset & ~(1 << node)
            arrival_time = dp[prefix, node] + query_distance_numba(locs[node], locs[start_node], dp[prefix, node],
                                                                   matrix, horizon)
            if arrival_time < dp[final_subset, start_node]:
                dp[final_subset, start_node] = arrival_time
        lengths.append(dp[final_subset, start_node])

    os.makedirs(f"{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}", exist_ok=True)
    with open(f"{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}/{test_idx}_{total_idx}.pkl",
              "wb") as f:
        pickle.dump(lengths, f)


def prepare_tsp(loc, num, prob_instance_num, test_idx, total_idx):
    print(f"Starting {test_idx}/ {total_idx}.")
    device = torch.device("cpu")
    torch.random.manual_seed(0)
    env = TDTSPEnv(time_matrix_params={'data': loc, 'scale': 1, 'interpolate': "linear", 'downsample': 1000},
                   generator_params={"num_loc": num}).to(device)
    torch.serialization.add_safe_globals([TensorDictDataset])
    test_set = torch.load(f'{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}.pt')
    test_set_loader = DataLoader(test_set, batch_size=1, collate_fn=test_set.collate_fn, shuffle=False)
    matrix = env.matrix.matrix.numpy()
    horizon = env.matrix.matrix.shape[-1]
    n_nodes = num

    n_subsets = 1 << n_nodes
    start_node = 0

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

    @nb.njit
    def query_distance_numba(loc1, loc2, time, matrix, horizon):
        time_idx = math.floor(time)
        surplus = time - time_idx
        time_mod = time_idx % horizon
        time_next = (time_idx + 1) % horizon
        start_cost = matrix[loc1][loc2][time_mod]  # Use separate brackets for each dimension
        end_cost = matrix[loc1][loc2][time_next]  # Use separate brackets for each dimension
        return start_cost + (end_cost - start_cost) * surplus

    @nb.njit
    def query_distance_static(loc1, loc2, matrix):
        return matrix[loc1][loc2][0]


    lengths = []

    start_prob_instance_num = max(test_idx * np.ceil(len(test_set_loader) / total_idx), 0)
    end_prob_instance_num = min((test_idx + 1) * np.ceil(len(test_set_loader) / total_idx), len(test_set_loader))
    for batch_idx, batch in tqdm(enumerate(test_set_loader)):
        if batch_idx < start_prob_instance_num or batch_idx >= end_prob_instance_num:
            continue
        td = env.reset(batch)
        locs = td["locs"][0].numpy().reshape(-1)
        dp = torch.full((n_subsets, n_nodes), float('inf'), device=device).numpy()
        parent = np.full((n_subsets, n_nodes), 0, dtype=np.int32)
        dp[0][start_node] = 0
        for size in range(1, n_nodes):
            for prefix in prefix_sets[size - 1]:
                for node in to_visit[prefix]:
                    for last_node in visited[prefix]:
                        new_prefix = prefix & ~(1 << last_node)
                        arrival_time = dp[new_prefix, last_node] + query_distance_static(locs[last_node], locs[node], matrix)
                        if arrival_time < dp[prefix, node]:
                            dp[prefix, node] = arrival_time
                            parent[prefix, node] = last_node

        final_subset = (1 << n_nodes) - 1
        for node in range(1, n_nodes):
            prefix = final_subset & ~(1 << node)
            arrival_time = dp[prefix, node] + query_distance_static(locs[node], locs[start_node], matrix)
            if arrival_time < dp[final_subset, start_node]:
                dp[final_subset, start_node] = arrival_time
                parent[final_subset, start_node] = node

        # Reconstruct the tour
        best_tours = np.full((n_nodes,), 0, dtype=np.int32)
        prefixes = final_subset
        best_tours[-1] = parent[final_subset, start_node]
        prefixes = prefixes & ~(1 << best_tours[-1])
        for i in range(n_nodes - 2, 0, -1):
            best_tours[i] = parent[prefixes, best_tours[i + 1]]
            prefixes = prefixes & ~(1 << best_tours[i])
        # print(best_tours)
        length = 0
        for i in range(n_nodes - 1):
            length = length + query_distance_numba(locs[best_tours[i]], locs[best_tours[i + 1]], length, matrix, horizon)
        length = length + query_distance_numba(locs[best_tours[-1]], locs[start_node], length, matrix, horizon)
        lengths.append(length)

    os.makedirs(f"{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}", exist_ok=True)
    with open(f"{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}/tsp_{test_idx}_{total_idx}.pkl",
              "wb") as f:
        pickle.dump(lengths, f)


def summarize_opt(loc, num, prob_instance_num, total_idx):
    lengths = []
    for i in range(total_idx):
        with open(f"{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}/{i}_{total_idx}.pkl",
                  "rb") as f:
            lengths += pickle.load(f)
    lengths = np.array(lengths)
    np.save(f"{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}/lengths.npy", lengths)
    print(f"Mean: {np.mean(lengths)}, Std: {np.std(lengths)}, total: {len(lengths)}")
    print(f"Summary saved to {PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}/lengths.npy")


def summarize_tsp(loc, num, prob_instance_num, total_idx):
    lengths = []
    for i in range(total_idx):
        with open(f"{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}/tsp_{i}_{total_idx}.pkl",
                  "rb") as f:
            lengths += pickle.load(f)
    lengths = np.array(lengths)
    np.save(f"{PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}/tsp_lengths.npy", lengths)
    print(f"Mean: {np.mean(lengths)}, Std: {np.std(lengths)}, total: {len(lengths)}")
    print(f"Summary saved to {PROJECT_ROOT}/rl4co/testcases/{loc}_{num}_dataset_{prob_instance_num}/tsp_lengths.npy")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--project_root", type=str)
    parser.add_argument("--city", type=str)
    parser.add_argument("--num_points", type=int)
    parser.add_argument("--prob_instance_num", type=int)
    parser.add_argument("--mode", type=str)

    parser.add_argument("--test_idx", type=int)
    parser.add_argument("--total_idx", type=int)
    args = parser.parse_args()

    PROJECT_ROOT = args.project_root

    if args.mode == "opt":
        prepare_opt(args.city, args.num_points, args.prob_instance_num, args.test_idx, args.total_idx)
    elif args.mode == "data":
        print(
            f"Preparing {args.mode} for {args.city} with {args.num_points} points and {args.prob_instance_num} instances")
        prepare_data(args.city, args.num_points, args.prob_instance_num)
    elif args.mode == 'tsp':
        prepare_tsp(args.city, args.num_points, args.prob_instance_num, args.test_idx, args.total_idx)
    else:
        assert args.mode == "summarize"
        print(
            f"Summarizing {args.mode} for {args.city} with {args.num_points} points and {args.prob_instance_num} instances")
        # summarize_opt(args.city, args.num_points, args.prob_instance_num, args.total_idx)
        summarize_tsp(args.city, args.num_points, args.prob_instance_num, args.total_idx)