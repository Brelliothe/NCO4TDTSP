import numpy as np
import torch
import math
from scipy.interpolate import CubicSpline
import json
import os
from rl4co.envs.routing import TDTSPEnv
from rl4co.envs.routing.tdtsp.generator import TDTSPGenerator
from rl4co.baselines.optimal import OptimalBaseline, SubOptimalBaseline
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tensordict.tensordict import TensorDict


def downsample(matrix, samples):
    # downsample the matrix to the given number of samples
    num_nodes, _, horizon = matrix.shape
    sampled = [0]
    for i in range(1, samples):
        furthest, furthest_distance = -1, 0
        for j in range(num_nodes):
            if j in sampled:
                continue
            distance = float('inf')
            for node in sampled:
                distance = min(distance, matrix[node, j, 0])
            if distance > furthest_distance:
                furthest = j
                furthest_distance = distance
        sampled.append(furthest)
    sampled = sorted(sampled)
    new_matrix = np.zeros((len(sampled), len(sampled), horizon))
    for i in range(len(sampled)):
        for j in range(len(sampled)):
            new_matrix[i, j] = matrix[sampled[i], sampled[j]]
    return new_matrix


def extract_beijing():
    filename = '~/ACO/data.csv'
    target_name = '~/rl4co/data/tdtsp/beijing.npy'
    if os.path.isfile(target_name):
        print(f"{target_name} already exists, skip")
        return
    print('extracting matrix from', filename)

    raw_data = np.loadtxt(filename, delimiter=',', skiprows=0)
    num_nodes = int(math.sqrt(raw_data.shape[0]))
    assert num_nodes * num_nodes == raw_data.shape[0], "The number of nodes must be a perfect square"
    horizon = raw_data.shape[1]
    matrix = np.zeros((num_nodes, num_nodes, horizon))
    for i, line in enumerate(raw_data):
        matrix[i // num_nodes, i % num_nodes] = line[:horizon] * raw_data.shape[1]

    print(f'saving matrix to {target_name}')
    np.save(target_name, matrix)


# this file tries to analyze the distribution of dataset
def extract_matrix(filename):
    target_name = filename.split('.')[0] + ".npy"
    # if os.path.isfile(target_name):
    #     print(f"{target_name} already exists, skip")
    #     return
    print('extracting matrix from', filename)
    # the function is used to get matrix from json file and save it for later usage
    with open(filename, 'r') as f:
        data = json.load(f)
    num_nodes = int(math.sqrt(len(data)))
    assert num_nodes * num_nodes == len(data), "The number of nodes must be a perfect square"

    # valid that the time matrix is collected with 3pm to 10pm
    horizon = 0
    leaving_time_collect = []
    for d in data:
        leaves = d['atf']['atf_leave_time']
        horizon = max(horizon, len(leaves))
        leaving_time_collect += leaves
    leaving_time_collect = list(set(leaving_time_collect))
    leaving_time_collect.remove(0)
    assert min(leaving_time_collect) == 54000000, "the minimum leaving time must be 54000000"
    assert max(leaving_time_collect) == 79200000, "the maximum leaving time must be 79200000"

    def query(time, leave_times, arrive_times):
        time = 54000000 + time * 10 * 60 * 1000  # convert to milliseconds
        for index, value in enumerate(leave_times):
            if value > time:
                pre_start = leave_times[index - 1]
                pre_cost = arrive_times[index - 1] - leave_times[index - 1]
                post_start = value
                post_cost = arrive_times[index] - leave_times[index]
                # linear interpolation
                cost = pre_cost + (post_cost - pre_cost) * (time - pre_start) / (post_start - pre_start)
                return cost
        # if time is greater than all leaving times, return the last cost
        return arrive_times[-1] - leave_times[-1]

    matrix = np.zeros((num_nodes, num_nodes, 43))  # sample every 10 minutes
    for d in data:
        start = int(d['from']) if d['from'] != 'depot' else 0
        end = int(d['to']) if d['to'] != 'depot' else 0
        leave = d['atf']['atf_leave_time']
        arrive = d['atf']['atf_arrive_time']
        for i in range(43):
            matrix[start, end, i] = query(i, leave, arrive) / 1000 / 60 / 10  # convert to 10 minutes

    matrix = downsample(matrix, 100)
    print('saving matrix to', target_name)
    np.save(target_name, matrix)


def analyze_data(env, horizon, step):
    size = 1000
    tsp = SubOptimalBaseline(env)
    opt = OptimalBaseline(env)
    dataset = env.dataset(size)
    dataset_loader = DataLoader(dataset, batch_size=1024, collate_fn=dataset.collate_fn)

    gap = torch.zeros(size, horizon // step, dtype=torch.float32)
    for t in tqdm(range(0, horizon, step)):
        tsp_length, opt_length = [], []
        locs = []
        for batch in dataset_loader:
            td = env.reset(TensorDict({'locs': batch['locs'], 'start_time': torch.ones(batch['locs'].shape[0]) * t}, batch_size=batch['locs'].shape[0], device=batch['locs'].device))
            solution = tsp.solve(td.clone())
            solution2 = opt.solve(td.clone())
            tsp_length.append(solution['tour_lengths'])
            opt_length.append(solution2['tour_lengths'])
            locs.append(td['locs'])
        locs = torch.cat(locs, dim=0)
        tsp_length = torch.cat(tsp_length, dim=0)
        opt_length = torch.cat(opt_length, dim=0)
        gap[:, t // step] = torch.nn.functional.relu(tsp_length - opt_length) / tsp_length
        assert gap[:, t // step].min() >= 0, \
            f"TSP gives tour length {tsp_length[gap[:, t // step] < 0]} smaller than optimal {opt_length[gap[:, t // step] < 0]} at time {t} on case {locs[gap[:, t // step] < 0]}"
    assert gap.min() >= 0, "The gap should be non-negative"
    return gap


def plot(gap, city):
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    fontsize = 40
    parameters = {
        'font.family': 'cmr10',
        'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize,
        'axes.axisbelow': True
    }
    plt.rcParams.update(parameters)
    colors = sns.color_palette()

    half = gap.shape[0] // 2
    gap = gap.sort(dim=0)[0]
    print(gap.mean(dim=0))
    upper = gap.max(dim=1).values[half:].numpy() * 100
    print(upper.mean())
    lower = gap.min(dim=1).values[half:].numpy() * 100
    print(lower.mean())
    # for t in range(gap.shape[1]):
    #     plt.plot(np.arange(0.5, 1, 1 / gap.shape[0]), torch.sort(gap[half:, t] * 100, dim=0)[0].cpu().numpy())
    plt.plot(np.arange(0.5, 1, 1 / gap.shape[0]), upper, label='UB', color=colors[0])
    plt.plot(np.arange(0.5, 1, 1 / gap.shape[0]), lower, label='LB', color=colors[1])
    plt.fill_between(np.arange(0.5, 1, 1 / gap.shape[0]), upper, lower, alpha=0.2, color=colors[2])
    plt.xlabel("Percentile")
    plt.ylabel("Savings (%)")
    if '_' in city:
        name = ' '.join([x[0].upper() + x[1:] for x in city.split('_')])
    else:
        name = city[0].upper() + city[1:]
    plt.title(f"{name}")
    plt.tight_layout()
    # plt.savefig(f'~/figures/{name}.pdf')
    plt.show()
    plt.close()


def analyze_benchmark(city):
    dir_path = "~/vrptdt-benchmark-main/instances/"
    json_file = os.path.join(dir_path, city + "_1000_tt.json")
    npy_file = os.path.join(dir_path, city + "_1000_tt.npy")
    assert os.path.isfile(json_file), f"{json_file} does not exist"
    if not os.path.isfile(npy_file) or True:
        extract_matrix(json_file)

    env = TDTSPEnv(
        time_matrix_params={"data": city, "downsample": 1000},
        generator_params={"num_loc": 10},
    )
    gap = analyze_data(env, 25, 1)
    plot(gap, city)


def analyze_beijing():
    filename = '~/ACO/data.csv'
    env = TDTSPEnv(time_matrix_params={'data': 'beijing'}, generator_params={"num_loc": 10})
    gap = analyze_data(env, 10, 1)
    print(gap.mean(dim=0), torch.topk(gap, k=int(gap.shape[0] * 0.2), largest=True, dim=0).values.mean(dim=0))
    plot(gap, 'beijing')


def extract_lyon():
    dir_path = '~/TDTSPBenchmark/Matrices/'
    filename = dir_path + 'matrix20.txt'
    target_name = dir_path + 'lyon.npy'
    # if os.path.isfile(target_name):
    #     print(f"{target_name} already exists, skip")
    #     return

    with open(filename, 'r') as f:
        lines = f.readlines()

    num_nodes, horizon, duration = lines[0].split()
    num_nodes = int(num_nodes)
    horizon = int(horizon)
    duration = int(duration)

    matrix = np.zeros((num_nodes, num_nodes, horizon))
    for i in range(num_nodes):
        for j in range(num_nodes):
            data = lines[i * (num_nodes + 1) + j + 1].split()
            assert len(data) == horizon, f"from {i} to {j} only has {len(data)} time steps, less than horizon {horizon}: {lines[i * num_nodes + j + 1]}"
            for t in range(horizon):
                matrix[i, j, t] = int(data[t]) / duration

    print(f'saving matrix to {target_name}')
    np.save(target_name, matrix)


def analyze_lyon():
    extract_lyon()
    env = TDTSPEnv(
        time_matrix_params={"data": 'lyon', "downsample": 1000},
        generator_params={"num_loc": 5},
    )
    gap = analyze_data(env, 90, 6)
    plot(gap, 'lyon')


# analyze_lyon()
# analyze_beijing()
for location in [
     'berlin',
#      'cincinnati',
#      'kyiv',
#      'london',
#      'madrid',
#      'nairobi',
#      'new_york',
#      'san_francisco',
#      'sao_paulo',
#      'seattle'
    ]:
    analyze_benchmark(location)