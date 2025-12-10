# this file contains the experiments for the TMatNet
from rl4co.envs import TDTSPEnv
from rl4co.baselines.aco import ACOBaseline, TACOBaseline
from rl4co.baselines.random import RandomBaseline
from rl4co.baselines.greedy import GreedyBaseline
from rl4co.baselines.sa import SimulatedAnnealingBaseline, BatchSABaseline
from rl4co.baselines.optimal import OptimalBaseline, SubOptimalBaseline, FastSubOptimalBaseline, ATSPBaseline
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.models.zoo import TimeMatNet, MatNet, MatNetREINFORCE
from pytorch_lightning.profilers import SimpleProfiler
import torch
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader
from tensordict.tensordict import TensorDict
from tqdm import tqdm


def eval_matnet(name, dataloader, env):
    matnet = MatNet.load_from_checkpoint(f"~/rl4co/ckpts/{name}").policy.to(env.device)
    matnet.eval()
    matnet_lengths = []
    matnet_tours = []
    for batch in tqdm(dataloader):
        env.pomo = True
        td = env.reset(batch.clone().to(env.device))
        env.pomo = False
        env.matrix.static = True
        rl_solution = matnet(td, env, phase="test", num_starts=10)
        env.matrix.static = False
        reward = unbatchify(rl_solution["reward"], (0, 10))
        actions = unbatchify(rl_solution["actions"], (0, 10))
        max_reward, max_idxs = reward.max(dim=-1)
        actions = gather_by_index(actions, max_idxs, dim=max_idxs.dim())
        idx = torch.argmax((actions == 0).float(), dim=1)
        positions = torch.arange(actions.shape[-1], device=actions.device).unsqueeze(0).expand(actions.shape[0], -1)
        positions = (positions + idx.unsqueeze(1)) % actions.shape[1]
        actions = torch.gather(actions, 1, positions)
        matnet_lengths.append(env.get_tour_length(td, actions))
        matnet_tours.append(actions)
    matnet_lengths = torch.cat(matnet_lengths, dim=0)
    matnet_tours = torch.cat(matnet_tours, dim=0)
    return matnet_tours, matnet_lengths


def eval_matnet_reinforce(name, dataloader, env):
    matnet = MatNetREINFORCE.load_from_checkpoint(f"~/rl4co/ckpts/{name}").policy.to(env.device)
    matnet.eval()
    matnet_lengths = []
    matnet_tours = []
    for batch in tqdm(dataloader):
        td_init = env.reset(batch.clone().to(env.device))
        rl_solution = matnet(td_init, env, phase="test", decode_type="greedy")
        matnet_lengths.append(-rl_solution["reward"])
        matnet_tours.append(torch.cat([torch.zeros((rl_solution['actions'].shape[0], 1), dtype=torch.long, device=env.device), rl_solution["actions"]], dim=1))
    matnet_lengths = torch.cat(matnet_lengths, dim=0)
    matnet_tours = torch.cat(matnet_tours, dim=0)
    return matnet_tours, matnet_lengths


def eval_tmatnet(name, dataloader, env):
    tmatnet = TimeMatNet.load_from_checkpoint(f"~/rl4co/ckpts/{name}").policy.to(env.device)
    tmatnet.eval()
    tmatnet_lengths = []
    tmatnet_tours = []
    for batch in tqdm(dataloader):
        td_init = env.reset(batch.clone().to(env.device))
        rl_solution = tmatnet(td_init, env, phase="test", decode_type="greedy")
        tmatnet_lengths.append(-rl_solution["reward"])
        tmatnet_tours.append(torch.cat([torch.zeros((rl_solution['actions'].shape[0], 1), dtype=torch.long, device=env.device), rl_solution["actions"]], dim=1))
    tmatnet_lengths = torch.cat(tmatnet_lengths, dim=0)
    tmatnet_tours = torch.cat(tmatnet_tours, dim=0)
    return tmatnet_tours, tmatnet_lengths


def eval_policy(name, dataset, env):
    policy_set = {"ACO": ACOBaseline,
                  "TACO": TACOBaseline,
                  "Random": RandomBaseline,
                  "Greedy": GreedyBaseline,
                  # "SA": SimulatedAnnealingBaseline,
                  "SA": BatchSABaseline,
                  "TSP": FastSubOptimalBaseline,
                  # "TSP": ATSPBaseline,
                  "Optimal": OptimalBaseline}
    assert name in policy_set, f"Policy {name} not implemented, only support {policy_set.keys()} now"
    policy = policy_set[name](env)
    tours, tour_lengths = [], []
    for batch in tqdm(dataset):
        td = env.reset(batch)
        # td.set("start_time", torch.zeros(batch.batch_size, device=td.device))
        solution = policy.solve(td.clone())
        tour_lengths.append(solution["tour_lengths"])
        tours.append(solution["tours"])
    tour_lengths = torch.cat(tour_lengths, dim=0)
    tours = torch.cat(tours, dim=0)
    print(f"{name} Tour lengths: {tour_lengths.mean().item()}")
    return tours, tour_lengths


def moe(tours_a, lengths_a, tours_b, lengths_b):
    mask = lengths_a < lengths_b
    expanded_mask = mask.unsqueeze(1).expand_as(tours_a)
    selected_tours = torch.where(expanded_mask, tours_a, tours_b)
    selected_lengths = torch.where(mask, lengths_a, lengths_b)
    return selected_tours, selected_lengths


def local_search(tours, lengths, env, dataloader, iters):
    searcher = BatchSABaseline(env)
    idx = 0
    improved_tours, improved_tour_lengths = [], []
    for batch in tqdm(dataloader):
        td = env.reset(batch.clone())
        end_idx = min(idx + 1024, tours.shape[0])
        improved_tour, improved_tour_length = searcher.neighborhood_search(tours[idx:end_idx], lengths[idx:end_idx], td, iters=iters)
        improved_tours.append(improved_tour)
        improved_tour_lengths.append(improved_tour_length)
        idx = end_idx
    improved_tours = torch.cat(improved_tours, dim=0)
    improved_tour_lengths = torch.cat(improved_tour_lengths, dim=0)
    return improved_tours, improved_tour_lengths


def eval_moe_search(iters):
    device = torch.device("cpu")
    torch.random.manual_seed(0)
    env = TDTSPEnv(time_matrix_params={'data': 'beijing', 'scale': 1, 'interpolate': "linear", 'downsample': 1000},
                   generator_params={"num_loc": 10}).to(device)
    test_set = env.dataset(10000)
    test_set_loader = DataLoader(test_set, batch_size=1024, collate_fn=test_set.collate_fn)

    # do the test
    opt_tours, opt_tour_lengths = eval_policy("Optimal", test_set_loader, env)

    # atsp baseline
    tsp_tours, tsp_tour_lengths = eval_policy("TSP", test_set_loader, env)
    print(f"tsp_lengths: {tsp_tour_lengths.mean().item()}, tsp_gaps: {((tsp_tour_lengths - opt_tour_lengths) / opt_tour_lengths).mean().item()}")
    matnet_tours, matnet_tour_lengths = eval_matnet("matnet-beijing-10-pomo.ckpt", test_set_loader, env)
    print(f"matnet_lengths: {matnet_tour_lengths.mean().item()}, matnet_gaps: {((matnet_tour_lengths - opt_tour_lengths) / opt_tour_lengths).mean().item()}")

    # raw tmatnet
    tmatnet_tours, tmatnet_tour_lengths = eval_tmatnet("tmatnet-beijing-10-matnet-pomo.ckpt", test_set_loader, env)
    print(f"tmatnet_lengths: {tmatnet_tour_lengths.mean().item()}, tmatnet_gaps: {((tmatnet_tour_lengths - opt_tour_lengths) / opt_tour_lengths).mean().item()}")
    improved_tmatnet_tours, improved_tmatnet_tour_lengths = local_search(tmatnet_tours, tmatnet_tour_lengths, env, test_set_loader, iters=iters)
    print(f"improved_tmatnet_lengths: {improved_tmatnet_tour_lengths.mean().item()}, improved_tmatnet_gaps: {((improved_tmatnet_tour_lengths - opt_tour_lengths) / opt_tour_lengths).mean().item()}")

    # moe with tsp
    moe_tsp_tours, moe_tsp_tour_lengths = moe(tsp_tours, tsp_tour_lengths, tmatnet_tours, tmatnet_tour_lengths)
    print(f"moe_tsp_lengths: {moe_tsp_tour_lengths.mean().item()}, moe_tsp_gaps: {((moe_tsp_tour_lengths - opt_tour_lengths) / opt_tour_lengths).mean().item()}")
    improved_moe_tsp_tours, improved_moe_tsp_tour_lengths = local_search(moe_tsp_tours, moe_tsp_tour_lengths, env, test_set_loader, iters=iters)
    print(f"improved_moe_tsp_lengths: {improved_moe_tsp_tour_lengths.mean().item()}, improved_moe_tsp_gaps: {((improved_moe_tsp_tour_lengths - opt_tour_lengths) / opt_tour_lengths).mean().item()}")

    # moe with matnet
    moe_matnet_tours, moe_matnet_tour_lengths = moe(matnet_tours, matnet_tour_lengths, tmatnet_tours, tmatnet_tour_lengths)
    print(f"moe_matnet_lengths: {moe_matnet_tour_lengths.mean().item()}, moe_matnet_gaps: {((moe_matnet_tour_lengths - opt_tour_lengths) / opt_tour_lengths).mean().item()}")
    improved_moe_matnet_tours, improved_moe_matnet_tour_lengths = local_search(moe_matnet_tours, moe_matnet_tour_lengths, env, test_set_loader, iters=iters)
    print(f"improved_moe_matnet_lengths: {improved_moe_matnet_tour_lengths.mean().item()}, improved_moe_matnet_gaps: {((improved_moe_matnet_tour_lengths - opt_tour_lengths) / opt_tour_lengths).mean().item()}")


def eval_training():
    device = torch.device("cpu")
    torch.random.manual_seed(0)
    env = TDTSPEnv(time_matrix_params={'data': 'beijing', 'scale': 1, 'interpolate': "linear", 'downsample': 1000},
                   generator_params={"num_loc": 10}).to(device)
    test_set = env.dataset(10000)
    test_set_loader = DataLoader(test_set, batch_size=1024, collate_fn=test_set.collate_fn, shuffle=False)

    reinforce_tours, reinforce_lengths = eval_matnet_reinforce("matnet-beijing-10-reinforce.ckpt", test_set_loader, env)
    pomo_tours, pomo_lengths = eval_matnet("matnet-beijing-10-pomo.ckpt", test_set_loader, env)
    tmatnet_tours, tmatnet_lengths = eval_tmatnet("tmatnet-beijing-10-matnet-pomo.ckpt", test_set_loader, env)

    # baselines
    tsp_tours, tsp_lengths = eval_policy("TSP", test_set_loader, env)
    opt_tours, opt_lengths = eval_policy("Optimal", test_set_loader, env)

    tsp_gaps = (tsp_lengths - opt_lengths) / opt_lengths
    reinforce_gaps = (reinforce_lengths - opt_lengths) / opt_lengths
    pomo_gaps = (pomo_lengths - opt_lengths) / opt_lengths
    tmatnet_gaps = (tmatnet_lengths - opt_lengths) / opt_lengths

    # compare on static sets
    mask = tsp_gaps < 0.001
    tsp_low_avg = torch.mean(tsp_gaps[mask])
    reinforce_low_avg = torch.mean(reinforce_gaps[mask])
    pomo_gaps_low_avg = torch.mean(pomo_gaps[mask])
    tmatnet_low_avg = torch.mean(tmatnet_gaps[mask])

    # compare on large amount sets
    mask = tsp_gaps > 0.03
    tsp_high_avg = torch.mean(tsp_gaps[mask])
    reinforce_high_avg = torch.mean(reinforce_gaps[mask])
    pomo_gaps_high_avg = torch.mean(pomo_gaps[mask])
    tmatnet_high_avg = torch.mean(tmatnet_gaps[mask])

    print(f"tsp_gaps: {tsp_gaps.mean().item()}, tsp_low_avg: {tsp_low_avg.item()}, tsp_high_avg: {tsp_high_avg.item()}")
    print(f"reinforce_gaps: {reinforce_gaps.mean().item()}, reinforce_low_avg: {reinforce_low_avg.item()}, reinforce_high_avg: {reinforce_high_avg.item()}")
    print(f"pomo_gaps: {pomo_gaps.mean().item()}, pomo_low_avg: {pomo_gaps_low_avg.item()}, pomo_high_avg: {pomo_gaps_high_avg.item()}")
    print(f"tmatnet_gaps: {tmatnet_gaps.mean().item()}, tmatnet_low_avg: {tmatnet_low_avg.item()}, tmatnet_high_avg: {tmatnet_high_avg.item()}")


def eval_replay_buffer():
    device = torch.device("cpu")
    torch.random.manual_seed(0)
    env = TDTSPEnv(time_matrix_params={'data': 'beijing', 'scale': 1, 'interpolate': "linear", 'downsample': 1000},
                   generator_params={"num_loc": 10}).to(device)
    test_set = env.dataset(10000)
    test_set_loader = DataLoader(test_set, batch_size=1024, collate_fn=test_set.collate_fn, shuffle=False)

    # baselines
    tsp_tours, tsp_lengths = eval_policy("TSP", test_set_loader, env)
    opt_tours, opt_lengths = eval_policy("Optimal", test_set_loader, env)

    pomo_tours, pomo_lengths = eval_tmatnet("tmatnet-beijing-10-matnet-pomo.ckpt", test_set_loader, env)
    reinforce_tours, reinforce_lengths = eval_tmatnet('tmatnet-beijing-10-matnet-reinforce.ckpt', test_set_loader, env)
    topt_tours, topt_lengths = eval_tmatnet('tmatnet-beijing-10-opt.ckpt', test_set_loader, env)

    tsp_gaps = (tsp_lengths - opt_lengths) / opt_lengths
    reinforce_gaps = (reinforce_lengths - opt_lengths) / opt_lengths
    pomo_gaps = (pomo_lengths - opt_lengths) / opt_lengths
    topt_gaps = (topt_lengths - opt_lengths) / opt_lengths

    # compare on static sets
    mask = tsp_gaps < 0.001
    tsp_low_avg = torch.mean(tsp_gaps[mask])
    reinforce_low_avg = torch.mean(reinforce_gaps[mask])
    pomo_gaps_low_avg = torch.mean(pomo_gaps[mask])
    topt_low_avg = torch.mean(topt_gaps[mask])

    # compare on large amount sets
    mask = tsp_gaps > 0.03
    tsp_high_avg = torch.mean(tsp_gaps[mask])
    reinforce_high_avg = torch.mean(reinforce_gaps[mask])
    pomo_gaps_high_avg = torch.mean(pomo_gaps[mask])
    topt_high_avg = torch.mean(topt_gaps[mask])

    print(f"tsp_gaps: {tsp_gaps.mean().item()}, tsp_low_avg: {tsp_low_avg.item()}, tsp_high_avg: {tsp_high_avg.item()}")
    print(f"reinforce_gaps: {reinforce_gaps.mean().item()}, reinforce_low_avg: {reinforce_low_avg.item()}, reinforce_high_avg: {reinforce_high_avg.item()}")
    print(f"pomo_gaps: {pomo_gaps.mean().item()}, pomo_low_avg: {pomo_gaps_low_avg.item()}, pomo_high_avg: {pomo_gaps_high_avg.item()}")
    print(f"topt_gaps: {topt_gaps.mean().item()}, topt_low_avg: {topt_low_avg.item()}, topt_high_avg: {topt_high_avg.item()}")


def prepare_data(loc, num):
    device = torch.device("cpu")
    torch.random.manual_seed(0)
    env = TDTSPEnv(time_matrix_params={'data': loc, 'scale': 1, 'interpolate': "linear", 'downsample': 1000},
                   generator_params={"num_loc": num}).to(device)
    test_set = env.dataset(10000)
    torch.save(test_set, f'~/rl4co/testcases/{loc}_{num}_dataset.pt')


def prepare_opt(loc, num):
    device = torch.device("cpu")
    torch.random.manual_seed(0)
    env = TDTSPEnv(time_matrix_params={'data': loc, 'scale': 1, 'interpolate': "linear", 'downsample': 1000},
                   generator_params={"num_loc": num}).to(device)
    torch.serialization.add_safe_globals([TensorDictDataset])
    test_set = torch.load(f'~/rl4co/testcases/{loc}_{num}_dataset.pt')
    test_set_loader = DataLoader(test_set, batch_size=1024, collate_fn=test_set.collate_fn, shuffle=False)
    opt = OptimalBaseline(env)
    opt_tours, opt_lengths = [], []
    for batch in tqdm(test_set_loader):
        td = env.reset(batch.clone())
        solution = opt.solve(td.clone())
        opt_lengths.append(solution["tour_lengths"])
        opt_tours.append(solution["tours"])
    opt_lengths = torch.cat(opt_lengths, dim=0)
    opt_tours = torch.cat(opt_tours, dim=0)
    torch.save(opt_lengths, f'~/rl4co/testcases/{loc}_{num}_opt_lengths.pt')
    torch.save(opt_tours, f'~/rl4co/testcases/{loc}_{num}_opt_tours.pt')


def eval_gaps():
    device = torch.device("cuda")
    torch.random.manual_seed(0)
    env = TDTSPEnv(time_matrix_params={'data': 'beijing', 'scale': 1, 'interpolate': "linear", 'downsample': 1000},
                   generator_params={"num_loc": 20}).to(device)
    # test_set = env.dataset(10000)
    test_set = torch.load(f'~/rl4co/testcases/beijing_20_dataset_10000.pt', weights_only=False)
    test_set_loader = DataLoader(test_set, batch_size=512, collate_fn=test_set.collate_fn)

    # Do the test
    # _, tsp_lengths = eval_policy("TSP", test_set_loader, env)
    # _, opt_lengths = eval_policy("Optimal", test_set_loader, env)
    import numpy as np
    tsp_lengths = torch.from_numpy(np.load('~/rl4co/testcases/beijing_20_dataset_10000/tsp_lengths.npy')).to(device)
    opt_lengths = torch.from_numpy(np.load('~/rl4co/testcases/beijing_20_dataset_10000/lengths.npy')).to(device)

    matnet = MatNet.load_from_checkpoint(
        # "~/rl4co/logs/train/runs/tdtsp_pomo/2025.05.10-16.56.51/beijing_constant_10/checkpoints/last.ckpt"
        '~/rl4co/ckpts/matnet-beijing-20-pomo.ckpt'
    ).policy.to(device)
    matnet.eval()
    env.to(device)
    matnet_lengths = []
    for batch in tqdm(test_set_loader):
        env.pomo = True
        td = env.reset(batch.clone().to(device))
        env.pomo = False
        env.matrix.static = True
        matnet_solution = matnet(td, env, phase="test", num_starts=10)
        env.matrix.static = False
        reward = unbatchify(matnet_solution["reward"], (0, 10))
        actions = unbatchify(matnet_solution["actions"], (0, 10))
        max_reward, max_idxs = reward.max(dim=-1)
        actions = gather_by_index(actions, max_idxs, dim=max_idxs.dim())
        idx = torch.argmax((actions == 0).float(), dim=1)
        positions = torch.arange(actions.shape[-1], device=actions.device).unsqueeze(0).expand(actions.shape[0], -1)
        positions = (positions + idx.unsqueeze(1)) % actions.shape[1]
        actions = torch.gather(actions, 1, positions)
        matnet_lengths.append(env.get_tour_length(td, actions))
    matnet_lengths = torch.cat(matnet_lengths, dim=0).clone()
    print(f"MatNet Tour lengths: {matnet_lengths.mean()}")

    tmatnet = TimeMatNet.load_from_checkpoint(
        # "~/rl4co/logs/train/runs/tdtsp-mat/2025.05.11-09.53.14/beijing_linear_10/checkpoints/last.ckpt"
        '~/rl4co/ckpts/tmatnet-beijing-20-matnet-pomo.ckpt'
    ).policy
    tmatnet.env = env
    tmatnet = tmatnet.to(device)
    tmatnet.eval()
    env.to(device)
    tmat_lengths = []
    for batch in tqdm(test_set_loader):
        td_init = env.reset(batch.clone().to(device))
        tmat_solution = tmatnet(td_init, env, phase="test", decode_type="greedy")
        tmat_lengths.append(-tmat_solution["reward"])
    tmat_lengths = torch.cat(tmat_lengths, dim=0).clone()
    print(f"Ours Tour lengths: {tmat_lengths.mean()}")

    improved_lengths = []
    helper = BatchSABaseline(env)
    for batch in tqdm(test_set_loader):
        td_init = env.reset(batch.clone().to(device))
        tmat_solution = tmatnet(td_init, env, phase="test", decode_type="greedy")
        tmat_actions = torch.cat(
            [torch.zeros(tmat_solution["actions"].shape[0], 1, device=actions.device, dtype=torch.long),
             tmat_solution["actions"]], dim=1)
        tmat_reward = -tmat_solution["reward"]

        env.pomo = True
        td = env.reset(batch.clone().to(device))
        env.pomo = False
        env.matrix.static = True
        rl_solution = matnet(td, env, phase="test", num_starts=10)
        env.matrix.static = False
        reward = unbatchify(rl_solution["reward"], (0, 10))
        actions = unbatchify(rl_solution["actions"], (0, 10))
        # reward = policy.env.get_tour_length(td_init.unsqueeze(1).expand(2, 10).reshape(-1), actions.reshape(-1, 10)).reshape(-1, 10)
        max_reward, max_idxs = reward.max(dim=-1)
        actions = gather_by_index(actions, max_idxs, dim=max_idxs.dim())
        idx = torch.argmax((actions == 0).float(), dim=1)
        positions = torch.arange(actions.shape[-1], device=actions.device).unsqueeze(0).expand(actions.shape[0], -1)
        positions = (positions + idx.unsqueeze(1)) % actions.shape[1]
        mat_actions = torch.gather(actions, 1, positions)
        mat_reward = env.get_tour_length(td, mat_actions)

        # moe first
        mask = tmat_reward < mat_reward
        moe_reward = torch.where(mask, tmat_reward, mat_reward)
        moe_actions = torch.where(mask.unsqueeze(-1).expand_as(tmat_actions), tmat_actions, mat_actions)

        # improve
        imp_tour, imp_length = helper.neighborhood_search(moe_actions, moe_reward, td_init, 2)
        improved_lengths.append(imp_length)
    improved_lengths = torch.cat(improved_lengths, dim=0)
    print("Improved Tour lengths: ", improved_lengths.mean())

    tsp_gaps = (tsp_lengths - opt_lengths) / opt_lengths
    matnet_gaps = (matnet_lengths - opt_lengths) / opt_lengths
    tmatnet_gaps = (tmat_lengths - opt_lengths) / opt_lengths
    improved_gaps = (improved_lengths - opt_lengths) / opt_lengths

    low_mask = tsp_gaps < 0.001
    high_mask = tsp_gaps > 0.03
    print(f"tsp_low_gaps: {tsp_gaps[low_mask].mean()}, tsp_high_gaps: {tsp_gaps[high_mask].mean()}")
    print(f"matnet_low_gaps: {matnet_gaps[low_mask].mean()}, matnet_high_gaps: {matnet_gaps[high_mask].mean()}")
    print(f"tmatnet_low_gaps: {tmatnet_gaps[low_mask].mean()}, tmatnet_high_gaps: {tmatnet_gaps[high_mask].mean()}")
    print(f"improved_low_gaps: {improved_gaps[low_mask].mean()}, improved_high_gaps: {improved_gaps[high_mask].mean()}")


if __name__ == "__main__":
    # eval_moe_search(10)
    # eval_training()
    # eval_replay_buffer()
    eval_gaps()
