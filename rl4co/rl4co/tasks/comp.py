from rl4co.envs import TDTSPEnv
from rl4co.baselines.aco import ACOBaseline, TACOBaseline
from rl4co.baselines.random import RandomBaseline
from rl4co.baselines.greedy import GreedyBaseline
from rl4co.baselines.sa import SimulatedAnnealingBaseline, BatchSABaseline
from rl4co.baselines.optimal import OptimalBaseline, SubOptimalBaseline, FastSubOptimalBaseline, ATSPBaseline
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.models.zoo import TimeMatNet, MatNet
from pytorch_lightning.profilers import SimpleProfiler
import torch
from torch.utils.data import DataLoader
from tensordict.tensordict import TensorDict
from tqdm import tqdm


def eval_policy(name, dataset):
    policy_set = {"ACO": ACOBaseline,
                  "Random": RandomBaseline,
                  "Greedy": GreedyBaseline,
                  "SA": BatchSABaseline,
                  "TSP": FastSubOptimalBaseline,
                  "Optimal": OptimalBaseline}
    assert name in policy_set, f"Policy {name} not implemented, only support {policy_set.keys()} now"
    policy = policy_set[name](env)
    tour_lengths = []
    for batch in tqdm(dataset):
        td = env.reset(batch)
        # td.set("start_time", torch.zeros(batch.batch_size, device=td.device))
        solution = policy.solve(td.clone())
        tour_lengths.append(solution["tour_lengths"])
    tour_lengths = torch.cat(tour_lengths, dim=0)
    print(f"{name} Tour lengths: {tour_lengths.mean().item()}")
    return tour_lengths


if __name__ == "__main__":
    # Create environment and baseline
    profiler = SimpleProfiler()
    # device = torch.device("cuda")
    device = torch.device("cpu")
    torch.random.manual_seed(0)
    env = TDTSPEnv(time_matrix_params={'data': 'beijing', 'scale': 1, 'interpolate': "linear", 'downsample': 1000},
                   generator_params={"num_loc": 10}).to(device)
    # test_set = env.dataset(10000)
    test_set = torch.load('/home/ubuntu/Temporal-TSP/rl4co/testcases/beijing_10_dataset_10000.pt', weights_only=False)
    test_set_loader = DataLoader(test_set, batch_size=1024, collate_fn=test_set.collate_fn, shuffle=False)

    # Do the test
    with profiler.profile("ACO"):
        aco_lengths = eval_policy("ACO", test_set_loader)
    with profiler.profile("Random"):
        random_length = eval_policy("Random", test_set_loader)
    with profiler.profile("Greedy"):
        greedy_lengths = eval_policy("Greedy", test_set_loader)
    with profiler.profile("SA"):
        sa_lengths = eval_policy("SA", test_set_loader)
    with profiler.profile("TSP"):
        tsp_lengths = eval_policy("TSP", test_set_loader)
    with profiler.profile("Optimal"):
        opt_lengths = eval_policy("Optimal", test_set_loader)

    matnet = MatNet.load_from_checkpoint(
        "~/rl4co/logs/train/runs/tdtsp_pomo/2025.05.10-16.56.51/beijing_constant_10/checkpoints/last.ckpt"
        # "~/rl4co/logs/train/runs/tdtsp_pomo/2025.05.10-19.22.53/beijing_linear_20/checkpoints/last.ckpt"
        # "~/rl4co/logs/train/runs/tdtsp_pomo/2025.05.12-00.36.29/beijing_constant_50/checkpoints/last.ckpt"
    ).policy.to(device)
    matnet.eval()
    env.to(device)
    with profiler.profile("MatNet"):
        matnet_lengths = []
        for batch in tqdm(test_set_loader):
            env.pomo = True
            td = env.reset(batch.clone().to(device))
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
    matnet_lengths = torch.cat(matnet_lengths, dim=0).clone()
    print(f"MatNet Tour lengths: {matnet_lengths.mean()}")

    tmatnet = TimeMatNet.load_from_checkpoint(
        "~/rl4co/logs/train/runs/tdtsp-mat/2025.05.11-09.53.14/beijing_linear_10/checkpoints/last.ckpt"
        # "~/rl4co/logs/train/runs/tdtsp-mat/2025.05.12-16.24.49/beijing_linear_50/checkpoints/last.ckpt"
    ).policy.to(device)
    tmatnet.env = env
    tmatnet = tmatnet.to(device)
    tmatnet.eval()
    env.to(device)
    with profiler.profile("RL"):
        tmat_lengths = []
        for batch in tqdm(test_set_loader):
            td_init = env.reset(batch.clone().to(device))
            rl_solution = tmatnet(td_init, env, phase="test", decode_type="greedy")
            tmat_lengths.append(-rl_solution["reward"])
    tmat_lengths = torch.cat(tmat_lengths, dim=0).clone()
    print(f"Ours Tour lengths: {tmat_lengths.mean()}")

    helper = BatchSABaseline(env)
    with profiler.profile("Improved"):
        improved_lengths = []
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

    gaps = {
        "ACO": (aco_lengths - opt_lengths) / opt_lengths,
        "Greedy": (greedy_lengths - opt_lengths) / opt_lengths,
        "SA": (sa_lengths - opt_lengths) / opt_lengths,
        "TSP": (tsp_lengths - opt_lengths) / opt_lengths,
        "MatNet": (matnet_lengths - opt_lengths) / opt_lengths,
        "Ours": (tmat_lengths - opt_lengths) / opt_lengths,
        "Improved": (improved_lengths - opt_lengths) / opt_lengths,
    }
    for name, gap in gaps.items():
        print(f"{name} gap: {gap.mean().item() * 100:.2f}%")

    print(profiler.summary())