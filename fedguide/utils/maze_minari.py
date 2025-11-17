import minari
import numpy as np
import torch
from torch.utils.data import Dataset


# load Minari PointMaze dataset
def load_minari_pointmaze(dataset_id="D4RL/pointmaze/medium-v2", download=True):

    def flatten_obs(obs_dict):
        # Minari PointMaze obs format: {observation, achieved_goal, desired_goal}
        obs = obs_dict["observation"]
        ag = obs_dict["achieved_goal"]
        dg = obs_dict["desired_goal"]
        return np.concatenate([obs, ag, dg], axis=-1).astype(np.float32)
    
    ds = minari.load_dataset(dataset_id, download=download)

    trajs = []
    for ep in ds.iterate_episodes():
        obs = np.stack(flatten_obs(ep.observations))
        actions = np.array(ep.actions, dtype=np.float32)
        rewards = np.array(ep.rewards, dtype=np.float32)
        dones = np.logical_or(ep.terminations, ep.truncations).astype(np.float32)

        # next states
        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)

        trajs.append(dict(
            s=obs,
            a=actions,
            r=rewards,
            s_next=next_obs,
            d=dones,
        ))

    return trajs

# map trajectory to category index based on final position
def traj_category(traj, n_bins=4):
        # final state position used as category
        final_s = traj["s"][-1]
        agent_x = final_s[4]
        agent_y = final_s[5]

        xy = np.array([agent_x, agent_y], dtype=np.float32)

        # assume maze normalized around [-1,1], map to [0,1]
        xy_norm = (xy + 1.0) / 2.0
        xy_norm = np.clip(xy_norm, 0, 0.9999)

        gx = int(xy_norm[0] * n_bins)
        gy = int(xy_norm[1] * n_bins)
        return gx * n_bins + gy

# dirichlet-based non-iid split
def split_trajs_dirichlet(trajs, n_clients=8, alpha=0.5, n_bins=4, seed=42):
    
    rng = np.random.default_rng(seed)
    K = n_bins * n_bins

    cat_to_trajs = [[] for _ in range(K)]
    for tr in trajs:
        c = traj_category(tr, n_bins=n_bins)
        cat_to_trajs[c].append(tr)

    client_trajs = [[] for _ in range(n_clients)]
    for trajs_c in cat_to_trajs:
        if not trajs_c:
            continue
        p = rng.dirichlet(alpha * np.ones(n_clients))
        counts = rng.multinomial(len(trajs_c), p)

        idx = 0
        for i in range(n_clients):
            n_i = counts[i]
            if n_i > 0:
                client_trajs[i].extend(trajs_c[idx: idx+n_i])
                idx += n_i

    return client_trajs

# Dataset class for transitions
class TransitionDataset(Dataset):
    def __init__(self, trajs):
        self.data = []
        for tr in trajs:
            s, a, r, s_next, d = tr["s"], tr["a"], tr["r"], tr["s_next"], tr["d"]
            T = len(a)  # actions define number of transitions (there are t-1 actions for t states)
            for t in range(T):
                self.data.append((s[t], a[t], r[t], s_next[t], d[t]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, a, r, s_next, d = self.data[idx]
        return {
            "s": torch.tensor(s, dtype=torch.float32),
            "a": torch.tensor(a, dtype=torch.float32),
            "r": torch.tensor(r, dtype=torch.float32),
            "s_": torch.tensor(s_next, dtype=torch.float32),
            "d": torch.tensor(d, dtype=torch.float32),
        }

# create Minari PointMaze datasets for clients
def make_maze_minari_datasets(
    n_clients=8,
    dataset_id="D4RL/pointmaze/medium-v2",
    alpha=0.5,
    seed=42,
):
    trajs = load_minari_pointmaze(dataset_id, download=True)
    client_trajs = split_trajs_dirichlet(trajs, n_clients=n_clients, alpha=alpha, seed=seed)
    return [TransitionDataset(ct) for ct in client_trajs]


# Unit tests
if __name__ == "__main__":
    print("\nMinari Maze Unit Tests\n")

    # TEST 1: load_minari_pointmaze
    print("Test 1: Loading Minari dataset...")

    try:
        trajs = load_minari_pointmaze("D4RL/pointmaze/medium-v2") #, download=True)
        print("  Trajectory 0 keys:", trajs[0].keys())
        print("  s shape:", trajs[0]["s"].shape)
        print("  a shape:", trajs[0]["a"].shape)
        print("  r shape:", trajs[0]["r"].shape)
        print("  s_next shape:", trajs[0]["s_next"].shape)
        print("  d shape:", trajs[0]["d"].shape)

        print(f"  Loaded {len(trajs)} trajectories")
        assert len(trajs) > 0
        assert "s" in trajs[0]
        assert "a" in trajs[0]
        obs_dim = trajs[0]["s"].shape[-1]
        act_dim = trajs[0]["a"].shape[-1]
        print(f"  Observation dim = {obs_dim}, Action dim = {act_dim}")
    except Exception as e:
        print("  FAILED:", e)
        raise

    # TEST 2: traj_category
    print("\nTest 2: Testing trajectory category mapping...")

    try:
        s0 = trajs[0]["s"][-1]
        print("DEBUG final state shape:", s0.shape)
        c = traj_category(trajs[0], n_bins=4)
        print(f"  Category index = {c}")
        assert 0 <= c < 16
    except Exception as e:
        print("  FAILED:", e)
        raise

    # TEST 3: split_trajs_dirichlet
    print("\nTest 3: Testing Dirichlet split...")

    try:
        n_clients = 8
        splits = split_trajs_dirichlet(trajs, n_clients=n_clients, alpha=0.5, seed=42)
        assert len(splits) == n_clients
        print(f"  Created {n_clients} client splits")

        total = sum(len(s) for s in splits)
        print(f"  Total trajs after split: {total}")
        assert total == len(trajs)

        for i, s in enumerate(splits):
            print(f"    Client {i}: {len(s)} trajs")
    except Exception as e:
        print("  FAILED:", e)
        raise

    # TEST 4: TransitionDataset
    print("\nTest 4: Testing TransitionDataset...")

    try:
        ds = TransitionDataset(splits[0])
        print(f"  Client 0 dataset size (transitions): {len(ds)}")
        assert len(ds) > 0

        sample = ds[0]
        assert "s" in sample
        assert "a" in sample
        assert "r" in sample
        assert "s_" in sample
        assert "d" in sample
        print(f"  Sample transition keys ok: {list(sample.keys())}")
        print(f"  s shape: {sample['s'].shape}, a shape: {sample['a'].shape}")
    except Exception as e:
        print("  FAILED:", e)
        raise

    # TEST 5: make_maze_minari_datasets
    print("\nTest 5: Testing make_maze_minari_datasets...")

    try:
        client_datasets = make_maze_minari_datasets(
            n_clients=4,
            dataset_id="D4RL/pointmaze/medium-v2",
            alpha=0.5,
            seed=123,
        )
        assert len(client_datasets) == 4
        print("  Generated 4 TransitionDatasets")

        for i, ds in enumerate(client_datasets):
            print(f"    Client {i}: {len(ds)} transitions")
            assert len(ds) > 0
    except Exception as e:
        print("  FAILED:", e)
        raise

    print("\n passed unit tests\n")

### Sample Output:
# Test 1: Loading Minari dataset...
#   Trajectory 0 keys: dict_keys(['s', 'a', 'r', 's_next', 'd'])
#   s shape: (329, 8)
#   a shape: (328, 2)
#   r shape: (328,)
#   s_next shape: (329, 8)
#   d shape: (328,)
#   Loaded 4752 trajectories
#   Observation dim = 8, Action dim = 2

# Test 2: Testing trajectory category mapping...
# DEBUG final state shape: (8,)
#   Category index = 3

# Test 3: Testing Dirichlet split...
#   Created 8 client splits
#   Total trajs after split: 4752
#     Client 0: 575 trajs
#     Client 1: 1164 trajs
#     Client 2: 365 trajs
#     Client 3: 644 trajs
#     Client 4: 693 trajs
#     Client 5: 307 trajs
#     Client 6: 468 trajs
#     Client 7: 536 trajs

# Test 4: Testing TransitionDataset...
#   Client 0 dataset size (transitions): 114877
#   Sample transition keys ok: ['s', 'a', 'r', 's_', 'd']
#   s shape: torch.Size([8]), a shape: torch.Size([2])

# Test 5: Testing make_maze_minari_datasets...
#   Generated 4 TransitionDatasets
#     Client 0: 167395 transitions
#     Client 1: 93573 transitions
#     Client 2: 114065 transitions
#     Client 3: 624967 transitions

#  passed unit tests