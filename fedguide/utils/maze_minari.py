"""
DEPRECATED: This module has been moved to fedguide.datasets.minari_loader
Please update your imports to use: from fedguide.datasets import make_minari_datasets
"""
import warnings
warnings.warn(
    "fedguide.utils.maze_minari is deprecated. "
    "Please use fedguide.datasets.minari_loader or fedguide.datasets instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
from fedguide.datasets.minari_loader import (
    load_minari_pointmaze,
    make_maze_minari_datasets,
    make_minari_datasets,
)
from fedguide.datasets.base import (
    TrajectoryWindowDataset,
    TransitionDataset,
)
from fedguide.datasets.heterogeneity import (
    traj_category,
    split_trajs_dirichlet,
)

# Keep original imports for backward compatibility
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

class TrajectoryWindowDataset(Dataset):
    """
    Yields fixed-length windows of (state sequence, action sequence).

    Each item:
      - s: [H, obs_dim]
      - a: [H, act_dim]
    """
    def __init__(self, trajs, horizon=32, stride=16):
        self.windows = []
        self.horizon = horizon

        for tr in trajs:
            s = tr["s"]          # [T_s, obs_dim]
            a = tr["a"]          # [T_a, act_dim]
            T = a.shape[0]       # number of transitions
            s = s[:T]            # align states with actions

            if T < horizon:
                continue  # skip very short trajectories, or pad if you prefer

            for t in range(0, T - horizon + 1, stride):
                s_win = s[t:t + horizon]       # [H, obs_dim]
                a_win = a[t:t + horizon]       # [H, act_dim]
                self.windows.append((s_win, a_win))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s_win, a_win = self.windows[idx]
        return {
            "s": torch.tensor(s_win, dtype=torch.float32),   # [H, obs_dim]
            "a": torch.tensor(a_win, dtype=torch.float32),   # [H, act_dim]
        }

# create Minari PointMaze datasets for clients
def make_maze_minari_datasets(
    n_clients=8,
    dataset_id="D4RL/pointmaze/medium-v2",
    alpha=0.5,
    seed=42,
    horizon=32,
    stride=16,
):
    trajs = load_minari_pointmaze(dataset_id, download=True)
    client_trajs = split_trajs_dirichlet(trajs, n_clients=n_clients, alpha=alpha, seed=seed)
    return [TrajectoryWindowDataset(ct, horizon=horizon, stride=stride) for ct in client_trajs]


# Unit tests
if __name__ == "__main__":
    print("\nMinari Maze Unit Tests\n")

    # TEST 1: load_minari_pointmaze
    print("Test 1: Loading Minari dataset...")
    dataset_id = "D4RL/pointmaze/medium-v2"

    try:
        trajs = load_minari_pointmaze(dataset_id) #, download=True)
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
        print("\n[TEST] Testing TrajectoryWindowDataset...")
        horizon = 32
        stride = 16

        ds = TrajectoryWindowDataset(trajs, horizon=horizon, stride=stride)

        print(f"[TEST] Number of windows: {len(ds)}")
        assert len(ds) > 0, "No trajectory windows were produced!"

        sample = ds[0]
        s = sample["s"]
        a = sample["a"]

        print(f"[TEST] s shape: {s.shape}, a shape: {a.shape}")

        assert s.shape[0] == horizon, "State window length should equal horizon"
        assert a.shape[0] == horizon, "Action window length should equal horizon"
        assert s.ndim == 2, "s must be [H, obs_dim]"
        assert a.ndim == 2, "a must be [H, act_dim]"
        assert s.shape[1] == a.shape[1] + (s.shape[1] - a.shape[1]), "Dims must be consistent"
    except Exception as e:
        print("  FAILED:", e)
        raise

    # TEST 5: make_maze_minari_datasets
    print("\nTest 5: Testing make_maze_minari_datasets...")

    try:
        print("\n[TEST] Testing full dataset creation...")

        clients = make_maze_minari_datasets(
            n_clients=4,
            dataset_id=dataset_id,
            alpha=0.5,
            seed=123,
            horizon=32,
            stride=16,
        )

        assert len(clients) == 4, "Should have 4 client datasets"

        for cid, ds in enumerate(clients):
            print(f"[TEST] Client {cid} dataset size: {len(ds)}")
            assert len(ds) > 0, f"Client {cid} received no windows!"

            sample = ds[0]
            s = sample["s"]
            a = sample["a"]

            assert s.ndim == 2 and a.ndim == 2, "Samples must be trajectory windows"
            assert s.shape[0] == a.shape[0], "State/action window mismatch"
            assert s.shape[0] == 32, "Window length should be 32"
    except Exception as e:
        print("  FAILED:", e)
        raise

    print("\n passed unit tests\n")

### Sample Output:
# Minari Maze Unit Tests

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

# [TEST] Testing TrajectoryWindowDataset...
# [TEST] Number of windows: 55640
# [TEST] s shape: torch.Size([32, 8]), a shape: torch.Size([32, 2])

# Test 5: Testing make_maze_minari_datasets...

# [TEST] Testing full dataset creation...
# [TEST] Client 0 dataset size: 9331
# [TEST] Client 1 dataset size: 5162
# [TEST] Client 2 dataset size: 6234
# [TEST] Client 3 dataset size: 34913

#  passed unit tests