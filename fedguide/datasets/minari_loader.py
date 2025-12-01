"""Minari dataset loading."""
import minari
import numpy as np
from .base import TrajectoryWindowDataset
from .heterogeneity import split_trajs_dirichlet


def load_minari_pointmaze(dataset_id="D4RL/pointmaze/medium-v2", download=True):
    """Load Minari PointMaze dataset."""
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


def make_minari_datasets(
    n_clients=8,
    dataset_id="D4RL/pointmaze/medium-v2",
    alpha=0.5,
    seed=42,
    horizon=32,
    stride=16,
):
    """Create Minari PointMaze datasets for clients."""
    trajs = load_minari_pointmaze(dataset_id, download=True)
    client_trajs = split_trajs_dirichlet(trajs, n_clients=n_clients, alpha=alpha, seed=seed)
    return [TrajectoryWindowDataset(ct, horizon=horizon, stride=stride) for ct in client_trajs]


# Alias for backward compatibility
make_maze_minari_datasets = make_minari_datasets


# Unit tests
if __name__ == "__main__":
    print("\nMinari Maze Unit Tests\n")

    # TEST 1: load_minari_pointmaze
    print("Test 1: Loading Minari dataset...")
    dataset_id = "D4RL/pointmaze/medium-v2"

    try:
        trajs = load_minari_pointmaze(dataset_id)
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
    from .heterogeneity import traj_category

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

    # TEST 4: TrajectoryWindowDataset
    print("\nTest 4: Testing TrajectoryWindowDataset...")

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
    except Exception as e:
        print("  FAILED:", e)
        raise

    # TEST 5: make_minari_datasets
    print("\nTest 5: Testing make_minari_datasets...")

    try:
        print("\n[TEST] Testing full dataset creation...")

        clients = make_minari_datasets(
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

