"""Minari dataset loading."""
import minari
import numpy as np
from typing import List, Dict, Any, Optional
from .base import TrajectoryWindowDataset
from .heterogeneity import split_trajs_dirichlet


def _flatten_goal_conditioned_obs(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten goal-conditioned observation dict to a single vector.
    
    Handles observation format: {observation, achieved_goal, desired_goal}
    """
    obs = obs_dict.get("observation", obs_dict.get("obs", None))
    ag = obs_dict.get("achieved_goal", None)
    dg = obs_dict.get("desired_goal", None)
    
    parts = []
    if obs is not None:
        parts.append(obs)
    if ag is not None:
        parts.append(ag)
    if dg is not None:
        parts.append(dg)
    
    if len(parts) == 0:
        raise ValueError(f"Could not extract observation from dict with keys: {list(obs_dict.keys())}")
    
    return np.concatenate(parts, axis=-1).astype(np.float32)


def _detect_observation_format(observations: Any) -> str:
    """Detect the format of observations from the first observation.
    
    Returns:
        "dict": Dictionary format (e.g., goal-conditioned)
        "array": Direct array format
    """
    if isinstance(observations, dict):
        return "dict"
    elif isinstance(observations, (list, tuple)) and len(observations) > 0:
        first_obs = observations[0]
        if isinstance(first_obs, dict):
            return "dict"
        else:
            return "array"
    else:
        return "array"


def load_minari_dataset(
    dataset_id: str,
    download: bool = True,
    flatten_obs: bool = True
) -> List[Dict[str, np.ndarray]]:
    """Load a Minari dataset and convert to trajectory format.
    
    This is a generic loader that works with various Minari datasets including:
    - pointmaze (goal-conditioned)
    - maze2d (goal-conditioned)
    - antmaze (goal-conditioned)
    - Other environments with standard observation spaces
    
    Args:
        dataset_id: Minari dataset ID (e.g., "D4RL/pointmaze/medium-v2")
        download: Whether to download the dataset if not found locally
        flatten_obs: Whether to flatten goal-conditioned observations (if dict format)
    
    Returns:
        List of trajectory dictionaries, each with keys:
        - s: observations [T, obs_dim]
        - a: actions [T, act_dim]
        - r: rewards [T]
        - s_next: next observations [T, obs_dim]
        - d: done flags [T]
    """
    ds = minari.load_dataset(dataset_id, download=download)
    
    trajs = []
    for ep in ds.iterate_episodes():
        # Detect observation format
        obs_list = list(ep.observations)
        obs_format = _detect_observation_format(obs_list[0] if len(obs_list) > 0 else None)
        
        # Convert observations to array format
        if obs_format == "dict":
            if flatten_obs:
                # Flatten goal-conditioned observations
                obs_array = np.stack([_flatten_goal_conditioned_obs(obs) for obs in obs_list])
            else:
                # Keep as dict (not recommended for most use cases)
                obs_array = obs_list
        else:
            # Direct array format
            obs_array = np.stack([np.asarray(obs, dtype=np.float32) for obs in obs_list])
        
        # Convert other fields
        actions = np.array(ep.actions, dtype=np.float32)
        rewards = np.array(ep.rewards, dtype=np.float32)
        dones = np.logical_or(ep.terminations, ep.truncations).astype(np.float32)
        
        # Compute next states
        if isinstance(obs_array, np.ndarray):
            next_obs = np.concatenate([obs_array[1:], obs_array[-1:]], axis=0)
        else:
            # For dict format, just copy the last state
            next_obs = obs_array
        
        trajs.append(dict(
            s=obs_array,
            a=actions,
            r=rewards,
            s_next=next_obs,
            d=dones,
        ))
    
    return trajs


def load_minari_pointmaze(dataset_id="D4RL/pointmaze/medium-v2", download=True):
    """Load Minari PointMaze dataset (backward compatibility wrapper)."""
    return load_minari_dataset(dataset_id, download=download, flatten_obs=True)


def load_minari_maze2d(dataset_id: str = "D4RL/maze2d/umaze-v1", download: bool = True):
    """Load Minari Maze2D dataset.
    
    Args:
        dataset_id: Minari dataset ID for maze2d (e.g., "D4RL/maze2d/umaze-v1")
        download: Whether to download the dataset if not found locally
    
    Returns:
        List of trajectory dictionaries
    """
    return load_minari_dataset(dataset_id, download=download, flatten_obs=True)


def load_minari_antmaze(dataset_id: str = "D4RL/antmaze/umaze-v0", download: bool = True):
    """Load Minari AntMaze dataset.
    
    Args:
        dataset_id: Minari dataset ID for antmaze (e.g., "D4RL/antmaze/umaze-v0")
        download: Whether to download the dataset if not found locally
    
    Returns:
        List of trajectory dictionaries
    """
    return load_minari_dataset(dataset_id, download=download, flatten_obs=True)


def make_minari_datasets(
    n_clients=8,
    dataset_id="D4RL/pointmaze/medium-v2",
    alpha=0.5,
    seed=42,
    horizon=32,
    stride=16,
    download=True,
):
    """Create Minari datasets for multiple clients with trajectory splitting.
    
    Supports various Minari dataset types:
    - pointmaze: "D4RL/pointmaze/medium-v2"
    - maze2d: "D4RL/maze2d/umaze-v1", "D4RL/maze2d/medium-v1", etc.
    - antmaze: "D4RL/antmaze/umaze-v0", "D4RL/antmaze/medium-play-v0", etc.
    
    Args:
        n_clients: Number of client datasets to create
        dataset_id: Minari dataset ID
        alpha: Dirichlet distribution parameter for trajectory splitting (smaller = more heterogeneous)
        seed: Random seed for splitting
        horizon: Window length for trajectory windows
        stride: Stride between trajectory windows
        download: Whether to download the dataset if not found locally
    
    Returns:
        List of TrajectoryWindowDataset objects, one per client
    """
    # Use generic loader for all dataset types
    trajs = load_minari_dataset(dataset_id, download=download, flatten_obs=True)
    client_trajs = split_trajs_dirichlet(trajs, n_clients=n_clients, alpha=alpha, seed=seed)
    return [TrajectoryWindowDataset(ct, horizon=horizon, stride=stride) for ct in client_trajs]


# Convenience functions for specific environments
def make_maze2d_minari_datasets(
    n_clients=8,
    dataset_id="D4RL/maze2d/umaze-v1",
    alpha=0.5,
    seed=42,
    horizon=32,
    stride=16,
    download=True,
):
    """Create Minari Maze2D datasets for multiple clients."""
    return make_minari_datasets(
        n_clients=n_clients,
        dataset_id=dataset_id,
        alpha=alpha,
        seed=seed,
        horizon=horizon,
        stride=stride,
        download=download,
    )


def make_antmaze_minari_datasets(
    n_clients=8,
    dataset_id="D4RL/antmaze/umaze-v0",
    alpha=0.5,
    seed=42,
    horizon=32,
    stride=16,
    download=True,
):
    """Create Minari AntMaze datasets for multiple clients."""
    return make_minari_datasets(
        n_clients=n_clients,
        dataset_id=dataset_id,
        alpha=alpha,
        seed=seed,
        horizon=horizon,
        stride=stride,
        download=download,
    )


# Alias for backward compatibility
make_maze_minari_datasets = make_minari_datasets


# Unit tests
if __name__ == "__main__":
    print("\nMinari Dataset Loader Unit Tests\n")

    # TEST 1: load_minari_pointmaze
    print("Test 1: Loading PointMaze dataset...")
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
        print("  (This may fail if dataset is not available)")
        # Don't raise - allow tests to continue

    # TEST 1b: load_minari_maze2d
    print("\nTest 1b: Loading Maze2D dataset...")
    try:
        maze2d_dataset_id = "D4RL/maze2d/umaze-v1"
        trajs = load_minari_maze2d(maze2d_dataset_id)
        print(f"  Loaded {len(trajs)} trajectories")
        if len(trajs) > 0:
            print("  s shape:", trajs[0]["s"].shape)
            print("  a shape:", trajs[0]["a"].shape)
            obs_dim = trajs[0]["s"].shape[-1]
            act_dim = trajs[0]["a"].shape[-1]
            print(f"  Observation dim = {obs_dim}, Action dim = {act_dim}")
    except Exception as e:
        print(f"  SKIPPED: {e}")
        print("  (This may fail if maze2d dataset is not available in Minari)")

    # TEST 1c: load_minari_antmaze
    print("\nTest 1c: Loading AntMaze dataset...")
    try:
        antmaze_dataset_id = "D4RL/antmaze/umaze-v0"
        trajs = load_minari_antmaze(antmaze_dataset_id)
        print(f"  Loaded {len(trajs)} trajectories")
        if len(trajs) > 0:
            print("  s shape:", trajs[0]["s"].shape)
            print("  a shape:", trajs[0]["a"].shape)
            obs_dim = trajs[0]["s"].shape[-1]
            act_dim = trajs[0]["a"].shape[-1]
            print(f"  Observation dim = {obs_dim}, Action dim = {act_dim}")
    except Exception as e:
        print(f"  SKIPPED: {e}")
        print("  (This may fail if antmaze dataset is not available in Minari)")

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
        print("  (This may fail if dataset is not available)")

    # TEST 6: Test generic load_minari_dataset function
    print("\nTest 6: Testing generic load_minari_dataset function...")
    try:
        # Test with pointmaze
        trajs = load_minari_dataset("D4RL/pointmaze/medium-v2", download=False, flatten_obs=True)
        print(f"  ✓ Loaded {len(trajs)} trajectories using generic loader")
        if len(trajs) > 0:
            print(f"  Observation shape: {trajs[0]['s'].shape}")
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # TEST 7: Test convenience functions
    print("\nTest 7: Testing convenience functions...")
    try:
        # Test maze2d convenience function
        try:
            clients = make_maze2d_minari_datasets(
                n_clients=2,
                dataset_id="D4RL/maze2d/umaze-v1",
                horizon=32,
                stride=16,
            )
            print(f"  ✓ make_maze2d_minari_datasets created {len(clients)} client datasets")
        except Exception as e:
            print(f"  ⚠ make_maze2d_minari_datasets skipped: {e}")

        # Test antmaze convenience function
        try:
            clients = make_antmaze_minari_datasets(
                n_clients=2,
                dataset_id="D4RL/antmaze/umaze-v0",
                horizon=32,
                stride=16,
            )
            print(f"  ✓ make_antmaze_minari_datasets created {len(clients)} client datasets")
        except Exception as e:
            print(f"  ⚠ make_antmaze_minari_datasets skipped: {e}")
    except Exception as e:
        print(f"  SKIPPED: {e}")

    print("\n✓ Unit tests completed\n")

