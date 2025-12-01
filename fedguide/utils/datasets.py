"""
DEPRECATED: This module has been moved to fedguide.datasets.d4rl_loader
Please update your imports to use: from fedguide.datasets import make_d4rl_datasets
"""
import warnings
warnings.warn(
    "fedguide.utils.datasets is deprecated. "
    "Please use fedguide.datasets.d4rl_loader or fedguide.datasets instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
from fedguide.datasets.d4rl_loader import make_d4rl_datasets, _make_d4rl_datasets
from fedguide.datasets.base import TrajectoryDataset

# Keep original imports for backward compatibility
import gymnasium as gym
import d4rl
import numpy as np
import json, os
from copy import deepcopy
from torch.utils.data import Dataset


# TrajectoryDataset is now imported from fedguide.datasets.base above
# _make_d4rl_datasets is now imported from fedguide.datasets.d4rl_loader above


if __name__ == "__main__":
    from fedguide.datasets.heterogeneity import build_hetero_config, load_hetero_config
    build_hetero_config(
        env_name='reacher',
        num_clients=3,
        hetero_type="both"
    )
    env = load_hetero_config(client_id=2)
    obs, _ = env.reset()
    print("obs shape:", obs.shape)

    datasets = _make_d4rl_datasets(
        env_group="reacher",
        n_clients=3,
        hetero_modes=("state_region", "dyn_shift")
    )
    print(len(datasets))
