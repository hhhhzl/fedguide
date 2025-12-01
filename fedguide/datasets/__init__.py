"""Unified dataset loading interface for d4rl and minari."""
from fedguide.datasets.base import TrajectoryDataset, TrajectoryWindowDataset, TransitionDataset

# from fedguide.datasets.loader import make_datasets, DatasetType
# from fedguide.datasets.heterogeneity import (
#     build_hetero_config,
#     load_hetero_config,
#     split_trajs_dirichlet,
#     traj_category
# )

# Placeholders to avoid import errors
try:
    from fedguide.datasets.loader import make_datasets, DatasetType
except (ImportError, RuntimeError, OSError):
    make_datasets = None
    DatasetType = None

try:
    from fedguide.datasets.heterogeneity import (
        build_hetero_config,
        load_hetero_config,
        split_trajs_dirichlet,
        traj_category
    )
except (ImportError, RuntimeError, OSError):
    build_hetero_config = None
    load_hetero_config = None
    split_trajs_dirichlet = None
    traj_category = None

# Backward compatibility aliases
# _make_d4rl_datasets = make_d4rl_datasets

__all__ = [
    "make_datasets",
    # "make_d4rl_datasets",
    # "make_minari_datasets",
    # "make_maze_minari_datasets",  # backward compatibility
    # "_make_d4rl_datasets",  # backward compatibility
    "DatasetType",
    "TrajectoryDataset",
    "TrajectoryWindowDataset",
    "TransitionDataset",
    "build_hetero_config",
    "load_hetero_config",
    "split_trajs_dirichlet",
    "traj_category",
]
