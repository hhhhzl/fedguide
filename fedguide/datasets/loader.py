"""Unified dataset loader interface."""
from enum import Enum
from typing import List, Optional
from torch.utils.data import Dataset


class DatasetType(Enum):
    """Dataset type enumeration."""
    D4RL = "d4rl"
    MINARI = "minari"


def make_datasets(
    dataset_type: DatasetType,
    n_clients: int,
    dataset_id: Optional[str] = None,
    env_group: Optional[str] = None,
    **kwargs
) -> List[Dataset]:
    """
    Unified interface to load datasets from d4rl or minari.
    
    Args:
        dataset_type: DatasetType.D4RL or DatasetType.MINARI
        n_clients: Number of client datasets to create
        dataset_id: For minari, the dataset ID (e.g., "D4RL/pointmaze/medium-v2")
                    For d4rl, this can be used as env_group if env_group is None
        env_group: For d4rl, the environment group name (e.g., "reacher", "maze2d")
        **kwargs: Additional arguments passed to specific loaders
                  For minari: alpha, seed, horizon, stride
                  For d4rl: hetero_modes, save_json
        
    Returns:
        List of Dataset objects, one per client
    """
    if dataset_type == DatasetType.MINARI:
        from .minari_loader import make_minari_datasets
        if dataset_id is None:
            raise ValueError("dataset_id is required for MINARI dataset type")
        return make_minari_datasets(
            n_clients=n_clients,
            dataset_id=dataset_id,
            **kwargs
        )
    elif dataset_type == DatasetType.D4RL:
        from .d4rl_loader import make_d4rl_datasets
        env_group = env_group or dataset_id
        if env_group is None:
            raise ValueError("env_group or dataset_id is required for D4RL dataset type")
        return make_d4rl_datasets(
            n_clients=n_clients,
            env_group=env_group,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

