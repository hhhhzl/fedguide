from typing import Any, Dict, Optional, Callable, Iterable
import numpy as np
import torch


def set_all_seeds(seed: Optional[int], env: Any = None):
    if seed is None:
        return
    import random as pyrandom
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Gym-like env seeding if available
    try:
        if hasattr(env, "reset"):
            try:
                env.reset(seed=seed)
            except TypeError:
                # old Gym API
                env.seed(seed)
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
    except Exception:
        pass
