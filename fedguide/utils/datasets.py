import gymnasium as gym
import d4rl
import numpy as np
import json, os
from copy import deepcopy
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, observations, actions):
        self.obs = np.asarray(observations, dtype=np.float32)
        self.acts = np.asarray(actions, dtype=np.float32)

    def __len__(self): return len(self.obs)

    def __getitem__(self, i):
        return np.concatenate([self.obs[i], self.acts[i]], axis=-1)


def _make_d4rl_datasets(
        env_group="reacher",
        n_clients=3,
        hetero_modes=("task", "state_region", "dyn_shift"),
        save_json="./configs/clients/fedguide_d4rl.json"
):
    """
    Create per-client heterogeneous D4RL datasets
    + Save heterogeneity info for reproducibility
    """
    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    hetero_info = {}

    # Base env name mapping
    base_names = {
        "reacher": "reacher-medium-v2",
        "maze2d": "maze2d-medium-v1",
        "antmaze": "antmaze-medium-play-v0",
    }
    base_env = base_names.get(env_group, "reacher-medium-v2")
    env = gym.make(base_env)
    data = env.get_dataset()

    datasets = []
    idx = np.arange(len(data["observations"]))

    for i in range(n_clients):
        print(f">>>> Make Hetero envs config for client {i}.........")
        obs, acts = deepcopy(data["observations"]), deepcopy(data["actions"])

        # ---------- Task heterogeneity ----------
        variant = base_env
        if "task" in hetero_modes:
            variant_list = [
                base_env.replace("medium", "expert"),
                base_env.replace("medium", "random"),
                base_env
            ]
            variant = variant_list[i % len(variant_list)]
            env_local = gym.make(variant)
            d_local = env_local.get_dataset()
            obs, acts = d_local["observations"], d_local["actions"]

        # ---------- State region heterogeneity ----------
        if "state_region" in hetero_modes:
            x = obs[:, 0]
            mask = x < np.median(x) if i % 2 == 0 else x >= np.median(x)
            obs, acts = obs[mask], acts[mask]

        # ---------- Dynamics heterogeneity ----------
        if "dyn_shift" in hetero_modes:
            scale = np.random.choice([0.95, 1.05])
            acts = acts * scale + np.random.normal(0, 0.01, acts.shape)

        datasets.append(TrajectoryDataset(obs, acts))
        hetero_info[f"client_{i}"] = {
            "env_variant": variant,
            "data_size": len(obs),
            "scaling": float(scale if "dyn_shift" in hetero_modes else 1.0),
            "state_split": "left" if i % 2 == 0 else "right"
        }
    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    json.dump(hetero_info, open(save_json, "w"), indent=2)
    print(f"[FedGuide] Saved D4RL heterogeneity metadata → {save_json}")
    return datasets


if __name__ == "__main__":
    from fedguide.utils.herero import build_hetero_config, load_hetero_config
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
