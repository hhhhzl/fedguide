import numpy as np
import json, os
from gymnasium.wrappers import TimeLimit
from fedguide.envs.reacher import generate_reacher_heterogeneity, CustomizedReacherEnv


def build_hetero_config(
        env_name='reacher',
        num_clients=8,
        hetero_type="both",
        variants=("medium-v2", "expert-v2", "random-v2"),
        save_path="./configs/clients/reacher_hetero.json"
):
    """Generate heterogeneity meta for all clients and save to JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    configs = {}
    for i in range(num_clients):
        print(f">>>> build config for client {i}......")
        qpos_range, act_noise, rew_scale, ang_noise = generate_reacher_heterogeneity(i, hetero_type)
        configs[f"client_{i}"] = {
            "variant": variants[i % len(variants)],
            "qpos_high_low": qpos_range,
            "action_noise": act_noise.tolist(),
            "reward_scale": rew_scale,
            "angle_noise": ang_noise
        }
    json.dump(configs, open(save_path, "w"), indent=2)
    print(f"[FedGuide] Saved heterogeneity config → {save_path}")
    return configs


def load_hetero_config(
        client_id,
        env_name='reacher',
        config_path="./configs/clients/reacher_hetero.json",
        max_episode_steps=50
):
    """Load customized Reacher environment based on saved config."""
    cfg = json.load(open(config_path))[f"client_{client_id}"]
    env = TimeLimit(
        CustomizedReacherEnv(
            qpos_high_low=cfg["qpos_high_low"],
            action_noise=np.array(cfg["action_noise"]),
            reward_scale=cfg["reward_scale"],
            angle_noise=cfg["angle_noise"],
            variant=cfg["variant"]
        ),
        max_episode_steps=max_episode_steps
    )
    return env
