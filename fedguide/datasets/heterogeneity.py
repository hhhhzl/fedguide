"""Heterogeneity utilities for dataset splitting and environment configuration."""
import numpy as np
import json
import os
from gymnasium.wrappers import TimeLimit
from fedguide.envs.reacher import generate_reacher_heterogeneity, CustomizedReacherEnv


def traj_category(traj, n_bins=4):
    """Map trajectory to category index based on final position."""
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


def split_trajs_dirichlet(trajs, n_clients=8, alpha=0.5, n_bins=4, seed=42):
    """Dirichlet-based non-iid split."""
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

