import gymnasium as gym
import numpy as np
import json, os
from typing import Optional

from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv
from gymnasium.wrappers import TimeLimit
from gymnasium.spaces import Box, Discrete
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch


# ====================================================
# HETEROGENEITY GENERATOR
# ====================================================

def generate_reacher_heterogeneity(client_id, hetero_type="both"):
    """
    hetero_type ∈ ['iid', 'init-state', 'dynamics', 'reward', 'both']
    """
    qpos_high_low = [[-0.2, 0.2], [-0.2, 0.2]]
    action_noise = np.zeros(2)
    reward_scale = 1.0
    angle_noise = 0.0

    if hetero_type == "iid":
        return qpos_high_low, action_noise, reward_scale, angle_noise

    # ---------------- Goal region heterogeneity ----------------
    if hetero_type in ["init-state", "both"]:
        n = 8  # 8x8 grid (64 clients)
        if client_id >= n * n:
            raise ValueError("client_id exceeds grid size (64 max)")
        row, col = divmod(client_id, n)
        x = -0.2 + row * 0.05
        y = 0.2 - col * 0.05
        qpos_high_low = [[x, x + 0.05], [y - 0.05, y]]

    # ---------------- Dynamics heterogeneity ----------------
    if hetero_type in ["dynamics", "both"]:
        # Use client_id as seed for deterministic generation
        rng = np.random.RandomState(client_id)
        action_noise = np.clip(rng.normal(0.0, 0.3, 2), -1.0, 1.0)

    # ---------------- Reward heterogeneity ----------------
    if hetero_type in ["reward", "both"]:
        # Use client_id as seed for deterministic generation
        rng = np.random.RandomState(client_id + 1000)  # Offset to avoid correlation with action_noise
        reward_scale = rng.uniform(0.8, 1.2)

    # ---------------- Randomized angle ----------------
    # Use client_id as seed for deterministic generation
    rng = np.random.RandomState(client_id + 2000)  # Offset to avoid correlation
    angle_noise = rng.uniform(-0.05, 0.05)

    return qpos_high_low, action_noise, reward_scale, angle_noise


def wrap_vector_envs(env):
    return env

    env.step_ = env.step
    env.reset_ = env.reset

    def new_step(action):
        obs, r, d, i = env.step_(action)
        new_obs = []
        for o in obs:
            new_obs.append(remove_state_about_target_position(o))
        obs = np.array(new_obs)
        return obs, r, d, i

    def new_reset():
        obs = env.reset_()
        new_obs = []
        for o in obs:
            new_obs.append(remove_state_about_target_position(o))
        obs = np.array(new_obs)
        return obs

    env.step = new_step
    env.reset = new_reset
    return env


def remove_state_about_target_position(state):
    state = np.concatenate([state[0:2], state[4:10]], axis=0)
    return state


class Reacher(object):
    def __init__(
            self,
            seed=None,
            qpos_high_low=[[-0.2, 0.2], [-0.2, 0.2]],
            qvel_high_low=[-0.005, 0.005],
            action_noise=np.zeros(2)
    ):
        self.seed = seed

        # Parallel envs for fast rollout.
        def make_env(seed):
            def _f():
                env = TimeLimit(
                    CustomizedReacherEnv(
                        qpos_high_low, qvel_high_low, action_noise),
                    max_episode_steps=50)
                env.seed(seed)
                return env

            return _f

        # Warmup and make sure subprocess is ready, if any.
        self.make_env = make_env
        self.env = DummyVecEnv([make_env(seed)])
        self.env = wrap_vector_envs(self.env)
        self.env.reset()

        # Create environment meta.
        env = make_env(0)()
        # env = wrap_vector_envs(env)
        # self.state_dim = 8  # self.env.observation_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Box):
            self.num_actions = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, Discrete):
            self.num_actions = self.env.action_space.n
        self.is_continuous = True
        # Dataset.
        # state = env.reset()
        self.env_sample = {
            'observations': [[np.zeros(shape=self.state_dim)]],
            'actions': [np.zeros(shape=self.num_actions)],
            'seq_mask': [0],
            'reward': [[0]],
            'dfr': [0],
        }
        self.output_types = {
            'observations': torch.dtypes.float32,
            'actions': torch.dtypes.float32,
            'seq_mask': torch.dtypes.int32,
            'reward': torch.dtypes.float32,
            'dfr': torch.dtypes.float32,
        }
        self.output_shapes = {
            'observations': [None, self.state_dim],
            'actions': [None, self.num_actions],
            'seq_mask': [None],
            'reward': [None, 1],
            'dfr': [None],
        }
        env.close()

    def get_single_envs(self):
        return self.env

    def get_parallel_envs(self, parallel):
        envs = SubprocVecEnv(
            [self.make_env(self.seed + 1 + j) for j in range(parallel)],
            start_method='fork')
        envs = wrap_vector_envs(envs)
        return envs

    def is_solved(self, episode_history):
        return False

    def render(self):
        return self.env.render()

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, r, d, i = self.env.step([action])
        return obs[0], r[0], d[0], i[0]

    def cleanup(self):
        self.env.close()


def make_hetero_reacher_env_from_metadata(
    metadata_path: str,
    client_index: int,
    seed: Optional[int] = None,
):
    """
    Build a single-client Reacher env from data/reacher/metadata.json (or compatible file).
    Used by federated clients so each mapped_client_id matches metadata.clients[client_index].
    """
    from gymnasium.wrappers import TimeLimit

    from fedguide.utils.seeds import set_all_seeds

    with open(metadata_path, "r") as f:
        meta = json.load(f)
    clients = meta.get("clients", [])
    if client_index < 0 or client_index >= len(clients):
        raise ValueError(
            f"client_index {client_index} out of range for metadata ({len(clients)} clients)"
        )
    cfg = clients[client_index]
    env = TimeLimit(
        CustomizedReacherEnv(
            qpos_high_low=cfg["qpos_high_low"],
            action_noise=np.asarray(cfg["action_noise"], dtype=np.float64),
            reward_scale=float(cfg["reward_scale"]),
            angle_noise=float(cfg["angle_noise"]),
            variant=cfg.get("variant", "medium-v2"),
        ),
        max_episode_steps=50,
    )
    if seed is not None:
        set_all_seeds(seed, env)
    return env


class CustomizedReacherEnv(ReacherEnv):
    """Customized Reacher Environment with heterogeneity support."""

    def __init__(self,
                 qpos_high_low=[[-0.2, 0.2], [-0.2, 0.2]],
                 qvel_high_low=[-0.005, 0.005],
                 action_noise=np.zeros(2),
                 reward_scale=1.0,
                 angle_noise=0.0,
                 variant="medium-v2",
                 **kwargs):
        super().__init__(**kwargs)
        self.qpos_high_low = qpos_high_low
        self.qvel_high_low = qvel_high_low
        self.action_noise = action_noise
        self.reward_scale = reward_scale
        self.angle_noise = angle_noise
        self.variant = variant

    def reset_model(self):
        """Randomize target goal and joint angles."""
        qpos = (
                self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
                + self.init_qpos
        )

        # ---- Goal Position ----
        while True:
            x = self.np_random.uniform(self.qpos_high_low[0][0],
                                       self.qpos_high_low[0][1], size=1)[0]
            y = self.np_random.uniform(self.qpos_high_low[1][0],
                                       self.qpos_high_low[1][1], size=1)[0]
            self.goal = np.array([x, y])
            if np.linalg.norm(self.goal) < 0.25:
                break
        qpos[-2:] = self.goal

        # ---- Random Initial Angle ----
        qpos[:2] += np.random.uniform(-self.angle_noise, self.angle_noise, 2)

        qvel = self.init_qvel + self.np_random.uniform(
            low=self.qvel_high_low[0],
            high=self.qvel_high_low[1],
            size=self.model.nv,
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        """Add heterogeneity-induced action noise + reward scaling."""
        noisy_action = action + self.action_noise
        obs, reward, terminated, truncated, info = super().step(noisy_action)
        return obs, reward * self.reward_scale, terminated, truncated, info
