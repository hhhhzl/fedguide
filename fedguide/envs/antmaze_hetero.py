"""
AntMaze D4RL heterogeneity (metadata.json) and dense-reward helpers.

Metadata shape mirrors data/reacher/metadata.json: top-level n_clients, hetero_type,
seed, variants, clients[] with per-client variant, qpos_high_low, action_noise,
reward_scale, angle_noise. For AntMaze, variant is a full D4RL env id (e.g.
antmaze-umaze-v0). qpos_high_low matches Reacher's grid for compatibility;
action_noise is 8-D (Ant action dim); reward_scale / angle_noise follow Reacher
semantics (angle_noise is stored for parity; not applied to MuJoCo in v1).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import gym as gym_legacy

# Default task variants (cycle by client_id) — umaze / medium / diverse
DEFAULT_ANTMAZE_VARIANTS: Tuple[str, ...] = (
    "antmaze-umaze-v0",
    "antmaze-medium-play-v0",
    "antmaze-umaze-diverse-v0",
)


def generate_antmaze_heterogeneity(
    client_id: int,
    hetero_type: str = "both",
) -> Tuple[List[List[List[float]]], np.ndarray, float, float]:
    """
    Same hetero_type categories as Reacher: iid, init-state, dynamics, reward, both.
    Returns (qpos_high_low, action_noise_8d, reward_scale, angle_noise).
    """
    qpos_high_low = [[-0.2, 0.2], [-0.2, 0.2]]
    action_noise = np.zeros(8, dtype=np.float64)
    reward_scale = 1.0
    angle_noise = 0.0

    if hetero_type == "iid":
        return qpos_high_low, action_noise, reward_scale, angle_noise

    if hetero_type in ("init-state", "both"):
        n = 8
        if client_id >= n * n:
            raise ValueError("client_id exceeds grid size (64 max)")
        row, col = divmod(client_id, n)
        x = -0.2 + row * 0.05
        y = 0.2 - col * 0.05
        qpos_high_low = [[x, x + 0.05], [y - 0.05, y]]

    if hetero_type in ("dynamics", "both"):
        rng = np.random.RandomState(client_id)
        action_noise = np.clip(rng.normal(0.0, 0.12, 8), -0.5, 0.5)

    if hetero_type in ("reward", "both"):
        rng = np.random.RandomState(client_id + 1000)
        reward_scale = float(rng.uniform(0.8, 1.2))

    rng = np.random.RandomState(client_id + 2000)
    angle_noise = float(rng.uniform(-0.05, 0.05))

    return qpos_high_low, action_noise, reward_scale, angle_noise


def make_hetero_antmaze_env_from_metadata(
    metadata_path: str,
    client_index: int,
    seed: Optional[int] = None,
    reward_type: Optional[str] = None,
    render_eval: bool = False,
) -> Any:
    """
    Build one AntMaze D4RL env from metadata.clients[client_index].
    Applies dense/sparse via reward_type (defaults to metadata or 'dense').
    """
    import d4rl  # noqa: F401 — register envs

    from fedguide.utils.mujoco_headless import ensure_mujoco_headless_gl_if_needed
    from fedguide.utils.seeds import set_all_seeds

    ensure_mujoco_headless_gl_if_needed()

    with open(metadata_path, "r") as f:
        meta = json.load(f)
    clients: List[Dict[str, Any]] = meta.get("clients", [])
    if client_index < 0 or client_index >= len(clients):
        raise ValueError(
            f"client_index {client_index} out of range for metadata ({len(clients)} clients)"
        )
    cfg = clients[client_index]
    env_name = str(cfg.get("variant", "antmaze-umaze-v0"))
    rt = reward_type if reward_type is not None else meta.get("reward_type", "dense")

    env = gym_legacy.make(env_name, reward_type=rt)
    env = _apply_d4rl_obs_fix(env, seed)
    env = _HeteroAntMazeActionRewardWrapper(
        env,
        action_noise=np.asarray(cfg["action_noise"], dtype=np.float32),
        reward_scale=float(cfg["reward_scale"]),
    )

    if render_eval:
        _set_gym_mujoco_render_mode_rgb(env)

    if seed is not None:
        set_all_seeds(seed, env)
    return env


class _HeteroAntMazeActionRewardWrapper(gym_legacy.Wrapper):
    """Action noise + reward scaling (old gym API)."""

    def __init__(self, env, action_noise: np.ndarray, reward_scale: float):
        super().__init__(env)
        self.action_noise = np.asarray(action_noise, dtype=np.float32).reshape(-1)
        self.reward_scale = float(reward_scale)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_noise.size:
            if self.action_noise.shape[0] == a.shape[0]:
                a = a + self.action_noise
            else:
                m = min(self.action_noise.shape[0], a.shape[0])
                a[:m] = a[:m] + self.action_noise[:m]
        low = self.action_space.low
        high = self.action_space.high
        a = np.clip(a, low, high)
        out = self.env.step(a)
        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
        else:
            obs, rew, done, info = out
            terminated = bool(done)
            truncated = False
        rew = float(rew) * self.reward_scale
        if len(out) == 5:
            return obs, rew, terminated, truncated, info
        return obs, rew, done, info


def _apply_d4rl_obs_fix(env: Any, seed: Optional[int]) -> Any:
    """Match fedguide.runner.factories._create_d4rl_obs fix for AntMaze observation dim."""
    class _D4RLObservationSpaceFix(gym_legacy.Wrapper):
        def __init__(self, env, obs_dim: int):
            super().__init__(env)
            self.observation_space = gym_legacy.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        env.reset()
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed") and seed is not None:
            env.action_space.seed(seed)
        out = env.reset()
    o0 = out[0] if isinstance(out, tuple) else out
    actual_dim = int(np.asarray(o0, dtype=np.float32).ravel().shape[0])
    decl_dim = int(np.asarray(env.observation_space.shape).prod())
    if actual_dim != decl_dim:
        env = _D4RLObservationSpaceFix(env, actual_dim)
    return env


def _set_gym_mujoco_render_mode_rgb(env) -> None:
    try:
        from gym.envs.mujoco import mujoco_env as _gym_mujoco
    except ImportError:
        return
    cur = env
    for _ in range(32):
        if isinstance(cur, _gym_mujoco.MujocoEnv):
            cur.render_mode = "rgb_array"
            return
        nxt = getattr(cur, "env", None) or getattr(cur, "_wrapped_env", None)
        if nxt is None:
            break
        cur = nxt


def build_d4rl_make_kwargs(
    env_name: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge d4rl_env_kwargs and reward_type for gym.make (reward_type only for antmaze)."""
    extra: Dict[str, Any] = dict(config.get("d4rl_env_kwargs") or {})
    if str(env_name).startswith("antmaze-"):
        if config.get("reward_type") is not None:
            extra["reward_type"] = config["reward_type"]
        elif "reward_type" not in extra:
            extra["reward_type"] = "dense"
    return extra
