"""
Federated HalfCheetah heterogeneity (metadata.json): dynamics + reward preference.

Metadata shape: top-level ``env: "halfcheetah"``, ``clients[]`` with per-client
``mass_scale``, ``damping_scale``, ``ground_friction``, ``action_gain``,
``forward_reward_weight``, ``ctrl_cost_weight``, ``unstable_cost_weight``,
``reset_noise_scale``. Shared state/action space (Gymnasium HalfCheetah-v4).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import Wrapper
except Exception:  # pragma: no cover
    import gym
    from gym import Wrapper


def metadata_is_halfcheetah(metadata_path: str) -> bool:
    if not metadata_path or not os.path.isfile(metadata_path):
        return False
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return str(meta.get("env", "")).lower() == "halfcheetah"


def _apply_dynamics_mujoco(
    mj_model: Any,
    mass_scale: float,
    damping_scale: float,
    ground_friction_scale: float,
) -> None:
    """Scale body masses, dof damping, and floor friction in-place."""
    try:
        import mujoco
    except ImportError as e:  # pragma: no cover
        raise ImportError("mujoco is required for HalfCheetah heterogeneity") from e

    ms = float(mass_scale)
    ds = float(damping_scale)
    gf = float(ground_friction_scale)

    for i in range(1, mj_model.nbody):
        mj_model.body_mass[i] *= ms

    mj_model.dof_damping[:] *= ds

    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if gid >= 0:
        mj_model.geom_friction[gid, :] *= gf


class HalfCheetahActionUnstableWrapper(Wrapper):
    """Apply action gain before physics; subtract unstable cost from reward."""

    def __init__(
        self,
        env: Any,
        *,
        action_gain: float = 1.0,
        unstable_cost_weight: float = 0.0,
    ):
        super().__init__(env)
        self.action_gain = float(action_gain)
        self.unstable_cost_weight = float(unstable_cost_weight)

    @staticmethod
    def _unstable_cost(data: Any) -> float:
        """Pitch (rooty) deviation + angular velocity — mild stability proxy."""
        qpos = data.qpos
        qvel = data.qvel
        pitch = float(qpos[2])
        pitch_vel = float(qvel[2])
        return pitch * pitch + 0.25 * pitch_vel * pitch_vel

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        a = a * self.action_gain
        low = self.action_space.low
        high = self.action_space.high
        a = np.clip(a, low, high)
        obs, rew, terminated, truncated, info = self.env.step(a)
        inner = self.env.unwrapped
        if hasattr(inner, "data") and self.unstable_cost_weight > 0.0:
            u = self._unstable_cost(inner.data)
            rew = float(rew) - self.unstable_cost_weight * u
            info = dict(info) if isinstance(info, dict) else {}
            info["reward_unstable"] = float(u)
        return obs, rew, terminated, truncated, info


def make_hetero_halfcheetah_env_from_metadata(
    metadata_path: str,
    client_index: int,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    render_eval: bool = False,
) -> Any:
    """
    Build one Gymnasium HalfCheetah env from ``metadata.clients[client_index]``.
    """
    from fedguide.utils.mujoco_headless import ensure_mujoco_headless_gl_if_needed
    from fedguide.utils.seeds import set_all_seeds

    if render_mode in ("rgb_array", "video") or render_eval:
        ensure_mujoco_headless_gl_if_needed()

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)
    clients: List[Dict[str, Any]] = meta.get("clients") or []
    if client_index < 0 or client_index >= len(clients):
        raise ValueError(
            f"client_index {client_index} out of range for metadata ({len(clients)} clients)"
        )
    cfg = clients[client_index]
    env_name = str(cfg.get("env_name") or meta.get("env_name") or "HalfCheetah-v4")

    mkw: Dict[str, Any] = {
        "forward_reward_weight": float(cfg["forward_reward_weight"]),
        "ctrl_cost_weight": float(cfg["ctrl_cost_weight"]),
        "reset_noise_scale": float(cfg["reset_noise_scale"]),
    }
    if render_eval and render_mode:
        rm = str(render_mode).lower()
        if rm == "video":
            rm = "rgb_array"
        if rm in ("rgb_array", "human"):
            mkw["render_mode"] = rm

    env = gym.make(env_name, **mkw)
    inner = env.unwrapped
    if not hasattr(inner, "model"):
        raise RuntimeError("Expected MujocoEnv inside HalfCheetah stack")

    _apply_dynamics_mujoco(
        inner.model,
        mass_scale=float(cfg["mass_scale"]),
        damping_scale=float(cfg["damping_scale"]),
        ground_friction_scale=float(cfg["ground_friction"]),
    )

    env = HalfCheetahActionUnstableWrapper(
        env,
        action_gain=float(cfg["action_gain"]),
        unstable_cost_weight=float(cfg["unstable_cost_weight"]),
    )

    if seed is not None:
        set_all_seeds(seed, env)
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

    return env


def make_halfcheetah_env_if_applicable(
    metadata_path: Optional[str],
    client_id: Optional[int],
    seed: Optional[int],
    render_mode: Optional[str] = None,
    render_eval: bool = False,
) -> Optional[Any]:
    """If ``metadata_path`` is federated HalfCheetah metadata, build env; else ``None``."""
    if not metadata_path or not metadata_is_halfcheetah(metadata_path):
        return None
    idx = client_id if client_id is not None else 0
    return make_hetero_halfcheetah_env_from_metadata(
        metadata_path,
        idx,
        seed=seed,
        render_mode=render_mode,
        render_eval=render_eval,
    )
