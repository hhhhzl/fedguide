"""Federated MuJoCo locomotion (Walker2D, Ant, Hopper) heterogeneity loader.

Mirrors ``halfcheetah_hetero`` for the three other locomotion envs. Each
client's ``metadata.clients[i]`` provides:
  * ``env_name``: gym id (``Walker2d-v4`` / ``Ant-v4`` / ``Hopper-v4``)
  * ``mass_scale``, ``damping_scale``, ``ground_friction``: dynamics
  * ``action_gain``: applied as multiplicative gain before clipping
  * ``forward_reward_weight``, ``ctrl_cost_weight``, ``reset_noise_scale``: rewards
  * ``contact_cost_weight``: ant-only
  * ``unstable_cost_weight``: extra pitch/roll penalty (Walker / Hopper / Ant /
    HalfCheetah-style; default 0)

The metadata top-level ``env`` field selects the loader path.
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


_LOCOMOTION_ENVS = {"walker", "ant", "hopper"}


def metadata_is_locomotion(metadata_path: str) -> Optional[str]:
    """Return the env name (``walker`` / ``ant`` / ``hopper``) if metadata is
    for one of these locomotion envs, else None."""
    if not metadata_path or not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    name = str(meta.get("env", "")).lower()
    return name if name in _LOCOMOTION_ENVS else None


def _apply_dynamics_mujoco(
    mj_model: Any,
    mass_scale: float,
    damping_scale: float,
    ground_friction_scale: float,
) -> None:
    try:
        import mujoco
    except ImportError as e:  # pragma: no cover
        raise ImportError("mujoco is required for locomotion heterogeneity") from e

    for i in range(1, mj_model.nbody):
        mj_model.body_mass[i] *= float(mass_scale)
    mj_model.dof_damping[:] *= float(damping_scale)
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if gid >= 0:
        mj_model.geom_friction[gid, :] *= float(ground_friction_scale)


class _LocomotionWrapper(Wrapper):
    """Apply action gain + optional unstable-cost shaping. Generic for
    Walker2D / Ant / Hopper / HalfCheetah-style envs.
    """

    def __init__(self, env: Any, *, action_gain: float = 1.0, unstable_cost_weight: float = 0.0):
        super().__init__(env)
        self.action_gain = float(action_gain)
        self.unstable_cost_weight = float(unstable_cost_weight)

    @staticmethod
    def _unstable_cost(data: Any) -> float:
        # Pitch (rooty / orientation) and its derivative — penalize large lean.
        try:
            qpos = data.qpos
            qvel = data.qvel
            # walker / hopper / halfcheetah have rooty at qpos[2]; ant uses
            # quaternions starting at qpos[3] but qpos[2] is z height -- skip
            # the pitch term for ant (just penalize angular velocity magnitude).
            tilt = float(qpos[2]) if qpos.shape[0] > 2 else 0.0
            ang_v = float(np.linalg.norm(qvel[2:5])) if qvel.shape[0] >= 5 else 0.0
            return tilt * tilt + 0.25 * ang_v * ang_v
        except Exception:
            return 0.0

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        a = a * self.action_gain
        a = np.clip(a, self.action_space.low, self.action_space.high)
        obs, rew, terminated, truncated, info = self.env.step(a)
        inner = self.env.unwrapped
        if hasattr(inner, "data") and self.unstable_cost_weight > 0.0:
            u = self._unstable_cost(inner.data)
            rew = float(rew) - self.unstable_cost_weight * u
            info = dict(info) if isinstance(info, dict) else {}
            info["reward_unstable"] = float(u)
        return obs, rew, terminated, truncated, info


def make_hetero_locomotion_env_from_metadata(
    metadata_path: str,
    client_index: int,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    render_eval: bool = False,
) -> Any:
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
    env_name = str(cfg.get("env_name") or meta.get("env_name"))

    mkw: Dict[str, Any] = {
        "ctrl_cost_weight": float(cfg["ctrl_cost_weight"]),
        "reset_noise_scale": float(cfg["reset_noise_scale"]),
    }
    is_ant = env_name.lower().startswith("ant")
    # Ant-v4 does not accept forward_reward_weight via gym.make; we still
    # capture it for the wrapper's post-hoc reward scaling.
    if not is_ant:
        mkw["forward_reward_weight"] = float(cfg["forward_reward_weight"])
    if "contact_cost_weight" in cfg:
        mkw["contact_cost_weight"] = float(cfg["contact_cost_weight"])
    if is_ant:
        # D4RL `ant-medium-v2` (which we use for the prior + BC pretrain)
        # carries the full 111-dim observation INCLUDING contact forces.
        # Ant-v4 default is 27-dim (contact forces excluded), which makes the
        # diffusion-prior checkpoint un-loadable at federation time (shape
        # mismatch on the first conv: checkpoint expects 111+8=119 channels,
        # current env emits 27+8=35). Force contact forces on so runtime
        # obs_dim matches pretrain obs_dim.
        mkw["use_contact_forces"] = True
        # Ant-v4 default healthy_reward=1.0 adds +1 per step (max +1000 per
        # episode for the full 1000-step rollout). That alive bonus is a
        # floor: even a policy that stands still and never walks earns ~+1000.
        # For locomotion benchmarking the bonus masks training progress
        # (FedAvg looks "at +1000" while not actually walking). We zero it so
        # the reported eval/return == forward_reward - ctrl_cost - contact_cost,
        # i.e. genuine locomotion quality.
        mkw["healthy_reward"] = 0.0

    if render_eval and render_mode:
        rm = str(render_mode).lower()
        if rm == "video":
            rm = "rgb_array"
        if rm in ("rgb_array", "human"):
            mkw["render_mode"] = rm

    env = gym.make(env_name, **mkw)
    inner = env.unwrapped
    if not hasattr(inner, "model"):
        raise RuntimeError(f"Expected MujocoEnv inside {env_name} stack")

    _apply_dynamics_mujoco(
        inner.model,
        mass_scale=float(cfg["mass_scale"]),
        damping_scale=float(cfg["damping_scale"]),
        ground_friction_scale=float(cfg.get("ground_friction", 1.0)),
    )

    env = _LocomotionWrapper(
        env,
        action_gain=float(cfg.get("action_gain", 1.0)),
        unstable_cost_weight=float(cfg.get("unstable_cost_weight", 0.0)),
    )

    if seed is not None:
        set_all_seeds(seed, env)
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
    return env


def make_locomotion_env_if_applicable(
    metadata_path: Optional[str],
    client_id: Optional[int],
    seed: Optional[int],
    render_mode: Optional[str] = None,
    render_eval: bool = False,
) -> Optional[Any]:
    if not metadata_path:
        return None
    name = metadata_is_locomotion(metadata_path)
    if name is None:
        return None
    idx = client_id if client_id is not None else 0
    return make_hetero_locomotion_env_from_metadata(
        metadata_path,
        idx,
        seed=seed,
        render_mode=render_mode,
        render_eval=render_eval,
    )
