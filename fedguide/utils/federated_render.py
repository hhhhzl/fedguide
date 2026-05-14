"""
Optional evaluation rendering for federated baselines (MuJoCo / Gymnasium).

Mirrors the behavior of CentralPPOTrainer: save mp4 when render_mode is \"video\",
collect rgb frames when render_mode is \"rgb_array\" or \"video\", and call env.render()
for \"human\".
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

import numpy as np


def should_render_this_round(
    server_round: int,
    render_eval: bool,
    render_every_n_rounds: int,
    *,
    include_first_round: bool = False,
) -> bool:
    if not render_eval:
        return False
    if render_every_n_rounds == -1:
        return True
    if render_every_n_rounds == 0:
        # Match CentralPPOTrainer: last-round-only is not handled here
        return False
    # Do not force round 1 by default: N Flower clients × MuJoCo rgb + video
    # on one machine often stalls round 1 for a long time.
    if include_first_round and server_round == 1:
        return True
    return server_round % render_every_n_rounds == 0


def _federated_render_skip_client(client_tag: str) -> bool:
    """
    Skip rendering unless this client is selected.

    Default: only mapped client "0". Set FEDGUIDE_FEDERATED_RENDER_ALL_CLIENTS=1
    for every client, or FEDGUIDE_FEDERATED_RENDER_CLIENT_TAG=k for one client id.
    """
    if os.environ.get("FEDGUIDE_FEDERATED_RENDER_ALL_CLIENTS", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    allowed = os.environ.get("FEDGUIDE_FEDERATED_RENDER_CLIENT_TAG", "0").strip()
    if allowed == "":
        return False
    return str(client_tag) != allowed


def _env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
        return obs, reward, done, info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def _sanitize_tag(tag: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(tag))
    return s[:96] if len(s) > 96 else s


def maybe_save_federated_eval_video(
    env: Any,
    *,
    server_round: int,
    render_eval: bool,
    render_mode: str,
    render_save_dir: Optional[str],
    render_every_n_rounds: int,
    render_episodes: int,
    eval_episodes: int,
    client_tag: str,
    act_fn: Callable[[Any], Any],
    max_steps: int = 2000,
) -> Optional[str]:
    """
    Run short eval rollouts with optional frame capture; save mp4 when mode is \"video\".

    act_fn(obs) must return an action suitable for env.step (numpy or scalar).
    """
    if _federated_render_skip_client(client_tag):
        return None
    if not should_render_this_round(server_round, render_eval, render_every_n_rounds):
        return None
    if render_mode not in ("video", "rgb_array", "human"):
        return None

    frames: List[np.ndarray] = []
    n_vis = min(int(render_episodes), int(eval_episodes), 32)

    for ep_idx in range(n_vis):
        episode_frames: List[np.ndarray] = []
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        steps = 0
        try:
            if render_mode in ("rgb_array", "video"):
                fr0 = env.render()
                if fr0 is not None:
                    episode_frames.append(np.asarray(fr0))
        except Exception:
            pass
        while not done and steps < max_steps:
            action = act_fn(obs)
            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action[0]
            try:
                if render_mode in ("rgb_array", "video"):
                    fr = env.render()
                    if fr is not None:
                        episode_frames.append(np.asarray(fr))
                elif render_mode == "human":
                    env.render()
            except Exception:
                pass
            obs, _r, done, _info = _env_step(env, action)
            steps += 1
        if episode_frames:
            frames.extend(episode_frames)

    video_path = None
    if (
        frames
        and render_save_dir
        and render_mode == "video"
    ):
        try:
            import imageio

            sub = os.path.join(str(render_save_dir), f"client_{_sanitize_tag(client_tag)}")
            os.makedirs(sub, exist_ok=True)
            video_path = os.path.join(sub, f"round_{int(server_round):04d}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(
                f"  [FederatedRender] Saved evaluation video to {video_path}",
                flush=True,
            )
        except ImportError:
            print(
                "  [FederatedRender] imageio not installed; cannot save video.",
                flush=True,
            )
        except Exception as e:
            print(f"  [FederatedRender] Failed to save video: {e}", flush=True)

    return video_path


def reacher_env_render_mode_from_config(render_eval: bool, render_mode_yaml: str) -> Optional[str]:
    """Map YAML render_mode to gymnasium env render_mode for Reacher (rgb_array enables env.render())."""
    if not render_eval:
        return None
    rm = (render_mode_yaml or "video").lower()
    if rm in ("video", "rgb_array"):
        return "rgb_array"
    if rm == "human":
        return "human"
    return None
