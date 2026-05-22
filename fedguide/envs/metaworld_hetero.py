"""MetaWorld ML10 federated heterogeneity loader (stub).

Each client = one of 10 manipulation tasks from MetaWorld ML10. Tasks share a
unified observation space (39D after preprocessing) and action space (4D)
which makes them straightforward to plug into the existing fedguide
PPO + diffusion-prior pipeline.

Metadata shape (8 or 10 clients):
  {
    "env": "metaworld_ml10",
    "n_clients": 10,
    "clients": [{"client_id": i, "task": <task_name>}, ...]
  }

NOTE: this module currently *only* provides the env loader. To finish the
infrastructure you still need:
  1. Install ``metaworld`` (``pip install metaworld@git+https://github.com/Farama-Foundation/Metaworld.git``).
  2. Generate ``data/metaworld/metadata.json`` listing 10 ML10 task names.
  3. Pretrain the diffusion prior + SDICE on the per-task scripted policies
     (MetaWorld provides them via ``env.policy``).
  4. Add ``configs/metaworld/main/fedguide_*.yaml`` and a chain script.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


METAWORLD_ML10_TRAIN_TASKS = (
    "reach-v3", "push-v3", "pick-place-v3", "door-open-v3",
    "drawer-close-v3", "button-press-topdown-v3", "peg-insert-side-v3",
    "window-open-v3", "sweep-v3", "basketball-v3",
)


def metadata_is_metaworld(metadata_path: str) -> bool:
    if not metadata_path or not os.path.isfile(metadata_path):
        return False
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return str(meta.get("env", "")).lower().startswith("metaworld")


def make_hetero_metaworld_env_from_metadata(
    metadata_path: str,
    client_index: int,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> Any:
    # MuJoCo offscreen renderer needs MUJOCO_GL set before context creation;
    # otherwise env.render() throws inside federated_render's try/except and
    # silently produces no frames (so no mp4 ever lands in render_save_dir).
    from fedguide.utils.mujoco_headless import ensure_mujoco_headless_gl_if_needed

    ensure_mujoco_headless_gl_if_needed()

    try:
        import metaworld  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "metaworld package is required for MetaWorld envs; install it via "
            "pip install metaworld@git+https://github.com/Farama-Foundation/Metaworld.git"
        ) from e

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)
    clients: List[Dict[str, Any]] = meta.get("clients") or []
    if client_index < 0 or client_index >= len(clients):
        raise ValueError(
            f"client_index {client_index} out of range for metadata "
            f"({len(clients)} clients)"
        )
    cfg = clients[client_index]
    task_name = cfg["task"]

    ml10 = metaworld.ML10(seed=seed)
    env_cls = ml10.train_classes.get(task_name) or ml10.test_classes.get(task_name)
    if env_cls is None:
        raise KeyError(f"task {task_name} not in MetaWorld ML10")
    try:
        env = env_cls(render_mode=render_mode) if render_mode else env_cls()
    except TypeError:
        env = env_cls()
    # MetaWorld envs need an explicit task setup before reset.
    tasks_for_name = [t for t in (ml10.train_tasks + ml10.test_tasks) if t.env_name == task_name]
    if not tasks_for_name:
        raise RuntimeError(f"no MetaWorld task instances found for {task_name}")
    env.set_task(tasks_for_name[0])
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:
            pass
    return env


def make_metaworld_env_if_applicable(
    metadata_path: Optional[str],
    client_id: Optional[int],
    seed: Optional[int],
    render_mode: Optional[str] = None,
) -> Optional[Any]:
    if not metadata_path or not metadata_is_metaworld(metadata_path):
        return None
    idx = client_id if client_id is not None else 0
    return make_hetero_metaworld_env_from_metadata(
        metadata_path, idx, seed=seed, render_mode=render_mode
    )