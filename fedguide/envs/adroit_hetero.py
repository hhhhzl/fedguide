"""D4RL Adroit federated heterogeneity loader (stub).

Adroit = 4 dexterous-hand manipulation tasks (door, hammer, pen, relocate)
each with {human, cloned, expert} D4RL datasets. Federated split:
  client_i ↔ task ∈ {door,hammer,pen,relocate} × dataset ∈ {human,cloned,expert}.

Metadata shape:
  {
    "env": "adroit",
    "n_clients": 8,
    "clients": [
      {"client_id": 0, "task": "door-human-v1"},
      {"client_id": 1, "task": "door-expert-v1"},
      ...
    ]
  }

Each client gets its own dataset for offline prior + SDICE pretrain. Online
federated PPO operates on the corresponding live env. State/action dims
differ per task (door 39D/28D-act, hammer 46D/26D-act, pen 45D/24D-act,
relocate 39D/30D-act) — paper-side this is fine because the per-client prior
is task-specific (no cross-task prior aggregation by default; OT-MoE
collapses to FedAvg-of-experts when shapes mismatch).

NOTE: this module currently *only* provides the env loader. To finish:
  1. ``pip install d4rl`` (already required for halfcheetah/antmaze).
  2. Generate ``data/adroit/metadata.json`` mapping each client to one Adroit
     env+dataset combo.
  3. Pretrain the diffusion prior + SDICE on the matching D4RL dataset (use
     ``_collect_d4rl_dataset`` from halfcheetah pretrain as template).
  4. Add ``configs/adroit/main/fedguide_*.yaml`` and a chain script.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


ADROIT_TASKS = ("door", "hammer", "pen", "relocate")
ADROIT_QUALITIES = ("human", "cloned", "expert")


def metadata_is_adroit(metadata_path: str) -> bool:
    if not metadata_path or not os.path.isfile(metadata_path):
        return False
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return str(meta.get("env", "")).lower() == "adroit"


def make_hetero_adroit_env_from_metadata(
    metadata_path: str,
    client_index: int,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> Any:
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    import gym as old_gym
    import d4rl  # noqa: F401  — registers envs

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)
    clients: List[Dict[str, Any]] = meta.get("clients") or []
    if client_index < 0 or client_index >= len(clients):
        raise ValueError(
            f"client_index {client_index} out of range for metadata "
            f"({len(clients)} clients)"
        )
    cfg = clients[client_index]
    env_id = cfg["task"]
    env = old_gym.make(env_id)
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:
            pass
    return env


def make_adroit_env_if_applicable(
    metadata_path: Optional[str],
    client_id: Optional[int],
    seed: Optional[int],
    render_mode: Optional[str] = None,
) -> Optional[Any]:
    if not metadata_path or not metadata_is_adroit(metadata_path):
        return None
    idx = client_id if client_id is not None else 0
    return make_hetero_adroit_env_from_metadata(
        metadata_path, idx, seed=seed, render_mode=render_mode
    )
