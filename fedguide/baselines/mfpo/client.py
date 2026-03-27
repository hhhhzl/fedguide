"""
Flower client for MFPO — same wiring as FedRL (NumPyClient + parameter dict).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from fedguide.baselines.fedrl.client import FedRLClient, _make_env
from fedguide.baselines.mfpo.agent import (
    MFPOAgent,
    MFPOContinuousWorker,
    MFPODiscreteCartPoleWorker,
)
from fedguide.baselines.mfpo.trainer import MFPTrainer
from fedguide.utils.gym_space_utils import is_box1d as _is_box1d


def _default_method_conf(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "learning_rate_a": float(config.get("learning_rate_a", config.get("lr_a", 1e-4))),
        "learning_rate_c": float(config.get("learning_rate_c", config.get("lr_c", 1e-4))),
        "gamma": float(config.get("gamma", 0.99)),
        "eps": float(config.get("eps", 1e-5)),
        "average_type": str(config.get("average_type", "target")),
        "fault_type": config.get("fault_type"),
        "c": float(config.get("c", 3.0)),
        "decay_rate": float(config.get("decay_rate", 0.99)),
        "decay_start_iter_id": int(config.get("decay_start_iter_id", 500)),
        "mfpo_test_episodes": int(config.get("mfpo_test_episodes", 10)),
    }


def client_fn_builder(
    env_id: str,
    *,
    batch_size: int = 20,
    local_update: int = 10,
    device: str = "cpu",
    num_clients: Optional[int] = None,
    cid_mapping_file: Optional[str] = None,
    metadata_path: Optional[str] = None,
    render_eval: bool = False,
    render_mode: str = "video",
    render_save_dir: Optional[str] = None,
    render_every_n_rounds: int = 10,
    render_episodes: int = 5,
    reacher_render_mode: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    metrics_collector: Optional[Any] = None,
    **config_overrides: Any,
):
    """Build Flower client_fn for MFPO (1:1 with MFPO-INFOCOM24)."""

    def client_fn(context) -> Any:
        cid = str(getattr(context, "client_id", None) or getattr(context, "node_id", None) or "0")

        num_c = num_clients or 4
        if cid_mapping_file:
            from fedguide.utils.client_id_mapping import get_mapped_client_id

            mapped_client_id = get_mapped_client_id(cid, num_c, cid_mapping_file)
        else:
            if num_clients is not None:
                mapped_client_id = abs(hash(cid)) % num_clients
            else:
                mapped_client_id = abs(hash(cid)) % 100

        base = 42 + (abs(hash(cid)) % 10000)
        random.seed(base)
        np.random.seed(base)
        torch.manual_seed(base)

        merged = {**config_overrides}
        method_conf = _default_method_conf(merged)

        env = _make_env(
            env_id,
            seed=base,
            client_id=mapped_client_id,
            num_clients=num_clients,
            metadata_path=metadata_path,
            render_mode=reacher_render_mode,
        )

        try:
            from gymnasium.spaces import Discrete
        except Exception:
            from gym.spaces import Discrete

        is_discrete = isinstance(env.action_space, Discrete)
        is_cartpole = str(env_id).lower() in ("cartpole-v1", "cartpole_v1")

        if is_discrete and is_cartpole:
            worker = MFPODiscreteCartPoleWorker(env, method_conf, device=device)
        elif _is_box1d(env.action_space) and _is_box1d(env.observation_space):
            worker = MFPOContinuousWorker(env, method_conf, device=device)
        else:
            raise ValueError(
                f"MFPO baseline supports continuous Box envs or CartPole-v1; got env_id={env_id}"
            )

        agent = MFPOAgent(worker, average_type=method_conf["average_type"], device=device)
        trainer = MFPTrainer(
            agent=agent,
            env=env,
            batch_size=batch_size,
            local_update=local_update,
            device=device,
        )

        client = FedRLClient(
            agent=agent,
            env=env,
            trainer=trainer,
            run_name=run_name or f"{env_id}-mfpo-cid{cid}",
            seed=base,
            device=device,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            metrics_collector=metrics_collector,
        )
        client.cid = cid
        return client.to_client()

    return client_fn
