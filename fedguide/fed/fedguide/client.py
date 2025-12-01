from __future__ import annotations

from typing import Any, Dict, Optional, Callable, Iterable
import random
import numpy as np
import torch

try:
    import gymnasium as gym
except Exception:
    import gym  # fallback to classic gym if needed

from fedguide.agents.fedguide_agent import FedguideAgent
from fedguide.trainers.fedguide_trainer import FedguideTrainer
from fedguide.fed.client import FedRLClient


# --------- Helpers ---------
def _is_box1d(space) -> bool:
    try:
        from gymnasium.spaces import Box
    except Exception:
        from gym.spaces import Box
    return isinstance(space, Box) and len(space.shape) == 1


def _make_env(env_id: str, seed: Optional[int] = None):
    if env_id.lower() in ["bandit2d", "bandit_2d", "2dbandit"]:
        from fedguide.envs.bandit2d import Bandit2D
        env = Bandit2D(K=4, sigma=0.2, seed=seed)
        if seed is not None:
            env.reset(seed=seed)
        return env
    
    env = gym.make(env_id)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass
    return env


class FedGuideClient(FedRLClient):
    def __init__(
        self,
        agent: Any,
        env: Any,
        trainer: Any,
        *,
        aggregate_mode: str = "policy",  # 'policy' | 'prior' | 'policy+prior' | 'policy_value' | 'all'
        run_name: Optional[str] = None,
        seed: Optional[int] = None,
        device: Optional[str] = "auto",
        logger: Optional[Any] = None,
        callbacks: Optional[Iterable[Callable[[Dict[str, Any]], None]]] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        logger_level: int = None,
        metrics_collector: Optional[Any] = None,  # Bandit2DMetricsCollector instance
    ):
        super().__init__(
            agent=agent,
            env=env,
            trainer=trainer,
            run_name=run_name,
            seed=seed,
            device=device,
            logger=logger,
            callbacks=callbacks,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            logger_level=(logger_level or 20),
        )
        self.aggregate_mode = (aggregate_mode or "policy").lower()
        self.metrics_collector = metrics_collector

    def get_parameters(self, config: Dict[str, Any]):
        if not hasattr(self.agent, "get_parameters"):
            return super().get_parameters(config)

        full = self.agent.get_parameters()
        mode = self.aggregate_mode

        def pick(keys):
            return {k: v for k, v in full.items() if k in keys and k in full}

        if mode == "policy":
            return pick({"policy", "log_std"})
        elif mode == "policy_value":
            return pick({"policy", "log_std", "value"})
        elif mode == "prior":
            return pick({"prior_adapt"})
        elif mode in ("policy+prior", "policy_prior", "policy-prior"):
            return pick({"policy", "log_std", "prior_adapt"})
        elif mode == "all":
            return full
        else:
            return pick({"policy", "log_std"})

    def set_parameters(self, parameters):
        if not hasattr(self.agent, "set_parameters"):
            return super().set_parameters(parameters)

        mode = self.aggregate_mode
        if not isinstance(parameters, dict):
            return super().set_parameters(parameters)

        allowed = set()
        if mode == "policy":
            allowed = {"policy", "log_std"}
        elif mode == "policy_value":
            allowed = {"policy", "log_std", "value"}
        elif mode == "prior":
            allowed = {"prior_adapt"}
        elif mode in ("policy+prior", "policy_prior", "policy-prior"):
            allowed = {"policy", "log_std", "prior_adapt"}
        elif mode == "all":
            allowed = {"policy", "log_std", "value", "prior_adapt", "guidance"}

        filtered = {k: v for k, v in parameters.items() if k in allowed}
        if filtered:
            self.agent.set_parameters(filtered)
    
    def fit(self, parameters, config):
        """Override fit to collect actions for metrics."""
        # Call parent fit
        result = super().fit(parameters, config)
        
        # Collect actions for metrics visualization (if collector is available)
        if self.metrics_collector is not None:
            try:
                # Get client ID
                cid = getattr(self, "cid", config.get("cid", "unknown"))
                client_id = int(cid) if isinstance(cid, (int, str)) and str(cid).isdigit() else hash(cid) % 10000
                
                # Get actions from trainer's last rollout
                if hasattr(self.trainer, 'last_actions') and self.trainer.last_actions is not None:
                    actions = self.trainer.last_actions
                    self.metrics_collector.collect_client_actions(client_id, actions)
            except Exception as e:
                # Silently fail if collection fails
                pass
        
        return result


# --------- client_fn_builder ----------
def client_fn_builder(
    env_id: str,
    algo: str = "ppo",
    *,
    aggregate_mode: str = "policy",
    n_steps: int = 200,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    lambda_local: float = 0.0,
    lambda_guide: float = 1.0,
    online_guidance: bool = False,
    online_prior: bool = False,
    # logging
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    metrics_collector: Optional[Any] = None,  # Bandit2DMetricsCollector instance
):

    def client_fn(context) -> Any:
        # 1) per-client seed
        cid = str(getattr(context, "client_id", None) or getattr(context, "node_id", None) or "0")
        base = 42 + (abs(hash(cid)) % 10000)
        random.seed(base)
        np.random.seed(base)
        torch.manual_seed(base)

        # 2) env
        # TODO: load env from config
        env = _make_env(env_id, seed=base)
        obs_space, act_space = env.observation_space, env.action_space
        assert _is_box1d(obs_space) and _is_box1d(act_space), "Only Support 1D Box spaces."

        state_dim = int(obs_space.shape[0])
        action_dim = int(act_space.shape[0])

        prior, guidance = None, None
        # Todo: load prior and guidance

        # 4) agent
        agent = FedguideAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            prior=prior,
            guidance=guidance,
            # lr=3e-4, clip_eps=0.2, entropy_coef=0.02, value_coef=0.5, ...
        )

        # 5) trainer
        trainer = FedguideTrainer(
            agent=agent,
            env=env,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            lambda_local=lambda_local,
            lambda_guide=lambda_guide,
            online_guidance=online_guidance,
            online_prior=online_prior,
        )

        # client
        client = FedGuideClient(
            agent=agent,
            env=env,
            trainer=trainer,
            aggregate_mode=aggregate_mode,
            run_name=run_name or f"{env_id}-{algo}-cid{cid}",
            seed=base,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            metrics_collector=metrics_collector,
        )
        # Store client_id for metrics collection
        client.cid = cid
        return client.to_client() if hasattr(client, "to_client") else client

    return client_fn
