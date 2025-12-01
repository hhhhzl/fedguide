"""
FedKL Client Implementation

This module implements the FedKL client that extends FedRLClient
and provides client function builder for easy instantiation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Callable, Iterable
import random
import numpy as np
import torch

try:
    import gymnasium as gym
except Exception:
    import gym  # fallback to classic gym if needed

from fedguide.fed.client import FedRLClient


# --------- Helpers ---------
def _is_box1d(space) -> bool:
    try:
        from gymnasium.spaces import Box
    except Exception:
        from gym.spaces import Box
    return isinstance(space, Box) and len(space.shape) == 1


def _make_env(env_id: str, seed: Optional[int] = None):
    env = gym.make(env_id)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass
    return env


class FedKLClient(FedRLClient):
    """
    FedKL Client implementation.
    
    This client extends FedRLClient and only aggregates the policy parameters,
    keeping the value network local. The KL penalties are handled in the trainer.
    """
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        trainer: Any,
        *,
        aggregate_mode: str = "policy",  #no need for this really
        run_name: Optional[str] = None,
        seed: Optional[int] = None,
        device: Optional[str] = "auto",
        logger: Optional[Any] = None,
        callbacks: Optional[Iterable[Callable[[Dict[str, Any]], None]]] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        logger_level: int = None,
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
        self.aggregate_mode = aggregate_mode
    
    def get_parameters(self, config: Dict[str, Any]):
        """Get parameters for federated aggregation (policy only by default)."""
        if not hasattr(self.agent, "get_parameters"):
            return super().get_parameters(config)
        
        full_params = self.agent.get_parameters()
        
        if self.aggregate_value:
            # Return all parameters
            return full_params
        
        mode = self.aggregate_mode
        def pick(keys):
            return {k: v for k, v in full_params.items() if k in keys and k in full_params}

        if mode == "policy":
            return pick({"policy", "log_std"})
        else:
            return pick({"policy", "log_std"})
    
    def set_parameters(self, parameters):
        """Set parameters from federated aggregation."""
        if not hasattr(self.agent, "set_parameters"):
            return super().set_parameters(parameters)
        
        if not isinstance(parameters, dict):
            return super().set_parameters(parameters)
        
        # If not aggregating value, remove value parameters
        parameters = {
            k: v for k, v in parameters.items()
            if k in ["policy", "log_std"]
        }
        
        if parameters:
            self.agent.set_parameters(parameters)
    

# --------- client_fn_builder ----------
def client_fn_builder(
    env_id: str,
    algo: str = "fedkl",
    *,
    aggregate_mode: str = "policy",
    n_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    update_epochs: int = 10,
    minibatch_size: int = 64,
    lambda_global: float = 0.1,  # Global KL penalty
    lambda_local: float = 0.05,  # Local KL penalty
    max_grad_norm: float = 0.5,
    # Network architecture
    hidden_dim: int = 256,
    lr: float = 3e-4,
    # Logging
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
):
    """
    Build client function for FedKL.
    
    Args:
        env_id: Environment ID (e.g., "HalfCheetah-v4")
        algo: Algorithm name (default: "fedkl")
        aggregate_value: Whether to aggregate value network
        n_steps: Number of steps per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_eps: PPO clip epsilon
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
        update_epochs: Number of policy update epochs
        minibatch_size: Minibatch size for updates
        lambda_global: Global KL penalty coefficient
        lambda_local: Local KL penalty coefficient
        max_grad_norm: Maximum gradient norm for clipping
        hidden_dim: Hidden layer dimension
        lr: Learning rate
        use_wandb: Use wandb logging
        wandb_project: Wandb project name
        run_name: Run name prefix
    """
    
    def client_fn(context) -> Any:
        # Import here to avoid circular imports
        from fedguide.baselines.fedKL.agent import FedKLAgent
        from fedguide.baselines.fedKL.trainer import FedKLTrainer
        
        # 1) per-client seed
        cid = str(getattr(context, "client_id", None) or getattr(context, "node_id", None) or "0")
        base = 42 + (abs(hash(cid)) % 10000)
        random.seed(base)
        np.random.seed(base)
        torch.manual_seed(base)
        
        # 2) env
        env = _make_env(env_id, seed=base)
        obs_space, act_space = env.observation_space, env.action_space
        assert _is_box1d(obs_space) and _is_box1d(act_space), "Only support 1D Box spaces."
        
        state_dim = int(obs_space.shape[0])
        action_dim = int(act_space.shape[0])
        
        # 3) agent
        agent = FedKLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            device="cpu",  # Will be moved to correct device by client
        )
        
        # 4) trainer
        trainer = FedKLTrainer(
            agent=agent,
            env=env,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            lambda_global=lambda_global,
            lambda_local=lambda_local,
            max_grad_norm=max_grad_norm,
            device="cpu",
        )
        
        # 5) client
        client = FedKLClient(
            agent=agent,
            env=env,
            trainer=trainer,
            aggregate_mode=aggregate_mode,
            run_name=run_name or f"{env_id}-{algo}-cid{cid}",
            seed=base,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )
        return client.to_client() if hasattr(client, "to_client") else client
    
    return client_fn