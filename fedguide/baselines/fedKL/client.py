"""
FedKL Client Implementation

This module implements the FedKL client that extends FedRLClient.
Only the policy parameters are aggregated; value networks remain local.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Callable, Iterable
import random
import numpy as np
import torch

try:
    import gymnasium as gym
except Exception:
    import gym

from fedguide.fed.client import FedRLClient

def _is_box1d(space) -> bool:
    try:
        from gymnasium.spaces import Box
    except Exception:
        from gym.spaces import Box
    return isinstance(space, Box) and len(space.shape) == 1


def _make_env(env_id: str, seed: Optional[int] = None):
    # Support for Bandit2D custom environment
    if env_id.lower() in ["bandit2d", "bandit_2d", "2dbandit"]:
        from fedguide.envs.bandit2d import Bandit2D
        env = Bandit2D(K=4, sigma=0.2, seed=seed)
        if seed is not None:
            env.reset(seed=seed)
        return env
    
    # Standard gymnasium environments
    env = gym.make(env_id)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass
    return env


class FedKLClient(FedRLClient):
    """
    FedKL Client implementation.
    
    Key design principles:
    - Only POLICY parameters are aggregated across clients
    - VALUE networks remain LOCAL to each client
    - This is because value functions are environment-specific and don't
      generalize well across different client distributions
    """
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        trainer: Any,
        *,
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
    def get_parameters(self, config: Dict[str, Any]):
        """
        Get parameters for federated aggregation.
        
        Returns only policy parameters (policy network + log_std).
        Value network parameters are excluded and remain local.
        """
        if not hasattr(self.agent, "get_parameters"):
            return super().get_parameters(config)
        
        # Get all parameters from agent
        full_params = self.agent.get_parameters()
        
        # Filter to only include policy-related parameters
        policy_params = {}
        for key, value in full_params.items():
            # Include policy network weights and log_std
            if key.startswith("policy.") or key == "log_std":
                policy_params[key] = value
        
        return policy_params
    
    def set_parameters(self, parameters):
        """
        Set parameters from federated aggregation.
        
        Only updates policy parameters. Value network is not modified
        during aggregation (remains local).
        """
        if not hasattr(self.agent, "set_parameters"):
            return super().set_parameters(parameters)
        
        if not isinstance(parameters, dict):
            return super().set_parameters(parameters)
        
        # Filter out any value parameters that might have been included
        policy_params = {}
        for key, value in parameters.items():
            if key.startswith("policy.") or key == "log_std":
                policy_params[key] = value
        
        if policy_params:
            self.agent.set_parameters(policy_params)


def client_fn_builder(
    env_id: str,
    algo: str = "fedkl",
    *,
    n_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    update_epochs: int = 10,
    minibatch_size: int = 64,
    lambda_global: float = 0.1,  # Global KL penalty (divergence from global policy)
    lambda_local: float = 0.05,  # Local KL penalty (divergence from start of round)
    max_grad_norm: float = 0.5,
    hidden_dim: int = 256,
    lr: float = 3e-4,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    metrics_collector: Optional[Any] = None,  # Bandit2DMetricsCollector instance (for backward compatibility)
    num_clients: Optional[int] = None,  # Total number of clients for ID mapping
):
    """
    Build client function for FedKL.
    
    Args:
        env_id: Environment ID (e.g., "HalfCheetah-v4")
        algo: Algorithm name
        n_steps: Steps per rollout before policy update
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_eps: PPO clipping parameter
        entropy_coef: Entropy bonus coefficient
        value_coef: Value loss coefficient
        update_epochs: Number of epochs to update policy per round
        minibatch_size: Minibatch size for SGD updates
        lambda_global: Penalty for KL(local_policy || global_policy)
        lambda_local: Penalty for KL(current_policy || policy_at_round_start)
        max_grad_norm: Maximum gradient norm for clipping
        hidden_dim: Neural network hidden dimension
        lr: Learning rate
        use_wandb: Enable wandb logging
        wandb_project: Wandb project name
        run_name: Run name prefix
        
    Returns:
        client_fn: Function that creates FedKLClient instances
    """
    
    def client_fn(context) -> Any:
        from fedguide.baselines.fedKL.agent import FedKLAgent
        from fedguide.baselines.fedKL.trainer import FedKLTrainer
        
        # Generate client-specific seed
        cid = str(getattr(context, "client_id", None) or getattr(context, "node_id", None) or "0")
        base_seed = 42 + (abs(hash(cid)) % 10000)
        
        # Set all random seeds
        random.seed(base_seed)
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)
        
        # Create environment
        env = _make_env(env_id, seed=base_seed)
        obs_space = env.observation_space
        act_space = env.action_space
        
        assert _is_box1d(obs_space) and _is_box1d(act_space), \
            "FedKL currently only supports 1D Box spaces"
        
        state_dim = int(obs_space.shape[0])
        action_dim = int(act_space.shape[0])
        
        # Create agent
        agent = FedKLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            device="cpu",
        )
        
        # Create trainer
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
        
        # Get collector from global variable if not passed directly
        # Note: This works because the module is imported before client_fn is called
        # Use nonlocal to modify the outer scope variable
        nonlocal metrics_collector
        if metrics_collector is None:
            try:
                # Try to import and access global collector from run script
                # Use importlib to ensure we get the module even if it's already imported
                import importlib
                try:
                    run_module = importlib.import_module('scripts.envs.bandit2d.run_fedkl_bandit2d')
                    metrics_collector = getattr(run_module, '_metrics_collector_global', None)
                except (ImportError, AttributeError):
                    try:
                        run_module = importlib.import_module('scripts.envs.bandit2d.run_fedguide_bandit2d')
                        metrics_collector = getattr(run_module, '_metrics_collector_global', None)
                    except (ImportError, AttributeError):
                        pass
            except Exception:
                pass
        
        # Create client
        client = FedKLClient(
            agent=agent,
            env=env,
            trainer=trainer,
            run_name=run_name or f"{env_id}-{algo}-cid{cid}",
            seed=base_seed,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            metrics_collector=metrics_collector,
        )
        
        # Register agent with metrics collector for visualization
        if metrics_collector is not None:
            # Map Flower client ID to sequential ID for metrics
            if num_clients is not None:
                mapped_id = abs(hash(cid)) % num_clients
            else:
                mapped_id = abs(hash(cid)) % 100
            metrics_collector.register_client_agent(mapped_id, agent)
        
        return client.to_client() if hasattr(client, "to_client") else client
    
    return client_fn
