"""
FedKL Client Implementation (Matches FedGuide style exactly)

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


class FedKLClient(FedRLClient):
    """
    FedKL Client implementation (matches FedGuide structure).
    
    Key design principles:
    - Only POLICY parameters are aggregated across clients
    - VALUE networks remain LOCAL to each client
    """
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        trainer: Any,
        *,
        run_name: Optional[str] = None,
        seed: Optional[int] = None,
        device: Optional[str] = "auto",  #fix this
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
        self.metrics_collector = metrics_collector
    
    def get_parameters(self, config: Dict[str, Any]):
        """Get parameters as a list for Flower compatibility (same as FedGuide)."""
        if not hasattr(self.agent, "get_parameters"):
            return super().get_parameters(config)
        
        # Get parameters as list from agent
        flat_params = []
        param_dict = self.agent.get_parameters()
        for module_params in param_dict.values():
            if isinstance(module_params, dict):
                for v in module_params.values():
                    if isinstance(v, torch.Tensor):
                        flat_params.append(v.numpy())
                    elif hasattr(v, "numpy"):
                        flat_params.append(v.numpy())
                    else:
                        flat_params.append(v)
            elif isinstance(module_params, torch.Tensor):
                flat_params.append(module_params.numpy())
            elif hasattr(module_params, "numpy"):
                flat_params.append(module_params.numpy())
            else:
                flat_params.append(module_params)
        
        
        # Verify return type - must be list of numpy arrays
        # Make sure all are independent copies (not views) for serialization
        verified_params = []
        for p in flat_params:
            try:
                if isinstance(p, np.ndarray):
                    # Make a copy to ensure it's independent and C-contiguous
                    verified_params.append(np.ascontiguousarray(p.copy()))
                elif hasattr(p, 'numpy'):
                    arr = p.numpy()
                    verified_params.append(np.ascontiguousarray(arr.copy()))
                else:
                    # Try to convert to numpy
                    arr = np.asarray(p)
                    verified_params.append(np.ascontiguousarray(arr.copy()))
            except Exception:
                # Fallback: return original
                verified_params.append(p)
        
        return verified_params
    
    def list_to_parameter_dict(self, lst):
        """Convert flat list from server into FedGuide-style parameter dict."""
        param_dict = {}

        # 1) Policy parameters
        policy_state = self.agent.policy.state_dict()
        new_policy_state = {}

        idx = 0
        for k, v in policy_state.items():
            new_policy_state[k] = torch.tensor(lst[idx], dtype=v.dtype)
            idx += 1
        param_dict["policy"] = new_policy_state

        # 2) log_std (last item)
        param_dict["log_std"] = torch.tensor(lst[idx])

        return param_dict

    def set_parameters(self, parameters):
        """Set parameters, handling both dict and list formats (same as FedGuide)."""
        if not hasattr(self.agent, "set_parameters"):
            return super().set_parameters(parameters)
            # If agent has dict-based API (FedGuide style)

        # If parameters is a list, convert it
        if isinstance(parameters, list):
            parameters = self.list_to_parameter_dict(parameters)
        # Now pass the dict to the agent
        self.agent.set_parameters(parameters)
      

    def fit(self, parameters, config):
        """Override fit to handle parameters and collect actions for metrics (matches FedGuide)."""
        cid = getattr(self, "cid", config.get("cid", "unknown"))
        rnd = int(config.get("server_round", 0))
        
        # Handle parameters - Flower may pass Parameters object or list (same as FedGuide)
        if parameters is not None:
            try:
                # Convert Parameters object to list if needed
                from flwr.common import parameters_to_ndarrays
                if hasattr(parameters, 'tensors') or hasattr(parameters, 'tensor_type'):
                    # It's a Parameters object, convert to list
                    param_list = parameters_to_ndarrays(parameters)
                else:
                    # It's already a list or dict
                    param_list = parameters
                self.set_parameters(param_list)
            except Exception as e:
                # Continue anyway - agent will use current parameters
                print("Parameter loading failed in fit():", e)
        if hasattr(self, "metrics"):
            self.metrics.set_step(rnd)
        
        # Train one round (same as FedGuide)
        train_result = self.trainer.train_one_round()
        
        # Extract loss and other metrics from trainer result (same logic as FedGuide)
        if isinstance(train_result, dict):
            # Try multiple possible loss keys
            loss = None
            for key in ["loss", "train/loss", "train/loss/total", "loss/total", "train/loss/policy", "train/loss/value"]:
                if key in train_result:
                    val = train_result[key]
                    # Check if value is valid (not None, not nan, not inf)
                    if val is not None:
                        try:
                            loss_float = float(val)
                            # Check for nan: nan != nan is True
                            # Check for inf
                            if loss_float == loss_float and loss_float != float('inf') and loss_float != float('-inf'):
                                loss = loss_float
                                break
                        except (TypeError, ValueError):
                            continue
            
            # If still None or invalid, try to use return as a proxy
            if loss is None:
                if "train/return" in train_result:
                    train_return_val = train_result["train/return"]
                    if train_return_val is not None:
                        try:
                            # Use negative return as loss
                            loss = -float(train_return_val)
                        except (TypeError, ValueError):
                            loss = 0.0
                    else:
                        loss = 0.0
                else:
                    loss = 0.0
            elif isinstance(loss, float):
                # Check for nan/inf
                if loss != loss or loss == float('inf') or loss == float('-inf'):
                    # Try to use return as fallback
                    if "train/return" in train_result:
                        train_return_val = train_result["train/return"]
                        if train_return_val is not None:
                            try:
                                loss = -float(train_return_val)
                            except (TypeError, ValueError):
                                loss = 0.0
                        else:
                            loss = 0.0
                    else:
                        loss = 0.0
            
            train_return = train_result.get("train/return", train_result.get("return", None))
            eval_return = train_result.get("eval/return", None)
        else:
            # If train_result is not a dict, it should be a scalar loss value
            if train_result is not None:
                try:
                    loss = float(train_result)
                    # Check for nan/inf
                    if loss != loss or loss == float('inf') or loss == float('-inf'):
                        loss = 0.0
                except (TypeError, ValueError):
                    loss = 0.0
            else:
                loss = 0.0
            train_return = None
            eval_return = None
        
        # Eval/save (same as FedGuide)
        success = self.trainer.save_eval(cid, rnd)
        samples = int(getattr(self.trainer, "n_steps", 0))
        
        # Ensure loss is a valid float (same as FedGuide)
        if loss is None:
            loss = 0.0
        else:
            try:
                loss = float(loss)
                # Check for nan/inf
                if loss != loss or loss == float('inf') or loss == float('-inf'):
                    loss = 0.0
            except (TypeError, ValueError):
                loss = 0.0
        
        # Print client loss for debugging (same as FedGuide)
        print(f"[FedKLClient {cid}] Round {rnd}: loss = {loss}, train_return = {train_return}, eval_return = {eval_return}, success = {success}")
        
        # Build metrics dict (same format as FedGuide)
        fit_metrics = {
            "loss": loss,
            "success": int(bool(success)),
        }
        if train_return is not None:
            fit_metrics["train/return"] = float(train_return)
        if eval_return is not None:
            fit_metrics["eval/return"] = float(eval_return)
        
        # Get new parameters (as list)
        new_params_list = self.get_parameters(config)
        
        # Ensure new_params_list is a list, not a dict
        if isinstance(new_params_list, dict):
            new_params_list = [np.asarray(v) for v in new_params_list.values()]
        
        # Verify it's a list
        if not isinstance(new_params_list, list):
            new_params_list = [np.asarray(new_params_list)] if not isinstance(new_params_list, list) else new_params_list
        
        # Collect actions for metrics visualization (same as FedGuide)
        if self.metrics_collector is not None:
            try:
                client_id = int(cid) if isinstance(cid, (int, str)) and str(cid).isdigit() else hash(cid) % 10000
                if hasattr(self.trainer, 'last_actions') and self.trainer.last_actions is not None:
                    actions = self.trainer.last_actions
                    self.metrics_collector.collect_client_actions(client_id, actions)
            except Exception:
                pass
        
        # Also pass actions through Flower metrics for server-side collection (same as FedGuide)
        if hasattr(self.trainer, 'last_actions') and self.trainer.last_actions is not None:
            try:
                import json
                actions = self.trainer.last_actions
                # Convert to list for JSON serialization
                if isinstance(actions, np.ndarray):
                    actions_list = actions.tolist()
                elif isinstance(actions, (list, tuple)):
                    actions_list = [a.tolist() if isinstance(a, np.ndarray) else a for a in actions]
                else:
                    actions_list = actions
                # Store in metrics as JSON string
                fit_metrics["client_actions"] = json.dumps(actions_list)
                # Also store client_id for mapping
                mapped_id = abs(hash(cid)) % (getattr(self, '_num_clients', 100) if hasattr(self, '_num_clients') else 100)
                fit_metrics["client_id_mapped"] = mapped_id
            except Exception:
                # Silently fail if serialization fails
                pass
        
        # Evaluate policy on grid and pass through metrics (same as FedGuide)
        if self.metrics_collector is not None:
            try:
                import json
                # Evaluate agent on grid
                grid_metrics = self.metrics_collector.evaluate_on_grid(
                    agent=self.agent,
                    client_id=mapped_id if 'mapped_id' in locals() else None,
                    round_num=rnd
                )
                # Serialize grid metrics to JSON
                serialized_metrics = {}
                for key, value in grid_metrics.items():
                    if isinstance(value, np.ndarray):
                        serialized_metrics[key] = json.dumps(value.tolist())
                    else:
                        serialized_metrics[key] = json.dumps(value)
                # Store in fit_metrics with prefix
                for key, value in serialized_metrics.items():
                    fit_metrics[f"client_grid_{key}"] = value
            except Exception:
                # Silently fail if evaluation fails
                pass
        
        return new_params_list, samples, fit_metrics


# --------- client_fn_builder ----------
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
    lambda_global: float = 0.1,
    lambda_local: float = 0.05,
    max_grad_norm: float = 0.5,
    hidden_dim: int = 256,
    lr: float = 3e-4,
    # logging (same as FedGuide)
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    metrics_collector: Optional[Any] = None,
    num_clients: Optional[int] = None,  # For ID mapping
):
    """
    Build client function for FedKL (matches FedGuide structure exactly).
    """
    
    def client_fn(context) -> Any:
        # Import here to avoid circular imports
        from fedguide.baselines.fedKL.agent import FedKLAgent
        from fedguide.baselines.fedKL.trainer import FedKLTrainer
        
        # 1) per-client seed and ID mapping (same as FedGuide)
        cid = str(getattr(context, "client_id", None) or getattr(context, "node_id", None) or "0")
        
        # Map Flower's client ID to 0, 1, 2, 3...
        if num_clients is not None:
            mapped_client_id = abs(hash(cid)) % num_clients
        else:
            mapped_client_id = abs(hash(cid)) % 100
        
        base = 42 + (abs(hash(cid)) % 10000)
        random.seed(base)
        np.random.seed(base)
        torch.manual_seed(base)
        
        # 2) env (same as FedGuide)
        env = _make_env(env_id, seed=base)
        obs_space, act_space = env.observation_space, env.action_space
        assert _is_box1d(obs_space) and _is_box1d(act_space), "Only Support 1D Box spaces."
        
        state_dim = int(obs_space.shape[0])
        action_dim = int(act_space.shape[0])
        
        # 3) agent (FedKL doesn't use pretrained prior/guidance)
        agent = FedKLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            device="cpu",
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
        
        # Get collector from global variable if not passed directly (same as FedGuide)
        nonlocal metrics_collector
        if metrics_collector is None:
            try:
                import importlib
                try:
                    run_module = importlib.import_module('scripts.envs.bandit2d.run_fedkl_bandit2d')
                    metrics_collector = getattr(run_module, '_metrics_collector_global', None)
                except (ImportError, AttributeError):
                    pass
            except Exception:
                pass
        
        # 5) client (same structure as FedGuide)
        client = FedKLClient(
            agent=agent,
            env=env,
            trainer=trainer,
            run_name=run_name or f"{env_id}-{algo}-cid{cid}",
            seed=base,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            metrics_collector=metrics_collector,
        )
        # Store client_id for metrics collection (same as FedGuide)
        client.cid = cid
        
        # Register agent with metrics collector for visualization (same as FedGuide)
        if metrics_collector is not None:
            if num_clients is not None:
                mapped_id = abs(hash(cid)) % num_clients
            else:
                mapped_id = abs(hash(cid)) % 100
            
            # Register agent if method exists
            if hasattr(metrics_collector, 'register_client_agent'):
                metrics_collector.register_client_agent(mapped_id, agent)
        
        # Convert NumPyClient to Client 
        return client.to_client()
    
    return client_fn