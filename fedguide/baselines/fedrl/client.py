"""
FedRL Client Implementation

This module implements the FedRL client that extends FedRLClient.
Supports both DQN (discrete actions) and DDPG (continuous actions).
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

from fedguide.fed.client import FedRLClient as BaseFedRLClient


# --------- Helpers ---------
def _is_box1d(space) -> bool:
    """Check if space is 1D Box (continuous action space)."""
    try:
        from gymnasium.spaces import Box
    except Exception:
        from gym.spaces import Box
    return isinstance(space, Box) and len(space.shape) == 1


def _is_discrete(space) -> bool:
    """Check if space is discrete."""
    try:
        from gymnasium.spaces import Discrete
    except Exception:
        from gym.spaces import Discrete
    return isinstance(space, Discrete)


def _make_env(env_id: str, seed: Optional[int] = None):
    """Create environment."""
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


class FedRLClient(BaseFedRLClient):  # Inherit from fedguide.fed.client.FedRLClient
    """
    FedRL Client implementation.
    
    Supports both DQN (discrete actions) and DDPG (continuous actions).
    Only policy/Q-network parameters are aggregated; other networks (if any) remain local.
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
        self.metrics_collector = metrics_collector
    
    def get_parameters(self, config: Dict[str, Any]):
        """Get parameters as a list for Flower compatibility."""
        if not hasattr(self.agent, "get_parameters"):
            return super().get_parameters(config)
        
        # Get parameters as dict from agent
        param_dict = self.agent.get_parameters()
        
        # Convert dict to flat list of numpy arrays
        flat_params = []
        for module_name, module_params in param_dict.items():
            if isinstance(module_params, dict):
                # It's a state_dict (e.g., {"q_net": {...}} or {"actor": {...}, "critic": {...}})
                for key in sorted(module_params.keys()):  # Sort for consistency
                    param_tensor = module_params[key]
                    if isinstance(param_tensor, torch.Tensor):
                        flat_params.append(param_tensor.detach().cpu().numpy())
                    elif hasattr(param_tensor, "numpy"):
                        flat_params.append(param_tensor.numpy())
                    else:
                        flat_params.append(np.asarray(param_tensor))
            elif isinstance(module_params, torch.Tensor):
                # It's a single tensor (e.g., log_std)
                flat_params.append(module_params.detach().cpu().numpy())
            elif hasattr(module_params, "numpy"):
                flat_params.append(module_params.numpy())
            else:
                flat_params.append(np.asarray(module_params))
        
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
    
    def list_to_parameter_dict(self, lst: list) -> Dict[str, Any]:
        """Convert flat list from server into FedRL-style parameter dict."""
        param_dict = {}
        
        # Get agent's parameter structure to understand layout
        agent_param_dict = self.agent.get_parameters()
        
        # Reconstruct dict from flat list
        idx = 0
        for module_name, module_params in agent_param_dict.items():
            if isinstance(module_params, dict):
                # It's a state_dict (e.g., {"q_net": {...}} or {"actor": {...}})
                new_module_params = {}
                for key in sorted(module_params.keys()):  # Must match order in get_parameters
                    original_tensor = module_params[key]
                    new_module_params[key] = torch.tensor(lst[idx], dtype=original_tensor.dtype)
                    idx += 1
                param_dict[module_name] = new_module_params
            elif isinstance(module_params, torch.Tensor):
                # It's a single tensor (e.g., log_std)
                param_dict[module_name] = torch.tensor(lst[idx], dtype=module_params.dtype)
                idx += 1
            else:
                # Fallback: try to convert
                param_dict[module_name] = torch.tensor(lst[idx])
                idx += 1
        
        return param_dict
    
    def set_parameters(self, parameters):
        """Set parameters, handling both dict and list formats."""
        if not hasattr(self.agent, "set_parameters"):
            return super().set_parameters(parameters)
        
        # If parameters is a list, convert it to dict
        if isinstance(parameters, list):
            parameters = self.list_to_parameter_dict(parameters)
        
        # Now pass the dict to the agent
        self.agent.set_parameters(parameters)
        
        # Rebuild optimizer after setting parameters
        if hasattr(self.agent, "rebuild_optimizer"):
            self.agent.rebuild_optimizer()
    
    def fit(self, parameters, config):
        """Override fit to handle parameters and collect actions for metrics."""
        cid = getattr(self, "cid", config.get("cid", "unknown"))
        rnd = int(config.get("server_round", 0))
        
        # Handle parameters - Flower may pass Parameters object or list
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
                print(f"[FedRLClient {cid}] Parameter loading failed in fit(): {e}")
        
        if hasattr(self, "metrics"):
            self.metrics.set_step(rnd)
        
        # Train one round
        train_result = self.trainer.train_one_round()
        
        # Extract loss and other metrics from trainer result
        if isinstance(train_result, dict):
            # Try multiple possible loss keys
            loss = None
            for key in ["loss", "train/loss", "train/loss/total", "loss/total", "loss/actor", "loss/critic"]:
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
        
        # Eval/save
        success = self.trainer.save_eval(cid, rnd)
        samples = int(getattr(self.trainer, "n_steps", 0))
        
        # Ensure loss is a valid float
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
        
        # Print client loss for debugging
        print(f"[FedRLClient {cid}] Round {rnd}: loss = {loss:.6f}, train_return = {train_return}, eval_return = {eval_return}, success = {success}")
        
        # Build metrics dict
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
        
        # Collect actions for metrics visualization (if collector is available)
        if self.metrics_collector is not None:
            try:
                client_id = int(cid) if isinstance(cid, (int, str)) and str(cid).isdigit() else hash(cid) % 10000
                if hasattr(self.trainer, 'last_actions') and self.trainer.last_actions is not None:
                    actions = self.trainer.last_actions
                    self.metrics_collector.collect_client_actions(client_id, actions)
            except Exception:
                pass
        
        # Also pass actions through Flower metrics for server-side collection
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
        
        # Evaluate policy on grid and pass through metrics (for Bandit2D)
        if self.metrics_collector is not None:
            try:
                import json
                # Evaluate agent on grid
                mapped_id = fit_metrics.get("client_id_mapped", None)
                grid_metrics = self.metrics_collector.evaluate_on_grid(
                    agent=self.agent,
                    client_id=mapped_id,
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
    algo: str = "dqn",  # "dqn" or "ddpg"
    *,
    # DQN/DDPG hyperparameters
    gamma: float = 0.9,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    # DQN-specific
    epsilon: float = 1.0,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.01,
    # DDPG-specific
    tau: float = 0.001,
    threshold: float = 2.0,
    aggregate_critic: bool = False,
    # Training hyperparameters
    batch_size: int = 16,
    replay_size: int = 1000,
    replay_initial: int = None,  # Default: 2 * batch_size for DQN, 1000 for DDPG
    sync_interval: int = 10,
    merge_interval: int = 16,  # Number of steps per round (E in FedRL)
    eval_episodes: int = 1,
    add_noise: bool = True,  # For DDPG exploration
    # logging
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    metrics_collector: Optional[Any] = None,
    num_clients: Optional[int] = None,  # For ID mapping
    device: str = "cpu",
):
    """
    Build client function for FedRL (supports both DQN and DDPG).
    
    Args:
        env_id: Environment ID
        algo: Algorithm type ("dqn" or "ddpg")
        ... (other hyperparameters)
    
    Returns:
        client_fn function for Flower
    """
    
    def client_fn(context) -> Any:
        # Import here to avoid circular imports
        from fedguide.baselines.fedrl.agent import DQNAgent, DDPGAgent
        from fedguide.baselines.fedrl.trainer import DQNTrainer, DDPGTrainer
        
        # 1) per-client seed and ID mapping
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
        
        # 2) env
        env = _make_env(env_id, seed=base)
        obs_space, act_space = env.observation_space, env.action_space
        
        # Determine state and action dimensions
        if algo.lower() == "dqn":
            assert _is_discrete(act_space), "DQN requires discrete action space"
            # Get state dimension (handle both Box and Discrete observation spaces)
            try:
                state_dim = int(obs_space.shape[0])
            except (AttributeError, IndexError):
                # Fallback: try to get from n if it's a Discrete space
                try:
                    state_dim = int(obs_space.n)
                except AttributeError:
                    raise ValueError(f"Unsupported observation space for DQN: {type(obs_space)}")
            action_dim = int(act_space.n)
        elif algo.lower() == "ddpg":
            assert _is_box1d(act_space), "DDPG requires continuous action space"
            assert _is_box1d(obs_space), "DDPG requires continuous observation space"
            state_dim = int(obs_space.shape[0])
            action_dim = int(act_space.shape[0])
        else:
            raise ValueError(f"Unsupported algorithm: {algo}. Must be 'dqn' or 'ddpg'")
        
        # 3) agent
        if algo.lower() == "dqn":
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                lr=lr,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                sync_interval=sync_interval,
                device=device,
            )
        else:  # ddpg
            agent = DDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=lr,
                gamma=gamma,
                tau=tau,
                threshold=threshold,
                device=device,
                aggregate_critic=aggregate_critic,
            )
        
        # 4) trainer
        if algo.lower() == "dqn":
            trainer_replay_initial = replay_initial if replay_initial is not None else 2 * batch_size
            trainer = DQNTrainer(
                agent=agent,
                env=env,
                device=device,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                batch_size=batch_size,
                replay_size=replay_size,
                sync_interval=sync_interval,
                merge_interval=merge_interval,
                eval_episodes=eval_episodes,
                replay_initial=trainer_replay_initial,
            )
        else:  # ddpg
            trainer_replay_initial = replay_initial if replay_initial is not None else 1000
            trainer = DDPGTrainer(
                agent=agent,
                env=env,
                device=device,
                gamma=gamma,
                batch_size=batch_size,
                replay_size=replay_size,
                replay_initial=trainer_replay_initial,
                merge_interval=merge_interval,
                tau=tau,
                eval_episodes=eval_episodes,
                add_noise=add_noise,
            )
        
        # Get collector from global variable if not passed directly
        nonlocal metrics_collector
        if metrics_collector is None:
            try:
                import importlib
                try:
                    run_module = importlib.import_module(f'scripts.envs.bandit2d.run_fedrl_{algo}_bandit2d')
                    metrics_collector = getattr(run_module, '_metrics_collector_global', None)
                except (ImportError, AttributeError):
                    pass
            except Exception:
                pass
        
        # 5) client
        client = FedRLClient(
            agent=agent,
            env=env,
            trainer=trainer,
            run_name=run_name or f"{env_id}-{algo}-cid{cid}",
            seed=base,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            metrics_collector=metrics_collector,
        )
        # Store client_id for metrics collection
        client.cid = cid
        
        # Register agent with metrics collector for visualization
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

