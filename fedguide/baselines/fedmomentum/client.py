"""
FedMomentum Client Implementation

This module implements the FedMomentum client that extends FedRLClient.
The client computes policy gradients and transmits them to the server for momentum-based aggregation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Callable, Iterable, List
import random
import numpy as np
import torch
import json

try:
    import gymnasium as gym
except Exception:
    import gym

from fedguide.fed.client import FedRLClient as BaseFedRLClient
from fedguide.utils.gym_space_utils import is_box1d as _is_box1d


# --------- Helpers ---------
def _make_env(
    env_id: str,
    seed: Optional[int] = None,
    client_id: Optional[int] = None,
    num_clients: Optional[int] = None,
    sigma: float = 0.2,
    metadata_path: Optional[str] = None,
    render_mode: Optional[str] = None,
):
    """Create environment."""
    if env_id.lower() in ["bandit2d", "bandit_2d", "2dbandit"]:
        from fedguide.envs.bandit2d import Bandit2D
        preferred_peak = (client_id % 4) if (client_id is not None and num_clients is not None) else None
        env = Bandit2D(K=4, sigma=sigma, seed=seed, preferred_peak=preferred_peak)
        if seed is not None:
            env.reset(seed=seed)
        return env

    if env_id.lower() == "reacher_hetero" and metadata_path:
        import os
        from fedguide.envs.reacher import make_hetero_reacher_env_from_metadata

        if os.path.isfile(metadata_path):
            idx = client_id if client_id is not None else 0
            return make_hetero_reacher_env_from_metadata(
                metadata_path, idx, seed=seed, render_mode=render_mode
            )

    from fedguide.envs.halfcheetah_hetero import make_halfcheetah_env_if_applicable

    _hc_env = make_halfcheetah_env_if_applicable(
        metadata_path, client_id, seed, render_mode, render_eval=False
    )
    if _hc_env is not None:
        return _hc_env

    env = gym.make(env_id)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass
    return env


class FedMomentumClient(BaseFedRLClient):
    """
    FedMomentum Client implementation.
    
    Key features:
    - Computes policy gradients using SVRPG trainer
    - Transmits gradients (not parameters) to server
    - Server aggregates gradients with momentum
    
    Note: Still returns parameters for compatibility, but server primarily uses gradients.
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
                # It's a state_dict (e.g., {"fc1.weight": ..., "fc1.bias": ...})
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
        """Convert flat list from server into parameter dict."""
        param_dict = {}
        
        # Get agent's parameter structure to understand layout
        agent_param_dict = self.agent.get_parameters()
        
        # Reconstruct dict from flat list
        idx = 0
        for module_name, module_params in agent_param_dict.items():
            if isinstance(module_params, dict):
                # It's a state_dict (e.g., {"fc1.weight": ..., "fc1.bias": ...})
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
    
    def _gradient_to_dict(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Convert gradient dictionary from torch tensors to numpy arrays.
        
        Args:
            gradient: Dictionary of gradients (keyed by parameter name)
        
        Returns:
            Dictionary of gradients as numpy arrays
        """
        grad_dict = {}
        for key, tensor in gradient.items():
            if isinstance(tensor, torch.Tensor):
                grad_dict[key] = tensor.detach().cpu().numpy()
            elif hasattr(tensor, "numpy"):
                grad_dict[key] = tensor.numpy()
            else:
                grad_dict[key] = np.asarray(tensor)
        return grad_dict
    
    def _serialize_gradient(self, gradient: Dict[str, np.ndarray]) -> str:
        """
        Serialize gradient dictionary to JSON string for transmission.
        
        Args:
            gradient: Dictionary of gradients as numpy arrays
        
        Returns:
            JSON string representation
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_grad = {}
        for key, value in gradient.items():
            if isinstance(value, np.ndarray):
                serializable_grad[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                serializable_grad[key] = list(value)
            else:
                serializable_grad[key] = value
        
        return json.dumps(serializable_grad)
    
    def fit(self, parameters, config):
        """
        Override fit to compute and transmit policy gradients.
        
        Key difference from standard FedRL:
        - Computes policy gradients using SVRPG trainer
        - Transmits gradients via metrics (not just parameters)
        - Server uses gradients for momentum-based aggregation
        """
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
                print(f"[FedMomentumClient {cid}] Parameter loading failed in fit(): {e}")
        
        if hasattr(self, "metrics"):
            self.metrics.set_step(rnd)

        if hasattr(self.trainer, "set_server_round"):
            self.trainer.set_server_round(rnd)

        use_strict = bool(config.get("use_fedsvrpgm_strict", False))
        if use_strict:
            u_json = config.get("u_r_flat_json", "[]")
            tp_json = config.get("theta_prev_flat_json", "[]")
            u_raw = json.loads(u_json) if isinstance(u_json, str) else u_json
            tp_raw = json.loads(tp_json) if isinstance(tp_json, str) else tp_json
            fed = {
                "u_r_flat": [np.asarray(x, dtype=np.float32) for x in u_raw],
                "theta_prev_flat": [np.asarray(x, dtype=np.float32) for x in tp_raw],
            }
            train_result = self.trainer.train_one_round(fed)
        else:
            train_result = self.trainer.train_one_round()
        
        # Extract loss and other metrics from trainer result
        delta_flat: Optional[List[np.ndarray]] = None
        if isinstance(train_result, tuple) and len(train_result) == 2:
            tr_dict, delta_flat = train_result[0], train_result[1]
            train_result = tr_dict
        if isinstance(train_result, dict):
            loss = train_result.get("loss", train_result.get("train/loss", 0.0))
            train_return = train_result.get("train/return", None)
            eval_return = train_result.get("eval/return", None)
        else:
            loss = float(train_result) if train_result is not None else 0.0
            train_return = None
            eval_return = None
        
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
        
        # Get policy gradient from trainer (SVRPG path; not used for FedSVRPG-M strict)
        policy_gradient = None
        if not use_strict:
            try:
                if hasattr(self.trainer, "get_policy_gradient"):
                    policy_gradient = self.trainer.get_policy_gradient()

                    grad_dict_np = self._gradient_to_dict(policy_gradient)

                    grad_json = self._serialize_gradient(grad_dict_np)
                else:
                    print(f"[FedMomentumClient {cid}] Warning: Trainer does not have get_policy_gradient method")
            except Exception as e:
                print(f"[FedMomentumClient {cid}] Warning: Failed to get policy gradient: {e}")
                import traceback
                traceback.print_exc()
        
        # Eval/save
        success = self.trainer.save_eval(cid, rnd)
        samples = int(getattr(self.trainer, "n_steps", 0))
        
        # Print client metrics for debugging
        print(f"[FedMomentumClient {cid}] Round {rnd}: loss = {loss:.6f}, "
              f"train_return = {train_return}, eval_return = {eval_return}, "
              f"success = {success}, gradient_computed = {policy_gradient is not None}")
        
        # Build metrics dict
        fit_metrics = {
            "loss": loss,
            "success": int(bool(success)),
        }
        if train_return is not None:
            fit_metrics["train/return"] = float(train_return)
        if eval_return is not None:
            fit_metrics["eval/return"] = float(eval_return)

        # Strict FedSVRPG-M: eval/return must come from evaluate(global θ), not post-local train
        if use_strict and "eval/return" in fit_metrics:
            del fit_metrics["eval/return"]
        
        # Add policy gradient to metrics (if available)
        if policy_gradient is not None:
            try:
                grad_dict_np = self._gradient_to_dict(policy_gradient)
                grad_json = self._serialize_gradient(grad_dict_np)
                fit_metrics["policy_gradient"] = grad_json
                print(f"[FedMomentumClient {cid}] Policy gradient serialized (size: {len(grad_json)} bytes)")
            except Exception as e:
                print(f"[FedMomentumClient {cid}] Warning: Failed to serialize gradient: {e}")

        if use_strict and delta_flat is not None:
            try:
                fit_metrics["policy_delta"] = json.dumps([arr.tolist() for arr in delta_flat])
            except Exception as e:
                print(f"[FedMomentumClient {cid}] Warning: Failed to serialize policy_delta: {e}")
        
        # Get new parameters (as list) - still needed for compatibility
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
                pass
        
        # Evaluate policy on grid and pass through metrics (for Bandit2D)
        if self.metrics_collector is not None:
            try:
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
                pass
        
        return new_params_list, samples, fit_metrics

    def evaluate(self, parameters, config):
        """
        Distributed eval of the *global* model θ_r (broadcast by server before local training).

        Flower records this under History.metrics_distributed (evaluate), not metrics_distributed_fit.
        Do not confuse with train/return from fit (on-policy rollouts during local updates).
        """
        from flwr.common import parameters_to_ndarrays

        if parameters is not None:
            try:
                if hasattr(parameters, "tensors") or hasattr(parameters, "tensor_type"):
                    param_list = parameters_to_ndarrays(parameters)
                else:
                    param_list = parameters
                self.set_parameters(param_list)
            except Exception as e:
                print(f"[FedMomentumClient evaluate] set_parameters failed: {e}")

        rnd = int(config.get("server_round", 0))
        cid = getattr(self, "cid", config.get("cid", "unknown"))
        n_ep = max(1, int(getattr(self.trainer, "eval_episodes", 1)))
        if not hasattr(self.trainer, "_eval_episode"):
            return float("nan"), 1, {}
        total = 0.0
        for _ in range(n_ep):
            total += float(self.trainer._eval_episode())
        avg = total / float(n_ep)
        print(f"[FedMomentumClient {cid}] evaluate round {rnd}: eval/return (global θ) = {avg:.4f}")
        return float("nan"), 1, {"eval/return": float(avg)}


# --------- client_fn_builder ----------
def client_fn_builder(
    env_id: str,
    *,
    # PPO hyperparameters
    n_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    max_grad_norm: float = 0.5,
    # Network architecture
    hidden_dim: int = 256,
    lr: float = 3e-4,
    # Algorithm selection
    algorithm: str = "svrpg",  # "svrpg" or "hapg"
    # SVRPG-specific parameters
    reference_update_freq: int = 5,
    use_svrpg: bool = True,  # Only used if algorithm="svrpg"
    # HAPG-specific parameters
    hessian_alpha: float = 0.1,
    use_diagonal_approx: bool = True,
    fisher_update_freq: int = 1,
    use_fisher_info: bool = True,
    # Evaluation
    eval_episodes: int = 1,
    # Logging
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    metrics_collector: Optional[Any] = None,
    num_clients: Optional[int] = None,
    device: str = "cpu",
    cid_mapping_file: Optional[str] = None,
    sigma: float = 0.2,
    metadata_path: Optional[str] = None,
    render_eval: bool = False,
    render_mode: str = "video",
    render_save_dir: Optional[str] = None,
    render_every_n_rounds: int = 10,
    render_episodes: int = 5,
    reacher_render_mode: Optional[str] = None,
    use_fedsvrpgm_strict: bool = False,
    fedsvrpgm_eta: float = 0.01,
    fedsvrpgm_beta: float = 0.2,
    local_steps_k: int = 5,
    fedsvrpgm_max_horizon: int = 500,
):
    """
    Build client function for FedMomentum (SVRPG-based).
    
    Args:
        env_id: Environment ID
        ... (other hyperparameters)
    
    Returns:
        client_fn function for Flower
    """
    
    def client_fn(context) -> Any:
        # Import here to avoid circular imports
        from fedguide.baselines.fedmomentum.agent import FedMomentumAgent
        from fedguide.baselines.fedmomentum.trainer import SVRPGTrainer, HAPGTrainer
        
        # 1) per-client seed and ID mapping
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
        
        # 2) env
        env = _make_env(
            env_id,
            seed=base,
            client_id=mapped_client_id,
            num_clients=num_clients,
            sigma=sigma,
            metadata_path=metadata_path,
            render_mode=reacher_render_mode,
        )
        obs_space, act_space = env.observation_space, env.action_space
        
        # Determine state and action dimensions
        assert _is_box1d(act_space), "FedMomentum requires continuous action space"
        assert _is_box1d(obs_space), "FedMomentum requires continuous observation space"
        state_dim = int(obs_space.shape[0])
        action_dim = int(act_space.shape[0])
        
        # 3) agent
        agent = FedMomentumAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            clip_eps=clip_eps,
            gae_lambda=gae_lambda,
            ent_coef=entropy_coef,  # Map entropy_coef to ent_coef (agent uses ent_coef)
            vf_coef=value_coef,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        
        # 4) trainer (select based on algorithm type)
        if use_fedsvrpgm_strict:
            from fedguide.baselines.fedmomentum.fedsvrpgm_strict import FedSVRPGMStrictTrainer

            trainer = FedSVRPGMStrictTrainer(
                agent=agent,
                env=env,
                device=device,
                gamma=gamma,
                eta=fedsvrpgm_eta,
                beta=fedsvrpgm_beta,
                local_steps_k=local_steps_k,
                max_horizon=fedsvrpgm_max_horizon,
                eval_episodes=eval_episodes,
                render_eval=render_eval,
                render_mode=render_mode,
                render_save_dir=render_save_dir,
                render_every_n_rounds=render_every_n_rounds,
                render_episodes=render_episodes,
                render_client_tag=str(mapped_client_id),
            )
        else:
            algorithm_lower = algorithm.lower()
            if algorithm_lower == "hapg":
                trainer = HAPGTrainer(
                    agent=agent,
                    env=env,
                    device=device,
                    n_steps=n_steps,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_eps=clip_eps,
                    entropy_coef=entropy_coef,
                    value_coef=value_coef,
                    update_epochs=update_epochs,
                    minibatch_size=minibatch_size,
                    max_grad_norm=max_grad_norm,
                    eval_episodes=eval_episodes,
                    hessian_alpha=hessian_alpha,
                    use_diagonal_approx=use_diagonal_approx,
                    fisher_update_freq=fisher_update_freq,
                    use_fisher_info=use_fisher_info,
                    reference_update_freq=reference_update_freq,
                    use_svrpg=False,
                    render_eval=render_eval,
                    render_mode=render_mode,
                    render_save_dir=render_save_dir,
                    render_every_n_rounds=render_every_n_rounds,
                    render_episodes=render_episodes,
                    render_client_tag=str(mapped_client_id),
                )
            else:
                trainer = SVRPGTrainer(
                    agent=agent,
                    env=env,
                    device=device,
                    n_steps=n_steps,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_eps=clip_eps,
                    entropy_coef=entropy_coef,
                    value_coef=value_coef,
                    update_epochs=update_epochs,
                    minibatch_size=minibatch_size,
                    max_grad_norm=max_grad_norm,
                    eval_episodes=eval_episodes,
                    reference_update_freq=reference_update_freq,
                    use_svrpg=use_svrpg,
                    render_eval=render_eval,
                    render_mode=render_mode,
                    render_save_dir=render_save_dir,
                    render_every_n_rounds=render_every_n_rounds,
                    render_episodes=render_episodes,
                    render_client_tag=str(mapped_client_id),
                )
        
        # Get collector from global variable if not passed directly
        nonlocal metrics_collector
        if metrics_collector is None:
            try:
                import importlib
                try:
                    run_module = importlib.import_module(f'scripts.envs.bandit2d.run_fedmomentum_bandit2d')
                    metrics_collector = getattr(run_module, '_metrics_collector_global', None)
                except (ImportError, AttributeError):
                    pass
            except Exception:
                pass
        
        # 5) client
        client = FedMomentumClient(
            agent=agent,
            env=env,
            trainer=trainer,
            run_name=run_name or f"{env_id}-fedmomentum-cid{cid}",
            seed=base,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            metrics_collector=metrics_collector,
        )
        # Store client_id for metrics collection
        client.cid = cid
        
        # Register agent with metrics collector for visualization
        if metrics_collector is not None:
            if hasattr(metrics_collector, 'register_client_agent'):
                metrics_collector.register_client_agent(mapped_client_id, agent)
        
        # Convert NumPyClient to Client
        return client.to_client()
    
    return client_fn

