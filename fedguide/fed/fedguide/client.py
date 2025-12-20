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
    # Import d4rl to register all d4rl environments (maze2d, antmaze, flow, etc.)
    try:
        import d4rl
    except ImportError:
        pass  # d4rl not available, continue with other options

    # Handle custom environments
    if env_id.lower() in ["bandit2d", "bandit_2d", "2dbandit"]:
        from fedguide.envs.bandit2d import Bandit2D
        env = Bandit2D(K=4, sigma=0.2, seed=seed)
        if seed is not None:
            env.reset(seed=seed)
        return env
    
    # Use gym.make for standard gym/gymnasium/d4rl environments
    # This works for: maze2d, antmaze, flow, and other registered environments
    env = gym.make(env_id)
    try:
        env.reset(seed=seed)
    except TypeError:
        # Some environments may not support seed parameter in reset
        pass
    except Exception as e:
        # If it's a flow environment and not registered, provide helpful error
        if "flow" in env_id.lower() or "figureeight" in env_id.lower():
            raise ValueError(
                f"Flow environment {env_id} not registered. "
                "Make sure flow figureeight environments are registered in d4rl.flow"
            ) from e
        raise
    return env


class FedGuideClient(FedRLClient):
    def __init__(
        self,
        agent: Any,
        env: Any,
        trainer: Any,
        *,
        aggregate_mode: str = "policy",  # 'policy' | 'prior' | 'policy+prior' | 'prior+guidance' | 'policy_value' | 'all'
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
        """Get parameters as a list for Flower compatibility.
        
        The dict format is also stored in metrics["modules"] for OT-MoE aggregation.
        """
        if not hasattr(self.agent, "get_parameters"):
            return super().get_parameters(config)

        full = self.agent.get_parameters()
        mode = self.aggregate_mode

        def pick(keys):
            return {k: v for k, v in full.items() if k in keys and k in full}

        # Get parameters as dict first
        if mode == "policy":
            param_dict = pick({"policy", "log_std"})
        elif mode == "policy_value":
            param_dict = pick({"policy", "log_std", "value"})
        elif mode == "prior":
            param_dict = pick({"prior_adapt"})
        elif mode in ("policy+prior", "policy_prior", "policy-prior"):
            param_dict = pick({"policy", "log_std", "prior_adapt"})
        elif mode in ("prior+guidance", "prior_guidance", "prior-guidance"):
            param_dict = pick({"prior_adapt", "guidance"})
        elif mode == "all":
            param_dict = pick({"policy", "log_std", "prior_adapt", "guidance"})
        else:
            param_dict = pick({"policy", "log_std"})
        
        # Convert dict to list for Flower compatibility
        # The dict format is used in metrics["modules"] for OT-MoE aggregation
        import torch
        flat_params = []
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
        import numpy as np
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

    def set_parameters(self, parameters):
        """Set parameters, handling both dict and list formats."""
        if not hasattr(self.agent, "set_parameters"):
            return super().set_parameters(parameters)

        mode = self.aggregate_mode
        
        # If parameters is a list (from server's flattened format), try to reconstruct dict
        # using layout from previous round's metrics, or skip if not available
        if not isinstance(parameters, dict):
            # For module-based aggregation modes, we need dict format
            # If we receive a list, we can't easily reconstruct without layout
            # Skip setting for now - agent will use its current parameters
            # This is OK for the first round when there are no aggregated parameters yet
            if mode in ("prior+guidance", "prior_guidance", "prior-guidance", "prior", "all"):
                # For first round, parameters might be None or empty list - that's OK
                if parameters is None or (isinstance(parameters, list) and len(parameters) == 0):
                    return
                # For subsequent rounds, we'd need layout to reconstruct - skip for now
                # TODO: Implement layout-based reconstruction if needed
                return
            # For non-module modes, use parent's implementation
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
        elif mode in ("prior+guidance", "prior_guidance", "prior-guidance"):
            allowed = {"prior_adapt", "guidance"}
        elif mode == "all":
            allowed = {"policy", "log_std", "prior_adapt", "guidance"}

        filtered = {k: v for k, v in parameters.items() if k in allowed}
        if filtered:
            self.agent.set_parameters(filtered)
    
    def fit(self, parameters, config):
        """Override fit to handle module-based parameters and collect actions for metrics."""
        cid = getattr(self, "cid", config.get("cid", "unknown"))
        rnd = int(config.get("server_round", 0))
        
        # Set parameters first
        # Handle parameters - Flower may pass Parameters object or list
        if parameters is not None:
            try:
                # Convert Parameters object to list if needed
                from flwr.common import parameters_to_ndarrays
                if hasattr(parameters, 'tensors') or hasattr(parameters, 'tensor_type'):
                    # It's a Parameters object, convert to list
                    param_list = parameters_to_ndarrays(parameters)
                    self.set_parameters(param_list)
                else:
                    # It's already a list or dict
                    self.set_parameters(parameters)
            except Exception:
                # Continue anyway - agent will use current parameters
                pass
        
        if hasattr(self, "metrics"):
            self.metrics.set_step(rnd)
        
        # Train one round
        train_result = self.trainer.train_one_round()
        
        # Extract loss and other metrics from trainer result
        if isinstance(train_result, dict):
            # Try multiple possible loss keys (different agents/trainers may use different names)
            # Note: FedguideAgent.update() returns dict with "loss/total", which becomes "train/loss/total" in trainer output
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
            
            # If still None or invalid, try to use return as a proxy (negative return = higher loss)
            if loss is None:
                if "train/return" in train_result:
                    train_return_val = train_result["train/return"]
                    if train_return_val is not None:
                        try:
                            # Use negative return as loss (higher return = lower loss)
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
        
        # Get parameters in module format for OT-MoE aggregation
        # Get dict format directly from agent (not from get_parameters which returns list)
        if hasattr(self.agent, "get_parameters"):
            full_param_dict = self.agent.get_parameters()
            mode = self.aggregate_mode
            
            def pick(keys):
                return {k: v for k, v in full_param_dict.items() if k in keys and k in full_param_dict}
            
            # Pick relevant modules based on aggregate_mode
            if mode == "policy":
                param_dict = pick({"policy", "log_std"})
            elif mode == "policy_value":
                param_dict = pick({"policy", "log_std", "value"})
            elif mode == "prior":
                param_dict = pick({"prior_adapt"})
            elif mode in ("policy+prior", "policy_prior", "policy-prior"):
                param_dict = pick({"policy", "log_std", "prior_adapt"})
            elif mode in ("prior+guidance", "prior_guidance", "prior-guidance"):
                param_dict = pick({"prior_adapt", "guidance"})
            elif mode == "all":
                param_dict = pick({"policy", "log_std", "prior_adapt", "guidance"})
            else:
                param_dict = pick({"policy", "log_std"})
        else:
            param_dict = {}
        
        # Convert dict to modules format: {module_name: [numpy arrays]}
        # This format is expected by FedGuideStrategy._modules_from_metrics
        modules_dict = {}
        if isinstance(param_dict, dict):
            import torch
            import json
            import numpy as np
            for module_name, module_params in param_dict.items():
                if isinstance(module_params, dict):
                    # Convert state_dict to list of numpy arrays
                    arrays = []
                    for v in module_params.values():
                        if isinstance(v, torch.Tensor):
                            arrays.append(v.numpy().tolist())
                        elif hasattr(v, "numpy"):
                            arrays.append(v.numpy().tolist())
                        elif isinstance(v, np.ndarray):
                            arrays.append(v.tolist())
                        else:
                            arrays.append(v)
                    modules_dict[module_name] = arrays
                elif isinstance(module_params, torch.Tensor):
                    modules_dict[module_name] = [module_params.numpy().tolist()]
                elif hasattr(module_params, "numpy"):
                    modules_dict[module_name] = [module_params.numpy().tolist()]
                elif isinstance(module_params, np.ndarray):
                    modules_dict[module_name] = [module_params.tolist()]
                else:
                    modules_dict[module_name] = [module_params]
        
        # Serialize modules_dict to JSON string for Flower metrics compatibility
        # Flower metrics only support basic types (int, float, str, bytes, bool, list)
        # Server will deserialize it using json.loads
        modules_json = json.dumps(modules_dict) if modules_dict else None
        
        # Ensure loss is a valid float (not None, not nan, not inf)
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
        print(f"[FedGuideClient {cid}] Round {rnd}: loss = {loss}, train_return = {train_return}, eval_return = {eval_return}, success = {success}")
        
        # Build metrics dict
        fit_metrics = {
            "loss": loss,
            "success": int(bool(success)),
        }
        if modules_json is not None:
            fit_metrics["modules"] = modules_json  # JSON string for OT-MoE aggregation
        if train_return is not None:
            fit_metrics["train/return"] = float(train_return)
        if eval_return is not None:
            fit_metrics["eval/return"] = float(eval_return)
        
        # Get new parameters (as list)
        new_params_list = self.get_parameters(config)
        
        # Ensure new_params_list is a list, not a dict
        if isinstance(new_params_list, dict):
            import numpy as np
            new_params_list = [np.asarray(v) for v in new_params_list.values()]
        
        # Verify it's a list
        if not isinstance(new_params_list, list):
            import numpy as np
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
        # This allows the server to collect actions even if collector is not shared
        if hasattr(self.trainer, 'last_actions') and self.trainer.last_actions is not None:
            try:
                import json
                import numpy as np
                actions = self.trainer.last_actions
                # Convert to list for JSON serialization
                if isinstance(actions, np.ndarray):
                    actions_list = actions.tolist()
                elif isinstance(actions, (list, tuple)):
                    actions_list = [a.tolist() if isinstance(a, np.ndarray) else a for a in actions]
                else:
                    actions_list = actions
                # Store in metrics as JSON string (Flower only allows basic types)
                fit_metrics["client_actions"] = json.dumps(actions_list)
                # Also store client_id for mapping
                mapped_id = abs(hash(cid)) % (getattr(self, '_num_clients', 100) if hasattr(self, '_num_clients') else 100)
                fit_metrics["client_id_mapped"] = mapped_id
            except Exception as e:
                # Silently fail if serialization fails
                pass
        
        # Evaluate policy on grid and pass through metrics for server-side visualization
        # This allows server to collect client_metrics even if collector is not shared
        if self.metrics_collector is not None:
            try:
                import json
                import numpy as np
                # Evaluate agent on grid
                grid_metrics = self.metrics_collector.evaluate_on_grid(
                    agent=self.agent,
                    client_id=mapped_id if 'mapped_id' in locals() else None,
                    round_num=rnd
                )
                # Serialize grid metrics to JSON (Flower only allows basic types)
                # Convert numpy arrays to lists and flatten for JSON
                serialized_metrics = {}
                for key, value in grid_metrics.items():
                    if isinstance(value, np.ndarray):
                        # Flatten and convert to list
                        serialized_metrics[key] = json.dumps(value.tolist())
                    else:
                        serialized_metrics[key] = json.dumps(value)
                # Store in fit_metrics with prefix to identify as client metrics
                for key, value in serialized_metrics.items():
                    fit_metrics[f"client_grid_{key}"] = value
            except Exception as e:
                # Silently fail if evaluation fails
                pass
        
        return new_params_list, samples, fit_metrics


# --------- client_fn_builder ----------
def client_fn_builder(
    env_id: str,
    algo: str = "ppo",
    *,
    aggregate_mode: str = "prior+guidance",
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
    metrics_collector: Optional[Any] = None,  # Bandit2DMetricsCollector instance (for backward compatibility)
    num_clients: Optional[int] = None,  # Total number of clients for ID mapping
):

    def client_fn(context) -> Any:
        # 1) per-client seed and ID mapping
        cid = str(getattr(context, "client_id", None) or getattr(context, "node_id", None) or "0")
        
        # Map Flower's client ID to 0, 1, 2, 3... for pretrained model loading
        if num_clients is not None:
            mapped_client_id = abs(hash(cid)) % num_clients
        else:
            mapped_client_id = abs(hash(cid)) % 100
        
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

        # 3) Load pretrained prior and guidance models
        prior, guidance = None, None
        prior_ckpt, guidance_ckpt = None, None
        
        import os
        # Map env_id to pretrain model directory name
        env_name_map = {
            "bandit2d": "Bandit2D",
            "bandit_2d": "Bandit2D",
            "2dbandit": "Bandit2D",
        }
        env_name = env_name_map.get(env_id.lower(), env_id)
        
        # Build checkpoint paths using mapped client ID (0, 1, 2, 3...)
        try:
            # Use mapped_client_id instead of raw cid for model path
            client_id = mapped_client_id
            # Use relative path to match pretrain script: ./model/models_prior
            base_dir = "./model/models_prior"
            client_dir = os.path.join(base_dir, env_name, f"client_{client_id}", "final")
            
            prior_path = os.path.join(client_dir, "torch_prior.pth")
            guidance_path = os.path.join(client_dir, "guidance_sdice.pth")
            
            # Load prior if checkpoint exists
            if os.path.isfile(prior_path):
                # Try to detect model type from checkpoint
                sd = torch.load(prior_path, map_location="cpu")
                is_simple_prior = False
                
                if isinstance(sd, dict):
                    # SimpleDiffusionPrior format: has "prior" key or "state_dim" but no "unet"
                    if "prior" in sd or ("state_dim" in sd and "unet" not in sd):
                        is_simple_prior = True
                    # Also check if it's a direct state dict with encoder/decoder keys
                    elif not ("unet" in sd or "scheduler_config" in sd):
                        # Check if keys suggest SimpleDiffusionPrior structure
                        if any("encoder" in k or "decoder" in k for k in sd.keys()):
                            is_simple_prior = True
                
                if is_simple_prior:
                    # Load SimpleDiffusionPrior
                    from fedguide.guidance.diffusion_prior import SimpleDiffusionPrior
                    
                    # Extract hyperparameters from checkpoint if available
                    if isinstance(sd, dict) and "prior" in sd:
                        hidden_dim = sd.get("hidden_dim", 256)
                        timesteps = sd.get("timesteps", 1000)
                    elif isinstance(sd, dict) and "state_dim" in sd:
                        hidden_dim = sd.get("hidden_dim", 256)
                        timesteps = sd.get("timesteps", 1000)
                    else:
                        # Default hyperparameters (should match pretrain settings)
                        hidden_dim = 256
                        timesteps = 1000
                    
                    prior = SimpleDiffusionPrior(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        hidden_dim=hidden_dim,
                        timesteps=timesteps
                    )
                    prior_ckpt = prior_path
                    print(f"[Client {cid} (mapped to {client_id})] Found pretrained SimpleDiffusionPrior at: {prior_path}")
                else:
                    # Load DiffusionGuidance
                    from fedguide.guidance.diffusion_prior import DiffusionGuidance
                    # Default hyperparameters (should match pretrain settings)
                    prior = DiffusionGuidance(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        hidden_dim=256,  # Default from pretrain_bandit2d.py
                        timesteps=1000,
                        horizon=64,  # Default from pretrain_bandit2d.py
                    )
                    prior_ckpt = prior_path
                    print(f"[Client {cid} (mapped to {client_id})] Found pretrained DiffusionGuidance at: {prior_path}")
            else:
                print(f"[Client {cid} (mapped to {client_id})] No pretrained prior found at: {prior_path} (will train from scratch)")
            
            # Load guidance if checkpoint exists (optional - only if pretrained with guidance_mode)
            if os.path.isfile(guidance_path):
                from fedguide.guidance.model import SDICE_Critic
                # Create args object for SDICE_Critic
                class _C:
                    pass
                c = _C()
                c.device = "cuda" if torch.cuda.is_available() else "cpu"
                c.q_ensemble_num = 0
                c.value_lr = 1e-4
                c.wt_lr = 1e-4
                c.weight_decay = 1e-4
                c.use_lr_schedule = 0
                c.train_epoch = 1
                c.min_value_lr = 1e-5
                c.M = 8
                c.alpha = 0.5
                c.hidden_dim = 256
                
                guidance = SDICE_Critic(adim=action_dim, sdim=state_dim, args=c)
                guidance_ckpt = guidance_path
                print(f"[Client {cid} (mapped to {client_id})] Found pretrained guidance at: {guidance_path}")
            else:
                # Guidance is optional - only created if pretrained with guidance_mode != "off"
                print(f"[Client {cid} (mapped to {client_id})] No pretrained guidance found at: {guidance_path} (optional, will use prior only)")
        except Exception as e:
            print(f"[Client {cid} (mapped to {mapped_client_id})] Failed to load pretrained models: {e}")
            prior, guidance = None, None
            prior_ckpt, guidance_ckpt = None, None

        # 4) agent
        agent = FedguideAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            prior=prior,
            guidance=guidance,
            prior_ckpt=prior_ckpt,
            guidance_ckpt=guidance_ckpt,
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
                    run_module = importlib.import_module('scripts.envs.bandit2d.run_fedguide_bandit2d')
                    metrics_collector = getattr(run_module, '_metrics_collector_global', None)
                except (ImportError, AttributeError):
                    try:
                        run_module = importlib.import_module('scripts.envs.bandit2d.run_fedkl_bandit2d')
                        metrics_collector = getattr(run_module, '_metrics_collector_global', None)
                    except (ImportError, AttributeError):
                        pass
            except Exception:
                pass
        
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
        
        # Register agent with metrics collector for visualization
        if metrics_collector is not None:
            # Map Flower client ID to sequential ID for metrics
            if num_clients is not None:
                mapped_id = abs(hash(cid)) % num_clients
            else:
                mapped_id = abs(hash(cid)) % 100
            metrics_collector.register_client_agent(mapped_id, agent)
        
        # Convert NumPyClient to Client
        return client.to_client()

    return client_fn
