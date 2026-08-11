from __future__ import annotations

from typing import Any, Dict, Optional, Callable, Iterable
import json
import os
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
from fedguide.utils.gym_space_utils import is_box1d as _is_box1d


def _make_env(
    env_id: str,
    seed: Optional[int] = None,
    client_id: Optional[int] = None,
    num_clients: Optional[int] = None,
    sigma: float = 0.2,
    metadata_path: Optional[str] = None,
    render_mode: Optional[str] = None,
    reward_type: Optional[str] = None,
    origin_client_id: Optional[int] = None,
):
    # Handle custom environments
    if env_id.lower() in ["bandit2d", "bandit_2d", "2dbandit"]:
        from fedguide.envs.bandit2d import Bandit2D
        if origin_client_id is not None and client_id == int(origin_client_id):
            env = Bandit2D(K=1, sigma=sigma, seed=seed, preferred_peak=None)
            env.mu = np.zeros((1, 2), dtype=np.float64)
            env.peak_weights = np.ones(1, dtype=np.float32)
            if seed is not None:
                env.reset(seed=seed)
            return env
        # Client-specific heterogeneity: each client prefers one peak (client_id % K)
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
        metadata_path, client_id, seed, render_mode, render_eval=(render_mode is not None)
    )
    if _hc_env is not None:
        return _hc_env

    # Walker2D / Ant / Hopper share the locomotion-hetero loader
    from fedguide.envs.mujoco_locomotion_hetero import make_locomotion_env_if_applicable

    _loco_env = make_locomotion_env_if_applicable(
        metadata_path, client_id, seed, render_mode, render_eval=(render_mode is not None)
    )
    if _loco_env is not None:
        return _loco_env

    # MetaWorld ML10
    from fedguide.envs.metaworld_hetero import make_metaworld_env_if_applicable

    _mw_env = make_metaworld_env_if_applicable(
        metadata_path, client_id, seed, render_mode
    )
    if _mw_env is not None:
        return _mw_env

    if env_id.lower() == "pointmazenarrow":
        from fedguide.envs.pointmaze_narrow import PointMazeNarrow
        env = PointMazeNarrow()
        if seed is not None:
            try:
                env.reset(seed=seed)
            except TypeError:
                env.reset()
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
        mapped_client_id: Optional[int] = None,  # Deterministic ID for env/metrics (0..N-1)
        num_clients: Optional[int] = None,
        policy_save_dir: Optional[str] = None,
        policy_save_every: int = 0,
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
        self._mapped_client_id = mapped_client_id
        self._num_clients = num_clients or 4
        self._incoming_layout: Optional[Dict[str, Any]] = None
        self._policy_save_dir = policy_save_dir
        self._policy_save_every = int(policy_save_every or 0)

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
        
        if not isinstance(parameters, dict):
            # Reconstruct module dict from server-provided layout when available.
            if mode in ("prior+guidance", "prior_guidance", "prior-guidance", "prior", "all"):
                if parameters is None or (isinstance(parameters, list) and len(parameters) == 0):
                    return
                layout = self._incoming_layout
                if layout is None:
                    return
                if isinstance(layout, str):
                    try:
                        layout = json.loads(layout)
                    except Exception:
                        return
                modules = self._unflatten_to_modules(parameters, layout)
                if modules:
                    parameters = modules
                else:
                    return
            else:
                return super().set_parameters(parameters)

        allowed = set()
        if mode == "policy":
            allowed = {"policy", "log_std"}
        elif mode == "policy_value":
            allowed = {"policy", "log_std", "value"}
        elif mode == "prior":
            allowed = {"prior_adapt", "prior_mixture"}
        elif mode in ("policy+prior", "policy_prior", "policy-prior"):
            allowed = {"policy", "log_std", "prior_adapt", "prior_mixture"}
        elif mode in ("prior+guidance", "prior_guidance", "prior-guidance"):
            allowed = {"prior_adapt", "prior_mixture", "guidance"}
        elif mode == "all":
            allowed = {"policy", "log_std", "prior_adapt", "prior_mixture", "guidance"}

        filtered = {k: v for k, v in parameters.items() if k in allowed}
        if filtered:
            self.agent.set_parameters(filtered)

    def _unflatten_to_modules(self, flat_params, layout: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct {module: state_dict_like} from flattened list using layout."""
        if not isinstance(layout, dict) or "order" not in layout:
            return {}
        try:
            full = self.agent.get_parameters()
        except Exception:
            return {}

        idx = 0
        out: Dict[str, Any] = {}
        for module_name, count in layout.get("order", []):
            count = int(count)
            chunk = flat_params[idx: idx + count]
            idx += count
            if module_name == "prior_mixture" and count == 3:
                out[module_name] = [
                    torch.as_tensor(np.asarray(arr), dtype=torch.float32)
                    for arr in chunk
                ]
                continue
            if module_name not in full:
                continue
            module_params = full[module_name]
            if isinstance(module_params, dict):
                keys = list(module_params.keys())
                if len(keys) != count:
                    continue
                rebuilt = {}
                for key, arr in zip(keys, chunk):
                    if isinstance(module_params[key], torch.Tensor):
                        rebuilt[key] = torch.tensor(
                            np.asarray(arr), dtype=module_params[key].dtype
                        )
                    else:
                        rebuilt[key] = np.asarray(arr)
                out[module_name] = rebuilt
            else:
                if count > 0:
                    arr0 = chunk[0]
                    if isinstance(module_params, torch.Tensor):
                        out[module_name] = torch.tensor(
                            np.asarray(arr0), dtype=module_params.dtype
                        )
                    else:
                        out[module_name] = np.asarray(arr0)
        return out
    
    def fit(self, parameters, config):
        """Override fit to handle module-based parameters and collect actions for metrics."""
        cid = getattr(self, "cid", config.get("cid", "unknown"))
        rnd = int(config.get("server_round", 0))
        self._incoming_layout = config.get("layout")
        if "client_id_mapped" in config:
            try:
                self._mapped_client_id = int(config.get("client_id_mapped"))
            except (TypeError, ValueError):
                pass
        if bool(config.get("routing_debug", False)):
            print(
                f"[FedGuideClientRouting] round={rnd} cid={cid} "
                f"mapped={self._mapped_client_id} expert={config.get('expert_id', None)} "
                f"has_layout={self._incoming_layout is not None}"
            )
        
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
        
        # Set server round for lambda_guide annealing (if trainer supports it)
        if hasattr(self.trainer, "set_server_round"):
            self.trainer.set_server_round(rnd)
        
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

        # Persist per-client policy each round (overwrites; last round = final).
        # FedGuide keeps the policy LOCAL (only the prior is OT-MoE aggregated),
        # so without this checkpoint we lose the trained per-client policies
        # when Ray actors are torn down at sim end. Path:
        #   <output_dir>/client_<cid>/round_<rnd>/policy.pth (and final/)
        try:
            import os as _os
            output_dir = getattr(self.trainer, "output_dir", None)
            if output_dir is None:
                output_dir = config.get("output_dir") if isinstance(config, dict) else None
            if output_dir:
                mapped = getattr(self, "_mapped_client_id", None)
                client_id_str = str(mapped if mapped is not None else cid)
                ckpt_dir = _os.path.join(output_dir, f"client_{client_id_str}", "final")
                _os.makedirs(ckpt_dir, exist_ok=True)
                policy_sd = self.agent.policy.state_dict() if hasattr(self.agent, "policy") else None
                log_std = getattr(self.agent, "log_std", None)
                if policy_sd is not None:
                    payload = {
                        "policy": {k: v.detach().cpu() for k, v in policy_sd.items()},
                        "round": int(rnd),
                        "mapped_client_id": int(self.mapped_client_id) if hasattr(self, "mapped_client_id") else -1,
                    }
                    if log_std is not None:
                        payload["log_std"] = log_std.detach().cpu()
                    if hasattr(self.agent, "value_fn"):
                        payload["value_fn"] = {k: v.detach().cpu() for k, v in self.agent.value_fn.state_dict().items()}
                    torch.save(payload, _os.path.join(ckpt_dir, "policy.pth"))
        except Exception as e:
            print(f"[FedGuideClient {cid}] policy ckpt save failed: {e}")

        # Per-client policy snapshot at fixed cadence (rounds 20/40/...): preserved
        # alongside the overwriting `final/` ckpt above. Lands in
        # `<policy_save_dir>/client_<id>/round_XXXX.pth`.
        if (
            self._policy_save_dir
            and self._policy_save_every > 0
            and rnd > 0
            and (rnd % self._policy_save_every == 0)
        ):
            try:
                mid = self._mapped_client_id if self._mapped_client_id is not None else cid
                client_dir = os.path.join(self._policy_save_dir, f"client_{mid}")
                os.makedirs(client_dir, exist_ok=True)
                ckpt_path = os.path.join(client_dir, f"round_{int(rnd):04d}.pth")
                policy_state = {k: v.detach().cpu() for k, v in self.agent.policy.state_dict().items()}
                payload = {
                    "round": int(rnd),
                    "mapped_client_id": mid,
                    "cid": str(cid),
                    "policy_state_dict": policy_state,
                }
                log_std = getattr(self.agent, "log_std", None)
                if log_std is not None and hasattr(log_std, "detach"):
                    payload["log_std"] = log_std.detach().cpu()
                torch.save(payload, ckpt_path)
                print(f"  [FedGuideClient {mid}] Saved policy snapshot to {ckpt_path}", flush=True)
            except Exception as e:
                print(f"  [FedGuideClient] Warning: failed to save policy snapshot: {e}", flush=True)

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
            import json
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
        collect_every = int(config.get("collect_metrics_every", 1) or 1)
        if self.metrics_collector is not None and rnd % collect_every == 0:
            try:
                client_id = (
                    self._mapped_client_id
                    if self._mapped_client_id is not None
                    else (int(cid) if isinstance(cid, (int, str)) and str(cid).isdigit() else hash(cid) % 10000)
                )
                if hasattr(self.trainer, 'last_actions') and self.trainer.last_actions is not None:
                    actions = self.trainer.last_actions
                    self.metrics_collector.collect_client_actions(client_id, actions)
            except Exception:
                pass

        # Use deterministic mapped ID (must match env preferred_peak for heterogeneity)
        mapped_id = (
            self._mapped_client_id
            if self._mapped_client_id is not None
            else abs(hash(str(cid))) % (self._num_clients or 4)
        )
        
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
    prior_coef: float = 1.0,
    lambda_guide_anneal: bool = False,
    lambda_guide_decay_rounds: int = 40,
    init_log_std: float = 0.0,
    online_guidance: bool = False,
    online_prior: bool = False,
    prior_adapt_fallback_all: bool = False,
    # logging
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    metrics_collector: Optional[Any] = None,  # Bandit2DMetricsCollector instance (for backward compatibility)
    num_clients: Optional[int] = None,  # Total number of clients for ID mapping
    cid_mapping_file: Optional[str] = None,  # File for deterministic cid->0..N-1 mapping
    sigma: float = 0.2,  # Bandit2D reward width
    use_pretrained_models: bool = True,
    metadata_path: Optional[str] = None,
    reward_type: Optional[str] = None,
    render_eval: bool = False,
    render_mode: str = "video",
    render_save_dir: Optional[str] = None,
    render_every_n_rounds: int = 10,
    render_episodes: int = 5,
    render_all_clients: bool = False,
    reacher_render_mode: Optional[str] = None,
    policy_save_dir: Optional[str] = None,
    policy_save_every: int = 0,
    output_dir: Optional[str] = None,
    origin_client_id: Optional[int] = None,
    origin_prior_path: Optional[str] = None,
    # Opt-in policy architecture knobs (defaults preserve legacy behaviour).
    policy_activation: str = "tanh",
    action_clamp_low: Optional[float] = None,
    action_clamp_high: Optional[float] = None,
    log_std_anneal: bool = False,
    log_std_anneal_target: float = -2.0,
    log_std_anneal_rounds: int = 40,
    # Where to look for pretrained prior/guidance ckpts. Defaults to the
    # legacy SimpleDiffusionPrior path; set to ``./model/models_prior_gauss``
    # to load Gaussian priors instead.
    prior_dir: str = "./model/models_prior",
    # If set, the agent's policy will be initialized from a per-client BC
    # checkpoint at <bc_dir>/<env>/client_<cid>/final/policy.pth (the format
    # produced by `scripts/envs/reacher/_bc_pretrain.py`). This is the warm-
    # start equivalent of bandit2d's "policy.bias ← prior.head_mu" trick for
    # state-conditional priors.
    bc_dir: Optional[str] = None,
    bc_env_name: Optional[str] = None,  # subdir under bc_dir; defaults to env mapping
    bc_blend_alpha: float = 1.0,
    # Guide-align loss weight (MSE between μ(s) and a + η·∇W). The original
    # default (1.0) is too aggressive on bandit2d where the SDICE_Critic
    # is undertrained and the gradient is noisy. Lower values (0.05–0.1)
    # let the prior dominate and use the guidance only as a fine adjustment.
    guide_coef: float = 1.0,
    guidance_eta: float = 0.1,
    prior_reshape: bool = False,
    reshape_beta: float = 0.1,
    dice_reward_eta: float = 0.0,
    dice_v_blend_alpha: float = 1.0,
    dice_adv_beta: float = 0.0,
):

    if render_all_clients:
        os.environ["FEDGUIDE_FEDERATED_RENDER_ALL_CLIENTS"] = "1"

    def client_fn(context) -> Any:
        # 1) per-client seed and ID mapping
        cid = str(getattr(context, "client_id", None) or getattr(context, "node_id", None) or "0")
        
        # Map Flower's long-int cid to 0..num_clients-1 (file-based to avoid collisions)
        num_c = num_clients or 4
        if cid_mapping_file:
            from fedguide.utils.client_id_mapping import get_mapped_client_id

            mapped_client_id = get_mapped_client_id(cid, num_c, cid_mapping_file)
        else:
            try:
                if str(cid).isdigit() and int(cid) < 10000:
                    mapped_client_id = int(cid) % num_c
                else:
                    import hashlib

                    h = int(hashlib.sha256(str(cid).encode()).hexdigest()[:8], 16)
                    mapped_client_id = h % num_c
            except (ValueError, TypeError):
                mapped_client_id = abs(hash(str(cid))) % num_c
        
        base = 42 + (abs(hash(cid)) % 10000)
        random.seed(base)
        np.random.seed(base)
        torch.manual_seed(base)

        # 2) env (client-specific heterogeneity for Bandit2D / Reacher)
        env = _make_env(
            env_id,
            seed=base,
            client_id=mapped_client_id,
            num_clients=num_clients,
            sigma=sigma,
            metadata_path=metadata_path,
            render_mode=reacher_render_mode,
            reward_type=reward_type,
            origin_client_id=origin_client_id,
        )
        obs_space, act_space = env.observation_space, env.action_space
        assert _is_box1d(obs_space) and _is_box1d(act_space), "Only Support 1D Box spaces."

        state_dim = int(obs_space.shape[0])
        action_dim = int(act_space.shape[0])

        # 3) Load pretrained prior and guidance models
        prior, guidance = None, None
        prior_ckpt, guidance_ckpt = None, None
        if not use_pretrained_models:
            prior_ckpt, guidance_ckpt = None, None

        import os
        # Map env_id to pretrain model directory name
        env_name_map = {
            "bandit2d": "Bandit2D",
            "bandit_2d": "Bandit2D",
            "2dbandit": "Bandit2D",
            "reacher_hetero": "Reacher",
            "reacher": "Reacher",
            "halfcheetah-v4": "HalfCheetah",
            "halfcheetah-v3": "HalfCheetah",
            "walker2d-v4": "Walker2D",
            "hopper-v4": "Hopper",
            # MetaWorld ML10 — every task shares the same prior subdir.
            "metaworld_ml10": "MetaWorld",
            "metaworld-ml10": "MetaWorld",
            "reach-v3": "MetaWorld",
            "push-v3": "MetaWorld",
            "pick-place-v3": "MetaWorld",
            "door-open-v3": "MetaWorld",
            "drawer-close-v3": "MetaWorld",
            "button-press-topdown-v3": "MetaWorld",
            "peg-insert-side-v3": "MetaWorld",
            "window-open-v3": "MetaWorld",
            "sweep-v3": "MetaWorld",
            "basketball-v3": "MetaWorld",
        }
        env_name = env_name_map.get(env_id.lower(), env_id)
        
        # Build checkpoint paths using mapped client ID (0, 1, 2, 3...)
        try:
            if not use_pretrained_models:
                raise FileNotFoundError("pretrained model loading disabled by config")
            # Use mapped_client_id instead of raw cid for model path
            client_id = mapped_client_id
            base_dir = prior_dir
            client_dir = os.path.join(base_dir, env_name, f"client_{client_id}", "final")
            is_origin_client = (
                origin_client_id is not None
                and mapped_client_id == int(origin_client_id)
            )
            prior_path = (
                str(origin_prior_path)
                if is_origin_client and origin_prior_path
                else os.path.join(client_dir, "torch_prior.pth")
            )
            guidance_path = (
                "" if is_origin_client
                else os.path.join(client_dir, "guidance_sdice.pth")
            )

            # Load prior if checkpoint exists
            if os.path.isfile(prior_path):
                # Try to detect model type from checkpoint
                sd = torch.load(prior_path, map_location="cpu")
                is_gaussian_prior = False
                is_simple_prior = False

                is_diffusion_unet = False
                if isinstance(sd, dict):
                    pt = sd.get("prior_type")
                    if pt == "gaussian":
                        is_gaussian_prior = True
                    elif pt in ("diffusion", "diffusion_unet", "diffusion_guidance"):
                        is_diffusion_unet = True
                    else:
                        inner = sd.get("prior") if "prior" in sd and isinstance(sd["prior"], dict) else sd
                        if isinstance(inner, dict) and any(k in inner for k in ("head_mu", "head_log_sigma")):
                            is_gaussian_prior = True
                        elif isinstance(inner, dict) and any(k.startswith("model.") for k in inner.keys()):
                            # DiffusionGuidance saves UNet under "model.*"
                            is_diffusion_unet = True
                        elif "prior" in sd or ("state_dim" in sd and "unet" not in sd):
                            is_simple_prior = True
                        elif not ("unet" in sd or "scheduler_config" in sd):
                            if any("encoder" in k or "decoder" in k for k in sd.keys()):
                                is_simple_prior = True

                if is_gaussian_prior:
                    from fedguide.guidance.diffusion_prior import GaussianBehaviorPrior

                    prior = GaussianBehaviorPrior(state_dim=state_dim, action_dim=action_dim)
                    prior_ckpt = prior_path
                    print(f"[Client {cid} (mapped to {client_id})] Found pretrained GaussianBehaviorPrior at: {prior_path}")
                elif is_diffusion_unet:
                    from fedguide.guidance.diffusion_prior import DiffusionGuidance

                    hidden_dim = sd.get("hidden_dim", 64) if isinstance(sd, dict) else 64
                    timesteps = sd.get("timesteps", 1000) if isinstance(sd, dict) else 1000
                    horizon = sd.get("horizon", 64) if isinstance(sd, dict) else 64
                    prior = DiffusionGuidance(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        hidden_dim=hidden_dim,
                        timesteps=timesteps,
                        horizon=horizon,
                    )
                    prior_ckpt = prior_path
                    print(f"[Client {cid} (mapped to {client_id})] Found pretrained DiffusionGuidance(UNet) at: {prior_path}")
                elif is_simple_prior:
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
            if not use_pretrained_models:
                print(f"[Client {cid} (mapped to {mapped_client_id})] Pretrained loading disabled; training from scratch.")
            else:
                print(f"[Client {cid} (mapped to {mapped_client_id})] Failed to load pretrained models: {e}")
            prior, guidance = None, None
            prior_ckpt, guidance_ckpt = None, None

        # Optional: BC warm-start checkpoint per client (reacher's
        # state-conditional analogue of bandit2d's "policy.bias ← prior.head_mu").
        actor_ckpt = None
        if bc_dir:
            try:
                env_subdir = bc_env_name or {
                    "Bandit2D": "Bandit2D",
                    "bandit2d": "Bandit2D",
                    "Reacher": "Reacher",
                    "reacher_hetero": "Reacher",
                }.get(env_id, env_id)
                cand = os.path.join(bc_dir, env_subdir,
                                    f"client_{mapped_client_id}",
                                    "final", "policy.pth")
                if os.path.isfile(cand):
                    actor_ckpt = cand
                    print(f"[FedGuide cid={mapped_client_id}] BC warm-start ← {cand}")
                else:
                    print(f"[FedGuide cid={mapped_client_id}] BC ckpt not found: {cand} (training without warm-start)")
            except Exception as e:
                print(f"[FedGuide cid={mapped_client_id}] BC lookup skipped: {e}")

        # 4) agent
        agent = FedguideAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            prior=prior,
            guidance=guidance,
            prior_ckpt=prior_ckpt,
            guidance_ckpt=guidance_ckpt,
            actor_ckpt=actor_ckpt,
            init_log_std=init_log_std,
            prior_coef=prior_coef,
            prior_adapt_fallback_all=prior_adapt_fallback_all,
            policy_activation=policy_activation,
            action_clamp_low=action_clamp_low,
            action_clamp_high=action_clamp_high,
            log_std_anneal=log_std_anneal,
            log_std_anneal_target=log_std_anneal_target,
            log_std_anneal_rounds=log_std_anneal_rounds,
            guide_coef=guide_coef,
            guidance_eta=guidance_eta,
            prior_reshape=prior_reshape,
            reshape_beta=reshape_beta,
            bc_blend_alpha=bc_blend_alpha,
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
            lambda_guide_anneal=lambda_guide_anneal,
            lambda_guide_decay_rounds=lambda_guide_decay_rounds,
            online_guidance=online_guidance,
            online_prior=online_prior,
            render_eval=render_eval,
            render_mode=render_mode,
            render_save_dir=render_save_dir,
            render_every_n_rounds=render_every_n_rounds,
            render_episodes=render_episodes,
            render_client_tag=str(mapped_client_id),
            dice_reward_eta=dice_reward_eta,
            dice_v_blend_alpha=dice_v_blend_alpha,
            dice_adv_beta=dice_adv_beta,
        )
        # Federated clients are reconstructed inside Ray actors, so explicitly
        # attach the configured model directory used by FedGuideClient.fit().
        # This keeps trained local policies under the same output tree as the
        # existing YAML runner instead of silently dropping them.
        trainer.output_dir = output_dir

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
            mapped_client_id=mapped_client_id,
            num_clients=num_clients,
            policy_save_dir=policy_save_dir,
            policy_save_every=policy_save_every,
        )
        # Store client_id for metrics collection
        client.cid = cid
        
        # Register agent with metrics collector for visualization
        if metrics_collector is not None:
            metrics_collector.register_client_agent(mapped_client_id, agent)
        
        # Convert NumPyClient to Client
        return client.to_client()

    return client_fn
