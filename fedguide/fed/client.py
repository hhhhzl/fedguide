import time
import logging
import torch
import flwr as fl
from typing import Any, Dict, Optional, Callable, Iterable
from fedguide.utils.logger import BaseLogger, MetricsBus, StdLogger, WandbLogger
from fedguide.utils.seeds import set_all_seeds


class FedRLClient(fl.client.NumPyClient):
    """Flower client wrapper for FedRL.

    This keeps the existing (agent, env, trainer) design. The trainer is expected to
    expose:
      - train_one_round() -> loss
      - save_eval(cid, rnd) -> success_flag (bool/int)
      - n_steps (int)  # number of local samples/steps used this round
    The agent is expected to expose:
      - get_parameters() -> list[np.ndarray] or Flower-compatible
      - set_parameters(params)
      - optionally `.to(device)` to move to compute device
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
        logger: Optional[BaseLogger] = None,
        callbacks: Optional[Iterable[Callable[[Dict[str, Any]], None]]] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        logger_level: int = logging.INFO,
        metrics_collector: Optional[Any] = None,  # Bandit2DMetricsCollector instance
    ):
        super().__init__()
        self.agent = agent
        self.env = env
        self.trainer = trainer

        # Device & seed
        self.device = device
        if hasattr(self.agent, "to"):
            try:
                self.agent.to(self.device)
            except Exception:
                pass
        set_all_seeds(seed, env)

        # Metrics/Logging
        if logger is None:
            if use_wandb:
                logger = WandbLogger(run_name=run_name, project=wandb_project)
            else:
                logger = StdLogger(run_name or "fedrl", level=logger_level)
        self.metrics = MetricsBus(logger=logger, callbacks=callbacks)
        self.metrics_collector = metrics_collector

        # Optional: allow trainer to push metrics via injected bus, without changing signatures
        if hasattr(self.trainer, "set_metrics_bus"):
            try:
                self.trainer.set_metrics_bus(self.metrics)
            except Exception:
                pass

    def get_parameters(self, config: Dict[str, Any]):
        if hasattr(self.agent, "get_parameters"):
            return self.agent.get_parameters()
        state = getattr(self.agent, "state_dict", lambda: {})()
        return [v.detach().cpu().numpy() for v in state.values()]

    def set_parameters(self, parameters):
        # Handle Parameters object from Flower
        from flwr.common import parameters_to_ndarrays
        if hasattr(parameters, 'tensors'):  # It's a Parameters object
            parameters = parameters_to_ndarrays(parameters)
        
        if hasattr(self.agent, "set_parameters"):
            # Check if agent expects dict or list
            import inspect
            sig = inspect.signature(self.agent.set_parameters)
            params = list(sig.parameters.values())
            # If agent.set_parameters expects a dict (has type hint or annotation), convert
            if params and 'Dict' in str(params[0].annotation):
                # Agent expects dict, but we have list - can't convert without layout
                # So use state_dict approach instead
                pass
            else:
                # Try calling with list first
                try:
                    return self.agent.set_parameters(parameters)
                except (TypeError, ValueError):
                    # If it fails, fall through to state_dict approach
                    pass
        
        # Fallback: use state_dict approach for list format
        if isinstance(parameters, list):
            sd = self.agent.state_dict()
            new_sd = {}
            for (k, v), arr in zip(sd.items(), parameters):
                tensor = torch.tensor(arr, dtype=v.dtype, device=v.device)
                new_sd[k] = tensor.view_as(v)
            self.agent.load_state_dict(new_sd, strict=False)
        else:
            # It's already a dict, pass through
            if hasattr(self.agent, "set_parameters"):
                return self.agent.set_parameters(parameters)

    # -------------------------
    # Federated operations
    # -------------------------
    def fit(self, parameters, config):
        start = time.time()
        self.set_parameters(parameters)

        # Round/context
        rnd = int(config.get("server_round", 0))
        cid = getattr(self, "cid", config.get("cid", "unknown"))
        self.metrics.set_step(rnd)

        # Train one round
        train_result = self.trainer.train_one_round()
        
        # Extract loss and other metrics from trainer result
        if isinstance(train_result, dict):
            # Debug: Print available keys for first round to understand structure
            if rnd == 1 and cid == "0":
                print(f"[DEBUG] train_result keys: {list(train_result.keys())}")
                for k, v in train_result.items():
                    if "loss" in k.lower():
                        print(f"[DEBUG] {k} = {v} (type: {type(v)})")
            
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
                                if rnd == 1 and cid == "0":
                                    print(f"[DEBUG] Found loss from key '{key}': {loss}")
                                break
                        except (TypeError, ValueError):
                            continue
            
            # If still None or invalid (nan/inf), try to use return as a proxy (negative return = higher loss)
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
            
            # Extract return metrics if available
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

        # Eval/save as the original code expects
        success = self.trainer.save_eval(cid, rnd)
        samples = int(getattr(self.trainer, "n_steps", 0))
        
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
        mapped_id = None
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
                mapped_id = abs(hash(cid)) % 100
                fit_metrics["client_id_mapped"] = mapped_id
            except Exception as e:
                # Silently fail if serialization fails
                pass
        
        # Evaluate policy on grid and pass through metrics for server-side visualization
        # This allows server to collect client_metrics even if collector is not shared
        if self.metrics_collector is not None and mapped_id is not None:
            try:
                import json
                import numpy as np
                # Evaluate agent on grid
                grid_metrics = self.metrics_collector.evaluate_on_grid(
                    agent=self.agent,
                    client_id=mapped_id,
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

        # Collect optional trainer-provided metrics if available
        extra = {}
        for key in ("return", "return_", "success_rate", "episode_len", "throughput"):
            if hasattr(self.trainer, key):
                try:
                    value = getattr(self.trainer, key)
                    # Handle property access
                    if callable(value) and not isinstance(value, (int, float, str, list, dict)):
                        try:
                            value = value()
                        except TypeError:
                            pass
                    # Normalize return_ to return
                    if key == "return_":
                        extra["return"] = value
                    else:
                        extra[key] = value
                except Exception:
                    pass
        dur = time.time() - start

        # Emit metrics
        self.metrics.emit({
            "round": rnd,
            "client_id": str(cid),
            "loss": float(loss) if loss is not None else float("nan"),
            "success": int(bool(success)),
            "samples": samples,
            "device": str(self.device),
            "duration_sec": dur,
            **extra,
        })

        # Build metrics dict for Flower server (include return for reward curve plotting)
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
        print(f"[Client {cid}] Round {rnd}: loss = {loss}, train_return = {train_return}, eval_return = {eval_return}, success = {success}")
        
        fit_metrics = {
            "loss": loss,
            "success": int(bool(success)),
        }
        if train_return is not None:
            fit_metrics["train/return"] = float(train_return)
        if eval_return is not None:
            fit_metrics["eval/return"] = float(eval_return)
        if "return" in extra:
            fit_metrics["return"] = float(extra["return"])

        # Return new parameters + num_examples + metrics for FL server
        new_params = self.get_parameters(config)
        return new_params, samples, fit_metrics

    def evaluate(self, parameters, config):
        """Evaluate the model on the local dataset.
        
        Returns:
            loss: Loss value (float)
            num_examples: Number of examples evaluated (int)
            metrics: Dictionary of metrics (dict)
        """
        # Convert parameters from Flower Parameters object to list of numpy arrays if needed
        from flwr.common import parameters_to_ndarrays
        if hasattr(parameters, 'tensors'):  # It's a Parameters object
            parameters = parameters_to_ndarrays(parameters)
        
        # Set parameters first
        self.set_parameters(parameters)
        
        # Get round number
        rnd = int(config.get("server_round", 0))
        cid = getattr(self, "cid", config.get("cid", "unknown"))
        
        # Run evaluation if trainer supports it
        loss = float("nan")
        num_examples = 0
        metrics = {}
        
        try:
            # Try to get evaluation metrics from trainer
            if hasattr(self.trainer, "save_eval"):
                # save_eval typically returns success flag, but we can use it for evaluation
                success = self.trainer.save_eval(cid, rnd)
                metrics["success"] = int(bool(success))
                num_examples = int(getattr(self.trainer, "n_steps", 0))
            
            # Try to get loss from trainer if available
            if hasattr(self.trainer, "last_loss"):
                loss = float(self.trainer.last_loss)
            elif hasattr(self.trainer, "loss"):
                loss_val = self.trainer.loss
                if callable(loss_val):
                    try:
                        loss = float(loss_val())
                    except (TypeError, ValueError):
                        pass
                else:
                    loss = float(loss_val)
        except Exception:
            pass
        
        # Return tuple: (loss, num_examples, metrics)
        return loss, num_examples, metrics

    def __del__(self):
        try:
            self.metrics.close()
        except Exception:
            pass