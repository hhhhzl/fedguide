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
        if hasattr(self.agent, "set_parameters"):
            return self.agent.set_parameters(parameters)
        sd = self.agent.state_dict()
        new_sd = {}
        for (k, v), arr in zip(sd.items(), parameters):
            tensor = torch.tensor(arr, dtype=v.dtype, device=v.device)
            new_sd[k] = tensor.view_as(v)
        self.agent.load_state_dict(new_sd, strict=False)

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
            loss = train_result.get("loss", train_result.get("train/loss", None))
            # Extract return metrics if available
            train_return = train_result.get("train/return", train_result.get("return", None))
            eval_return = train_result.get("eval/return", None)
        else:
            loss = train_result
            train_return = None
            eval_return = None

        # Eval/save as the original code expects
        success = self.trainer.save_eval(cid, rnd)
        samples = int(getattr(self.trainer, "n_steps", 0))

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
        fit_metrics = {
            "loss": float(loss) if loss is not None else float("nan"),
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
        # Keep the original behavior (NotImplemented). If needed, you can implement
        # an evaluation pass similar to fit and emit metrics via self.metrics.
        return NotImplemented

    def __del__(self):
        try:
            self.metrics.close()
        except Exception:
            pass