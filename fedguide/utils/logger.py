import logging
from typing import Any, Dict, Optional, Callable, Iterable

def get_logger(name="fedguide"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# -----------------------------
# Lightweight logging backends
# -----------------------------
class BaseLogger:
    def __init__(self, run_name: Optional[str] = None):
        self.run_name = run_name or "fedrl"

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        raise NotImplementedError

    def close(self):
        pass


class StdLogger(BaseLogger):
    def __init__(self, run_name: Optional[str] = None, level: int = logging.INFO):
        super().__init__(run_name)
        self.logger = logging.getLogger(self.run_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)
        self.logger.setLevel(level)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        prefix = f"step={step} " if step is not None else ""
        self.logger.info(prefix + ", ".join(f"{k}={v}" for k, v in metrics.items()))


class WandbLogger(BaseLogger):  # optional
    def __init__(self, run_name: Optional[str] = None, project: Optional[str] = None, **cfg):
        super().__init__(run_name)
        try:
            import wandb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("wandb is not installed. pip install wandb or use StdLogger.") from e
        self._wandb = wandb
        self._wandb.init(project=project or "fedrl", name=self.run_name, config=cfg)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        self._wandb.log(metrics, step=step)

    def close(self):
        self._wandb.finish()

# -----------------------------
# Pluggable metrics collector
# -----------------------------
class MetricsBus:
    """Fan-out metrics to multiple sinks and/or user-defined callbacks.

    Without changing agent/env/trainer, the client will emit a few standard metrics
    at round end. You may also call `emit` from the trainer if you pass this bus in.
    """

    def __init__(self, logger: Optional[BaseLogger] = None, callbacks: Optional[Iterable[Callable[[Dict[str, Any]], None]]] = None):
        self.logger = logger or StdLogger("fedrl")
        self.callbacks = list(callbacks) if callbacks is not None else []
        self._global_step = 0

    @property
    def step(self) -> int:
        return self._global_step

    def set_step(self, step: int):
        self._global_step = int(step)

    def emit(self, metrics: Dict[str, Any], step: Optional[int] = None):
        stp = self._global_step if step is None else int(step)
        # ensure JSON-serializable basic types for flwr/federated metadata
        safe = {k: (v.item() if isinstance(v, (np.generic,)) else float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v) for k, v in metrics.items()}
        self.logger.log(safe, step=stp)
        for cb in self.callbacks:
            try:
                cb({"step": stp, **safe})
            except Exception:  # pragma: no cover
                pass

    def close(self):
        try:
            self.logger.close()
        except Exception:
            pass