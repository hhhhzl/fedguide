"""MFPO (Momentum-assisted Federated Policy Optimization) baseline — INFOCOM 2024."""

from fedguide.baselines.mfpo.agent import MFPOAgent, MFPOContinuousWorker, MFPODiscreteCartPoleWorker
from fedguide.baselines.mfpo.trainer import MFPTrainer
from fedguide.baselines.mfpo.client import client_fn_builder

__all__ = [
    "MFPOAgent",
    "MFPOContinuousWorker",
    "MFPODiscreteCartPoleWorker",
    "MFPTrainer",
    "client_fn_builder",
]
