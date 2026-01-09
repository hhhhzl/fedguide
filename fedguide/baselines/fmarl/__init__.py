"""
FMARL Baseline Implementation

This module provides the FMARL (Federated Multi-Agent Reinforcement Learning)
baseline implementation for federated reinforcement learning.

FMARL features:
- Sequential federated training with old policy synchronization
- Weighted aggregation based on client timesteps (not num_examples)
- Old policy snapshot for KL divergence computation
"""

from fedguide.baselines.fmarl.client import FMARLClient, client_fn_builder
from fedguide.baselines.fmarl.server import FMARLStrategy, run_fmarl_server, FMARLServer
from fedguide.baselines.fmarl.agent import FMARLAgent, PolicyNetwork, ValueNetwork
from fedguide.baselines.fmarl.trainer import FMARLTrainer

__all__ = [
    "FMARLClient",
    "FMARLStrategy",
    "FMARLServer",
    "FMARLAgent",
    "FMARLTrainer",
    "PolicyNetwork",
    "ValueNetwork",
    "client_fn_builder",
    "run_fmarl_server",
]

