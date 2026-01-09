"""
FedRL Baseline Implementation

This module implements the FedRL algorithm (Federated Reinforcement Learning 
with Environment Heterogeneity) from AISTATS 2022.

Supports both DQN (discrete actions) and DDPG (continuous actions).
"""

from fedguide.baselines.fedrl.agent import (
    DQNNetwork,
    DQNAgent,
    DDPGActor,
    DDPGCritic,
    DDPGAgent,
    net_para_add,
    net_para_scale,
)

from fedguide.baselines.fedrl.trainer import (
    ReplayBuffer,
    DQNTrainer,
    DDPGTrainer,
)

from fedguide.baselines.fedrl.client import (
    FedRLClient,
    client_fn_builder,
)

from fedguide.baselines.fedrl.server import (
    FedRLStrategy,
    FedRLServer,
    run_fedrl_server,
)

__all__ = [
    # Agents
    "DQNNetwork",
    "DQNAgent",
    "DDPGActor",
    "DDPGCritic",
    "DDPGAgent",
    # Utilities
    "net_para_add",
    "net_para_scale",
    # Trainers
    "ReplayBuffer",
    "DQNTrainer",
    "DDPGTrainer",
    # Client and Server
    "FedRLClient",
    "client_fn_builder",
    "FedRLStrategy",
    "FedRLServer",
    "run_fedrl_server",
]

