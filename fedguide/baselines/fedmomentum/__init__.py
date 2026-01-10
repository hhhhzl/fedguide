"""
FedMomentum: Federated Reinforcement Learning with Momentum

This module implements FedSVRPG-M and FedHAPG-M algorithms from the paper:
"Momentum for the Win: Collaborative Federated Reinforcement Learning across Heterogeneous Environments"

The implementation includes:
- Server-side momentum aggregation
- Client-side policy gradient computation
- SVRPG (Stochastic Variance Reduced Policy Gradient) training
- HAPG (Hessian-Aware Policy Gradient) training
"""

from .server import FedMomentumStrategy, run_fedmomentum_server
from .client import FedMomentumClient, client_fn_builder
from .trainer import SVRPGTrainer, HAPGTrainer
from .agent import FedMomentumAgent

__all__ = [
    "FedMomentumStrategy",
    "run_fedmomentum_server",
    "FedMomentumClient",
    "client_fn_builder",
    "SVRPGTrainer",
    "HAPGTrainer",
    "FedMomentumAgent",
]

