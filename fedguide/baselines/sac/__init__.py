"""Centralized SAC baseline implementation.

This module implements a centralized SAC agent that learns from multiple clients' data
without federated aggregation. All client data is merged into a single replay buffer
for centralized training.
"""

from .agent import SACAgent
from .trainer import CentralSACTrainer

__all__ = ["SACAgent", "CentralSACTrainer"]

