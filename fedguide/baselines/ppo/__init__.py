"""Centralized PPO baseline implementation.

This module implements a centralized PPO agent that learns from multiple clients' data
without federated aggregation. All client data is merged into a single replay buffer
for centralized training.
"""

from .agent import PPOAgent
from .trainer import CentralPPOTrainer

__all__ = ["PPOAgent", "CentralPPOTrainer"]

