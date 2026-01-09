"""
Runner module for training RL agents across different environments.

This module provides a unified interface for running training experiments
across different environments (bandit2d, d4rl, minari, reacher) and algorithms 
(PPO, SAC, FedGuide, FedKL).

The module uses a registry system for automatic discovery of available runners.
New environments and algorithms can be added by simply creating the appropriate
runner file and registering it in the environment's __init__.py.
"""

from .run_from_config import main as run_from_config
from .registry import get_registry, register_env, register_runner

__all__ = ['run_from_config', 'get_registry', 'register_env', 'register_runner']

