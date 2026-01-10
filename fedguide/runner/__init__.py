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
from .runner import run_training, main as unified_main
from .factories import get_registry

# Keep old imports for backward compatibility
try:
    from .registry import register_env, register_runner
    __all__ = ['run_from_config', 'run_training', 'unified_main', 'get_registry', 'register_env', 'register_runner']
except ImportError:
    __all__ = ['run_from_config', 'run_training', 'unified_main', 'get_registry']

