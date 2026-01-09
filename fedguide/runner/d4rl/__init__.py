"""
D4RL environment runners for PPO and SAC algorithms.

This module automatically registers available runners with the global registry.
"""

from fedguide.runner.registry import register_env, register_runner

# Register environment type
register_env('d4rl', 'd4rl')

# Auto-register available runners
try:
    from . import ppo
    register_runner('d4rl', 'ppo')
except ImportError:
    pass

try:
    from . import sac
    register_runner('d4rl', 'sac')
except ImportError:
    pass

# Export main functions for direct usage
from .ppo import main as run_ppo
from .sac import main as run_sac

__all__ = ['run_ppo', 'run_sac']

