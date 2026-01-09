"""
Reacher environment runners for PPO and SAC algorithms.

This module automatically registers available runners with the global registry.
"""

from fedguide.runner.registry import register_env, register_runner

# Register environment type (note: config uses 'reacher_hetero' but module is 'reacher')
register_env('reacher_hetero', 'reacher')

# Auto-register available runners
# Catch all exceptions to allow this module to be imported even if dependencies are missing
try:
    from . import ppo
    register_runner('reacher_hetero', 'ppo')
except (ImportError, RuntimeError, ModuleNotFoundError, Exception):
    pass

try:
    from . import sac
    register_runner('reacher_hetero', 'sac')
except (ImportError, RuntimeError, ModuleNotFoundError, Exception):
    pass

# Export main functions for direct usage
from .ppo import main as run_ppo
from .sac import main as run_sac

__all__ = ['run_ppo', 'run_sac']

