"""
Bandit2D environment runners for PPO, SAC, FedGuide, and FedKL algorithms.

This module automatically registers available runners with the global registry.
"""

from fedguide.runner.registry import register_env, register_runner

# Register environment type
register_env('bandit2d', 'bandit2d')

# Auto-register available runners
try:
    from . import ppo
    register_runner('bandit2d', 'ppo')
except ImportError:
    pass

try:
    from . import sac
    register_runner('bandit2d', 'sac')
except ImportError:
    pass

try:
    from . import fedguide
    register_runner('bandit2d', 'fedguide')
except ImportError:
    pass

try:
    from . import fedkl
    register_runner('bandit2d', 'fedkl')
except ImportError:
    pass

# Export main functions for direct usage
from .ppo import main as run_ppo
from .sac import main as run_sac

try:
    from . import fedguide
    from .fedguide import main as run_fedguide
except ImportError:
    run_fedguide = None

try:
    from . import fedkl
    from .fedkl import main as run_fedkl
except ImportError:
    run_fedkl = None

__all__ = ['run_ppo', 'run_sac', 'run_fedguide', 'run_fedkl']

