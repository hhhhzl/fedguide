"""
Auto-discover and register all available runners.

This module provides lazy discovery of runners by importing environment modules
only when needed. Import errors are caught to allow environments with optional
dependencies (e.g., mujoco) to be skipped if dependencies are not available.
"""

import os
from pathlib import Path
from typing import List, Set

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def auto_discover_runners(env_types: Set[str] = None) -> None:
    """
    Automatically discover and register available runners.
    
    This function imports environment modules in fedguide/runner/,
    which triggers their auto-registration in their __init__.py files.
    
    Args:
        env_types: Optional set of environment types to discover.
                  If None, discovers all available environments.
                  Import errors are caught and logged but don't fail.
    """
    runner_dir = Path(__file__).parent
    
    for env_dir in runner_dir.iterdir():
        if not env_dir.is_dir() or env_dir.name.startswith('_') or env_dir.name == '__pycache__':
            continue
        
        env_type = env_dir.name
        
        # If specific env_types requested, skip others
        if env_types is not None and env_type not in env_types:
            continue
        
        # Try to import the module, which will trigger registration
        try:
            module_name = f"fedguide.runner.{env_type}"
            __import__(module_name)
        except (ImportError, RuntimeError, ModuleNotFoundError) as e:
            # Silently skip if module cannot be imported (e.g., missing optional dependencies)
            # This allows bandit2d to run even if mujoco is not installed
            pass
        except Exception as e:
            # Catch any other errors during import (e.g., mujoco_py build errors)
            # This prevents optional dependencies from breaking the system
            pass


# Don't auto-discover on import - only discover when explicitly called
# This prevents unnecessary imports when only using specific environments

