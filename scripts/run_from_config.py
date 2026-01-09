"""
Unified entry point for running RL training from YAML configuration files.

This script provides a single interface for running both PPO and SAC training
across all supported environments (bandit2d, d4rl, minari, reacher).

Usage:
    python scripts/run_from_config.py configs/bandit2d/ppo.yaml
    python scripts/run_from_config.py configs/bandit2d/sac.yaml --algorithm sac
    python scripts/run_from_config.py configs/bandit2d/ppo.yaml --algorithm ppo --seeds 0,1,2
"""

import sys
import os

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# Import and run the unified runner
from fedguide.runner.run_from_config import main

if __name__ == "__main__":
    main()

