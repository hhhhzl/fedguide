"""
Unified runner for training RL agents from YAML configuration files.

This module supports multiple algorithms (PPO, SAC, FedGuide, FedKL) and environments 
(bandit2d, d4rl, minari, reacher). Supports multiple random seeds - if seed is a list, 
runs training for each seed.

Uses the registry system for automatic discovery of available runners.
"""

import argparse
import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import List, Union, Dict, Any

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

# Import unified runner (old registry import kept for backward compatibility if needed)
try:
    from fedguide.runner.registry import get_registry
except ImportError:
    # If old registry isn't available, that's fine - we use unified runner now
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        # Try relative to configs directory
        alt_path = os.path.join(_project_root, "configs", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def normalize_seed(seed: Union[int, List[int]]) -> List[int]:
    """Convert seed to list format."""
    if isinstance(seed, int):
        return [seed]
    elif isinstance(seed, list):
        return seed
    else:
        raise ValueError(f"seed must be int or list of ints, got {type(seed)}")


def run_training(config: Dict[str, Any], seed: int, algorithm: str = None) -> bool:
    """
    Run training for a given seed using the unified runner.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        algorithm: Algorithm name. If None, inferred from config or config path.
    
    Returns:
        True if training succeeded, False otherwise
    """
    # Import unified runner
    from fedguide.runner.unified_runner import run_training as unified_run_training
    
    # Determine algorithm
    if algorithm is None:
        algorithm = config.get('algorithm', 'ppo')
    
    # Override seed in config
    config['seed'] = seed
    
    # Add seed subfolder to output directories
    for key in ['output_dir', 'metrics_dir', 'render_save_dir']:
        if key in config and config[key]:
            base_dir = config[key]
            config[key] = os.path.join(base_dir, f"seed_{seed}")
    
    try:
        # Use unified runner directly
        history = unified_run_training(config)
        return True
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False




def main():
    """Main entry point for unified config runner."""
    parser = argparse.ArgumentParser(
        description="Run RL training from YAML configuration file with multi-seed support"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file (e.g., configs/bandit2d/ppo.yaml)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Algorithm to use (ppo, sac, fedguide, fedkl, etc.). If not specified, inferred from config path."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override seeds from config (comma-separated list, e.g., '0,1,2,3,4')"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config_path
    if not os.path.exists(config_path):
        # Try relative to configs directory
        alt_path = os.path.join(_project_root, "configs", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {args.config_path}")
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Determine algorithm from config path if not specified
    algorithm = args.algorithm
    if algorithm is None:
        # Try to infer from config path
        config_path_lower = config_path.lower()
        if 'fedguide' in config_path_lower:
            algorithm = 'fedguide'
        elif 'fedkl' in config_path_lower:
            algorithm = 'fedkl'
        elif 'ppo' in config_path_lower:
            algorithm = 'ppo'
        elif 'sac' in config_path_lower:
            algorithm = 'sac'
        else:
            # Try to get from config
            algorithm = config.get('algorithm', 'ppo')  # Default
    
    # Determine seeds
    if args.seeds:
        # Override with command-line seeds
        seed_list = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        # Use seeds from config
        seed_list = normalize_seed(config.get("seed", 42))
    
    # Get environment type
    env_type = config.get("env_type", "bandit2d")
    
    print(f"\nConfiguration loaded:")
    print(f"  Environment type: {env_type}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Seeds to run: {seed_list}")
    print(f"  Total runs: {len(seed_list)}")
    print(f"  Using unified runner")
    
    # Run training for each seed
    results = []
    for i, seed in enumerate(seed_list, 1):
        print(f"\n{'#'*80}")
        print(f"# Run {i}/{len(seed_list)}: Seed {seed}")
        print(f"{'#'*80}")
        
        success = run_training(config, seed, algorithm)
        results.append((seed, success))
    
    # Print summary
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}")
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    for seed, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  Seed {seed:3d}: {status}")
    
    print(f"\nTotal: {len(results)} runs, {successful} successful, {failed} failed")
    print(f"{'='*80}\n")
    
    # Exit with error if any run failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

