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

# Import registry
from fedguide.runner.registry import get_registry
from fedguide.runner.auto_discover import auto_discover_runners


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


def _get_runner_module_path(env_type: str, algorithm: str) -> str:
    """
    Get the module path for a specific environment and algorithm runner using registry.
    
    Args:
        env_type: Environment type (bandit2d, d4rl, minari, reacher_hetero)
        algorithm: Algorithm name (ppo, sac, fedguide, fedkl, etc.)
    
    Returns:
        Module path (e.g., 'fedguide.runner.bandit2d.ppo')
    """
    registry = get_registry()
    return registry.get_runner_module(env_type, algorithm)


def run_training(config: Dict[str, Any], seed: int, algorithm: str = None) -> bool:
    """
    Run training for a given seed using the appropriate runner module.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        algorithm: Algorithm name (ppo, sac). If None, inferred from config path or defaults to ppo.
    
    Returns:
        True if training succeeded, False otherwise
    """
    # Determine algorithm
    if algorithm is None:
        # Try to infer from config path or default to ppo
        algorithm = config.get('algorithm', 'ppo')
    
    # Determine environment type
    env_type = config.get('env_type', 'bandit2d')
    
    # Try to import and call the runner module directly
    try:
        runner_module_path = _get_runner_module_path(env_type, algorithm)
        runner_module = __import__(runner_module_path, fromlist=['main'])
        main_func = runner_module.main
        
        # Temporarily modify sys.argv to pass config as --config argument
        import sys
        original_argv = sys.argv.copy()
        
        # Create a temporary config file or pass config via environment
        # For now, use subprocess which is more reliable for complex argument parsing
        sys.argv = original_argv
        return _run_training_subprocess(config, seed, env_type, algorithm)
        
    except (ImportError, AttributeError) as e:
        # Fallback to subprocess call
        print(f"Warning: Could not import runner module, using subprocess fallback: {e}")
        return _run_training_subprocess(config, seed, env_type, algorithm)


def _config_to_args_dict(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Convert config dictionary to arguments dictionary for runner."""
    args_dict = config.copy()
    
    # Override seed
    args_dict['seed'] = seed
    
    # Add seed subfolder to output directories
    for key in ['output_dir', 'metrics_dir', 'render_save_dir']:
        if key in args_dict and args_dict[key]:
            base_dir = args_dict[key]
            args_dict[key] = os.path.join(base_dir, f"seed_{seed}")
    
    return args_dict


def _run_training_subprocess(config: Dict[str, Any], seed: int, env_type: str, algorithm: str) -> bool:
    """
    Run training via subprocess using the registry system.
    
    This method uses the registry to discover runner modules and build command arguments.
    """
    registry = get_registry()
    
    # Get module path from registry
    try:
        module_path = registry.get_runner_module(env_type, algorithm)
    except (ValueError, KeyError) as e:
        raise ValueError(f"No runner module found for env_type={env_type}, algorithm={algorithm}. "
                        f"Supported envs: {registry.get_supported_envs()}, "
                        f"Supported algorithms: {registry.get_supported_algorithms()}") from e
    
    # Build command using -m flag
    cmd = [sys.executable, '-m', module_path]
    
    # Build arguments using registry builders
    args_dict = _config_to_args_dict(config, seed)
    
    # Build environment-specific arguments
    registry.build_env_args(env_type, config, cmd)
    
    # Build algorithm-specific arguments
    registry.build_algo_args(algorithm, config, cmd)
    
    # Add common output arguments
    cmd.extend(["--output_dir", args_dict.get("output_dir", f"./model/policy/{env_type}/{algorithm}")])
    cmd.extend(["--metrics_dir", args_dict.get("metrics_dir", f"./metrics/{env_type}/{algorithm}")])
    if "save_every" in config:
        cmd.extend(["--save_every", str(config.get("save_every", 10))])
    
    # Add device and seed
    cmd.extend(["--device", config.get("device", "auto")])
    cmd.extend(["--seed", str(seed)])
    
    # Add rendering arguments if specified
    if config.get("render_eval", False):
        cmd.append("--render_eval")
        if "render_mode" in config:
            cmd.extend(["--render_mode", config.get("render_mode", "video")])
        if "render_save_dir" in args_dict:
            cmd.extend(["--render_save_dir", args_dict.get("render_save_dir")])
        if "render_every_n_rounds" in config:
            cmd.extend(["--render_every_n_rounds", str(config.get("render_every_n_rounds", 10))])
        if "render_episodes" in config:
            cmd.extend(["--render_episodes", str(config.get("render_episodes", 1))])
    
    # Add logprob collection arguments if specified (for bandit2d and reacher)
    if env_type in ['bandit2d', 'reacher_hetero']:
        if config.get("collect_logprob", False):
            cmd.append("--collect_logprob")
        if "logprob_grid_size" in config:
            cmd.extend(["--logprob_grid_size", str(config.get("logprob_grid_size", 200))])
        if "logprob_bounds" in config:
            bounds = config.get("logprob_bounds", [-1.5, 1.5])
            cmd.extend(["--logprob_bounds", str(bounds[0]), str(bounds[1])])
    
    print(f"\n{'='*80}")
    print(f"Running {algorithm.upper()} training for {env_type} with seed={seed}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, check=False, cwd=_project_root)
    return result.returncode == 0


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
    
    # Get registry
    registry = get_registry()
    
    # Auto-discover runners for the specific environment (lazy loading)
    # This avoids importing environments with optional dependencies (e.g., mujoco)
    auto_discover_runners(env_types={env_type})
    
    print(f"\nConfiguration loaded:")
    print(f"  Environment type: {env_type}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Seeds to run: {seed_list}")
    print(f"  Total runs: {len(seed_list)}")
    
    # Show registry status
    supported_envs = registry.get_supported_envs()
    supported_algorithms = registry.get_supported_algorithms()
    if supported_envs or supported_algorithms:
        print(f"  Supported environments: {supported_envs}")
        print(f"  Supported algorithms: {supported_algorithms}")
    
    # Verify runner is registered (non-blocking warning)
    if not registry.is_runner_registered(env_type, algorithm):
        print(f"\nNote: Runner for env_type={env_type}, algorithm={algorithm} is not explicitly registered.")
        print(f"  Attempting to auto-discover: {registry.get_runner_module(env_type, algorithm)}")
    
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

