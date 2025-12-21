"""
Run SAC training from YAML configuration file.
Supports multiple random seeds - if seed is a list, runs training for each seed.
"""

import argparse
import os
import sys
import yaml
import subprocess
from pathlib import Path
from typing import List, Union, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
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


def run_bandit2d_training(config: Dict[str, Any], seed: int) -> None:
    """Run Bandit2D SAC training with given seed."""
    script_path = os.path.join(
        os.path.dirname(__file__),
        "envs", "bandit2d", "run_sac_centralized_bandit2d.py"
    )
    
    # Add seed subfolder to output directories
    base_output_dir = config.get("output_dir", "./model/policy/bandit2d/sac")
    base_metrics_dir = config.get("metrics_dir", "./metrics/bandit2d/sac")
    output_dir = os.path.join(base_output_dir, f"seed_{seed}")
    metrics_dir = os.path.join(base_metrics_dir, f"seed_{seed}")
    
    cmd = [
        sys.executable, script_path,
        "--num_clients", str(config.get("num_clients", 4)),
        "--data_dir", config.get("data_dir", "data/bandit2d"),
        "--rounds", str(config.get("rounds", 100)),
        "--update_steps", str(config.get("update_steps", 1000)),
        "--batch_size", str(config.get("batch_size", 256)),
        "--hidden_dim", str(config.get("hidden_dim", 256)),
        "--lr", str(config.get("lr", 3e-4)),
        "--gamma", str(config.get("gamma", 0.99)),
        "--tau", str(config.get("tau", 0.005)),
        "--alpha", str(config.get("alpha", 0.2)),
        "--K", str(config.get("K", 4)),
        "--sigma", str(config.get("sigma", 0.2)),
        "--eval_episodes", str(config.get("eval_episodes", 50)),
        "--output_dir", output_dir,
        "--metrics_dir", metrics_dir,
        "--save_every", str(config.get("save_every", 10)),
        "--device", config.get("device", "auto"),
        "--seed", str(seed),
    ]
    
    print(f"\n{'='*80}")
    print(f"Running Bandit2D SAC training with seed={seed}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Training failed for seed={seed} with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ Training completed successfully for seed={seed}")
        return True


def run_d4rl_training(config: Dict[str, Any], seed: int) -> None:
    """Run D4RL SAC training with given seed."""
    script_path = os.path.join(
        os.path.dirname(__file__),
        "envs", "run_sac_centralized_d4rl.py"
    )
    
    # Add seed subfolder to output directories
    base_output_dir = config.get("output_dir", "./model/policy/d4rl/sac")
    base_metrics_dir = config.get("metrics_dir", "./metrics/d4rl/sac")
    output_dir = os.path.join(base_output_dir, f"seed_{seed}")
    metrics_dir = os.path.join(base_metrics_dir, f"seed_{seed}")
    
    cmd = [
        sys.executable, script_path,
        "--env_name", config.get("env_name", ""),
        "--num_clients", str(config.get("num_clients", 1)),
        "--rounds", str(config.get("rounds", 100)),
        "--update_steps", str(config.get("update_steps", 1000)),
        "--batch_size", str(config.get("batch_size", 256)),
        "--hidden_dim", str(config.get("hidden_dim", 256)),
        "--lr", str(config.get("lr", 3e-4)),
        "--gamma", str(config.get("gamma", 0.99)),
        "--tau", str(config.get("tau", 0.005)),
        "--alpha", str(config.get("alpha", 0.2)),
        "--action_std", str(config.get("action_std", 0.1)),
        "--eval_episodes", str(config.get("eval_episodes", 10)),
        "--output_dir", output_dir,
        "--metrics_dir", metrics_dir,
        "--save_every", str(config.get("save_every", 10)),
        "--device", config.get("device", "auto"),
        "--seed", str(seed),
    ]
    
    print(f"\n{'='*80}")
    print(f"Running D4RL SAC training: {config.get('env_name')} with seed={seed}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Training failed for seed={seed} with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ Training completed successfully for seed={seed}")
        return True


def run_minari_training(config: Dict[str, Any], seed: int) -> None:
    """Run Minari SAC training with given seed."""
    script_path = os.path.join(
        os.path.dirname(__file__),
        "envs", "run_sac_centralized_minari.py"
    )
    
    # Add seed subfolder to output directories
    base_output_dir = config.get("output_dir", "./model/policy/minari/sac")
    base_metrics_dir = config.get("metrics_dir", "./metrics/minari/sac")
    output_dir = os.path.join(base_output_dir, f"seed_{seed}")
    metrics_dir = os.path.join(base_metrics_dir, f"seed_{seed}")
    
    cmd = [
        sys.executable, script_path,
        "--dataset_id", config.get("dataset_id", ""),
        "--num_clients", str(config.get("num_clients", 1)),
        "--rounds", str(config.get("rounds", 100)),
        "--update_steps", str(config.get("update_steps", 1000)),
        "--batch_size", str(config.get("batch_size", 256)),
        "--hidden_dim", str(config.get("hidden_dim", 256)),
        "--lr", str(config.get("lr", 3e-4)),
        "--gamma", str(config.get("gamma", 0.99)),
        "--tau", str(config.get("tau", 0.005)),
        "--alpha", str(config.get("alpha", 0.2)),
        "--action_std", str(config.get("action_std", 0.1)),
        "--eval_episodes", str(config.get("eval_episodes", 10)),
        "--output_dir", output_dir,
        "--metrics_dir", metrics_dir,
        "--save_every", str(config.get("save_every", 10)),
        "--device", config.get("device", "auto"),
        "--seed", str(seed),
    ]
    
    # Add optional env_name if specified
    if config.get("env_name"):
        cmd.extend(["--env_name", config.get("env_name")])
    
    # Add download flag if specified
    if config.get("download", True):
        cmd.append("--download")
    
    print(f"\n{'='*80}")
    print(f"Running Minari SAC training: {config.get('dataset_id')} with seed={seed}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Training failed for seed={seed} with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ Training completed successfully for seed={seed}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Run SAC training from YAML configuration file with multi-seed support"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file (e.g., configs/bandit2d/sac.yaml)"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override seeds from config (comma-separated list, e.g., '0,1,2,3,4,5')"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config_path
    if not os.path.exists(config_path):
        # Try relative to configs directory
        alt_path = os.path.join("configs", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {args.config_path}")
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Determine seeds
    if args.seeds:
        # Override with command-line seeds
        seed_list = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        # Use seeds from config
        seed_list = normalize_seed(config.get("seed", 42))
    
    print(f"\nConfiguration loaded:")
    print(f"  Environment type: {config.get('env_type', 'unknown')}")
    if config.get("env_type") == "d4rl":
        print(f"  Environment name: {config.get('env_name', 'unknown')}")
    elif config.get("env_type") == "minari":
        print(f"  Dataset ID: {config.get('dataset_id', 'unknown')}")
    print(f"  Seeds to run: {seed_list}")
    print(f"  Total runs: {len(seed_list)}")
    
    # Determine environment type
    env_type = config.get("env_type", "bandit2d")
    
    # Run training for each seed
    results = []
    for i, seed in enumerate(seed_list, 1):
        print(f"\n{'#'*80}")
        print(f"# Run {i}/{len(seed_list)}: Seed {seed}")
        print(f"{'#'*80}")
        
        if env_type == "bandit2d":
            success = run_bandit2d_training(config, seed)
        elif env_type == "d4rl":
            success = run_d4rl_training(config, seed)
        elif env_type == "minari":
            success = run_minari_training(config, seed)
        else:
            raise ValueError(f"Unknown environment type: {env_type}. Supported types: 'bandit2d', 'd4rl', 'minari'")
        
        results.append((seed, success))
    
    # Print summary
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}")
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    for seed, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  Seed {seed:3d}: {status}")
    
    print(f"\nTotal: {len(results)} runs, {successful} successful, {failed} failed")
    print(f"{'='*80}\n")
    
    # Exit with error if any run failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

