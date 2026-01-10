"""
Entry point for FedMomentum federated training on Bandit2D.

This script uses the unified runner system.
"""

import sys
import os
import argparse

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from fedguide.runner.unified_runner import run_training, load_config


def main():
    """Main entry point for FedMomentum federated training on Bandit2D."""
    parser = argparse.ArgumentParser(description="FedMomentum federated training for Bandit2D")
    
    # Config file (primary method)
    parser.add_argument("--config", type=str, default="configs/bandit2d/fedkl.yaml",  # FedMomentum uses similar config
                       help="Path to YAML configuration file")
    
    # Command-line overrides (optional)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device (overrides config)")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds (overrides config)")
    parser.add_argument("--num_clients", type=int, default=None, help="Number of clients (overrides config)")
    
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        alt_path = os.path.join(_project_root, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Set algorithm and environment type
    config['algorithm'] = 'fedmomentum'
    config['env_type'] = 'bandit2d'
    
    # Override with command-line args if provided
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['device'] = args.device
    if args.rounds is not None:
        config['rounds'] = args.rounds
    if args.num_clients is not None:
        config['num_clients'] = args.num_clients
    
    # Run training
    run_training(config, args)


if __name__ == "__main__":
    main()
