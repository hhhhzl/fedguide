"""
Run FedKL federated training for 2D Bandit environment.

This module trains a federated FedKL agent on Bandit2D using Flower simulation.
"""

import argparse
import os
import sys

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from fedguide.baselines.fedKL.server import run_fedkl_server
from fedguide.baselines.fedKL.client import client_fn_builder
from fedguide.runner.bandit2d._common import (
    create_metrics_collector,
    make_evaluate_fn,
    save_training_results
)


def main():
    """Main entry point for FedKL federated training on Bandit2D."""
    parser = argparse.ArgumentParser(description="FedKL federated training for Bandit2D")
    
    # Federated learning args
    parser.add_argument("--num_clients", type=int, default=4,
                       help="Number of federated clients")
    parser.add_argument("--rounds", type=int, default=60,
                       help="Number of federated learning rounds")
    parser.add_argument("--cpus_per_client", type=int, default=2,
                       help="Number of CPUs per client")
    
    # Training args
    parser.add_argument("--n_steps", type=int, default=200,
                       help="Number of steps per round")
    parser.add_argument("--lambda_global", type=float, default=15.0,
                       help="Global KL penalty coefficient")
    parser.add_argument("--lambda_local", type=float, default=0.05,
                       help="Local KL penalty coefficient")
    parser.add_argument("--update_epochs", type=int, default=10,
                       help="Number of update epochs per round")
    parser.add_argument("--minibatch_size", type=int, default=64,
                       help="Minibatch size for updates")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                       help="PPO clipping epsilon")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5,
                       help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="Maximum gradient norm for clipping")
    
    # Network args
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension for networks")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    
    # Logging args
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for logging")
    
    # Metrics collection args
    parser.add_argument("--metrics_dir", type=str, default="./metrics/bandit2d_fedkl",
                       help="Directory to save metrics for visualization")
    parser.add_argument("--collect_metrics_every", type=int, default=1,
                       help="Collect metrics every N rounds (0 to disable)")
    
    # Environment args (for compatibility with run_from_config, but not used)
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Data directory (not used for federated learning)")
    parser.add_argument("--K", type=int, default=None,
                       help="Number of peaks (not used, hardcoded in client)")
    parser.add_argument("--sigma", type=float, default=None,
                       help="Standard deviation (not used, hardcoded in client)")
    
    # Output and device args (for compatibility with run_from_config)
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (not used for federated learning)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (passed to client but may be overridden)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (passed to client)")
    
    args = parser.parse_args()
    
    # Create metrics collector
    metrics_collector = create_metrics_collector(
        metrics_dir=args.metrics_dir,
        collect_every=args.collect_metrics_every
    )
    
    # Create evaluate function for metrics collection
    evaluate_fn = make_evaluate_fn(
        collect_every=args.collect_metrics_every,
        collector=metrics_collector,
        algorithm="fedkl"
    )
    
    # Build client function
    client_fn = client_fn_builder(
        env_id="Bandit2D",
        algo="fedkl",
        n_steps=args.n_steps,
        lambda_global=args.lambda_global,
        lambda_local=args.lambda_local,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name or "bandit2d-fedkl",
        metrics_collector=metrics_collector,
        num_clients=args.num_clients,
    )
    
    print(f"Starting FedKL training:")
    print(f"  Environment: Bandit2D")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Steps per round: {args.n_steps}")
    
    # Run FedKL server
    history = run_fedkl_server(
        client_fn=client_fn,
        num_rounds=args.rounds,
        num_clients=args.num_clients,
        fraction_fit=1.0,
        min_fit_clients=args.num_clients,
        use_simulation=True,
        evaluate_fn=evaluate_fn,
    )
    
    print("\nTraining completed!")
    
    # Save results
    save_training_results(history, metrics_collector, args.metrics_dir, "fedkl")
    
    return history


if __name__ == "__main__":
    main()

