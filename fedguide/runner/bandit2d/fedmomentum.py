"""
Run FedMomentum federated training for 2D Bandit environment.

This module trains a federated FedMomentum agent on Bandit2D using Flower simulation.
Supports both SVRPG and HAPG algorithms with momentum-based aggregation.
"""

import argparse
import os
import sys

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from fedguide.baselines.fedmomentum.server import run_fedmomentum_server
from fedguide.baselines.fedmomentum.client import client_fn_builder
from fedguide.runner.bandit2d._common import (
    create_metrics_collector,
    make_evaluate_fn,
    save_training_results
)


def main():
    """Main entry point for FedMomentum federated training on Bandit2D."""
    parser = argparse.ArgumentParser(description="FedMomentum federated training for Bandit2D")
    
    # Federated learning args
    parser.add_argument("--num_clients", type=int, default=4,
                       help="Number of federated clients")
    parser.add_argument("--rounds", type=int, default=60,
                       help="Number of federated learning rounds")
    parser.add_argument("--cpus_per_client", type=int, default=2,
                       help="Number of CPUs per client")
    
    # Algorithm selection
    parser.add_argument("--algorithm", type=str, default="fedmomentum",
                       help="Main algorithm name (should be 'fedmomentum')")
    parser.add_argument("--algorithm_type", type=str, default="svrpg",
                       choices=["svrpg", "hapg"],
                       help="Sub-algorithm type: 'svrpg' or 'hapg'")
    
    # Training args
    parser.add_argument("--n_steps", type=int, default=200,
                       help="Number of steps per round")
    parser.add_argument("--update_epochs", type=int, default=4,
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
    
    # SVRPG-specific args
    parser.add_argument("--reference_update_freq", type=int, default=5,
                       help="Update reference policy every N rounds (SVRPG)")
    parser.add_argument("--use_svrpg", action="store_true", default=True,
                       help="Use SVRPG variance reduction (default: True for svrpg algorithm)")
    
    # HAPG-specific args
    parser.add_argument("--hessian_alpha", type=float, default=0.1,
                       help="Hessian correction coefficient (HAPG)")
    parser.add_argument("--use_diagonal_approx", action="store_true", default=True,
                       help="Use diagonal approximation for Hessian (HAPG)")
    parser.add_argument("--fisher_update_freq", type=int, default=1,
                       help="Update Fisher matrix every N rounds (HAPG)")
    parser.add_argument("--use_fisher_info", action="store_true", default=True,
                       help="Use Fisher Information Matrix (HAPG)")
    
    # Momentum args
    parser.add_argument("--momentum_beta", type=float, default=0.9,
                       help="Momentum coefficient for server aggregation")
    parser.add_argument("--server_lr", type=float, default=0.001,
                       help="Server learning rate for parameter updates")
    
    # Logging args
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for logging")
    
    # Metrics collection args
    parser.add_argument("--metrics_dir", type=str, default="./metrics/bandit2d_fedmomentum",
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
    
    # Use algorithm_type for sub-algorithm selection
    algorithm_type = args.algorithm_type if hasattr(args, 'algorithm_type') else args.algorithm
    
    # Set algorithm-specific defaults
    if algorithm_type == "hapg":
        args.use_svrpg = False  # HAPG can optionally combine with SVRPG, but default is False
    else:
        algorithm_type = "svrpg"  # Default to svrpg
    
    # Create metrics collector
    metrics_collector = create_metrics_collector(
        metrics_dir=args.metrics_dir,
        collect_every=args.collect_metrics_every
    )
    
    # Store collector in global scope for client access
    global _metrics_collector_global
    _metrics_collector_global = metrics_collector
    
    # Create evaluate function for metrics collection
    evaluate_fn = make_evaluate_fn(
        collect_every=args.collect_metrics_every,
        collector=metrics_collector,
        algorithm="fedmomentum"
    )
    
    # Build client function
    client_fn = client_fn_builder(
        env_id="bandit2d",  # Use lowercase for consistency
        # Algorithm selection
        algorithm=algorithm_type,  # Use algorithm_type for sub-algorithm
            # Training parameters
            n_steps=args.n_steps,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            # Network architecture
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            # SVRPG-specific
            reference_update_freq=args.reference_update_freq,
            use_svrpg=args.use_svrpg if algorithm_type == "svrpg" else False,
            # HAPG-specific
            hessian_alpha=args.hessian_alpha,
            use_diagonal_approx=args.use_diagonal_approx,
            fisher_update_freq=args.fisher_update_freq,
            use_fisher_info=args.use_fisher_info,
            # Evaluation
            eval_episodes=1,
            # Logging
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            run_name=args.run_name or f"bandit2d-fedmomentum-{algorithm_type}",
            metrics_collector=metrics_collector,
            num_clients=args.num_clients,
            device=args.device,
        )
    
    print(f"Starting FedMomentum training:")
    print(f"  Algorithm Type: {algorithm_type.upper()}")
    print(f"  Environment: Bandit2D")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Steps per round: {args.n_steps}")
    print(f"  Momentum beta: {args.momentum_beta}")
    print(f"  Server LR: {args.server_lr}")
    if algorithm_type == "svrpg":
        print(f"  Reference update freq: {args.reference_update_freq}")
    elif algorithm_type == "hapg":
        print(f"  Hessian alpha: {args.hessian_alpha}")
        print(f"  Fisher update freq: {args.fisher_update_freq}")
    
    # Run FedMomentum server
    history = run_fedmomentum_server(
        client_fn=client_fn,
        num_rounds=args.rounds,
        num_clients=args.num_clients,
        fraction_fit=1.0,
        min_fit_clients=args.num_clients,
        use_simulation=True,
        evaluate_fn=evaluate_fn,
        momentum_beta=args.momentum_beta,
        server_lr=args.server_lr,
    )
    
    print("\nTraining completed!")
    
    # Save results
    save_training_results(history, metrics_collector, args.metrics_dir, "fedmomentum")
    
    return history


# Global variable for metrics collector (accessible in client_fn)
_metrics_collector_global = None


if __name__ == "__main__":
    main()

