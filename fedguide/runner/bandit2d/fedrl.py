"""
Run FedRL federated training for 2D Bandit environment.

This module trains a federated FedRL agent (DQN or DDPG) on Bandit2D using Flower simulation.
"""

import argparse
import os
import sys

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from fedguide.baselines.fedrl.server import run_fedrl_server
from fedguide.baselines.fedrl.client import client_fn_builder
from fedguide.runner.bandit2d._common import (
    create_metrics_collector,
    make_evaluate_fn,
    save_training_results
)


def main():
    """Main entry point for FedRL federated training on Bandit2D."""
    parser = argparse.ArgumentParser(description="FedRL federated training for Bandit2D")
    
    # Algorithm selection
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ddpg"],
                       help="Algorithm: 'dqn' (discrete) or 'ddpg' (continuous)")
    
    # Federated learning args
    parser.add_argument("--num_clients", type=int, default=4,
                       help="Number of federated clients")
    parser.add_argument("--rounds", type=int, default=60,
                       help="Number of federated learning rounds")
    parser.add_argument("--cpus_per_client", type=int, default=2,
                       help="Number of CPUs per client")
    
    # Training args
    parser.add_argument("--merge_interval", type=int, default=16,
                       help="Number of steps per round (E in FedRL paper)")
    parser.add_argument("--gamma", type=float, default=0.9,
                       help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for updates")
    parser.add_argument("--replay_size", type=int, default=1000,
                       help="Replay buffer capacity")
    parser.add_argument("--replay_initial", type=int, default=None,
                       help="Minimum buffer size before training (default: 2*batch_size for DQN, 1000 for DDPG)")
    parser.add_argument("--eval_episodes", type=int, default=1,
                       help="Number of episodes for evaluation")
    
    # DQN-specific args
    parser.add_argument("--epsilon", type=float, default=1.0,
                       help="Initial epsilon for epsilon-greedy (DQN)")
    parser.add_argument("--epsilon_decay", type=float, default=0.99,
                       help="Epsilon decay rate (DQN)")
    parser.add_argument("--epsilon_min", type=float, default=0.01,
                       help="Minimum epsilon (DQN)")
    parser.add_argument("--sync_interval", type=int, default=10,
                       help="Steps between target network sync (DQN)")
    
    # DDPG-specific args
    parser.add_argument("--tau", type=float, default=0.001,
                       help="Soft update coefficient (DDPG)")
    parser.add_argument("--threshold", type=float, default=2.0,
                       help="Action clipping threshold (DDPG)")
    parser.add_argument("--aggregate_critic", action="store_true",
                       help="Aggregate critic parameters (DDPG, default: false)")
    parser.add_argument("--add_noise", action="store_true", default=True,
                       help="Add exploration noise (DDPG)")
    
    # Network args
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension for networks")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    
    # Logging args
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for logging")
    
    # Metrics collection args
    parser.add_argument("--metrics_dir", type=str, default="./metrics/bandit2d_fedrl",
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
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (passed to client)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (passed to client)")
    
    args = parser.parse_args()
    
    # Set default replay_initial based on algorithm
    if args.replay_initial is None:
        if args.algo == "dqn":
            args.replay_initial = 2 * args.batch_size
        else:  # ddpg
            args.replay_initial = 1000
    
    # Create metrics collector
    metrics_collector = create_metrics_collector(
        metrics_dir=args.metrics_dir,
        collect_every=args.collect_metrics_every
    )
    
    # Create evaluate function for metrics collection
    evaluate_fn = make_evaluate_fn(
        collect_every=args.collect_metrics_every,
        collector=metrics_collector,
        algorithm="fedrl"
    )
    
    # Build client function
    client_fn = client_fn_builder(
        env_id="Bandit2D",
        algo=args.algo,
        gamma=args.gamma,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        # DQN-specific
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        sync_interval=args.sync_interval,
        # DDPG-specific
        tau=args.tau,
        threshold=args.threshold,
        aggregate_critic=args.aggregate_critic,
        # Training
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        replay_initial=args.replay_initial,
        merge_interval=args.merge_interval,
        eval_episodes=args.eval_episodes,
        add_noise=args.add_noise,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name or f"bandit2d-fedrl-{args.algo}",
        metrics_collector=metrics_collector,
        num_clients=args.num_clients,
        device=args.device,
    )
    
    print(f"Starting FedRL-{args.algo.upper()} training:")
    print(f"  Environment: Bandit2D")
    print(f"  Algorithm: {args.algo.upper()}")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Steps per round (merge_interval): {args.merge_interval}")
    
    # Run FedRL server
    history = run_fedrl_server(
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
    save_training_results(history, metrics_collector, args.metrics_dir, "fedrl")
    
    return history


if __name__ == "__main__":
    main()

