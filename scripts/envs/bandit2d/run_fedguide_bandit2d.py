"""
Run FedGuide federated training for 2D Bandit environment.
"""
import argparse
import flwr as fl
from flwr.server import ServerConfig
from fedguide.fed.fedguide.server import FedGuideServer
from fedguide.fed.fedguide.client import client_fn_builder
from fedguide.utils.bandit2d_metrics import Bandit2DMetricsCollector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--cpus_per_client", type=int, default=2)
    
    # Training args
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--lambda_local", type=float, default=0.25)
    parser.add_argument("--lambda_guide", type=float, default=0.2)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=64)
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    
    # Metrics collection
    parser.add_argument("--metrics_dir", type=str, default="./metrics/bandit2d_fedguide",
                       help="Directory to save metrics for visualization")
    parser.add_argument("--collect_metrics_every", type=int, default=10,
                       help="Collect metrics every N rounds (0 to disable)")
    
    args = parser.parse_args()
    
    # Create metrics collector
    metrics_collector = None
    if args.collect_metrics_every > 0:
        metrics_collector = Bandit2DMetricsCollector(
            save_dir=args.metrics_dir,
            grid_size=200,
            bounds=(-1.5, 1.5)
        )
        print(f"Metrics collection enabled: saving to {args.metrics_dir}")
        print(f"  Collecting metrics every {args.collect_metrics_every} rounds")
    
    # Build client function
    client_fn = client_fn_builder(
        env_id="Bandit2D",
        algo="ppo",
        aggregate_mode="policy",
        n_steps=args.n_steps,
        lambda_local=args.lambda_local,
        lambda_guide=args.lambda_guide,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name or "bandit2d-fedguide",
        metrics_collector=metrics_collector,
    )
    
    # Create strategy
    strategy = FedGuideServer(
        fraction_fit=1.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
    )
    
    config = ServerConfig(num_rounds=args.rounds)
    
    print(f"Starting FedGuide training:")
    print(f"  Environment: Bandit2D")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Steps per round: {args.n_steps}")
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        strategy=strategy,
        config=config,
        client_resources={"num_cpus": args.cpus_per_client},
    )
    
    print("\nTraining completed!")
    
    # Save metrics if collector was used
    if metrics_collector is not None:
        metrics_collector.save("bandit2d_metrics.pkl")
        print(f"\nMetrics saved to {args.metrics_dir}/bandit2d_metrics.pkl")
        print("  To visualize, run:")
        print(f"    python scripts/envs/bandit2d/visualize_bandit2d.py --metrics_path {args.metrics_dir}/bandit2d_metrics.pkl")
    
    return history


if __name__ == "__main__":
    main()

