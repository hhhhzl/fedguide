"""
Run FedGuide federated training for 2D Bandit environment.
"""
import argparse
import flwr as fl
from flwr.server import ServerConfig
from fedguide.fed.fedguide.server import FedGuideServer
from fedguide.fed.fedguide.client import client_fn_builder


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
    
    args = parser.parse_args()
    
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
    return history


if __name__ == "__main__":
    main()

