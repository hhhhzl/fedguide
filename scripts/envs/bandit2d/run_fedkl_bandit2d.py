"""
Run FedKL baseline for 2D Bandit environment.
"""
import argparse
from fedguide.baselines.fedKL.server import run_fedkl_server
from fedguide.baselines.fedKL.client import client_fn_builder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--cpus_per_client", type=int, default=2)
    
    # Training args
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--lambda_global", type=float, default=0.1)
    parser.add_argument("--lambda_local", type=float, default=0.05)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    
    # Network args
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    
    args = parser.parse_args()
    
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
    )
    
    print("\nTraining completed!")
    return history


if __name__ == "__main__":
    main()

