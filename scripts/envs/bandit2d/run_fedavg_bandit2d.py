"""
Entry point for FedAvg federated training on Bandit2D.
FedAvg = FedKL with lambda_global=0, lambda_local=0 (no KL penalty).
"""
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from fedguide.runner.runner import run_training, load_config


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FedAvg federated training for Bandit2D")
    parser.add_argument("--config", type=str, default="configs/bandit2d/fedavg.yaml",
                        help="Config file (use fedavg_hetero.yaml for heterogeneous peaks)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--num_clients", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        alt_path = os.path.join(_project_root, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = load_config(config_path)
    config['algorithm'] = 'fedkl'  # Use FedKL implementation with lambda=0
    config['env_type'] = 'bandit2d'

    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['device'] = args.device
    if args.rounds is not None:
        config['rounds'] = args.rounds
    if args.num_clients is not None:
        config['num_clients'] = args.num_clients

    run_training(config, args)


if __name__ == "__main__":
    main()


# For heterogeneous Bandit2D (4 distinct peaks), use:
#   --config configs/bandit2d/fedavg_hetero.yaml --rounds 20
# Then visualize round 1 for early divergence:
#   python scripts/envs/bandit2d/visualize_bandit2d.py --metrics_path ./metrics/bandit2d/fedavg_hetero/bandit2d_metrics.pkl --client_policies --client_policies_output plots/bandit2d/client_policies_fedavg_hetero.png --round_num 1
# Standalone (no federation) verification with 4 distinct peaks:
#   python scripts/envs/bandit2d/verify_hetero_env.py
