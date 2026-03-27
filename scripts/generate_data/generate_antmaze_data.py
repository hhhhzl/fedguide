"""
Generate metadata.json for AntMaze D4RL clients (heterogeneity), mirroring Reacher layout.
"""
import argparse
import json
import os

from fedguide.envs.antmaze_hetero import DEFAULT_ANTMAZE_VARIANTS, generate_antmaze_heterogeneity


def generate_antmaze_metadata(
    n_clients: int = 8,
    hetero_type: str = "both",
    seed: int = 42,
    reward_type: str = "dense",
    variants=DEFAULT_ANTMAZE_VARIANTS,
    output_dir: str = "data/antmaze",
):
    client_configs = []
    for client_id in range(n_clients):
        qpos_high_low, action_noise, reward_scale, angle_noise = generate_antmaze_heterogeneity(
            client_id, hetero_type
        )
        variant = variants[client_id % len(variants)]
        client_configs.append(
            {
                "client_id": client_id,
                "variant": variant,
                "qpos_high_low": qpos_high_low,
                "action_noise": action_noise.tolist(),
                "reward_scale": float(reward_scale),
                "angle_noise": float(angle_noise),
            }
        )

    metadata = {
        "env": "antmaze",
        "n_clients": n_clients,
        "hetero_type": hetero_type,
        "seed": seed,
        "reward_type": reward_type,
        "variants": list(variants),
        "clients": client_configs,
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated metadata for {n_clients} clients → {path}")
    return metadata, client_configs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate AntMaze metadata.json for federated heterogeneity")
    p.add_argument("--n_clients", type=int, default=8)
    p.add_argument(
        "--hetero_type",
        type=str,
        default="both",
        choices=["iid", "init-state", "dynamics", "reward", "both"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reward_type", type=str, default="dense", choices=["dense", "sparse"])
    p.add_argument("--output_dir", type=str, default="data/antmaze")
    args = p.parse_args()
    generate_antmaze_metadata(
        n_clients=args.n_clients,
        hetero_type=args.hetero_type,
        seed=args.seed,
        reward_type=args.reward_type,
        output_dir=args.output_dir,
    )
