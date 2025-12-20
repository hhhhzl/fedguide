"""
Generate offline datasets for 2D bandit with client heterogeneity.

Each client i only sees data near μ_i (within local_radius).
"""
import numpy as np
import torch
import os
import json
# Direct import from base to avoid mujoco dependencies
from fedguide.datasets.base import TrajectoryDataset

import matplotlib.pyplot as plt


def visualize_bandit2d_datasets(datasets, mu, title="Bandit2D Client Datasets"):
    plt.figure(figsize=(6, 6))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink", "tab:gray"]

    for i, dataset in enumerate(datasets):
        obs = dataset.obs
        plt.scatter(
            obs[:, 0], obs[:, 1],
            s=8,
            alpha=0.6,
            color=colors[i % len(colors)],
            label=f"Client {i}"
        )

    plt.scatter(mu[:, 0], mu[:, 1], c="black", s=80, marker="x", label="Peak μ_i")

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect("equal", "box")

    plt.title(title)
    plt.legend()
    plt.show()


def generate_bandit2d_datasets(K=4, n_clients=4, samples_per_client=1000,
                               sigma=0.2, local_radius=0.3, seed=42, overlap_factor=1.33):
    """
    Generate offline datasets for 2D bandit with client heterogeneity.
    
    Args:
        K: Number of peaks
        n_clients: Number of clients
        samples_per_client: Number of samples per client
        sigma: Standard deviation for reward function
        local_radius: Radius for local data sampling around each peak
        seed: Random seed
        overlap_factor: Factor to create overlap between adjacent sectors (default 1.33 = 30% overlap)
                        For 50% overlap, use 1.5
    
    Returns:
        datasets: List of TrajectoryDataset objects
        mu: Array of peak locations
    """
    np.random.seed(seed)

    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    mu = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])

    r_min = 1.0 - local_radius
    r_max = 1.0 + local_radius

    datasets = []

    for client_id in range(n_clients):
        angle_center = angles[client_id % K]
        # Use overlap_factor to create overlapping sectors
        # angle_span = 2π / K * overlap_factor (e.g., 1.33 for 30% overlap, 1.5 for 50% overlap)
        angle_span = 2 * np.pi / K * overlap_factor
        theta_min = angle_center - angle_span / 2.0
        theta_max = angle_center + angle_span / 2.0

        observations = []
        actions = []

        for _ in range(samples_per_client):
            u = np.random.rand()
            r = np.sqrt((r_max ** 2 - r_min ** 2) * u + r_min ** 2)
            theta = np.random.uniform(theta_min, theta_max)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            action = np.array([x, y])
            action = np.clip(action, -1.5, 1.5)
            observations.append(action)
            actions.append(action)

        dataset = TrajectoryDataset(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32)
        )
        datasets.append(dataset)
        print(f"Client {client_id}: {len(dataset)} samples on arc near angle {angle_center:.3f}")

    return datasets, mu


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=4, help="Number of peaks")
    parser.add_argument("--n_clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--samples_per_client", type=int, default=1000,
                        help="Number of samples per client")
    parser.add_argument("--sigma", type=float, default=0.2,
                        help="Standard deviation for reward function")
    parser.add_argument("--local_radius", type=float, default=0.3,
                        help="Radius for local data sampling")
    parser.add_argument("--overlap_factor", type=float, default=1.33,
                        help="Overlap factor for sectors (1.33 = 30%% overlap, 1.5 = 50%% overlap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data/bandit2d",
                        help="Output directory for metadata")

    args = parser.parse_args()

    datasets, mu = generate_bandit2d_datasets(
        K=args.K,
        n_clients=args.n_clients,
        samples_per_client=args.samples_per_client,
        sigma=args.sigma,
        local_radius=args.local_radius,
        seed=args.seed,
        overlap_factor=args.overlap_factor
    )
    # visualize_bandit2d_datasets(datasets, mu)

    # Save metadata
    os.makedirs(args.output_dir, exist_ok=True)
    metadata = {
        "K": args.K,
        "n_clients": args.n_clients,
        "samples_per_client": args.samples_per_client,
        "mu": mu.tolist(),
        "sigma": args.sigma,
        "local_radius": args.local_radius,
        "overlap_factor": args.overlap_factor,
        "seed": args.seed
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {len(datasets)} client datasets")
    print(f"Metadata saved to {args.output_dir}/metadata.json")
