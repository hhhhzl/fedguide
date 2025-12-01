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


def generate_bandit2d_datasets(K=4, n_clients=4, samples_per_client=1000, 
                                sigma=0.2, local_radius=0.3, seed=42):
    """
    Generate offline datasets for 2D bandit with client heterogeneity.
    
    Args:
        K: Number of peaks
        n_clients: Number of clients
        samples_per_client: Number of samples per client
        sigma: Standard deviation for reward function
        local_radius: Radius for local data sampling around each peak
        seed: Random seed
    
    Returns:
        datasets: List of TrajectoryDataset objects
        mu: Array of peak locations
    """
    np.random.seed(seed)
    
    # Place K peaks on unit circle
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    mu = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
    
    datasets = []
    
    for client_id in range(n_clients):
        # Each client sees data near its corresponding peak
        mu_i = mu[client_id % K]  # Cycle if n_clients > K
        
        # Sample actions near μ_i
        observations = []
        actions = []
        
        for _ in range(samples_per_client):
            # Sample from Gaussian centered at μ_i
            action = np.random.multivariate_normal(
                mu_i, 
                cov=local_radius**2 * np.eye(2)
            )
            # Clip to valid range
            action = np.clip(action, -1.5, 1.5)
            
            # For bandit: state = action
            observations.append(action)
            actions.append(action)
        
        # Create dataset
        dataset = TrajectoryDataset(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32)
        )
        datasets.append(dataset)
        
        print(f"Client {client_id}: {len(dataset)} samples near μ_{client_id % K} = {mu_i}")
    
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
        seed=args.seed
    )
    
    # Save metadata
    os.makedirs(args.output_dir, exist_ok=True)
    metadata = {
        "K": args.K,
        "n_clients": args.n_clients,
        "samples_per_client": args.samples_per_client,
        "mu": mu.tolist(),
        "sigma": args.sigma,
        "local_radius": args.local_radius,
        "seed": args.seed
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGenerated {len(datasets)} client datasets")
    print(f"Metadata saved to {args.output_dir}/metadata.json")

