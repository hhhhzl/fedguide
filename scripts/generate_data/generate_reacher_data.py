"""
Generate metadata.json for Reacher environment with client heterogeneity.
Similar to data/bandit2d/metadata.json structure.
"""
import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt
from fedguide.envs.reacher import generate_reacher_heterogeneity


def visualize_reacher_clients(client_configs, title="Reacher Client Distribution"):
    """
    Visualize the distribution of Reacher clients.
    
    Args:
        client_configs: List of client configuration dictionaries
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(client_configs)))
    
    # Plot 1: Goal region distribution (qpos_high_low)
    ax1 = axes[0]
    for i, config in enumerate(client_configs):
        qpos = config["qpos_high_low"]
        # Calculate center of the region
        x_center = (qpos[0][0] + qpos[0][1]) / 2
        y_center = (qpos[1][0] + qpos[1][1]) / 2
        x_width = qpos[0][1] - qpos[0][0]
        y_width = qpos[1][1] - qpos[1][0]
        
        # Draw rectangle for goal region
        rect = plt.Rectangle(
            (qpos[0][0], qpos[1][0]), x_width, y_width,
            facecolor=colors[i], alpha=0.3, edgecolor=colors[i], linewidth=2
        )
        ax1.add_patch(rect)
        
        # Mark center
        ax1.scatter(x_center, y_center, c=colors[i], s=100, 
                   marker='x', linewidths=2, label=f"Client {i}")
    
    ax1.set_xlabel("X Position", fontsize=12)
    ax1.set_ylabel("Y Position", fontsize=12)
    ax1.set_title("Goal Region Distribution", fontsize=14)
    ax1.set_xlim(-0.25, 0.25)
    ax1.set_ylim(-0.25, 0.25)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", "box")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Action noise and reward scale
    ax2 = axes[1]
    action_noises = [config["action_noise"] for config in client_configs]
    reward_scales = [config["reward_scale"] for config in client_configs]
    
    # Scatter plot: action_noise magnitude vs reward_scale
    noise_magnitudes = [np.linalg.norm(noise) for noise in action_noises]
    ax2.scatter(noise_magnitudes, reward_scales, c=colors[:len(client_configs)], 
               s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
    
    # Annotate each point with client ID
    for i, (noise_mag, rew_scale) in enumerate(zip(noise_magnitudes, reward_scales)):
        ax2.annotate(f"C{i}", (noise_mag, rew_scale), 
                    fontsize=8, ha='center', va='center')
    
    ax2.set_xlabel("Action Noise Magnitude", fontsize=12)
    ax2.set_ylabel("Reward Scale", fontsize=12)
    ax2.set_title("Dynamics & Reward Heterogeneity", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def generate_reacher_metadata(
    n_clients=4,
    hetero_type="both",
    seed=42,
    variants=("medium-v2", "expert-v2", "random-v2"),
    output_dir="data/reacher"
):
    """
    Generate metadata.json for Reacher environment with client heterogeneity.
    
    Args:
        n_clients: Number of clients
        hetero_type: Type of heterogeneity ['iid', 'init-state', 'dynamics', 'reward', 'both']
        seed: Random seed (for reproducibility, though dynamics/reward now use client_id)
        variants: Tuple of variant names to cycle through
        output_dir: Output directory for metadata.json
    
    Returns:
        metadata: Dictionary containing all client configurations
        client_configs: List of client configuration dictionaries
    """
    # Note: seed is kept for compatibility, but dynamics/reward now use client_id
    # for deterministic generation
    
    # Generate configurations for each client
    client_configs = []
    for client_id in range(n_clients):
        qpos_high_low, action_noise, reward_scale, angle_noise = generate_reacher_heterogeneity(
            client_id, hetero_type
        )
        
        client_config = {
            "client_id": client_id,
            "variant": variants[client_id % len(variants)],
            "qpos_high_low": qpos_high_low,
            "action_noise": action_noise.tolist(),
            "reward_scale": float(reward_scale),
            "angle_noise": float(angle_noise)
        }
        client_configs.append(client_config)
    
    # Create metadata structure similar to bandit2d
    metadata = {
        "n_clients": n_clients,
        "hetero_type": hetero_type,
        "seed": seed,
        "variants": list(variants),
        "clients": client_configs
    }
    
    # Save metadata
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated metadata for {n_clients} clients")
    print(f"Metadata saved to {metadata_path}")
    
    return metadata, client_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate metadata.json for Reacher environment with heterogeneity"
    )
    parser.add_argument(
        "--n_clients", type=int, default=4,
        help="Number of clients"
    )
    parser.add_argument(
        "--hetero_type", type=str, default="both",
        choices=["iid", "init-state", "dynamics", "reward", "both"],
        help="Type of heterogeneity"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (for compatibility, dynamics/reward use client_id)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/reacher",
        help="Output directory for metadata.json"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize client distribution"
    )
    
    args = parser.parse_args()
    
    metadata, client_configs = generate_reacher_metadata(
        n_clients=args.n_clients,
        hetero_type=args.hetero_type,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    if args.visualize:
        visualize_reacher_clients(client_configs, 
                                 title=f"Reacher Client Distribution (hetero_type={args.hetero_type})")

