"""
Generate metadata.json for Reacher environment with client heterogeneity.
Similar to data/bandit2d/metadata.json structure.
"""
import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt


def visualize_reacher_clients(client_configs, title="Reacher Client Distribution"):
    """
    Visualize the distribution of Reacher clients.
    
    Args:
        client_configs: List of client configuration dictionaries
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Use continuous colormap for better visualization with many clients
    n_clients = len(client_configs)
    colormap = plt.cm.viridis  # Use viridis colormap for continuous colors
    colors = [colormap(i / max(1, n_clients - 1)) for i in range(n_clients)]
    
    # Plot 1: Goal region distribution (qpos_high_low)
    ax1 = axes[0]
    
    # Draw grid lines to show 8x8 structure (if applicable)
    # Check if we have a grid-like structure (8x8 = 64 clients)
    if n_clients == 64:
        # Draw 8x8 grid lines
        # Grid: X from -0.2 to 0.2 (step 0.05), Y from 0.2 to -0.2 (step -0.05)
        grid_size = 8
        cell_size = 0.05
        # Draw vertical lines (X direction)
        for i in range(grid_size + 1):
            x_pos = -0.2 + i * cell_size
            ax1.axvline(x_pos, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        # Draw horizontal lines (Y direction)
        for i in range(grid_size + 1):
            y_pos = 0.2 - i * cell_size  # Y goes from 0.2 down to -0.2
            ax1.axhline(y_pos, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    
    for i, config in enumerate(client_configs):
        qpos = config["qpos_high_low"]
        # Calculate center of the region
        x_center = (qpos[0][0] + qpos[0][1]) / 2
        y_center = (qpos[1][0] + qpos[1][1]) / 2
        x_width = qpos[0][1] - qpos[0][0]
        y_width = qpos[1][1] - qpos[1][0]
        
        # Draw rectangle for goal region with reduced alpha for better visibility
        rect = plt.Rectangle(
            (qpos[0][0], qpos[1][0]), x_width, y_width,
            facecolor=colors[i], alpha=0.4, edgecolor=colors[i], linewidth=1.5
        )
        ax1.add_patch(rect)
        
        # Mark center with larger marker for better visibility
        ax1.scatter(x_center, y_center, c=[colors[i]], s=80, 
                   marker='x', linewidths=2, zorder=5)
    
    ax1.set_xlabel("X Position", fontsize=12)
    ax1.set_ylabel("Y Position", fontsize=12)
    ax1.set_title("Goal Region Distribution", fontsize=14)
    ax1.set_xlim(-0.25, 0.25)
    ax1.set_ylim(-0.25, 0.25)
    ax1.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax1.set_aspect("equal", "box")
    
    # Add colorbar for client ID mapping
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=n_clients-1))
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=ax1, label='Client ID', shrink=0.8)
    
    # Plot 2: Action noise and reward scale
    ax2 = axes[1]
    action_noises = [config["action_noise"] for config in client_configs]
    reward_scales = [config["reward_scale"] for config in client_configs]
    
    # Scatter plot: action_noise magnitude vs reward_scale
    noise_magnitudes = [np.linalg.norm(noise) for noise in action_noises]
    
    # Use client_id for color mapping to match left plot
    scatter = ax2.scatter(noise_magnitudes, reward_scales, 
                         c=range(n_clients), cmap=colormap,
                         s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
    
    # Annotate each point with client ID (only if not too many clients)
    if n_clients <= 20:
        for i, (noise_mag, rew_scale) in enumerate(zip(noise_magnitudes, reward_scales)):
            ax2.annotate(f"C{i}", (noise_mag, rew_scale), 
                        fontsize=8, ha='center', va='center')
    
    ax2.set_xlabel("Action Noise Magnitude", fontsize=12)
    ax2.set_ylabel("Reward Scale", fontsize=12)
    ax2.set_title("Dynamics & Reward Heterogeneity", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for client ID mapping (matching left plot)
    cbar2 = plt.colorbar(scatter, ax=ax2, label='Client ID', shrink=0.8)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def load_reacher_metadata(metadata_path):
    """
    Load existing metadata.json file.
    
    Args:
        metadata_path: Path to metadata.json file
    
    Returns:
        metadata: Dictionary containing all metadata
        client_configs: List of client configuration dictionaries
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    client_configs = metadata.get("clients", [])
    return metadata, client_configs


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
    from fedguide.envs.reacher import generate_reacher_heterogeneity
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
        "--metadata_path", type=str, default=None,
        help="Path to existing metadata.json file (if provided, only visualize, don't generate)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize client distribution"
    )
    
    args = parser.parse_args()
    
    # If metadata_path is provided, load and visualize only
    if args.metadata_path:
        print(f"Loading metadata from {args.metadata_path}")
        metadata, client_configs = load_reacher_metadata(args.metadata_path)
        print(f"Loaded metadata for {len(client_configs)} clients")
        print(f"Hetero type: {metadata.get('hetero_type', 'unknown')}")
        
        title = f"Reacher Client Distribution (hetero_type={metadata.get('hetero_type', 'unknown')}, n_clients={len(client_configs)})"
        visualize_reacher_clients(client_configs, title=title)
    else:
        # Generate new metadata
        metadata, client_configs = generate_reacher_metadata(
            n_clients=args.n_clients,
            hetero_type=args.hetero_type,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
        if args.visualize:
            visualize_reacher_clients(client_configs, 
                                     title=f"Reacher Client Distribution (hetero_type={args.hetero_type})")

