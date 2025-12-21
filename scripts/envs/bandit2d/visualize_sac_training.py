"""
Visualize training history for Centralized SAC baseline.

This script reads the training history pickle file and plots:
- Return (eval/return) vs rounds
- Loss (total, actor, critic) vs rounds
- Q Value vs rounds
- Policy log probability distribution (heatmap) for selected rounds
"""

import argparse
import pickle
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union


def load_history(pickle_path: str) -> Dict:
    """
    Load training history from pickle file.
    
    Args:
        pickle_path: Path to training_history.pkl file
    
    Returns:
        Dictionary containing history and args
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"History file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def extract_metrics(history: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Extract metrics from history for plotting.
    
    Args:
        history: List of metric dictionaries from training
    
    Returns:
        Dictionary with arrays of metrics
    """
    rounds = []
    total_loss = []
    actor_loss = []
    critic_loss = []
    q_value = []
    eval_return = []
    
    for metrics in history:
        rounds.append(metrics.get('round', len(rounds) + 1))
        total_loss.append(metrics.get('loss', 0.0))
        actor_loss.append(metrics.get('train/loss/actor', 0.0))
        critic_loss.append(metrics.get('train/loss/critic', 0.0))
        q_value.append(metrics.get('train/q_value', 0.0))
        
        # Eval return might not be present in all rounds
        if 'eval/return' in metrics:
            eval_return.append(metrics['eval/return'])
        else:
            eval_return.append(np.nan)  # Use NaN instead of None for easier handling
    
    return {
        'rounds': np.array(rounds),
        'total_loss': np.array(total_loss),
        'actor_loss': np.array(actor_loss),
        'critic_loss': np.array(critic_loss),
        'q_value': np.array(q_value),
        'eval_return': np.array(eval_return),
    }


def plot_training_curves(metrics: Dict[str, np.ndarray], output_path: Optional[str] = None):
    """
    Plot training curves for return and loss.
    
    Args:
        metrics: Dictionary containing extracted metrics
        output_path: Optional path to save the figure
    """
    rounds = metrics['rounds']
    
    # Create figure with subplots (2x2 for training curves)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SAC Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Eval Return
    ax1 = axes[0, 0]
    eval_return = metrics['eval_return']
    valid_return = ~np.isnan(eval_return)
    if np.any(valid_return):
        ax1.plot(rounds[valid_return], eval_return[valid_return], 
                'b-', linewidth=2, label='Eval Return', marker='o', markersize=4)
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Return', fontsize=12)
        ax1.set_title('Evaluation Return vs Rounds', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No evaluation data available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Evaluation Return vs Rounds', fontsize=13, fontweight='bold')
    
    # Plot 2: Total Loss
    ax2 = axes[0, 1]
    ax2.plot(rounds, metrics['total_loss'], 'r-', linewidth=2, label='Total Loss', marker='o', markersize=4)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Total Loss vs Rounds', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Actor and Critic Loss
    ax3 = axes[1, 0]
    ax3.plot(rounds, metrics['actor_loss'], 'g-', linewidth=2, label='Actor Loss', marker='o', markersize=4)
    ax3.plot(rounds, metrics['critic_loss'], 'orange', linewidth=2, label='Critic Loss', marker='s', markersize=4)
    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Actor & Critic Loss vs Rounds', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Q Value
    ax4 = axes[1, 1]
    ax4.plot(rounds, metrics['q_value'], 'purple', linewidth=2, label='Q Value', marker='^', markersize=4)
    ax4.set_xlabel('Round', fontsize=12)
    ax4.set_ylabel('Q Value', fontsize=12)
    ax4.set_title('Q Value vs Rounds', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def plot_policy_logprob_distribution(history: List[Dict], round_nums: Optional[List[int]] = None, 
                                     output_path: Optional[str] = None):
    """
    Plot policy density distribution for selected rounds.
    Uses normalized probability density (0-1 range) for visualization, which is more intuitive
    and compatible with FedAvg visualizations.
    
    Args:
        history: List of metric dictionaries from training
        round_nums: List of round numbers to visualize (if None, use first, middle, last)
        output_path: Optional path to save the figure
    """
    # Find rounds with policy data (density or logprob)
    rounds_with_policy = []
    for i, metrics in enumerate(history):
        if 'policy/density_grid' in metrics or 'policy/logprob_grid' in metrics:
            rounds_with_policy.append((i, metrics.get('round', i + 1)))
    
    if not rounds_with_policy:
        print("Warning: No policy density or logprob data found in history. Skipping policy visualization.")
        return
    
    # Select rounds to visualize
    if round_nums is None:
        # Use first, middle, and last rounds with policy data
        if len(rounds_with_policy) == 1:
            selected_rounds = rounds_with_policy
        elif len(rounds_with_policy) == 2:
            selected_rounds = [rounds_with_policy[0], rounds_with_policy[-1]]
        else:
            mid_idx = len(rounds_with_policy) // 2
            selected_rounds = [
                rounds_with_policy[0],
                rounds_with_policy[mid_idx],
                rounds_with_policy[-1]
            ]
    else:
        # Use specified rounds
        selected_rounds = [(i, r) for i, r in rounds_with_policy if r in round_nums]
        if not selected_rounds:
            print(f"Warning: None of the specified rounds {round_nums} have policy data.")
            print(f"Available rounds with policy data: {[r for _, r in rounds_with_policy]}")
            return
    
    n_plots = len(selected_rounds)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle('Policy Density Distribution', fontsize=16, fontweight='bold')
    
    for plot_idx, (hist_idx, round_num) in enumerate(selected_rounds):
        metrics = history[hist_idx]
        
        # Extract density grid data (preferred for visualization)
        # Fall back to logprob if density not available
        if 'policy/density_grid' in metrics:
            density_grid = np.array(metrics['policy/density_grid'])
            use_density = True
        elif 'policy/logprob_grid' in metrics:
            # Convert logprob to density if needed (backward compatibility)
            logprob_grid = np.array(metrics['policy/logprob_grid'])
            logprob_flat = logprob_grid.ravel()
            logprob_flat = logprob_flat - logprob_flat.max()  # Normalize
            density_grid = np.exp(logprob_flat).reshape(logprob_grid.shape)
            use_density = True
        else:
            print(f"Warning: No policy density or logprob data found for round {round_num}. Skipping this round.")
            # Skip this round but continue with others
            if len(selected_rounds) == 1:
                # If only one round and it's missing data, show empty plot
                ax = axes[0] if n_plots > 1 else axes
                ax.text(0.5, 0.5, f'No policy data\nfor round {round_num}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Round {round_num}', fontsize=12, fontweight='bold')
            continue
        
        # Get grid coordinates if available
        if 'policy/grid_X' in metrics and 'policy/grid_Y' in metrics:
            X = np.array(metrics['policy/grid_X'])
            Y = np.array(metrics['policy/grid_Y'])
            extent = [X.min(), X.max(), Y.min(), Y.max()]
        else:
            # Infer from density grid shape
            grid_size = density_grid.shape[0]
            extent = [-1.5, 1.5, -1.5, 1.5]
            X, Y = np.meshgrid(np.linspace(-1.5, 1.5, grid_size),
                              np.linspace(-1.5, 1.5, grid_size))
        
        ax = axes[plot_idx]
        
        # Plot heatmap using density (0-1 range, more intuitive)
        im = ax.imshow(density_grid, origin='lower', extent=extent, 
                      cmap='viridis', aspect='equal', interpolation='bilinear',
                      vmin=0, vmax=1)
        
        # Determine axis labels based on action dimensions
        action_dims = metrics.get('policy/action_dims', [0, 1])
        action_dim = metrics.get('policy/action_dim', 2)
        
        if action_dim > 2:
            ax.set_xlabel(f'Action Dim {action_dims[0]}', fontsize=11)
            ax.set_ylabel(f'Action Dim {action_dims[1]}', fontsize=11)
            title = f'Round {round_num}\n(Marginal Distribution)'
        else:
            ax.set_xlabel('X Position', fontsize=11)
            ax.set_ylabel('Y Position', fontsize=11)
            title = f'Round {round_num}'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar_label = 'Policy Density' if use_density else 'Log Probability'
        plt.colorbar(im, ax=ax, label=cbar_label)
        
        # Add peak locations if available (for Bandit2D)
        try:
            from fedguide.envs.bandit2d import Bandit2D
            env = Bandit2D(K=4, sigma=0.2)
            peaks = env.get_peak_locations()
            ax.scatter(peaks[:, 0], peaks[:, 1], c='red', marker='*', 
                      s=150, edgecolors='white', linewidths=1, zorder=10, label='Peaks')
            ax.legend(loc='upper right', fontsize=9)
        except Exception:
            pass
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Policy density distribution figure saved to {output_path}")
    else:
        plt.show()


def plot_combined(metrics: Dict[str, np.ndarray], output_path: Optional[str] = None):
    """
    Plot a combined figure with return and loss on the same plot (dual y-axis).
    
    Args:
        metrics: Dictionary containing extracted metrics
        output_path: Optional path to save the figure
    """
    rounds = metrics['rounds']
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot loss on left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    line1 = ax1.plot(rounds, metrics['total_loss'], color=color, linewidth=2, 
                     label='Total Loss', marker='o', markersize=4)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot return on right y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Return', color=color, fontsize=12)
    eval_return = metrics['eval_return']
    valid_return = ~np.isnan(eval_return)
    if np.any(valid_return):
        line2 = ax2.plot(rounds[valid_return], eval_return[valid_return], 
                        color=color, linewidth=2, label='Eval Return', 
                        marker='s', markersize=4)
        ax2.tick_params(axis='y', labelcolor=color)
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
    else:
        ax1.legend(loc='upper left')
    
    plt.title('Training Progress: Loss and Return vs Rounds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined figure saved to {output_path}")
    else:
        plt.show()


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def normalize_seed(seed: Union[int, List[int]]) -> List[int]:
    """Convert seed to list format."""
    if isinstance(seed, int):
        return [seed]
    elif isinstance(seed, list):
        return seed
    else:
        raise ValueError(f"seed must be int or list of ints, got {type(seed)}")


def visualize_single_seed(history_path: str, output_dir: Optional[str], 
                         combined: bool, plot_logprob: bool, 
                         logprob_rounds: Optional[List[int]], seed: Optional[int] = None):
    """
    Visualize training history for a single seed.
    
    Args:
        history_path: Path to training_history.pkl file
        output_dir: Directory to save plots
        combined: Whether to create combined plot
        plot_logprob: Whether to plot logprob distribution
        logprob_rounds: Specific rounds to plot logprob
        seed: Seed number (for logging purposes)
    """
    seed_str = f" (seed {seed})" if seed is not None else ""
    print(f"\n{'='*80}")
    print(f"Visualizing{seed_str}")
    print(f"{'='*80}")
    
    # Load history
    print(f"Loading history from {history_path}...")
    if not os.path.exists(history_path):
        print(f"Warning: History file not found: {history_path}")
        return
    
    data = load_history(history_path)
    history = data.get('history', [])
    
    if not history:
        print(f"Error: No history found in the file!")
        return
    
    print(f"Loaded {len(history)} rounds of training data")
    
    # Extract metrics
    metrics = extract_metrics(history)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Total rounds: {len(metrics['rounds'])}")
    print(f"  Final total loss: {metrics['total_loss'][-1]:.4f}")
    print(f"  Final actor loss: {metrics['actor_loss'][-1]:.4f}")
    print(f"  Final critic loss: {metrics['critic_loss'][-1]:.4f}")
    print(f"  Final Q value: {metrics['q_value'][-1]:.4f}")
    
    eval_return = metrics['eval_return']
    valid_return = ~np.isnan(eval_return)
    if np.any(valid_return):
        valid_returns = eval_return[valid_return]
        print(f"  Final eval return: {valid_returns[-1]:.4f}")
        print(f"  Max eval return: {np.max(valid_returns):.4f}")
        print(f"  Mean eval return: {np.mean(valid_returns):.4f}")
    
    # Create output directory if needed
    output_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "training_curves.png")
    
    # Plot training curves
    print("\nGenerating plots...")
    plot_training_curves(metrics, output_path=output_path)
    
    # Plot combined figure if requested
    if combined:
        combined_path = None
        if output_dir:
            combined_path = os.path.join(output_dir, "training_combined.png")
        plot_combined(metrics, output_path=combined_path)
    
    # Plot policy density distribution if requested
    if plot_logprob:
        density_path = None
        if output_dir:
            density_path = os.path.join(output_dir, "policy_density_distribution.png")
        plot_policy_logprob_distribution(history, round_nums=logprob_rounds, 
                                       output_path=density_path)
    
    print(f"\nVisualization complete{seed_str}!")


def main():
    parser = argparse.ArgumentParser(description="Visualize SAC training history")
    
    # Config file or direct path
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file (if provided, will visualize all seeds)")
    parser.add_argument("--history_path", type=str, 
                       default="./results/sac_centralized_bandit2d/training_history.pkl",
                       help="Path to training_history.pkl file (used if --config not provided)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save plots (if None, display only)")
    parser.add_argument("--combined", action="store_true",
                       help="Also create a combined plot with dual y-axis")
    parser.add_argument("--plot_logprob", action="store_true", default=True,
                       help="Plot policy log probability distribution")
    parser.add_argument("--logprob_rounds", type=int, nargs='+', default=None,
                       help="Specific round numbers to plot logprob (if None, auto-select)")
    
    # Seed selection (when using config)
    parser.add_argument("--seeds", type=str, default=None,
                       help="Override seeds from config (comma-separated list, e.g., '0,1,2')")
    
    args = parser.parse_args()
    
    # If config file is provided, use it
    if args.config:
        # Load config
        if not os.path.exists(args.config):
            # Try relative to configs directory
            alt_path = os.path.join("configs", args.config)
            if os.path.exists(alt_path):
                config_path = alt_path
            else:
                raise FileNotFoundError(f"Configuration file not found: {args.config}")
        else:
            config_path = args.config
        
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
        
        # Determine seeds
        if args.seeds:
            # Override with command-line seeds
            seed_list = [int(s.strip()) for s in args.seeds.split(',')]
        else:
            # Use seeds from config
            seed_list = normalize_seed(config.get("seed", 42))
        
        # Get metrics directory
        base_metrics_dir = config.get("metrics_dir", "./metrics/bandit2d/sac")
        
        # Get output directory (if not specified, use metrics_dir)
        if args.output_dir:
            base_output_dir = args.output_dir
        else:
            base_output_dir = base_metrics_dir.replace("metrics", "plots")
        
        print(f"\nConfiguration loaded:")
        print(f"  Environment type: {config.get('env_type', 'unknown')}")
        print(f"  Seeds to visualize: {seed_list}")
        print(f"  Total visualizations: {len(seed_list)}")
        
        # Visualize each seed
        for seed in seed_list:
            seed_metrics_dir = os.path.join(base_metrics_dir, f"seed_{seed}")
            seed_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
            history_path = os.path.join(seed_metrics_dir, "training_history.pkl")
            
            visualize_single_seed(
                history_path=history_path,
                output_dir=seed_output_dir if args.output_dir else None,
                combined=args.combined,
                plot_logprob=args.plot_logprob,
                logprob_rounds=args.logprob_rounds,
                seed=seed
            )
        
        print(f"\n{'='*80}")
        print("All visualizations complete!")
        print(f"{'='*80}\n")
    
    else:
        # Use direct history path (single seed)
        visualize_single_seed(
            history_path=args.history_path,
            output_dir=args.output_dir,
            combined=args.combined,
            plot_logprob=args.plot_logprob,
            logprob_rounds=args.logprob_rounds,
            seed=None
        )


if __name__ == "__main__":
    main()

