"""
Visualize training history for Centralized SAC baseline.

This script reads the training history pickle file and plots:
- Return (eval/return) vs rounds
- Loss (total, actor, critic) vs rounds
"""

import argparse
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


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
    
    # Create figure with subplots
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


def main():
    parser = argparse.ArgumentParser(description="Visualize SAC training history")
    
    parser.add_argument("--history_path", type=str, 
                       default="./results/sac_centralized_bandit2d/training_history.pkl",
                       help="Path to training_history.pkl file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save plots (if None, display only)")
    parser.add_argument("--combined", action="store_true",
                       help="Also create a combined plot with dual y-axis")
    
    args = parser.parse_args()
    
    # Load history
    print(f"Loading history from {args.history_path}...")
    data = load_history(args.history_path)
    history = data.get('history', [])
    
    if not history:
        print("Error: No history found in the file!")
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
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "training_curves.png")
    
    # Plot training curves
    print("\nGenerating plots...")
    plot_training_curves(metrics, output_path=output_path)
    
    # Plot combined figure if requested
    if args.combined:
        combined_path = None
        if args.output_dir:
            combined_path = os.path.join(args.output_dir, "training_combined.png")
        plot_combined(metrics, output_path=combined_path)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

