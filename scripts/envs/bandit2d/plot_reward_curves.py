"""
Plot reward curves for Bandit2D federated learning experiments.

Supports both FedGuide and FedKL baselines.
"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List


def extract_rewards_from_history(history, metric_key: str = "return") -> Dict[int, List[float]]:
    """
    Extract reward metrics from Flower history object.
    
    Args:
        history: Flower History object from start_simulation
        metric_key: Key to extract from metrics (e.g., "return", "eval/return")
    
    Returns:
        Dictionary mapping round number to list of client rewards
    """
    rewards_by_round: Dict[int, List[float]] = {}
    
    # History structure: history.metrics_distributed_fit or history.metrics_centralized_fit
    if hasattr(history, 'metrics_distributed_fit'):
        for round_num, metrics_list in history.metrics_distributed_fit.items():
            round_rewards = []
            for metrics in metrics_list:
                # Try different possible keys
                if metric_key in metrics:
                    round_rewards.append(float(metrics[metric_key]))
                elif f"train/{metric_key}" in metrics:
                    round_rewards.append(float(metrics[f"train/{metric_key}"]))
                elif f"eval/{metric_key}" in metrics:
                    round_rewards.append(float(metrics[f"eval/{metric_key}"]))
            if round_rewards:
                rewards_by_round[round_num] = round_rewards
    
    return rewards_by_round


def extract_rewards_from_metrics_file(metrics_path: str) -> Optional[Dict[int, float]]:
    """
    Extract rewards from Bandit2DMetricsCollector pickle file if available.
    
    Note: This may not contain reward info, but we check anyway.
    """
    try:
        from fedguide.utils.bandit2d_metrics import Bandit2DMetricsCollector
        collector = Bandit2DMetricsCollector.load(metrics_path)
        # Check if rewards are stored in metrics_history
        rewards = {}
        for i, round_metrics in enumerate(collector.metrics_history):
            if 'rewards' in round_metrics:
                rewards[i] = round_metrics['rewards']
        return rewards if rewards else None
    except Exception:
        return None

def summarize_curve(label: str, rounds, means, stds):
    """Print simple numeric summary for a reward curve."""
    if not rounds:
        print(f"[SUMMARY] {label}: no data")
        return

    rounds = np.array(rounds, dtype=np.float32)
    means = np.array(means, dtype=np.float32)
    stds = np.array(stds, dtype=np.float32)

    final_mean = means[-1]
    final_std = stds[-1]
    best_mean = means.max()

    # Simple discrete AUC normalized by number of rounds
    auc = float(means.mean())

    print(
        f"[SUMMARY] {label}: "
        f"final = {final_mean:.3f} ± {final_std:.3f}, "
        f"best = {best_mean:.3f}, "
        f"mean-over-rounds (AUC proxy) = {auc:.3f}"
    )

def plot_reward_curves(
    fedguide_history_path: Optional[str] = None,
    fedkl_history_path: Optional[str] = None,
    fedguide_metrics_path: Optional[str] = None,
    fedkl_metrics_path: Optional[str] = None,
    output_path: Optional[str] = None,
    metric_key: str = "return",
    show_std: bool = True,
    window_size: int = 5,  # Moving average window
):
    """
    Plot reward curves comparing FedGuide and FedKL.
    
    Args:
        fedguide_history_path: Path to FedGuide training history pickle file
        fedkl_history_path: Path to FedKL training history pickle file
        fedguide_metrics_path: Path to FedGuide metrics pickle file (alternative)
        fedkl_metrics_path: Path to FedKL metrics pickle file (alternative)
        output_path: Path to save figure
        metric_key: Metric key to plot ("return", "eval/return", etc.)
        show_std: Whether to show standard deviation bands
        window_size: Moving average window size
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract rewards from history files
    fedguide_rewards = None
    fedkl_rewards = None
    
    if fedguide_history_path and Path(fedguide_history_path).exists():
        with open(fedguide_history_path, 'rb') as f:
            history = pickle.load(f)
        fedguide_rewards = extract_rewards_from_history(history, metric_key)
        print(f"Loaded FedGuide history from {fedguide_history_path}")
    
    if fedkl_history_path and Path(fedkl_history_path).exists():
        with open(fedkl_history_path, 'rb') as f:
            history = pickle.load(f)
        fedkl_rewards = extract_rewards_from_history(history, metric_key)
        print(f"Loaded FedKL history from {fedkl_history_path}")
    
    # If no history, try metrics files
    if fedguide_rewards is None and fedguide_metrics_path:
        fedguide_rewards = extract_rewards_from_metrics_file(fedguide_metrics_path)
    
    if fedkl_rewards is None and fedkl_metrics_path:
        fedkl_rewards = extract_rewards_from_metrics_file(fedkl_metrics_path)
    
    # Process and plot FedGuide
    if fedguide_rewards:
        rounds = sorted(fedguide_rewards.keys())
        means = []
        stds = []
        for rnd in rounds:
            rewards = fedguide_rewards[rnd]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))

        summarize_curve("FedGuide", rounds, means, stds)
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Moving average
        if window_size > 1 and len(means) >= window_size:
            means_ma = np.convolve(means, np.ones(window_size)/window_size, mode='valid')
            stds_ma = np.convolve(stds, np.ones(window_size)/window_size, mode='valid')
            rounds_ma = rounds[window_size-1:]
        else:
            means_ma = means
            stds_ma = stds
            rounds_ma = rounds
        
        ax.plot(rounds_ma, means_ma, label='FedGuide', color='tab:blue', linewidth=2)
        if show_std:
            ax.fill_between(rounds_ma, means_ma - stds_ma, means_ma + stds_ma, 
                          alpha=0.2, color='tab:blue')
        print(f"FedGuide: {len(rounds)} rounds, mean reward: {np.mean(means):.4f}")
    
    # Process and plot FedKL
    if fedkl_rewards:
        rounds = sorted(fedkl_rewards.keys())
        means = []
        stds = []
        for rnd in rounds:
            rewards = fedkl_rewards[rnd]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))

        summarize_curve("FedKL", rounds, means, stds)
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Moving average
        if window_size > 1 and len(means) >= window_size:
            means_ma = np.convolve(means, np.ones(window_size)/window_size, mode='valid')
            stds_ma = np.convolve(stds, np.ones(window_size)/window_size, mode='valid')
            rounds_ma = rounds[window_size-1:]
        else:
            means_ma = means
            stds_ma = stds
            rounds_ma = rounds
        
        ax.plot(rounds_ma, means_ma, label='FedKL', color='tab:orange', linewidth=2)
        if show_std:
            ax.fill_between(rounds_ma, means_ma - stds_ma, means_ma + stds_ma, 
                          alpha=0.2, color='tab:orange')
        print(f"FedKL: {len(rounds)} rounds, mean reward: {np.mean(means):.4f}")
    
    if fedguide_rewards is None and fedkl_rewards is None:
        print("Warning: No reward data found. Please provide history files or metrics files.")
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel(f'Average {metric_key.replace("/", " ").title()}', fontsize=12)
    ax.set_title('Bandit2D Federated Learning Reward Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Reward curve saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot reward curves for Bandit2D experiments")
    parser.add_argument("--fedguide_history", type=str, default=None,
                       help="Path to FedGuide training history pickle file")
    parser.add_argument("--fedkl_history", type=str, default=None,
                       help="Path to FedKL training history pickle file")
    parser.add_argument("--fedguide_metrics", type=str, default=None,
                       help="Path to FedGuide metrics pickle file (alternative)")
    parser.add_argument("--fedkl_metrics", type=str, default=None,
                       help="Path to FedKL metrics pickle file (alternative)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save figure (if None, display)")
    parser.add_argument("--metric_key", type=str, default="return",
                       help="Metric key to plot (default: 'return')")
    parser.add_argument("--no_std", action="store_true",
                       help="Don't show standard deviation bands")
    parser.add_argument("--window_size", type=int, default=5,
                       help="Moving average window size (default: 5)")
    
    args = parser.parse_args()
    
    # Auto-detect history files if not provided
    if args.fedguide_history is None:
        default_path = "./metrics/bandit2d_fedguide/training_history.pkl"
        if Path(default_path).exists():
            args.fedguide_history = default_path
            print(f"Auto-detected FedGuide history: {default_path}")
    
    if args.fedkl_history is None:
        default_path = "./metrics/bandit2d_fedkl/training_history.pkl"
        if Path(default_path).exists():
            args.fedkl_history = default_path
            print(f"Auto-detected FedKL history: {default_path}")
    
    plot_reward_curves(
        fedguide_history_path=args.fedguide_history,
        fedkl_history_path=args.fedkl_history,
        fedguide_metrics_path=args.fedguide_metrics,
        fedkl_metrics_path=args.fedkl_metrics,
        output_path=args.output_path,
        metric_key=args.metric_key,
        show_std=not args.no_std,
        window_size=args.window_size,
    )

