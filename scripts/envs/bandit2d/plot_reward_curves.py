"""
Plot reward curves for Bandit2D federated learning experiments.

Supports FedGuide, FedKL, FedAvg, FMARL, FedRL, FedRep, FedMomentum, and
centralized PPO/SAC (dict-format training_history.pkl).
"""
import argparse
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple


def resolve_training_history(metrics_dir: str) -> Optional[str]:
    """
    Pick the training_history.pkl to plot under metrics_dir.

    If both ./training_history.pkl and seed_*/training_history.pkl exist, prefer
    the file with the newer mtime (matches analyze_returns.py heuristic) so a
    fresh single-seed run at the root is not shadowed by stale multi-seed dirs.
    """
    metrics_dir = os.path.abspath(metrics_dir)
    seed_files = sorted(glob.glob(os.path.join(metrics_dir, "seed_*", "training_history.pkl")))
    root_file = os.path.join(metrics_dir, "training_history.pkl")
    if seed_files and os.path.isfile(root_file):
        root_mtime = os.path.getmtime(root_file)
        seed_mtimes = [os.path.getmtime(f) for f in seed_files]
        if root_mtime > max(seed_mtimes):
            return root_file
        # Default: first seed file for consistent ordering (or could average — not implemented)
        return seed_files[0]
    if seed_files:
        return seed_files[0]
    if os.path.isfile(root_file):
        return root_file
    return None


def extract_rewards_from_centralized_blob(blob: Dict[str, Any], metric_key: str = "eval/return") -> Dict[int, List[float]]:
    """PPO/SAC training_history.pkl shape: {'history': [ {metrics}, ...] }."""
    rewards_by_round: Dict[int, List[float]] = {}
    if not isinstance(blob, dict) or "history" not in blob:
        return rewards_by_round

    candidates = [metric_key]
    if "/" not in metric_key:
        candidates.extend([f"eval/{metric_key}", f"train/{metric_key}"])

    for row in blob["history"]:
        if not isinstance(row, dict):
            continue
        rnd = int(row.get("round", len(rewards_by_round) + 1))
        val = None
        for k in candidates:
            if k in row and row[k] is not None:
                try:
                    val = float(row[k])
                    break
                except (TypeError, ValueError):
                    continue
        if val is not None and val == val:  # not nan
            rewards_by_round.setdefault(rnd, []).append(val)

    return rewards_by_round


def load_rewards_for_plot(path: Optional[str], metric_key: str) -> Dict[int, List[float]]:
    """Load Flower History or centralized dict from a single pickle path."""
    if not path or not Path(path).exists():
        return {}
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"  Skip load (error): {path}: {e}")
        return {}

    if isinstance(obj, dict) and "history" in obj:
        return extract_rewards_from_centralized_blob(obj, metric_key=metric_key)
    return extract_rewards_from_history(obj, metric_key)


def extract_rewards_from_history(history, metric_key: str = "eval/return") -> Dict[int, List[float]]:
    """
    Extract reward metrics from Flower history object.

    Flower stores: metrics_distributed_fit[metric_key] = [(round, value), ...].
    Each round typically has one aggregated value; we treat it as [value] for compatibility.
    """
    rewards_by_round: Dict[int, List[float]] = {}
    metrics = getattr(history, "metrics_distributed_fit", None) or getattr(
        history, "metrics_centralized_fit", {}
    )
    if not metrics:
        return rewards_by_round

    # Resolve metric key (try exact, then common variants)
    pairs = None
    candidates = [metric_key]
    if "/" not in metric_key:
        candidates.extend([f"eval/{metric_key}", f"train/{metric_key}"])
    for key in candidates:
        if key in metrics and metrics[key]:
            pairs = metrics[key]
            break
    if pairs is None:
        return rewards_by_round

    # Flower format: [(round, value), ...]
    for round_num, value in pairs:
        rnd = int(round_num)
        val = float(value)
        rewards_by_round.setdefault(rnd, []).append(val)

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

def _plot_curve(ax, rewards_by_round: Dict[int, List[float]], label: str, color: str,
                show_std: bool, window_size: int):
    """Helper to plot one algorithm's curve."""
    if not rewards_by_round:
        return
    rounds = sorted(rewards_by_round.keys())
    means = [np.mean(rewards_by_round[r]) for r in rounds]
    stds = [np.std(rewards_by_round[r]) for r in rounds]
    summarize_curve(label, rounds, means, stds)
    means = np.array(means)
    stds = np.array(stds)
    # Use MA only when there are strictly more points than the window; otherwise
    # len==window would collapse the entire curve to a single point (misleading).
    if window_size > 1 and len(means) > window_size:
        means_ma = np.convolve(means, np.ones(window_size) / window_size, mode='valid')
        stds_ma = np.convolve(stds, np.ones(window_size) / window_size, mode='valid')
        rounds_ma = rounds[window_size - 1:]
    else:
        means_ma, stds_ma, rounds_ma = means, stds, rounds
    ax.plot(rounds_ma, means_ma, label=label, color=color, linewidth=2)
    if show_std:
        ax.fill_between(rounds_ma, means_ma - stds_ma, means_ma + stds_ma, alpha=0.2, color=color)


def plot_reward_curves(
    fedguide_history_path: Optional[str] = None,
    fedkl_history_path: Optional[str] = None,
    fedavg_history_path: Optional[str] = None,
    fedguide_metrics_path: Optional[str] = None,
    fedkl_metrics_path: Optional[str] = None,
    fedavg_metrics_path: Optional[str] = None,
    output_path: Optional[str] = None,
    metric_key: str = "return",
    show_std: bool = True,
    window_size: int = 5,  # Moving average window
):
    """
    Plot reward curves comparing FedGuide, FedKL, and FedAvg.
    
    Args:
        fedguide_history_path: Path to FedGuide training history pickle file
        fedkl_history_path: Path to FedKL training history pickle file
        fedavg_history_path: Path to FedAvg training history pickle file
        fedguide_metrics_path: Path to FedGuide metrics pickle file (alternative)
        fedkl_metrics_path: Path to FedKL metrics pickle file (alternative)
        fedavg_metrics_path: Path to FedAvg metrics pickle file (alternative)
        output_path: Path to save figure
        metric_key: Metric key to plot ("return", "eval/return", etc.)
        show_std: Whether to show standard deviation bands
        window_size: Moving average window size
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract rewards from history files
    fedguide_rewards = None
    fedkl_rewards = None
    fedavg_rewards = None
    
    if fedguide_history_path and Path(fedguide_history_path).exists():
        fedguide_rewards = load_rewards_for_plot(fedguide_history_path, metric_key)
        print(f"Loaded FedGuide history from {fedguide_history_path} ({len(fedguide_rewards)} rounds with data)")
    
    if fedkl_history_path and Path(fedkl_history_path).exists():
        fedkl_rewards = load_rewards_for_plot(fedkl_history_path, metric_key)
        print(f"Loaded FedKL history from {fedkl_history_path} ({len(fedkl_rewards)} rounds with data)")
    
    if fedavg_history_path and Path(fedavg_history_path).exists():
        fedavg_rewards = load_rewards_for_plot(fedavg_history_path, metric_key)
        print(f"Loaded FedAvg history from {fedavg_history_path} ({len(fedavg_rewards)} rounds with data)")
    
    # If no history, try metrics files
    if fedguide_rewards is None and fedguide_metrics_path:
        fedguide_rewards = extract_rewards_from_metrics_file(fedguide_metrics_path)
    
    if fedkl_rewards is None and fedkl_metrics_path:
        fedkl_rewards = extract_rewards_from_metrics_file(fedkl_metrics_path)
    
    if fedavg_rewards is None and fedavg_metrics_path:
        fedavg_rewards = extract_rewards_from_metrics_file(fedavg_metrics_path)
    
    # Process and plot FedGuide, FedKL, FedAvg
    _plot_curve(ax, fedguide_rewards, 'FedGuide', 'tab:blue', show_std, window_size)
    _plot_curve(ax, fedkl_rewards, 'FedKL', 'tab:orange', show_std, window_size)
    _plot_curve(ax, fedavg_rewards, 'FedAvg', 'tab:green', show_std, window_size)
    
    if not any([fedguide_rewards, fedkl_rewards, fedavg_rewards]):
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


def plot_extended_bandit2d_reward_curves(
    metrics_root: str = "./metrics/bandit2d",
    output_path: Optional[str] = None,
    metric_key: str = "eval/return",
    show_std: bool = True,
    window_size: int = 5,
) -> None:
    """
    One figure: common baselines under metrics_root, using resolve_training_history
    per algorithm subfolder. FMARL / FedRL-DDPG use the same layout as FedKL.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_root = os.path.abspath(metrics_root)

    curves: List[Tuple[str, str, Dict[int, List[float]]]] = []
    subdirs = [
        ("FedGuide", "tab:blue", "fedguide"),
        ("FedKL", "tab:orange", "fedkl"),
        ("FedAvg", "tab:green", "fedavg"),
        ("PPO", "tab:brown", "ppo"),
        ("SAC", "tab:pink", "sac"),
        ("FMARL", "tab:red", "fmarl"),
        ("FedRL-DDPG", "tab:purple", "fedrl_ddpg"),
        ("FedRep", "tab:gray", "fedrep"),
        ("FedMomentum", "tab:cyan", "fedmomentum"),
    ]

    for label, color, sub in subdirs:
        sub_path = os.path.join(metrics_root, sub)
        hist_path = resolve_training_history(sub_path)
        if not hist_path:
            print(f"[{label}] No training_history.pkl under {sub_path}")
            curves.append((label, color, {}))
            continue
        rewards = load_rewards_for_plot(hist_path, metric_key)
        mx = max(rewards.keys()) if rewards else 0
        print(f"[{label}] {hist_path} — rounds with metric: {len(rewards)}, max round {mx}")
        curves.append((label, color, rewards))

    any_data = False
    for label, color, rewards in curves:
        if rewards:
            any_data = True
        suffix = "" if rewards else " (no eval/return data)"
        _plot_curve(ax, rewards, label + suffix, color, show_std, window_size)

    if not any_data:
        ax.text(
            0.5, 0.5, "No eval/return data found under metrics_root",
            ha="center", va="center", transform=ax.transAxes, fontsize=14,
        )

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel(f'Average {metric_key.replace("/", " ").title()}', fontsize=12)
    ax.set_title("Bandit2D Federated Learning Reward Curves (Extended)", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Extended reward curve saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot reward curves for Bandit2D experiments")
    parser.add_argument("--fedguide_history", type=str, default=None,
                       help="Path to FedGuide training history pickle file")
    parser.add_argument("--fedkl_history", type=str, default=None,
                       help="Path to FedKL training history pickle file")
    parser.add_argument("--fedavg_history", type=str, default=None,
                       help="Path to FedAvg training history pickle file")
    parser.add_argument("--fedguide_metrics", type=str, default=None,
                       help="Path to FedGuide metrics pickle file (alternative)")
    parser.add_argument("--fedkl_metrics", type=str, default=None,
                       help="Path to FedKL metrics pickle file (alternative)")
    parser.add_argument("--fedavg_metrics", type=str, default=None,
                       help="Path to FedAvg metrics pickle file (alternative)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save figure (if None, display)")
    parser.add_argument("--metric_key", type=str, default="eval/return",
                       help="Metric key to plot (default: 'eval/return')")
    parser.add_argument("--no_std", action="store_true",
                       help="Don't show standard deviation bands")
    parser.add_argument("--window_size", type=int, default=5,
                       help="Moving average window size (default: 5)")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Plot all common curves (FedGuide, FedKL, FedAvg, PPO, SAC, FMARL, FedRL-DDPG, FedRep, FedMomentum) from --metrics_root",
    )
    parser.add_argument(
        "--metrics_root",
        type=str,
        default="./metrics/bandit2d",
        help="Root directory containing per-algorithm subfolders (used with --extended)",
    )
    
    args = parser.parse_args()

    if args.extended:
        plot_extended_bandit2d_reward_curves(
            metrics_root=args.metrics_root,
            output_path=args.output_path or "./plots/bandit2d/reward_curves.png",
            metric_key=args.metric_key,
            show_std=not args.no_std,
            window_size=args.window_size,
        )
    else:
        # Auto-detect history files if not provided
        for attr, default in [
            ("fedguide_history", "./metrics/bandit2d/fedguide/training_history.pkl"),
            ("fedkl_history", "./metrics/bandit2d/fedkl/training_history.pkl"),
            ("fedavg_history", "./metrics/bandit2d/fedavg/training_history.pkl"),
        ]:
            if getattr(args, attr) is None:
                resolved = resolve_training_history(str(Path(default).parent))
                if resolved:
                    setattr(args, attr, resolved)
                    print(f"Auto-detected: {attr.replace('_', ' ').title()} = {resolved}")
        
        plot_reward_curves(
            fedguide_history_path=args.fedguide_history,
            fedkl_history_path=args.fedkl_history,
            fedavg_history_path=args.fedavg_history,
            fedguide_metrics_path=args.fedguide_metrics,
            fedkl_metrics_path=args.fedkl_metrics,
            fedavg_metrics_path=args.fedavg_metrics,
            output_path=args.output_path,
            metric_key=args.metric_key,
            show_std=not args.no_std,
            window_size=args.window_size,
        )

