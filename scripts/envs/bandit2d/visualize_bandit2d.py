"""
Visualize Bandit2D experiment metrics.

This script generates visualization figures showing:
- Multi-client data distribution
- FedGuide diffusion prior
- FedGuide value guidance
- FedGuide policy
- Local-only policy
- FedAvg policy
- FedKL policy
"""

import numpy as np
import matplotlib.pyplot as plt
from fedguide.utils.bandit2d_metrics import Bandit2DMetricsCollector
import argparse


def visualize_bandit2d(metrics_path: str, output_path: str = None, round_num: int = -1):
    """
    Visualize Bandit2D experiment metrics.
    
    Args:
        metrics_path: Path to metrics pickle file
        output_path: Path to save figure (if None, display)
        round_num: Round number to visualize (-1 for last round)
    """
    # Load metrics
    collector = Bandit2DMetricsCollector.load(metrics_path)
    
    # Select round to visualize
    if round_num < 0:
        round_num = len(collector.metrics_history) - 1
    if round_num >= len(collector.metrics_history):
        print(f"Warning: Round {round_num} not available. Using last round.")
        round_num = len(collector.metrics_history) - 1
    
    metrics = collector.metrics_history[round_num]
    
    # Create figure (2x4 layout)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    # (a) Multi-client dataset distribution
    ax = axes[0]
    all_actions = []
    for i, (client_id, actions_list) in enumerate(collector.client_actions.items()):
        if actions_list:
            actions = np.concatenate(actions_list, axis=0)
            all_actions.append(actions)
            ax.scatter(actions[:, 0], actions[:, 1], alpha=0.3, s=5, 
                      color=colors[i % len(colors)], label=f"Client {client_id}")
    ax.set_title("Multi-client Dataset")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # (b) Federated diffusion prior
    if 'server_metrics' in metrics and 'prior_logprob' in metrics['server_metrics']:
        ax = axes[1]
        prior = np.exp(metrics['server_metrics']['prior_logprob'])
        im = ax.imshow(prior, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='viridis')
        ax.set_title("FedGuide Diffusion Prior")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        axes[1].text(0.5, 0.5, "Prior not available", ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("FedGuide Diffusion Prior")
    
    # (c) Federated value guidance
    if 'server_metrics' in metrics and 'value' in metrics['server_metrics']:
        ax = axes[2]
        value = metrics['server_metrics']['value']
        im = ax.imshow(value, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='plasma')
        ax.set_title("FedGuide Value Guidance")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        axes[2].text(0.5, 0.5, "Value not available", ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("FedGuide Value Guidance")
    
    # (d) FedGuide policy
    if 'server_metrics' in metrics and 'fedguide_policy_density' in metrics['server_metrics']:
        ax = axes[3]
        pi_fg = metrics['server_metrics']['fedguide_policy_density']
        im = ax.imshow(pi_fg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
        ax.set_title("FedGuide Policy\n(Prior + Value)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        axes[3].text(0.5, 0.5, "FedGuide policy not available", ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title("FedGuide Policy")
    
    # (e) Local-only policy (client 0)
    if 0 in metrics['client_metrics'] and 'policy_density' in metrics['client_metrics'][0]:
        ax = axes[4]
        pi_local = metrics['client_metrics'][0]['policy_density']
        im = ax.imshow(pi_local, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
        ax.set_title("Local Client Policy (0)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        axes[4].text(0.5, 0.5, "Local policy not available", ha='center', va='center', transform=axes[4].transAxes)
        axes[4].set_title("Local Client Policy (0)")
    
    # (f) FedAvg policy (average of all client policies)
    if metrics['client_metrics']:
        ax = axes[5]
        policy_densities = []
        for client_id, client_metrics in metrics['client_metrics'].items():
            if 'policy_density' in client_metrics:
                policy_densities.append(client_metrics['policy_density'])
        if policy_densities:
            pi_fedavg = np.mean(policy_densities, axis=0)
            im = ax.imshow(pi_fedavg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
            ax.set_title("FedAvg Policy\n(Avg Policy)")
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            axes[5].text(0.5, 0.5, "FedAvg policy not available", ha='center', va='center', transform=axes[5].transAxes)
            axes[5].set_title("FedAvg Policy")
    else:
        axes[5].text(0.5, 0.5, "No client metrics", ha='center', va='center', transform=axes[5].transAxes)
        axes[5].set_title("FedAvg Policy")
    
    # (g) FedKL-policy (direct average policy, no value guidance)
    # Note: FedKL and FedAvg aggregate policy the same way (both average policy parameters)
    # But we mark it separately for clarity
    if metrics['client_metrics']:
        ax = axes[6]
        policy_densities = []
        for client_id, client_metrics in metrics['client_metrics'].items():
            if 'policy_density' in client_metrics:
                policy_densities.append(client_metrics['policy_density'])
        if policy_densities:
            # FedKL also averages policy, so calculation is the same
            pi_fedkl = np.mean(policy_densities, axis=0)
            im = ax.imshow(pi_fedkl, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
            ax.set_title("FedKL Policy\n(Avg Policy, No Value)")
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            axes[6].text(0.5, 0.5, "FedKL policy not available", ha='center', va='center', transform=axes[6].transAxes)
            axes[6].set_title("FedKL Policy")
    else:
        axes[6].text(0.5, 0.5, "No client metrics", ha='center', va='center', transform=axes[6].transAxes)
        axes[6].set_title("FedKL Policy")
    
    # (h) FedGuide policy again (for side-by-side comparison)
    if 'server_metrics' in metrics and 'fedguide_policy_density' in metrics['server_metrics']:
        ax = axes[7]
        pi_fg = metrics['server_metrics']['fedguide_policy_density']
        im = ax.imshow(pi_fg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
        ax.set_title("FedGuide Policy\n(Prior + Value)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        axes[7].text(0.5, 0.5, "FedGuide policy not available", ha='center', va='center', transform=axes[7].transAxes)
        axes[7].set_title("FedGuide Policy")
    
    plt.suptitle(f"Bandit2D Visualization - Round {round_num}", fontsize=14, y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def visualize_comparison(metrics_fedguide_path: str, metrics_fedkl_path: str = None, 
                        output_path: str = None, round_num: int = -1):
    """
    Compare FedGuide and FedKL results.
    
    Args:
        metrics_fedguide_path: Path to FedGuide metrics file
        metrics_fedkl_path: Path to FedKL metrics file (optional, for comparison)
        output_path: Path to save figure
        round_num: Round number to visualize
    """
    # Load FedGuide metrics
    collector_fg = Bandit2DMetricsCollector.load(metrics_fedguide_path)
    
    if round_num < 0:
        round_num = len(collector_fg.metrics_history) - 1
    if round_num >= len(collector_fg.metrics_history):
        round_num = len(collector_fg.metrics_history) - 1
    metrics_fg = collector_fg.metrics_history[round_num]
    
    # Create comparison figure (2x4 layout)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    # First row: FedGuide components
    # (a) Multi-client dataset
    ax = axes[0]
    for i, (client_id, actions_list) in enumerate(collector_fg.client_actions.items()):
        if actions_list:
            actions = np.concatenate(actions_list, axis=0)
            ax.scatter(actions[:, 0], actions[:, 1], alpha=0.3, s=5,
                      color=colors[i % len(colors)], label=f"Client {client_id}")
    ax.set_title("Multi-client Dataset")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # (b) FedGuide Prior
    if 'server_metrics' in metrics_fg and 'prior_logprob' in metrics_fg['server_metrics']:
        ax = axes[1]
        prior = np.exp(metrics_fg['server_metrics']['prior_logprob'])
        im = ax.imshow(prior, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='viridis')
        ax.set_title("FedGuide Diffusion Prior")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    # (c) FedGuide Value
    if 'server_metrics' in metrics_fg and 'value' in metrics_fg['server_metrics']:
        ax = axes[2]
        value = metrics_fg['server_metrics']['value']
        im = ax.imshow(value, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='plasma')
        ax.set_title("FedGuide Value Guidance")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    # (d) FedGuide Policy
    if 'server_metrics' in metrics_fg and 'fedguide_policy_density' in metrics_fg['server_metrics']:
        ax = axes[3]
        pi_fg = metrics_fg['server_metrics']['fedguide_policy_density']
        im = ax.imshow(pi_fg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
        ax.set_title("FedGuide Policy\n(Prior + Value)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    # Second row: Baseline comparisons
    # (e) Local-only policy
    if 0 in metrics_fg['client_metrics'] and 'policy_density' in metrics_fg['client_metrics'][0]:
        ax = axes[4]
        pi_local = metrics_fg['client_metrics'][0]['policy_density']
        im = ax.imshow(pi_local, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
        ax.set_title("Local Client Policy (0)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    # (f) FedAvg policy (from FedGuide metrics, since FedAvg and FedKL aggregate policy the same way)
    if metrics_fg['client_metrics']:
        ax = axes[5]
        policy_densities = []
        for client_id, client_metrics in metrics_fg['client_metrics'].items():
            if 'policy_density' in client_metrics:
                policy_densities.append(client_metrics['policy_density'])
        if policy_densities:
            pi_fedavg = np.mean(policy_densities, axis=0)
            im = ax.imshow(pi_fedavg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
            ax.set_title("FedAvg Policy\n(Avg Policy)")
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    
    # (g) FedKL-policy
    if metrics_fedkl_path:
        # If FedKL metrics provided, use it
        try:
            collector_kl = Bandit2DMetricsCollector.load(metrics_fedkl_path)
            if round_num < len(collector_kl.metrics_history):
                metrics_kl = collector_kl.metrics_history[round_num]
            else:
                metrics_kl = collector_kl.metrics_history[-1]
            
            if 'server_metrics' in metrics_kl and 'policy_density' in metrics_kl['server_metrics']:
                # If FedKL has server metrics, use it
                ax = axes[6]
                pi_fedkl = metrics_kl['server_metrics']['policy_density']
                im = ax.imshow(pi_fedkl, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
                ax.set_title("FedKL Policy\n(Avg Policy, No Value)")
                plt.colorbar(im, ax=ax)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            elif 'client_metrics' in metrics_kl:
                # Otherwise compute average from client metrics
                ax = axes[6]
                policy_densities = []
                for client_id, client_metrics in metrics_kl['client_metrics'].items():
                    if 'policy_density' in client_metrics:
                        policy_densities.append(client_metrics['policy_density'])
                if policy_densities:
                    pi_fedkl = np.mean(policy_densities, axis=0)
                    im = ax.imshow(pi_fedkl, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
                    ax.set_title("FedKL Policy\n(Avg Policy, No Value)")
                    plt.colorbar(im, ax=ax)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
        except Exception as e:
            print(f"Warning: Failed to load FedKL metrics: {e}")
            # Fall back to computing from FedGuide metrics
            if metrics_fg['client_metrics']:
                ax = axes[6]
                policy_densities = []
                for client_id, client_metrics in metrics_fg['client_metrics'].items():
                    if 'policy_density' in client_metrics:
                        policy_densities.append(client_metrics['policy_density'])
                if policy_densities:
                    pi_fedkl = np.mean(policy_densities, axis=0)
                    im = ax.imshow(pi_fedkl, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
                    ax.set_title("FedKL Policy\n(Avg Policy, No Value)")
                    plt.colorbar(im, ax=ax)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
    else:
        # If no FedKL metrics provided, compute from FedGuide metrics (same aggregation)
        if metrics_fg['client_metrics']:
            ax = axes[6]
            policy_densities = []
            for client_id, client_metrics in metrics_fg['client_metrics'].items():
                if 'policy_density' in client_metrics:
                    policy_densities.append(client_metrics['policy_density'])
            if policy_densities:
                pi_fedkl = np.mean(policy_densities, axis=0)
                im = ax.imshow(pi_fedkl, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
                ax.set_title("FedKL Policy\n(Avg Policy, No Value)")
                plt.colorbar(im, ax=ax)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
    
    # (h) FedGuide Policy (display again for comparison)
    if 'server_metrics' in metrics_fg and 'fedguide_policy_density' in metrics_fg['server_metrics']:
        ax = axes[7]
        pi_fg = metrics_fg['server_metrics']['fedguide_policy_density']
        im = ax.imshow(pi_fg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
        ax.set_title("FedGuide Policy\n(Prior + Value)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    plt.suptitle(f"Bandit2D Comparison - Round {round_num}", fontsize=14, y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Bandit2D experiment metrics")
    parser.add_argument("--metrics_path", type=str, required=True,
                       help="Path to metrics pickle file")
    parser.add_argument("--metrics_fedkl_path", type=str, default=None,
                       help="Path to FedKL metrics pickle file (for comparison)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save figure (if None, display)")
    parser.add_argument("--round_num", type=int, default=-1,
                       help="Round number to visualize (-1 for last round)")
    parser.add_argument("--comparison", action="store_true",
                       help="Create comparison figure with FedKL")
    args = parser.parse_args()
    
    if args.comparison:
        visualize_comparison(
            args.metrics_path,
            args.metrics_fedkl_path,
            args.output_path,
            args.round_num
        )
    else:
        visualize_bandit2d(args.metrics_path, args.output_path, args.round_num)

