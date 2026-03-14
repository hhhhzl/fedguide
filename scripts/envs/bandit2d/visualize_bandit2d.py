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
    
    # Check if metrics_history is empty
    if len(collector.metrics_history) == 0:
        raise ValueError(f"No metrics found in {metrics_path}. The metrics_history is empty.")
    
    # Select round to visualize
    if round_num < 0:
        round_num = len(collector.metrics_history) - 1
    if round_num >= len(collector.metrics_history):
        print(f"Warning: Round {round_num} not available. Using last round.")
        round_num = len(collector.metrics_history) - 1
    if round_num < 0:
        round_num = 0

    metrics = collector.metrics_history[round_num]
    
    # If client_actions are in metrics but not in collector.client_actions, use metrics
    if 'client_actions' in metrics and not collector.client_actions:
        collector.client_actions = {k: [np.array(v)] if isinstance(v, list) else v 
                                   for k, v in metrics['client_actions'].items()}
    
    # Create figure (2x4 layout)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    # Pre-compute unified color scales for different visualization types
    # 1. For policy density visualizations (hot colormap)
    policy_densities_list = []
    if 'server_metrics' in metrics and 'fedguide_policy_density' in metrics['server_metrics']:
        policy_densities_list.append(np.array(metrics['server_metrics']['fedguide_policy_density']))
    if 'client_metrics' in metrics:
        for client_id, client_metrics in metrics['client_metrics'].items():
            if 'policy_density' in client_metrics:
                policy_densities_list.append(np.array(client_metrics['policy_density']))
    
    # Compute unified vmin/vmax for policy densities
    policy_vmin, policy_vmax = 0, 1
    if policy_densities_list:
        all_policy_values = np.concatenate([p.ravel() for p in policy_densities_list])
        valid_policy_values = all_policy_values[~np.isnan(all_policy_values)]
        if len(valid_policy_values) > 0:
            policy_vmin = 0
            policy_vmax = np.percentile(valid_policy_values, 99)  # Use 99th percentile to avoid outliers
    
    # 2. For hexbin visualizations (viridis colormap) - will be computed from data
    hexbin_vmin, hexbin_vmax = None, None
    
    # (a) Multi-client dataset distribution
    # Collect all actions for hexbin visualization
    ax = axes[0]
    all_actions = []
    # Use client_actions from metrics if collector.client_actions is empty
    actions_dict = collector.client_actions if collector.client_actions else {}
    if 'client_actions' in metrics and not actions_dict:
        actions_dict = metrics['client_actions']
    
    for i, (client_id, actions_list) in enumerate(actions_dict.items()):
        try:
            # Handle different formats: list of arrays, single array, or list of lists
            if isinstance(actions_list, list):
                if len(actions_list) > 0:
                    # Check if first element is array or list
                    if isinstance(actions_list[0], (np.ndarray, list)):
                        actions = np.concatenate([np.array(a) for a in actions_list], axis=0)
                    else:
                        actions = np.array(actions_list)
                else:
                    continue
            elif isinstance(actions_list, np.ndarray):
                actions = actions_list
            else:
                continue
            
            # Ensure 2D shape
            if len(actions.shape) == 1:
                actions = actions.reshape(-1, 2)
            elif len(actions.shape) == 2 and actions.shape[1] != 2:
                actions = actions.reshape(-1, 2)
            
            if actions.shape[0] > 0:
                all_actions.append(actions)
        except Exception as e:
            print(f"Warning: Failed to process actions for client {client_id}: {e}")
            continue
    
    # Use hexbin for density visualization (like visualize_prior_aggregated.py)
    if all_actions:
        all_actions_combined = np.concatenate(all_actions, axis=0)
        # Clip to bounds
        all_actions_combined = np.clip(all_actions_combined, -1.5, 1.5)
        
        # Calculate unified color scale for hexbin (like visualize_prior_aggregated.py)
        fig_temp = plt.figure()
        ax_temp = fig_temp.add_subplot(111)
        hb_temp = ax_temp.hexbin(
            all_actions_combined[:, 0],
            all_actions_combined[:, 1],
            gridsize=60,
            extent=(-1.5, 1.5, -1.5, 1.5),
        )
        counts = hb_temp.get_array()
        if counts is not None:
            valid_counts = counts[~np.isnan(counts)]
            hexbin_vmin = 0
            hexbin_vmax = max(valid_counts) if len(valid_counts) > 0 else 1
        else:
            hexbin_vmin = 0
            hexbin_vmax = 1
        plt.close(fig_temp)
        
        # Now plot with unified color scale
        hb = ax.hexbin(
            all_actions_combined[:, 0],
            all_actions_combined[:, 1],
            gridsize=60,
            extent=(-1.5, 1.5, -1.5, 1.5),
            cmap='viridis',
            vmin=hexbin_vmin,
            vmax=hexbin_vmax,
        )
        ax.set_title("Multi-client Dataset")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Density (hexbin count)")
        
        # Also add peak locations if available
        try:
            from fedguide.envs.bandit2d import Bandit2D
            env = Bandit2D(K=4, sigma=0.2)
            peaks = env.get_peak_locations()
            ax.scatter(
                peaks[:, 0],
                peaks[:, 1],
                c="red",
                marker="*",
                s=150,
                edgecolors="white",
                linewidths=1,
                zorder=10,
                label="Peaks"
            )
            ax.legend()
        except Exception:
            pass
    else:
        ax.text(0.5, 0.5, "No actions available", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Multi-client Dataset")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    # (b) Federated diffusion prior
    # Check if prior_logprob is available and valid (not all zeros)
    prior_available = False
    prior_data = None
    if 'server_metrics' in metrics and 'prior_logprob' in metrics['server_metrics']:
        prior_logprob = np.array(metrics['server_metrics']['prior_logprob'])
        # Check if prior is valid (not all zeros or all same value)
        if not np.allclose(prior_logprob, 0) and not np.allclose(prior_logprob, prior_logprob.flat[0]):
            prior_data = np.exp(prior_logprob)
            prior_available = True
    
    if prior_available:
        ax = axes[1]
        im = ax.imshow(prior_data, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='viridis')
        ax.set_title("FedGuide Diffusion Prior")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        # Add peak locations
        try:
            from fedguide.envs.bandit2d import Bandit2D
            env = Bandit2D(K=4, sigma=0.2)
            peaks = env.get_peak_locations()
            ax.scatter(
                peaks[:, 0],
                peaks[:, 1],
                c="red",
                marker="*",
                s=150,
                edgecolors="white",
                linewidths=1,
                zorder=10,
            )
        except Exception:
            pass
    else:
        axes[1].text(0.5, 0.5, "Prior not available\n(or all zeros)", ha='center', va='center', transform=axes[1].transAxes)
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
        pi_fg = np.array(metrics['server_metrics']['fedguide_policy_density'])
        im = ax.imshow(pi_fg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot',
                      vmin=policy_vmin, vmax=policy_vmax)
        ax.set_title("FedGuide Policy\n(Prior + Value)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        # Add peak locations
        try:
            from fedguide.envs.bandit2d import Bandit2D
            env = Bandit2D(K=4, sigma=0.2)
            peaks = env.get_peak_locations()
            ax.scatter(
                peaks[:, 0],
                peaks[:, 1],
                c="red",
                marker="*",
                s=150,
                edgecolors="white",
                linewidths=1,
                zorder=10,
            )
        except Exception:
            pass
    else:
        axes[3].text(0.5, 0.5, "FedGuide policy not available", ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title("FedGuide Policy")
    
    # (e) Local-only policy (client 0 or first available client)
    # Try to get from client_metrics first, otherwise estimate from client_actions
    client_0_id = None
    if metrics['client_metrics']:
        # Find client 0 or use the first available client_id
        if 0 in metrics['client_metrics']:
            client_0_id = 0
        else:
            # Use the first available client_id
            client_0_id = min(metrics['client_metrics'].keys())
    
    if client_0_id is not None and 'policy_density' in metrics['client_metrics'][client_0_id]:
        ax = axes[4]
        pi_local = np.array(metrics['client_metrics'][client_0_id]['policy_density'])
        im = ax.imshow(pi_local, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot',
                      vmin=policy_vmin, vmax=policy_vmax)
        ax.set_title(f"Local Client Policy ({client_0_id})")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        # Add peak locations
        try:
            from fedguide.envs.bandit2d import Bandit2D
            env = Bandit2D(K=4, sigma=0.2)
            peaks = env.get_peak_locations()
            ax.scatter(
                peaks[:, 0],
                peaks[:, 1],
                c="red",
                marker="*",
                s=150,
                edgecolors="white",
                linewidths=1,
                zorder=10,
            )
        except Exception:
            pass
    elif 'client_actions' in metrics and metrics['client_actions']:
        # Try to find client 0 in actions, or use first available
        action_client_id = None
        if 0 in metrics['client_actions']:
            action_client_id = 0
        else:
            action_client_id = min(metrics['client_actions'].keys())
        
        if action_client_id is not None:
            # Estimate policy density from actions using 2D histogram
            ax = axes[4]
            try:
                actions_list = metrics['client_actions'][action_client_id]
                # Handle different formats
                if isinstance(actions_list, list):
                    if len(actions_list) > 0 and isinstance(actions_list[0], (np.ndarray, list)):
                        actions = np.concatenate([np.array(a) for a in actions_list], axis=0)
                    else:
                        actions = np.array(actions_list)
                elif isinstance(actions_list, np.ndarray):
                    actions = actions_list
                else:
                    raise ValueError(f"Unexpected actions format: {type(actions_list)}")
                
                # Ensure 2D shape
                if len(actions.shape) == 1:
                    actions = actions.reshape(-1, 2)
                elif len(actions.shape) == 2 and actions.shape[1] != 2:
                    actions = actions.reshape(-1, 2)
                
                if actions.shape[0] > 0:
                    # Create 2D histogram
                    H, xedges, yedges = np.histogram2d(actions[:, 0], actions[:, 1], 
                                                       bins=collector.grid_size, 
                                                       range=[[-1.5, 1.5], [-1.5, 1.5]])
                    # Normalize
                    H = H / (H.sum() + 1e-10)
                    # Transpose for imshow (origin='lower')
                    H = H.T
                    im = ax.imshow(H, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot',
                                  vmin=policy_vmin, vmax=policy_vmax)
                    ax.set_title(f"Local Client Policy ({action_client_id})\n(Estimated from Actions)")
                    plt.colorbar(im, ax=ax)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    
                    # Add peak locations
                    try:
                        from fedguide.envs.bandit2d import Bandit2D
                        env = Bandit2D(K=4, sigma=0.2)
                        peaks = env.get_peak_locations()
                        ax.scatter(
                            peaks[:, 0],
                            peaks[:, 1],
                            c="red",
                            marker="*",
                            s=150,
                            edgecolors="white",
                            linewidths=1,
                            zorder=10,
                        )
                    except Exception:
                        pass
                else:
                    axes[4].text(0.5, 0.5, f"No actions for client {action_client_id}", ha='center', va='center', transform=axes[4].transAxes)
                    axes[4].set_title(f"Local Client Policy ({action_client_id})")
            except Exception as e:
                print(f"Warning: Failed to estimate policy from actions for client {action_client_id}: {e}")
                axes[4].text(0.5, 0.5, "Local policy not available", ha='center', va='center', transform=axes[4].transAxes)
                axes[4].set_title(f"Local Client Policy ({action_client_id})")
    else:
        axes[4].text(0.5, 0.5, "Local policy not available", ha='center', va='center', transform=axes[4].transAxes)
        axes[4].set_title("Local Client Policy (0)")
    
    # (f) FedAvg policy (average of all client policies)
    # Try to get from client_metrics first, otherwise estimate from client_actions
    policy_densities = []
    if metrics['client_metrics']:
        for client_id, client_metrics in metrics['client_metrics'].items():
            if 'policy_density' in client_metrics:
                policy_densities.append(client_metrics['policy_density'])
    
    # If no policy densities from metrics, estimate from actions
    if not policy_densities and 'client_actions' in metrics:
        for client_id, actions_list in metrics['client_actions'].items():
            try:
                # Handle different formats
                if isinstance(actions_list, list):
                    if len(actions_list) > 0 and isinstance(actions_list[0], (np.ndarray, list)):
                        actions = np.concatenate([np.array(a) for a in actions_list], axis=0)
                    else:
                        actions = np.array(actions_list)
                elif isinstance(actions_list, np.ndarray):
                    actions = actions_list
                else:
                    continue
                
                # Ensure 2D shape
                if len(actions.shape) == 1:
                    actions = actions.reshape(-1, 2)
                elif len(actions.shape) == 2 and actions.shape[1] != 2:
                    actions = actions.reshape(-1, 2)
                
                if actions.shape[0] > 0:
                    # Create 2D histogram
                    H, xedges, yedges = np.histogram2d(actions[:, 0], actions[:, 1], 
                                                       bins=collector.grid_size, 
                                                       range=[[-1.5, 1.5], [-1.5, 1.5]])
                    # Normalize
                    H = H / (H.sum() + 1e-10)
                    # Transpose for imshow (origin='lower')
                    H = H.T
                    policy_densities.append(H)
            except Exception as e:
                print(f"Warning: Failed to estimate policy from actions for client {client_id}: {e}")
                continue
    
    if policy_densities:
        ax = axes[5]
        pi_fedavg = np.mean(policy_densities, axis=0)
        im = ax.imshow(pi_fedavg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot',
                      vmin=policy_vmin, vmax=policy_vmax)
        ax.set_title("FedAvg Policy\n(Avg Policy)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        # Add peak locations
        try:
            from fedguide.envs.bandit2d import Bandit2D
            env = Bandit2D(K=4, sigma=0.2)
            peaks = env.get_peak_locations()
            ax.scatter(
                peaks[:, 0],
                peaks[:, 1],
                c="red",
                marker="*",
                s=150,
                edgecolors="white",
                linewidths=1,
                zorder=10,
            )
        except Exception:
            pass
    else:
        axes[5].text(0.5, 0.5, "FedAvg policy not available", ha='center', va='center', transform=axes[5].transAxes)
        axes[5].set_title("FedAvg Policy")
    
    # (g) FedKL-policy (direct average policy, no value guidance)
    # Note: FedKL and FedAvg aggregate policy the same way (both average policy parameters)
    # But we mark it separately for clarity
    # Reuse policy_densities from FedAvg calculation
    if policy_densities:
        ax = axes[6]
        # FedKL also averages policy, so calculation is the same
        pi_fedkl = np.mean(policy_densities, axis=0)
        im = ax.imshow(pi_fedkl, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot',
                      vmin=policy_vmin, vmax=policy_vmax)
        ax.set_title("FedKL Policy\n(Avg Policy, No Value)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        # Add peak locations
        try:
            from fedguide.envs.bandit2d import Bandit2D
            env = Bandit2D(K=4, sigma=0.2)
            peaks = env.get_peak_locations()
            ax.scatter(
                peaks[:, 0],
                peaks[:, 1],
                c="red",
                marker="*",
                s=150,
                edgecolors="white",
                linewidths=1,
                zorder=10,
            )
        except Exception:
            pass
    else:
        axes[6].text(0.5, 0.5, "FedKL policy not available", ha='center', va='center', transform=axes[6].transAxes)
        axes[6].set_title("FedKL Policy")
    
    # (h) FedGuide policy again (for side-by-side comparison)
    if 'server_metrics' in metrics and 'fedguide_policy_density' in metrics['server_metrics']:
        ax = axes[7]
        pi_fg = np.array(metrics['server_metrics']['fedguide_policy_density'])
        im = ax.imshow(pi_fg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot',
                      vmin=policy_vmin, vmax=policy_vmax)
        ax.set_title("FedGuide Policy\n(Prior + Value)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        # Add peak locations
        try:
            from fedguide.envs.bandit2d import Bandit2D
            env = Bandit2D(K=4, sigma=0.2)
            peaks = env.get_peak_locations()
            ax.scatter(
                peaks[:, 0],
                peaks[:, 1],
                c="red",
                marker="*",
                s=150,
                edgecolors="white",
                linewidths=1,
                zorder=10,
            )
        except Exception:
            pass
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
                        metrics_fedavg_path: str = None, output_path: str = None, round_num: int = -1):
    """
    Compare FedGuide, FedKL, and FedAvg results.
    
    Args:
        metrics_fedguide_path: Path to FedGuide metrics file
        metrics_fedkl_path: Path to FedKL metrics file (optional, for comparison)
        metrics_fedavg_path: Path to FedAvg metrics file (optional, for fair comparison)
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
    
    # (d) FedGuide Guided Density (Prior + Value)
    if 'server_metrics' in metrics_fg and 'fedguide_policy_density' in metrics_fg['server_metrics']:
        ax = axes[3]
        pi_fg = metrics_fg['server_metrics']['fedguide_policy_density']
        im = ax.imshow(pi_fg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
        ax.set_title("FedGuide Guided Density\n(Prior + Value)")
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
    
    # (f) FedAvg policy (from metrics_fedavg_path if provided, else from FedGuide client metrics)
    pi_fedavg = None
    if metrics_fedavg_path:
        try:
            collector_avg = Bandit2DMetricsCollector.load(metrics_fedavg_path)
            if round_num < len(collector_avg.metrics_history):
                metrics_avg = collector_avg.metrics_history[round_num]
            else:
                metrics_avg = collector_avg.metrics_history[-1] if collector_avg.metrics_history else {}
            if 'server_metrics' in metrics_avg and 'policy_density' in metrics_avg['server_metrics']:
                pi_fedavg = metrics_avg['server_metrics']['policy_density']
            elif 'client_metrics' in metrics_avg:
                policy_densities = []
                for cid, cm in metrics_avg['client_metrics'].items():
                    if 'policy_density' in cm:
                        policy_densities.append(cm['policy_density'])
                if policy_densities:
                    pi_fedavg = np.mean(policy_densities, axis=0)
        except Exception as e:
            print(f"Warning: Failed to load FedAvg metrics from {metrics_fedavg_path}: {e}")
    if pi_fedavg is None and metrics_fg['client_metrics']:
        policy_densities = []
        for client_id, client_metrics in metrics_fg['client_metrics'].items():
            if 'policy_density' in client_metrics:
                policy_densities.append(client_metrics['policy_density'])
        if policy_densities:
            pi_fedavg = np.mean(policy_densities, axis=0)
    if pi_fedavg is not None:
        ax = axes[5]
        im = ax.imshow(pi_fedavg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
        ax.set_title("FedAvg Policy\n(Avg Policy)" + (" [from FedAvg run]" if metrics_fedavg_path else ""))
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        axes[5].text(0.5, 0.5, "FedAvg policy not available", ha='center', va='center', transform=axes[5].transAxes)
        axes[5].set_title("FedAvg Policy")
    
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
    
    # (h) FedGuide Avg Policy Density (comparable to FedAvg/FedKL)
    pi_fedguide_avg = None
    if 'server_metrics' in metrics_fg and 'policy_density' in metrics_fg['server_metrics']:
        pi_fedguide_avg = metrics_fg['server_metrics']['policy_density']
    elif metrics_fg.get('client_metrics'):
        policy_densities = []
        for cid, cm in metrics_fg['client_metrics'].items():
            if 'policy_density' in cm:
                policy_densities.append(cm['policy_density'])
        if policy_densities:
            pi_fedguide_avg = np.mean(policy_densities, axis=0)
    if pi_fedguide_avg is not None:
        ax = axes[7]
        im = ax.imshow(pi_fedguide_avg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot')
        ax.set_title("FedGuide Avg Policy\n(Policy Density)")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        axes[7].text(0.5, 0.5, "FedGuide avg policy\nnot available", ha='center', va='center',
                    transform=axes[7].transAxes)
        axes[7].set_title("FedGuide Avg Policy")
    
    plt.suptitle(f"Bandit2D Comparison - Round {round_num}", fontsize=14, y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to {output_path}")
    else:
        plt.show()


def visualize_client_policies(metrics_path: str, output_path: str = None, round_num: int = -1,
                              algorithm_label: str = "FedGuide"):
    """
    Plot each client's policy_density in a 2x2 grid to check if they concentrate
    at the same location (single peak after avg) or different locations (multi-peak after avg).
    """
    collector = Bandit2DMetricsCollector.load(metrics_path)
    if len(collector.metrics_history) == 0:
        raise ValueError(f"No metrics found in {metrics_path}")
    if round_num < 0:
        round_num = len(collector.metrics_history) - 1
    round_num = min(round_num, len(collector.metrics_history) - 1)
    metrics = collector.metrics_history[round_num]

    client_ids = sorted([k for k in metrics.get('client_metrics', {}).keys()
                        if 'policy_density' in metrics['client_metrics'][k]])
    if not client_ids:
        raise ValueError(f"No client policy_density found in {metrics_path} round {round_num}")

    n = len(client_ids)
    ncols = 2
    nrows = (n + 1) // 2 + (1 if n > 2 else 0)  # 4 clients -> 2x2, add row for avg
    if n <= 2:
        nrows = n
    # Layout: 2x2 for 4 clients, plus optional 5th for average
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2x3: 4 clients + avg + empty or legend
    axes = axes.flatten()

    # Unified color scale across all clients
    all_densities = [np.array(metrics['client_metrics'][cid]['policy_density']) for cid in client_ids]
    vmin, vmax = 0, 1
    if all_densities:
        concat = np.concatenate([d.ravel() for d in all_densities])
        valid = concat[~np.isnan(concat)]
        if len(valid) > 0:
            vmax = np.percentile(valid, 99)

    for i, cid in enumerate(client_ids):
        ax = axes[i]
        pi = np.array(metrics['client_metrics'][cid]['policy_density'])
        im = ax.imshow(pi, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot',
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"Client {cid} Policy")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)

    # 5th subplot: average of all clients (same as server policy_density)
    ax_avg = axes[4]
    pi_avg = np.mean(all_densities, axis=0)
    im_avg = ax_avg.imshow(pi_avg, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5], cmap='hot',
                           vmin=vmin, vmax=vmax)
    ax_avg.set_title(f"{algorithm_label} Avg Policy\n(mean of {n} clients)")
    ax_avg.set_xlabel("x")
    ax_avg.set_ylabel("y")
    plt.colorbar(im_avg, ax=ax_avg)

    # Hide 6th subplot
    axes[5].set_visible(False)

    plt.suptitle(f"{algorithm_label} Per-Client Policy Density (Round {round_num})\n"
                 "If clients overlap → single peak after avg; if spread → multi-peak", fontsize=12, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Client policies figure saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Bandit2D experiment metrics")
    parser.add_argument("--metrics_path", type=str, required=True,
                       help="Path to metrics pickle file (FedGuide for comparison)")
    parser.add_argument("--metrics_fedkl_path", type=str, default=None,
                       help="Path to FedKL metrics pickle file (for comparison)")
    parser.add_argument("--metrics_fedavg_path", type=str, default=None,
                       help="Path to FedAvg metrics pickle file (for fair 3-way comparison)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save figure (if None, display)")
    parser.add_argument("--round_num", type=int, default=-1,
                       help="Round number to visualize (-1 for last round)")
    parser.add_argument("--comparison", action="store_true",
                       help="Create comparison figure with FedKL and optionally FedAvg")
    parser.add_argument("--client_policies", action="store_true",
                       help="Plot each client's policy_density (2x2 + avg) to check overlap")
    parser.add_argument("--client_policies_output", type=str, default=None,
                       help="Path to save client policies figure (default: same dir as output_path)")
    args = parser.parse_args()
    
    if args.comparison:
        visualize_comparison(
            args.metrics_path,
            args.metrics_fedkl_path,
            args.metrics_fedavg_path,
            args.output_path,
            args.round_num
        )
    if args.client_policies:
        out = args.client_policies_output
        if out is None and args.output_path:
            from pathlib import Path
            out = str(Path(args.output_path).parent / "client_policies.png")
        elif out is None:
            out = "client_policies.png"
        # Infer algorithm label from path (e.g. .../fedavg/... -> FedAvg)
        p = args.metrics_path.lower()
        label = "FedGuide"
        if "/fedavg/" in p or "fedavg" in p.split("/")[-2:]:
            label = "FedAvg"
        elif "/fedkl/" in p or "fedkl" in p.split("/")[-2:]:
            label = "FedKL"
        visualize_client_policies(args.metrics_path, out, args.round_num, algorithm_label=label)
    if not args.comparison and not args.client_policies:
        visualize_bandit2d(args.metrics_path, args.output_path, args.round_num)

