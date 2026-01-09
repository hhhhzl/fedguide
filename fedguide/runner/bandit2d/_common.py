"""
Common utilities for Bandit2D federated learning runners.

This module provides shared functionality for FedGuide and FedKL runners.
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Optional, Callable
from fedguide.utils.bandit2d_metrics import Bandit2DMetricsCollector


def create_metrics_collector(
    metrics_dir: str,
    collect_every: int,
    grid_size: int = 200,
    bounds: tuple = (-1.5, 1.5)
) -> Optional[Bandit2DMetricsCollector]:
    """
    Create and configure metrics collector.
    
    Args:
        metrics_dir: Directory to save metrics
        collect_every: Collect metrics every N rounds (0 to disable)
        grid_size: Grid size for metrics collection
        bounds: Bounds for grid evaluation
    
    Returns:
        Metrics collector instance or None if disabled
    """
    if collect_every <= 0:
        return None
    
    collector = Bandit2DMetricsCollector(
        save_dir=metrics_dir,
        grid_size=grid_size,
        bounds=bounds
    )
    
    print(f"Metrics collection enabled: saving to {metrics_dir}")
    print(f"  Collecting metrics every {collect_every} rounds")
    
    return collector


def make_evaluate_fn(
    collect_every: int,
    collector: Optional[Bandit2DMetricsCollector],
    algorithm: str = "fedguide"
) -> Optional[Callable]:
    """
    Create evaluate function for metrics collection.
    
    Args:
        collect_every: Collect metrics every N rounds
        collector: Metrics collector instance
        algorithm: Algorithm name ('fedguide', 'fedkl', 'fmarl', 'fedrl', or 'fedrep')
    
    Returns:
        Evaluate function or None if disabled
    """
    if collect_every <= 0 or collector is None:
        return None
    
    # Store collector in global scope for access in evaluate_fn
    # Note: This is a workaround for Ray actor isolation
    global _metrics_collector_global
    _metrics_collector_global = collector
    
    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate function called after each round."""
        print(f"[evaluate_fn] Called for round {server_round}")
        
        # Collect metrics every N rounds OR on the first round
        should_collect = (server_round % collect_every == 0) or (server_round == 1)
        
        if not should_collect:
            print(f"[evaluate_fn] Skipping collection for round {server_round} (collect_every={collect_every})")
            return None, {}
        
        print(f"[evaluate_fn] Collecting metrics for round {server_round}")
        
        try:
            # Access global collector
            global _metrics_collector_global
            collector = _metrics_collector_global
            if collector is None:
                print(f"[evaluate_fn] ERROR: Metrics collector not initialized for round {server_round}")
                return None, {}
            
            # Get collected actions from config (passed from strategy.evaluate)
            collected_actions = config.get('_collected_actions', {})
            
            # Get collected client metrics from config
            collected_client_metrics = config.get('_collected_client_metrics', {})
            
            # Create metrics entry
            round_metrics = {
                'round': server_round,
                'client_metrics': {},
                'server_metrics': {},
            }
            
            # Process client metrics
            if collected_client_metrics:
                # Aggregate client metrics to compute server_metrics
                server_prior_prob = None
                server_value = None
                server_policy_density = None
                
                for client_id, client_grid_metrics in collected_client_metrics.items():
                    # Convert numpy arrays to lists for serialization
                    client_metrics_dict = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in client_grid_metrics.items()
                    }
                    round_metrics['client_metrics'][client_id] = client_metrics_dict
                    
                    # Aggregate for server_metrics
                    if algorithm == 'fedguide' and 'prior_logprob' in client_grid_metrics:
                        prior_lp = np.array(client_grid_metrics['prior_logprob'])
                        prior_prob = np.exp(prior_lp)
                        if server_prior_prob is None:
                            server_prior_prob = prior_prob.copy()
                        else:
                            server_prior_prob = server_prior_prob + prior_prob
                    
                    if 'value' in client_grid_metrics:
                        value = np.array(client_grid_metrics['value'])
                        if server_value is None:
                            server_value = value.copy()
                        else:
                            server_value = server_value + value
                    
                    if 'policy_density' in client_grid_metrics:
                        policy_dens = np.array(client_grid_metrics['policy_density'])
                        if server_policy_density is None:
                            server_policy_density = policy_dens.copy()
                        else:
                            server_policy_density = server_policy_density + policy_dens
                
                # Average the aggregated metrics
                num_clients = len(collected_client_metrics)
                if num_clients > 0:
                    if server_prior_prob is not None:
                        server_prior_prob = server_prior_prob / num_clients
                        server_prior_logprob = np.log(server_prior_prob + 1e-12)
                    else:
                        server_prior_logprob = None
                    
                    if server_value is not None:
                        server_value = server_value / num_clients
                    if server_policy_density is not None:
                        server_policy_density = server_policy_density / num_clients
                    
                    # Build server_metrics
                    server_metrics = {}
                    if server_prior_logprob is not None:
                        server_metrics['prior_logprob'] = server_prior_logprob.tolist()
                    if server_value is not None:
                        server_metrics['value'] = server_value.tolist()
                    if server_policy_density is not None:
                        server_metrics['policy_density'] = server_policy_density.tolist()
                    
                    # Compute FedGuide policy (only for fedguide algorithm)
                    if algorithm == 'fedguide' and server_prior_logprob is not None and server_value is not None:
                        beta = 5.0  # Coefficient for FedGuide policy
                        prior_lp = np.array(server_prior_logprob)
                        value = np.array(server_value)
                        log_pi_fg = prior_lp + beta * value
                        log_pi_fg = log_pi_fg - log_pi_fg.max()  # Normalize to avoid overflow
                        pi_fg = np.exp(log_pi_fg)
                        server_metrics['fedguide_policy_density'] = pi_fg.tolist()
                    
                    round_metrics['server_metrics'] = server_metrics
                    print(f"[evaluate_fn] Computed server_metrics from {num_clients} clients: {list(server_metrics.keys())}")
            
            # Add client actions
            if collected_actions:
                round_metrics['client_actions'] = {
                    k: list(v) if isinstance(v, np.ndarray) else v 
                    for k, v in collected_actions.items()
                }
            elif collector.client_actions:
                round_metrics['client_actions'] = {
                    k: list(v) if isinstance(v, np.ndarray) else v 
                    for k, v in collector.client_actions.items()
                }
            
            # Append to history
            collector.metrics_history.append(round_metrics)
            print(f"[evaluate_fn] Appended round_metrics to history (round {server_round})")
            
        except Exception as e:
            print(f"[evaluate_fn] ERROR: Failed to collect metrics for round {server_round}: {e}")
            import traceback
            traceback.print_exc()
        
        return None, {}
    
    return evaluate_fn


# Global variable to store metrics collector (accessible in evaluate_fn)
_metrics_collector_global: Optional[Bandit2DMetricsCollector] = None


def save_training_results(
    history: Any,
    metrics_collector: Optional[Bandit2DMetricsCollector],
    metrics_dir: str,
    algorithm: str
) -> None:
    """
    Save training history and metrics.
    
    Args:
        history: Training history from Flower simulation
        metrics_collector: Metrics collector instance
        metrics_dir: Directory to save results
        algorithm: Algorithm name for visualization commands
    """
    # Save metrics if collector was used
    if metrics_collector is not None:
        print(f"\nBefore saving - Metrics collector state:")
        print(f"  metrics_history length: {len(metrics_collector.metrics_history)}")
        print(f"  client_agents: {len(metrics_collector.client_agents)}")
        print(f"  client_actions: {len(metrics_collector.client_actions)}")
        
        # If metrics_history is empty but we have actions, create at least one entry
        if len(metrics_collector.metrics_history) == 0:
            print("  WARNING: metrics_history is empty!")
            if metrics_collector.client_actions:
                print(f"  Creating metrics entry from {len(metrics_collector.client_actions)} clients' actions")
                round_metrics = {
                    'round': 'summary',
                    'client_metrics': {},
                    'server_metrics': {},
                    'client_actions': {
                        k: list(v) if isinstance(v, np.ndarray) else v 
                        for k, v in metrics_collector.client_actions.items()
                    }
                }
                metrics_collector.metrics_history.append(round_metrics)
        
        metrics_collector.save("bandit2d_metrics.pkl")
        print(f"\nMetrics saved to {metrics_dir}/bandit2d_metrics.pkl")
        print(f"  Final metrics_history length: {len(metrics_collector.metrics_history)}")
        print("  To visualize, run:")
        print(f"    python scripts/envs/bandit2d/visualize_bandit2d.py --metrics_path {metrics_dir}/bandit2d_metrics.pkl")
    
    # Save training history
    os.makedirs(metrics_dir, exist_ok=True)
    history_path = os.path.join(metrics_dir, "training_history.pkl")
    
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\nTraining history saved to {history_path}")
    print("  To plot reward curves, run:")
    if algorithm == 'fedguide':
        print(f"    python scripts/envs/bandit2d/plot_reward_curves.py \\")
        print(f"        --fedguide_history {history_path}")
    elif algorithm == 'fedkl':
        print(f"    python scripts/envs/bandit2d/plot_reward_curves.py \\")
        print(f"        --fedkl_history {history_path}")
    elif algorithm == 'fmarl':
        print(f"    python scripts/envs/bandit2d/plot_reward_curves.py \\")
        print(f"        --fedkl_history {history_path}  # FMARL uses same format as FedKL")
    elif algorithm == 'fedrl':
        print(f"    python scripts/envs/bandit2d/plot_reward_curves.py \\")
        print(f"        --fedkl_history {history_path}  # FedRL uses same format as FedKL")
    elif algorithm == 'fedrep':
        print(f"    python scripts/envs/bandit2d/plot_reward_curves.py \\")
        print(f"        --fedkl_history {history_path}  # FedRep uses same format as FedKL")

