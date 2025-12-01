"""
Run FedGuide federated training for 2D Bandit environment.
"""
import argparse
import os
import pickle
import torch
import numpy as np
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import parameters_to_ndarrays
from fedguide.fed.fedguide.server import FedGuideServer
from fedguide.fed.fedguide.client import client_fn_builder
from fedguide.utils.bandit2d_metrics import Bandit2DMetricsCollector
from fedguide.agents.fedguide_agent import FedguideAgent

# Global variable to store metrics collector (accessible in client_fn)
_metrics_collector_global = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--cpus_per_client", type=int, default=2)
    
    # Training args
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--lambda_local", type=float, default=0.25)
    parser.add_argument("--lambda_guide", type=float, default=0.2)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=64)
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    
    # Metrics collection
    parser.add_argument("--metrics_dir", type=str, default="./metrics/bandit2d_fedguide",
                       help="Directory to save metrics for visualization")
    parser.add_argument("--collect_metrics_every", type=int, default=1,
                       help="Collect metrics every N rounds (0 to disable)")
    
    args = parser.parse_args()
    
    # Create metrics collector
    global _metrics_collector_global
    metrics_collector = None
    if args.collect_metrics_every > 0:
        metrics_collector = Bandit2DMetricsCollector(
            save_dir=args.metrics_dir,
            grid_size=200,
            bounds=(-1.5, 1.5)
        )
        _metrics_collector_global = metrics_collector
        print(f"Metrics collection enabled: saving to {args.metrics_dir}")
        print(f"  Collecting metrics every {args.collect_metrics_every} rounds")
    
    # Build client function - pass collector directly (it should be serializable)
    # Each client will get a copy, but they'll all register their agents
    client_fn = client_fn_builder(
        env_id="Bandit2D",
        algo="ppo",
        aggregate_mode="policy",
        n_steps=args.n_steps,
        lambda_local=args.lambda_local,
        lambda_guide=args.lambda_guide,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name or "bandit2d-fedguide",
        metrics_collector=metrics_collector,  # Pass directly - will be serialized to each actor
        num_clients=args.num_clients,  # Pass num_clients for ID mapping
    )
    
    # Create evaluate function for metrics collection
    def make_evaluate_fn(collect_every):
        """Create evaluate function that collects metrics periodically."""
        if collect_every <= 0:
            return None
        
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
                collector = _metrics_collector_global
                if collector is None:
                    print(f"[evaluate_fn] ERROR: Metrics collector not initialized for round {server_round}")
                    return None, {}
                
                # Get collected actions from config (passed from strategy.evaluate)
                # This is a workaround for Ray actor isolation
                collected_actions = config.get('_collected_actions', {})
                
                print(f"[evaluate_fn] Collector found: client_agents={len(collector.client_agents)}, "
                      f"client_actions={len(collector.client_actions)}, "
                      f"collected_actions_from_server={len(collected_actions)}, "
                      f"metrics_history={len(collector.metrics_history)}")
                
                # Get collected client metrics from config (passed from strategy.evaluate)
                collected_client_metrics = config.get('_collected_client_metrics', {})
                
                # Always create a metrics entry, even if we don't have agents
                # This ensures we have at least some data for visualization
                round_metrics = {
                    'round': server_round,
                    'client_metrics': {},
                    'server_metrics': {},
                }
                
                # Add client metrics from collected grid evaluations
                if collected_client_metrics:
                    # Aggregate client metrics to compute server_metrics
                    server_prior_logprob = None
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
                        if 'prior_logprob' in client_grid_metrics:
                            prior_lp = np.array(client_grid_metrics['prior_logprob'])
                            if server_prior_logprob is None:
                                server_prior_logprob = prior_lp.copy()
                            else:
                                server_prior_logprob = server_prior_logprob + prior_lp
                        
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
                        if server_prior_logprob is not None:
                            server_prior_logprob = server_prior_logprob / num_clients
                        if server_value is not None:
                            server_value = server_value / num_clients
                        if server_policy_density is not None:
                            server_policy_density = server_policy_density / num_clients
                    
                    # Build server_metrics
                    if server_prior_logprob is not None or server_value is not None:
                        server_metrics = {}
                        if server_prior_logprob is not None:
                            server_metrics['prior_logprob'] = server_prior_logprob.tolist()
                        if server_value is not None:
                            server_metrics['value'] = server_value.tolist()
                        if server_policy_density is not None:
                            server_metrics['policy_density'] = server_policy_density.tolist()
                        
                        # Compute FedGuide policy: π_FG ∝ π_fed_prior * exp(β * V_fed)
                        if server_prior_logprob is not None and server_value is not None:
                            beta = 5.0  # Coefficient for FedGuide policy
                            prior_lp = np.array(server_prior_logprob)
                            value = np.array(server_value)
                            log_pi_fg = prior_lp + beta * value
                            log_pi_fg = log_pi_fg - log_pi_fg.max()  # Normalize to avoid overflow
                            pi_fg = np.exp(log_pi_fg)
                            server_metrics['fedguide_policy_density'] = pi_fg.tolist()
                        
                        round_metrics['server_metrics'] = server_metrics
                        print(f"[evaluate_fn] Computed server_metrics from {num_clients} clients: {list(server_metrics.keys())}")
                    
                    print(f"[evaluate_fn] Added client_metrics from server: {len(collected_client_metrics)} clients")
                
                # Add client actions if available (prefer current round's actions from server)
                if collected_actions:
                    # Priority: Use current round's actions from server (not accumulated)
                    round_metrics['client_actions'] = {k: list(v) if isinstance(v, np.ndarray) else v 
                                                      for k, v in collected_actions.items()}
                    print(f"[evaluate_fn] Added actions from server metrics (round {server_round}): {len(collected_actions)} clients")
                elif collector.client_actions:
                    # Fallback: Use accumulated actions from collector (for backward compatibility)
                    round_metrics['client_actions'] = {k: list(v) if isinstance(v, np.ndarray) else v 
                                                      for k, v in collector.client_actions.items()}
                    print(f"[evaluate_fn] Added actions from collector (accumulated): {len(collector.client_actions)} clients")
                
                # We already have client_metrics and server_metrics from collected_client_metrics
                # So we don't need to call collect_round_metrics (which requires agents in Ray actors)
                # Just append the round_metrics we built
                collector.metrics_history.append(round_metrics)
                print(f"[evaluate_fn] Appended round_metrics to history (round {server_round})")
                
                print(f"[evaluate_fn] Metrics history now has {len(collector.metrics_history)} entries")
                
            except Exception as e:
                print(f"[evaluate_fn] ERROR: Failed to collect metrics for round {server_round}: {e}")
                import traceback
                traceback.print_exc()
            
            return None, {}
        
        return evaluate_fn
    
    # Create evaluate function
    evaluate_fn = make_evaluate_fn(
        collect_every=args.collect_metrics_every,
    )
    
    # Create strategy - enable evaluation
    strategy = FedGuideServer(
        fraction_fit=1.0,
        fraction_evaluate=1.0,  # Enable evaluation
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,  # Set minimum evaluate clients
        min_available_clients=args.num_clients,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        evaluate_fn=evaluate_fn,
    )
    
    config = ServerConfig(num_rounds=args.rounds)
    
    print(f"Starting FedGuide training:")
    print(f"  Environment: Bandit2D")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Steps per round: {args.n_steps}")
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        strategy=strategy,
        config=config,
        client_resources={"num_cpus": args.cpus_per_client},
    )
    
    print("\nTraining completed!")
    
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
                # Create a summary entry with all collected actions
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
                print(f"  Created summary entry, metrics_history now has {len(metrics_collector.metrics_history)} entries")
            else:
                print("  ERROR: No actions collected either!")
        
        metrics_collector.save("bandit2d_metrics.pkl")
        print(f"\nMetrics saved to {args.metrics_dir}/bandit2d_metrics.pkl")
        print(f"  Final metrics_history length: {len(metrics_collector.metrics_history)}")
        print("  To visualize, run:")
        print(f"    python scripts/envs/bandit2d/visualize_bandit2d.py --metrics_path {args.metrics_dir}/bandit2d_metrics.pkl")
    
    # Save training history for reward curve plotting
    os.makedirs(args.metrics_dir, exist_ok=True)
    history_path = os.path.join(args.metrics_dir, "training_history.pkl")
    
    # Debug: Check history contents
    print(f"\nTraining history debug info:")
    print(f"  history type: {type(history)}")
    if hasattr(history, 'losses_distributed'):
        print(f"  losses_distributed: {len(history.losses_distributed) if history.losses_distributed else 0} entries")
    if hasattr(history, 'metrics_distributed_fit'):
        print(f"  metrics_distributed_fit: {len(history.metrics_distributed_fit) if history.metrics_distributed_fit else 0} entries")
    if hasattr(history, 'metrics_centralized'):
        print(f"  metrics_centralized: {len(history.metrics_centralized) if history.metrics_centralized else 0} entries")
    if hasattr(history, '__dict__'):
        print(f"  history attributes: {list(history.__dict__.keys())}")
    
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\nTraining history saved to {history_path}")
    print("  To plot reward curves, run:")
    print(f"    python scripts/envs/bandit2d/plot_reward_curves.py \\")
    print(f"        --fedguide_history {history_path}")
    
    return history


if __name__ == "__main__":
    main()

