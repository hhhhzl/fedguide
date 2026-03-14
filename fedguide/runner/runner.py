"""
Unified runner that supports all algorithms and environments.

This runner uses factory functions registered in a registry to dynamically create
agents, trainers, and environments based on configuration. It eliminates the need
for separate runner files for each (environment, algorithm) combination.
"""

import argparse
import os
import sys
import yaml
import torch
import pickle
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from fedguide.utils.seeds import set_all_seeds
from fedguide.runner.factories import get_registry


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        # Try relative to configs directory
        alt_path = os.path.join(_project_root, "configs", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _create_datasets(env_type: str, config: Dict[str, Any], seed: int):
    """Create datasets for the environment if needed."""
    # For on-policy methods (PPO, SAC), datasets might not be needed
    # For offline methods, load datasets here
    # This can be customized per environment type using hooks
    return []


def _save_checkpoint(agent, history: List, config: Dict[str, Any], round_num: int, output_dir: str):
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_round_{round_num}.pkl")
    
    agent_state = None
    if hasattr(agent, 'actor'):
        agent_state = agent.actor.state_dict()
    elif hasattr(agent, 'policy'):
        agent_state = agent.policy.state_dict()
    elif hasattr(agent, 'state_dict'):
        agent_state = agent.state_dict()
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'round': round_num,
            'history': history,
            'agent_state': agent_state,
            'args': config,
        }, f)
    print(f"  Saved checkpoint to {checkpoint_path}")


def _save_final_results(history: List, config: Dict[str, Any], env_type: str, algorithm: str):
    """Save final training results."""
    metrics_dir = config.get('metrics_dir', './metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    final_path = os.path.join(metrics_dir, "training_history.pkl")
    with open(final_path, 'wb') as f:
        pickle.dump({
            'history': history,
            'args': config,
            'final_metrics': history[-1] if history else {},
        }, f)
    
    print(f"\nTraining completed!")
    print(f"Results saved to {final_path}")
    
    # Print final statistics
    if history:
        final_metrics = history[-1]
        print(f"\nFinal Metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")


def _run_centralized_training(env_type: str, algorithm: str, config: Dict[str, Any], args, device: str):
    """Run centralized training (PPO, SAC, etc.)."""
    registry = get_registry()
    
    # Set seeds
    seed = config.get('seed', args.seed or 42)
    set_all_seeds(seed)
    
    # Create environment
    print(f"\nCreating {env_type} environment...")
    env = registry.create_env(env_type, config, seed=seed)
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = getattr(env.action_space, 'low', None)
    action_high = getattr(env.action_space, 'high', None)
    
    print(f"Environment dimensions: obs={obs_dim}, action={action_dim}")
    
    # Create agent
    print(f"\nCreating {algorithm.upper()} agent...")
    agent_config = {
        'state_dim': obs_dim,
        'action_dim': action_dim,
        'device': device,
        'action_low': action_low,
        'action_high': action_high,
        **config
    }
    agent = registry.create_agent(algorithm, env, agent_config)
    
    # Create datasets (if needed for offline training)
    datasets = _create_datasets(env_type, config, seed)
    
    # Create trainer
    print(f"\nCreating {algorithm.upper()} trainer...")
    trainer_config = {
        'device': device,
        **config
    }
    trainer = registry.create_trainer(algorithm, agent, env, datasets, trainer_config)
    
    # Setup hooks for environment-specific logic (e.g., bandit2d metrics)
    hooks = registry.get_hooks(env_type, algorithm)
    
    # Initialize hooks
    if hooks:
        for hook in hooks:
            if hasattr(hook, 'on_train_start'):
                hook.on_train_start(trainer, agent, env, config)
    
    # Run training loop
    rounds = config.get('rounds', 100)
    history = []
    save_every = config.get('save_every', 10)
    
    print(f"\n{'='*60}")
    print(f"Starting {algorithm.upper()} Training")
    print(f"{'='*60}")
    print(f"Rounds: {rounds}")
    print(f"Steps per round: {config.get('steps_per_round', 2000)}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    for round_num in range(1, rounds + 1):
        # Train one round
        metrics = trainer.train_one_round(round_num=round_num)
        metrics['round'] = round_num
        history.append(metrics)
        
        # Call hooks
        if hooks:
            for hook in hooks:
                if hasattr(hook, 'on_round_end'):
                    hook.on_round_end(round_num, metrics, trainer, agent, env, config)
        
        # Print progress
        if round_num % 10 == 0 or round_num == 1:
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{rounds}")
            print(f"{'='*60}")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not key.startswith('policy/'):
                    print(f"  {key}: {value:.4f}")
        
        # Collect policy logprob distribution if requested
        if config.get('collect_logprob', False) and (round_num % save_every == 0 or round_num == rounds or round_num == 1):
            if hasattr(trainer, 'evaluate_policy_logprob_on_grid'):
                try:
                    print(f"  [Round {round_num}] Computing policy logprob distribution on grid...", flush=True)
                    grid_size = config.get('logprob_grid_size', 200)
                    bounds = config.get('logprob_bounds', [-1.5, 1.5])
                    action_dims = config.get('logprob_action_dims')
                    
                    policy_metrics = trainer.evaluate_policy_logprob_on_grid(
                        grid_size=grid_size,
                        bounds=tuple(bounds),
                        action_dims=action_dims
                    )
                    if policy_metrics is not None:
                        metrics['policy/density_grid'] = policy_metrics['policy_density'].tolist()
                        metrics['policy/logprob_grid'] = policy_metrics['policy_logprob'].tolist()
                        metrics['policy/grid_X'] = policy_metrics['X'].tolist()
                        metrics['policy/grid_Y'] = policy_metrics['Y'].tolist()
                        if 'action_dims' in policy_metrics:
                            metrics['policy/action_dims'] = policy_metrics['action_dims']
                except Exception as e:
                    print(f"  Warning: Failed to compute policy logprob grid: {e}", flush=True)
        
        # Save checkpoint
        if round_num % save_every == 0 or round_num == rounds:
            output_dir = config.get('output_dir', './model/policy')
            _save_checkpoint(agent, history, config, round_num, output_dir)
    
    # Finalize hooks
    if hooks:
        for hook in hooks:
            if hasattr(hook, 'on_train_end'):
                hook.on_train_end(history, trainer, agent, env, config)
    
    # Save final results
    _save_final_results(history, config, env_type, algorithm)
    
    # Cleanup
    if hasattr(env, 'close'):
        env.close()
    
    return history


def _run_federated_training(env_type: str, algorithm: str, config: Dict[str, Any], args, device: str):
    """Run federated training (FedGuide, FedKL, etc.)."""
    import flwr as fl
    from flwr.server import ServerConfig
    
    registry = get_registry()
    
    # Set seeds
    seed = config.get('seed', args.seed or 42)
    set_all_seeds(seed)
    
    # Setup hooks for environment-specific logic (e.g., bandit2d metrics)
    hooks = registry.get_hooks(env_type, algorithm)
    metrics_collector = None
    
    # Deterministic client ID mapping for heterogeneous Bandit2D (Flower VCE uses long int cids)
    metrics_dir = config.get('metrics_dir', './metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    cid_mapping_file = os.path.abspath(os.path.join(metrics_dir, ".cid_mapping.json"))
    from fedguide.utils.client_id_mapping import clear_mapping_file
    clear_mapping_file(cid_mapping_file)
    config['cid_mapping_file'] = cid_mapping_file
    
    # Initialize hooks and get metrics collector if available
    if hooks:
        for hook in hooks:
            if hasattr(hook, 'on_federated_start'):
                metrics_collector = hook.on_federated_start(config)
    
    # Create federated client function
    print(f"\nCreating {algorithm.upper()} federated client function...")
    client_fn = registry.create_federated_client_fn(
        algorithm, 
        config, 
        env_type=env_type, 
        device=device,
        metrics_collector=metrics_collector
    )
    
    # Create federated server strategy
    print(f"\nCreating {algorithm.upper()} federated server strategy...")
    evaluate_fn = None
    if hooks:
        for hook in hooks:
            if hasattr(hook, 'create_evaluate_fn'):
                evaluate_fn = hook.create_evaluate_fn(config, metrics_collector, algorithm)
    
    server = registry.create_federated_server(
        algorithm, 
        config,
        evaluate_fn=evaluate_fn
    )
    
    # Create server config
    rounds = config.get('rounds', 60)
    num_clients = config.get('num_clients', 4)
    server_config = ServerConfig(num_rounds=rounds)
    
    print(f"\n{'='*60}")
    print(f"Starting {algorithm.upper()} Federated Training")
    print(f"{'='*60}")
    print(f"Environment: {env_type}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {rounds}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Run federated simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        strategy=server,
        config=server_config,
        client_resources={"num_cpus": config.get('cpus_per_client', 2)},
    )
    
    # Finalize hooks
    if hooks:
        for hook in hooks:
            if hasattr(hook, 'on_federated_end'):
                hook.on_federated_end(history, metrics_collector, config, algorithm)
    
    # Save results
    _save_federated_results(history, metrics_collector, config, env_type, algorithm)
    
    return history


def _save_federated_results(history: Any, metrics_collector: Any, config: Dict[str, Any], env_type: str, algorithm: str):
    """Save federated training results."""
    metrics_dir = config.get('metrics_dir', './metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save metrics collector if available
    if metrics_collector is not None:
        try:
            if hasattr(metrics_collector, 'save'):
                metrics_collector.save("bandit2d_metrics.pkl")
                print(f"\nMetrics saved to {metrics_dir}/bandit2d_metrics.pkl")
        except Exception as e:
            print(f"Warning: Failed to save metrics collector: {e}")
    
    # Save training history
    history_path = os.path.join(metrics_dir, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\nTraining history saved to {history_path}")
    print(f"\n✅ Federated training completed successfully!")


def run_training(config: Dict[str, Any], args=None, device: Optional[str] = None):
    """
    Unified training function that works for all algorithms and environments.
    
    Args:
        config: Configuration dictionary
        args: Optional argparse args
        device: Optional device override
    """
    # Determine environment type and algorithm
    env_type = config.get('env_type', 'bandit2d')
    algorithm = config.get('algorithm', 'ppo')
    
    # Set device
    if device is None:
        device = config.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"Environment: {env_type}, Algorithm: {algorithm}, Device: {device}")
    print(f"{'='*80}")
    
    # Check if it's a federated algorithm
    federated_algorithms = {'fedguide', 'fedkl', 'fmarl', 'fedrl', 'fedrep', 'fedmomentum'}
    is_federated = algorithm.lower() in federated_algorithms
    
    if is_federated:
        return _run_federated_training(env_type, algorithm, config, args, device)
    else:
        return _run_centralized_training(env_type, algorithm, config, args, device)


def main():
    """Main entry point for unified runner."""
    parser = argparse.ArgumentParser(description="Unified runner for all algorithms and environments")
    
    # Config file
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    
    # Common args (can override config)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        alt_path = os.path.join(_project_root, "configs", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Override with command-line args
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['device'] = args.device
    
    # Run training
    run_training(config, args)


if __name__ == "__main__":
    main()

