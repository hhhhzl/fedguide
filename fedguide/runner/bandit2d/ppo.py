"""
Run Centralized PPO baseline for 2D Bandit environment.

This module trains a central PPO agent on mixed data from multiple clients.
No federated aggregation is performed - pure centralized training.
"""

import argparse
import os
import sys
import pickle
import json
import yaml
import numpy as np
import torch
from typing import List, Dict, Any, Union

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from fedguide.baselines.ppo.agent import PPOAgent
from fedguide.baselines.ppo.trainer import CentralPPOTrainer
from fedguide.envs.bandit2d import Bandit2D
from fedguide.datasets.base import TransitionDataset, TrajectoryDataset
from fedguide.utils.seeds import set_all_seeds
from scripts.generate_data.generate_bandit2d_data import generate_bandit2d_datasets


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


def normalize_seed(seed: Union[int, List[int]]) -> List[int]:
    """Convert seed to list format."""
    if isinstance(seed, int):
        return [seed]
    elif isinstance(seed, list):
        return seed
    else:
        raise ValueError(f"seed must be int or list of ints, got {type(seed)}")


def convert_trajectory_to_transitions(
    trajectory_dataset: TrajectoryDataset,
    env: Bandit2D
) -> List[dict]:
    """
    Convert TrajectoryDataset to trajectory format for TransitionDataset.
    
    For Bandit2D, each sample is a single-step transition:
    - state = action (bandit property)
    - action = action
    - reward = computed from action
    - next_state = action (same as state)
    - done = True (bandit always terminates after one step)
    
    Args:
        trajectory_dataset: TrajectoryDataset with obs and actions
        env: Bandit2D environment for computing rewards
    
    Returns:
        List of trajectory dictionaries
    """
    trajs = []
    
    # Each sample in TrajectoryDataset is a single (obs, action) pair
    # For bandit, we treat each as a single-step trajectory
    for i in range(len(trajectory_dataset)):
        sample = trajectory_dataset[i]  # Concatenated [obs, action]
        action = sample[2:]  # Last 2 elements are action (assuming 2D obs + 2D action)
        state = sample[:2]  # First 2 elements are obs (which equals action for bandit)
        
        # Compute reward
        reward = env.compute_reward(action)
        
        # Create single-step trajectory
        traj = {
            's': np.array([state], dtype=np.float32),  # [1, 2]
            'a': np.array([action], dtype=np.float32),  # [1, 2]
            'r': np.array([reward], dtype=np.float32),  # [1]
            's_next': np.array([action], dtype=np.float32),  # [1, 2] (next state = action)
            'd': np.array([1.0], dtype=np.float32),  # [1] (always done)
        }
        trajs.append(traj)
    
    return trajs


def load_bandit2d_datasets(
    data_dir: str = "data/bandit2d",
    n_clients: int = 4
) -> List[TrajectoryDataset]:
    """
    Load bandit2d datasets from disk or generate if not found.
    
    Args:
        data_dir: Directory containing data
        n_clients: Number of clients
    
    Returns:
        List of TrajectoryDataset objects
    """
    metadata_path = os.path.join(data_dir, "metadata.json")
    
    # Check if data exists
    if os.path.exists(metadata_path):
        print(f"Loading datasets from {data_dir}")
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # For now, we'll generate data on the fly
        # In a full implementation, you'd load from saved files
        print("Note: Generating data on the fly (full loading not implemented)")
        
        datasets, mu = generate_bandit2d_datasets(
            K=metadata.get('K', 4),
            n_clients=n_clients,
            samples_per_client=metadata.get('samples_per_client', 1000),
            sigma=metadata.get('sigma', 0.2),
            local_radius=metadata.get('local_radius', 0.3),
            seed=metadata.get('seed', 42),
        )
        return datasets
    else:
        # Generate new data
        print(f"Data not found at {data_dir}, generating new data...")
        
        datasets, mu = generate_bandit2d_datasets(
            K=4,
            n_clients=n_clients,
            samples_per_client=1000,
            sigma=0.2,
            local_radius=0.3,
            seed=42,
        )
        
        # Save metadata
        os.makedirs(data_dir, exist_ok=True)
        metadata = {
            "K": 4,
            "n_clients": n_clients,
            "samples_per_client": 1000,
            "mu": mu.tolist(),
            "sigma": 0.2,
            "local_radius": 0.3,
            "seed": 42
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return datasets


def main():
    """Main entry point for Bandit2D PPO training."""
    parser = argparse.ArgumentParser(description="Centralized PPO for Bandit2D")
    
    # Config file argument
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file")
    
    # Data args
    parser.add_argument("--num_clients", type=int, default=4,
                       help="Number of clients")
    parser.add_argument("--data_dir", type=str, default="data/bandit2d",
                       help="Directory containing bandit2d data")
    
    # Training args
    parser.add_argument("--rounds", type=int, default=100,
                       help="Number of training rounds")
    parser.add_argument("--steps_per_round", type=int, default=2000,
                       help="Number of environment steps to collect per round (on-policy)")
    parser.add_argument("--update_epochs", type=int, default=4,
                       help="Number of epochs per update")
    parser.add_argument("--minibatch_size", type=int, default=None,
                       help="Minibatch size for PPO updates (if None, use full batch)")
    
    # Agent args
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension for networks")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                       help="PPO clipping epsilon")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5,
                       help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--action_std", type=float, default=0.1,
                       help="Initial standard deviation for action distribution (if learnable_std=False)")
    parser.add_argument("--learnable_std", action="store_true", default=True,
                       help="Use learnable action std (recommended)")
    parser.add_argument("--no_learnable_std", dest="learnable_std", action="store_false",
                       help="Use fixed action std")
    
    # Environment args
    parser.add_argument("--K", type=int, default=4,
                       help="Number of peaks in bandit")
    parser.add_argument("--sigma", type=float, default=0.2,
                       help="Standard deviation for reward function")
    
    # Evaluation args
    parser.add_argument("--eval_episodes", type=int, default=50,
                       help="Number of episodes for evaluation (increased for more stable evaluation)")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default=f"./model/policy/bandit2d/ppo",
                       help="Directory to save results")
    parser.add_argument("--metrics_dir", type=str, default=f"./metrics/bandit2d/ppo",
                       help="Directory to save results")
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save results every N rounds")
    
    # Logprob collection args
    parser.add_argument("--collect_logprob", action="store_true", default=False,
                       help="Collect policy logprob distribution on grid (for visualization)")
    parser.add_argument("--no_collect_logprob", dest="collect_logprob", action="store_false",
                       help="Disable logprob collection")
    parser.set_defaults(collect_logprob=True)  # Default to True
    parser.add_argument("--logprob_grid_size", type=int, default=200,
                       help="Grid size for logprob evaluation (grid_size x grid_size)")
    parser.add_argument("--logprob_bounds", type=float, nargs=2, default=[-1.5, 1.5],
                       help="Bounds for logprob grid evaluation [min, max]")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use ('cpu', 'cuda', or 'auto')")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Parse arguments to get config path first
    temp_args, _ = parser.parse_known_args()
    
    # Load config file if provided, and update parser defaults
    if temp_args.config:
        print(f"Loading configuration from: {temp_args.config}")
        config = load_config(temp_args.config)
        print(f"Configuration loaded successfully")
        
        # Update parser defaults with config values
        config_mapping = {
            'num_clients': 'num_clients',
            'data_dir': 'data_dir',
            'rounds': 'rounds',
            'steps_per_round': 'steps_per_round',
            'update_epochs': 'update_epochs',
            'minibatch_size': 'minibatch_size',
            'hidden_dim': 'hidden_dim',
            'lr': 'lr',
            'gamma': 'gamma',
            'clip_eps': 'clip_eps',
            'gae_lambda': 'gae_lambda',
            'entropy_coef': 'entropy_coef',
            'value_coef': 'value_coef',
            'max_grad_norm': 'max_grad_norm',
            'action_std': 'action_std',
            'learnable_std': 'learnable_std',
            'K': 'K',
            'sigma': 'sigma',
            'eval_episodes': 'eval_episodes',
            'output_dir': 'output_dir',
            'metrics_dir': 'metrics_dir',
            'save_every': 'save_every',
            'collect_logprob': 'collect_logprob',
            'logprob_grid_size': 'logprob_grid_size',
            'logprob_bounds': 'logprob_bounds',
            'device': 'device',
            'seed': 'seed',
        }
        
        # Set defaults from config
        for config_key, arg_key in config_mapping.items():
            if config_key in config:
                config_value = config[config_key]
                # Special handling for seed (if it's a list, use first element)
                if config_key == 'seed' and isinstance(config_value, list):
                    if len(config_value) > 0:
                        config_value = config_value[0]
                    else:
                        config_value = 42
                # Special handling for collect_logprob (boolean flag)
                if config_key == 'collect_logprob':
                    # For boolean flags, we need to update the default
                    parser.set_defaults(**{arg_key: config_value})
                    continue
                # Find the action in parser and update its default
                for action in parser._actions:
                    if action.dest == arg_key and not isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                        action.default = config_value
                        break
    
    # Now parse all arguments (command line will override config defaults)
    args = parser.parse_args()
    
    # Set random seeds FIRST, before any random operations
    print(f"Setting random seed: {args.seed}")
    set_all_seeds(args.seed)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    # Load datasets
    print(f"\nLoading datasets from {args.data_dir}...")
    trajectory_datasets = load_bandit2d_datasets(
        data_dir=args.data_dir,
        n_clients=args.num_clients
    )
    print(f"Loaded {len(trajectory_datasets)} client datasets")
    
    # Create environment for reward computation and evaluation
    env = Bandit2D(K=args.K, sigma=args.sigma, seed=args.seed)
    # Ensure environment is also seeded
    set_all_seeds(args.seed, env)
    
    # Convert TrajectoryDataset to TransitionDataset format
    print("\nConverting datasets to transition format...")
    transition_datasets = []
    for client_id, traj_dataset in enumerate(trajectory_datasets):
        trajs = convert_trajectory_to_transitions(traj_dataset, env)
        transition_dataset = TransitionDataset(trajs)
        transition_datasets.append(transition_dataset)
        print(f"Client {client_id}: {len(transition_dataset)} transitions")
    
    # Create PPO agent
    print(f"\nCreating PPO agent...")
    print(f"  State dim: 2, Action dim: 2")
    print(f"  Hidden dim: {args.hidden_dim}, LR: {args.lr}")
    print(f"  Clip eps: {args.clip_eps}, GAE lambda: {args.gae_lambda}")
    print(f"  Device: {device}")
    import sys
    sys.stdout.flush()
    
    try:
        agent = PPOAgent(
            state_dim=2,
            action_dim=2,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            clip_eps=args.clip_eps,
            gae_lambda=args.gae_lambda,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            action_std=args.action_std,
            learnable_std=args.learnable_std,
            device=device,
        )
        print("Agent created successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Create trainer
    print(f"\nCreating centralized trainer...")
    trainer = CentralPPOTrainer(
        agent=agent,
        datasets=transition_datasets,  # Kept for compatibility, but not used in on-policy
        env=env,
        steps_per_round=args.steps_per_round,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        eval_episodes=args.eval_episodes,
        device=device,
    )
    
    # Training loop
    print(f"\nStarting training for {args.rounds} rounds...")
    print(f"  Steps per round: {args.steps_per_round}")
    print(f"  Update epochs: {args.update_epochs}")
    print(f"  Minibatch size: {args.minibatch_size if args.minibatch_size is not None else 'full batch'}")
    print(f"  Learnable std: {args.learnable_std}")
    
    history = []
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{args.rounds}")
        print(f"{'='*60}")
        import sys
        sys.stdout.flush()
        
        # Train one round
        print(f"  [Round {round_num}] Starting training...", flush=True)
        metrics = trainer.train_one_round(round_num=round_num)
        metrics['round'] = round_num
        
        # Store history
        history.append(metrics)
        
        # Print progress (always print, not just every 10 rounds)
        print(f"\n  [Round {round_num}] Training completed:")
        print(f"    Loss: {metrics['loss']:.4f}")
        print(f"    Policy Loss: {metrics['train/loss/actor']:.4f}")
        print(f"    Value Loss: {metrics['train/loss/critic']:.4f}")
        print(f"    Entropy: {metrics['train/entropy']:.4f}")
        print(f"    Returns mean: {metrics['train/returns_mean']:.4f}")
        print(f"    V mean: {metrics['train/V_mean']:.4f}")
        if 'eval/return' in metrics:
            print(f"    Eval Return (deterministic): {metrics['eval/return']:.4f}")
        if 'eval/return_stochastic_mean' in metrics:
            print(f"    Eval Return (stochastic mean): {metrics['eval/return_stochastic_mean']:.4f}")
            print(f"    Eval Return (stochastic max): {metrics['eval/return_stochastic_max']:.4f}")
        sys.stdout.flush()
        
        # Collect policy logprob distribution (for visualization)
        if args.collect_logprob and (round_num % args.save_every == 0 or round_num == args.rounds or round_num == 1):
            try:
                print(f"  [Round {round_num}] Computing policy logprob distribution on grid...", flush=True)
                policy_metrics = trainer.evaluate_policy_logprob_on_grid(
                    grid_size=args.logprob_grid_size,
                    bounds=tuple(args.logprob_bounds)
                )
                if policy_metrics is not None:
                    # Store in metrics (convert to list for JSON serialization)
                    # Prefer density for visualization (0-1 range)
                    metrics['policy/density_grid'] = policy_metrics['policy_density'].tolist()
                    metrics['policy/logprob_grid'] = policy_metrics['policy_logprob'].tolist()
                    metrics['policy/grid_X'] = policy_metrics['X'].tolist()
                    metrics['policy/grid_Y'] = policy_metrics['Y'].tolist()
                    if 'action_dims' in policy_metrics:
                        metrics['policy/action_dims'] = policy_metrics['action_dims']
                    if 'action_dim' in policy_metrics:
                        metrics['policy/action_dim'] = int(policy_metrics['action_dim'])
                    print(f"  [Round {round_num}] Policy logprob distribution computed (action_dim={policy_metrics.get('action_dim', 'unknown')})", flush=True)
                else:
                    print(f"  [Round {round_num}] Policy logprob visualization skipped (unsuitable action space)", flush=True)
            except Exception as e:
                print(f"  Warning: Failed to compute policy logprob grid: {e}", flush=True)
                import traceback
                traceback.print_exc()
        
        # Save checkpoint
        if round_num % args.save_every == 0 or round_num == args.rounds:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_round_{round_num}.pkl")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'round': round_num,
                    'history': history,
                    'agent_state': agent.actor.state_dict(),
                    'args': vars(args),
                }, f)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Save final results
    final_path = os.path.join(args.metrics_dir, "training_history.pkl")
    with open(final_path, 'wb') as f:
        pickle.dump({
            'history': history,
            'args': vars(args),
            'final_metrics': history[-1] if history else {},
        }, f)
    
    print(f"\nTraining completed!")
    print(f"Results saved to {final_path}")
    
    # Print final statistics
    if history:
        final_metrics = history[-1]
        print(f"\nFinal Metrics:")
        print(f"  Loss: {final_metrics['loss']:.4f}")
        print(f"  Policy Loss: {final_metrics['train/loss/actor']:.4f}")
        print(f"  Value Loss: {final_metrics['train/loss/critic']:.4f}")
        print(f"  Entropy: {final_metrics['train/entropy']:.4f}")
        print(f"  Returns mean: {final_metrics['train/returns_mean']:.4f}")
        print(f"  V mean: {final_metrics['train/V_mean']:.4f}")
        if 'eval/return' in final_metrics:
            print(f"  Eval Return: {final_metrics['eval/return']:.4f}")
    
    return history


if __name__ == "__main__":
    main()

