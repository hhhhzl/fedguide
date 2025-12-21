"""
Run Centralized SAC baseline for Minari environments.

This script trains a central SAC agent on Minari datasets.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import gymnasium as gym
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fedguide.baselines.sac.agent import SACAgent
from fedguide.baselines.sac.trainer import CentralSACTrainer
from fedguide.datasets.base import TransitionDataset
from fedguide.datasets.minari_loader import load_minari_dataset
from fedguide.utils.seeds import set_all_seeds


def convert_minari_to_transitions(trajs: List[dict]):
    """Convert Minari trajectory format to TransitionDataset format."""
    # Minari trajectories already have the right format: s, a, r, s_next, d
    # Just need to ensure they're in the right format for TransitionDataset
    return trajs


def load_minari_data(dataset_id: str, n_clients: int = 1, download: bool = True, env_name: str = None):
    """Load Minari dataset and split into client datasets."""
    try:
        import minari
    except ImportError:
        raise ImportError("minari is required to load Minari datasets. Please install it with: pip install minari")
    
    # Load Minari dataset
    print(f"Loading Minari dataset: {dataset_id}")
    all_trajs = load_minari_dataset(dataset_id, download=download, flatten_obs=True)
    
    # Split into clients (simple split for now)
    if n_clients == 1:
        client_trajs = [all_trajs]
    else:
        trajs_per_client = len(all_trajs) // n_clients
        client_trajs = [all_trajs[i*trajs_per_client:(i+1)*trajs_per_client] 
                       for i in range(n_clients)]
        # Add remaining trajectories to last client
        if len(all_trajs) % n_clients > 0:
            client_trajs[-1].extend(all_trajs[n_clients*trajs_per_client:])
    
    # Convert to TransitionDataset
    transition_datasets = [TransitionDataset(trajs) for trajs in client_trajs]
    
    # Create environment from dataset or provided env_name
    if env_name:
        print(f"Using provided environment name: {env_name}")
        env = gym.make(env_name)
    else:
        try:
            ds = minari.load_dataset(dataset_id, download=False)
            # Try to get environment from dataset
            if hasattr(ds, 'env_spec') and hasattr(ds.env_spec, 'id'):
                env_name = ds.env_spec.id
                env = gym.make(env_name)
            elif hasattr(ds, 'env'):
                # Some minari datasets have env attribute
                env = ds.env
            else:
                raise AttributeError("Dataset does not have env_spec.id or env attribute")
        except Exception as e:
            print(f"Warning: Could not create environment from dataset: {e}")
            print("Trying to infer environment from dataset_id...")
            # Try to infer environment from dataset_id
            dataset_id_lower = dataset_id.lower()
            if "pointmaze" in dataset_id_lower:
                try:
                    env = gym.make("PointMaze_UMaze-v3")
                except:
                    env = gym.make("pointmaze-umaze-v1")
            elif "maze2d" in dataset_id_lower:
                env = gym.make("maze2d-umaze-v1")
            elif "antmaze" in dataset_id_lower:
                env = gym.make("antmaze-umaze-v0")
            else:
                raise ValueError(f"Could not infer environment from dataset_id: {dataset_id}. "
                               f"Please specify env_name in config or ensure dataset has env_spec.")
    
    return transition_datasets, env


def main():
    parser = argparse.ArgumentParser(description="Centralized SAC for Minari environments")
    
    # Environment args
    parser.add_argument("--dataset_id", type=str, required=True,
                       help="Minari dataset ID (e.g., 'D4RL/pointmaze/medium-v2', 'D4RL/maze2d/umaze-v1')")
    parser.add_argument("--env_name", type=str, default=None,
                       help="Optional: Environment name (if dataset doesn't provide it)")
    parser.add_argument("--num_clients", type=int, default=1,
                       help="Number of clients (for data splitting)")
    parser.add_argument("--download", action="store_true", default=True,
                       help="Download dataset if not found locally")
    
    # Training args
    parser.add_argument("--rounds", type=int, default=100,
                       help="Number of training rounds")
    parser.add_argument("--update_steps", type=int, default=1000,
                       help="Number of update steps per round")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for training")
    
    # Agent args
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension for networks")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                       help="Soft update coefficient")
    parser.add_argument("--alpha", type=float, default=0.2,
                       help="Temperature parameter")
    parser.add_argument("--action_std", type=float, default=0.1,
                       help="Action distribution standard deviation")
    
    # Evaluation args
    parser.add_argument("--eval_episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save models")
    parser.add_argument("--metrics_dir", type=str, default=None,
                       help="Directory to save metrics")
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save results every N rounds")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use ('cpu', 'cuda', or 'auto')")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Rendering args
    parser.add_argument("--render_eval", action="store_true",
                       help="Render evaluation episodes")
    parser.add_argument("--render_mode", type=str, default="video",
                       choices=["human", "rgb_array", "video"],
                       help="Rendering mode: 'human' (display), 'rgb_array' (collect frames), 'video' (save video)")
    parser.add_argument("--render_save_dir", type=str, default=None,
                       help="Directory to save rendered videos (if render_mode='video')")
    parser.add_argument("--render_every_n_rounds", type=int, default=10,
                       help="Render every N rounds (0 = only last round, -1 = all rounds)")
    parser.add_argument("--render_episodes", type=int, default=1,
                       help="Number of episodes to render per round")
    
    args = parser.parse_args()
    
    # Set random seeds FIRST, before any random operations
    print(f"Setting random seed: {args.seed}")
    set_all_seeds(args.seed)
    
    # Set output directory
    if args.output_dir is None:
        env_short = args.dataset_id.replace('/', '_').replace('-', '_')
        args.output_dir = f"./model/policy/{env_short}/sac"

    if args.metrics_dir is None:
        env_short = args.dataset_id.replace('/', '_').replace('-', '_')
        args.metrics_dir = f"./metrics/{env_short}/sac"
    
    # Set render save directory
    if args.render_eval and args.render_save_dir is None:
        env_short = args.dataset_id.replace('/', '_').replace('-', '_')
        args.render_save_dir = f"./videos/{env_short}/sac"
    
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
    print(f"\nLoading Minari dataset: {args.dataset_id}...")
    try:
        transition_datasets, env = load_minari_data(
            dataset_id=args.dataset_id,
            n_clients=args.num_clients,
            download=args.download,
            env_name=args.env_name
        )
        # Ensure environment is also seeded
        set_all_seeds(args.seed, env)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    print(f"Loaded {len(transition_datasets)} client datasets")
    for i, ds in enumerate(transition_datasets):
        print(f"  Client {i}: {len(ds)} transitions")
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, 'shape'):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    # Get action bounds
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        action_low = env.action_space.low
        action_high = env.action_space.high
    else:
        action_low = None
        action_high = None
    
    print(f"\nEnvironment info:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    if action_low is not None and action_high is not None:
        print(f"  Action bounds: [{action_low}, {action_high}]")
    else:
        print(f"  Action bounds: Not available (using defaults)")
    
    # Create SAC agent
    print(f"\nCreating SAC agent...")
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        device=device,
        action_low=action_low,
        action_high=action_high,
        action_std=args.action_std,
    )
    print("Agent created successfully")
    
    # Create trainer
    print(f"\nCreating centralized trainer...")
    trainer = CentralSACTrainer(
        agent=agent,
        datasets=transition_datasets,
        env=env,
        batch_size=args.batch_size,
        update_steps=args.update_steps,
        gamma=args.gamma,
        eval_episodes=args.eval_episodes,
        device=device,
        render_eval=args.render_eval,
        render_mode=args.render_mode,
        render_save_dir=args.render_save_dir,
        render_every_n_rounds=args.render_every_n_rounds,
        render_episodes=args.render_episodes,
    )
    
    # Training loop
    print(f"\nStarting training for {args.rounds} rounds...")
    print(f"  Update steps per round: {args.update_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total transitions: {trainer.total_transitions}")
    
    history = []
    
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{args.rounds}")
        print(f"{'='*60}")
        
        # Handle render_every_n_rounds == 0 (only render last round)
        if args.render_eval and args.render_every_n_rounds == 0 and round_num == args.rounds:
            # Temporarily set render_every_n_rounds to -1 for last round
            trainer.render_every_n_rounds = -1
        
        metrics = trainer.train_one_round(round_num=round_num)
        
        # Restore original value
        if args.render_eval and args.render_every_n_rounds == 0:
            trainer.render_every_n_rounds = 0
        metrics['round'] = round_num
        history.append(metrics)
        
        print(f"\n  [Round {round_num}] Metrics:")
        print(f"    Actor Loss: {metrics['train/loss/actor']:.4f}")
        print(f"    Critic Loss: {metrics['train/loss/critic']:.4f}")
        print(f"    Q Value: {metrics['train/q_value']:.4f}")
        if 'eval/return' in metrics:
            print(f"    Eval Return (deterministic): {metrics['eval/return']:.4f}")
        if 'eval/return_stochastic_mean' in metrics:
            print(f"    Eval Return (stochastic mean): {metrics['eval/return_stochastic_mean']:.4f}")
            print(f"    Eval Return (stochastic max): {metrics['eval/return_stochastic_max']:.4f}")
        
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
        }, f)
    
    print(f"\nTraining completed! Results saved to {final_path}")
    return history


if __name__ == "__main__":
    main()

