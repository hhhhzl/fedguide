"""
Run Centralized SAC baseline for D4RL environments (reacher, maze2d, antmaze, flow).

This script trains a central SAC agent on D4RL datasets.
"""

import argparse
import os
import sys
import pickle
import json
import numpy as np
import torch
import gymnasium as gym
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fedguide.baselines.sac.agent import SACAgent
from fedguide.baselines.sac.trainer import CentralSACTrainer
from fedguide.datasets.base import TransitionDataset


def convert_d4rl_to_transitions(obs, acts, rewards, terminals, next_obs):
    """Convert D4RL dataset format to trajectory format for TransitionDataset."""
    trajs = []
    current_traj = {
        's': [],
        'a': [],
        'r': [],
        's_next': [],
        'd': [],
    }
    
    for i in range(len(obs)):
        current_traj['s'].append(obs[i])
        current_traj['a'].append(acts[i])
        current_traj['r'].append(rewards[i])
        current_traj['s_next'].append(next_obs[i] if next_obs is not None else obs[i] if i+1 < len(obs) else obs[i])
        current_traj['d'].append(float(terminals[i]))
        
        # If terminal, end current trajectory
        if terminals[i] or (i + 1 == len(obs)):
            traj = {
                's': np.array(current_traj['s'], dtype=np.float32),
                'a': np.array(current_traj['a'], dtype=np.float32),
                'r': np.array(current_traj['r'], dtype=np.float32),
                's_next': np.array(current_traj['s_next'], dtype=np.float32),
                'd': np.array(current_traj['d'], dtype=np.float32),
            }
            trajs.append(traj)
            current_traj = {'s': [], 'a': [], 'r': [], 's_next': [], 'd': []}
    
    return trajs


def load_d4rl_data(env_name: str, n_clients: int = 1):
    """Load D4RL dataset and split into client datasets."""
    try:
        import d4rl
    except ImportError:
        raise ImportError("d4rl is required to load D4RL datasets. Please install it with: pip install d4rl")
    
    # Create environment to get dataset
    env = gym.make(env_name)
    dataset = env.get_dataset()
    
    obs = dataset['observations']
    acts = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    
    # Compute next_obs if not provided
    if 'next_observations' in dataset:
        next_obs = dataset['next_observations']
    else:
        next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
    
    # Convert to trajectories
    all_trajs = convert_d4rl_to_transitions(obs, acts, rewards, terminals, next_obs)
    
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
    
    return transition_datasets, env


def main():
    parser = argparse.ArgumentParser(description="Centralized SAC for D4RL environments")
    
    # Environment args
    parser.add_argument("--env_name", type=str, required=True,
                       help="D4RL environment name (e.g., 'reacher-medium-v2', 'maze2d-umaze-v1')")
    parser.add_argument("--num_clients", type=int, default=1,
                       help="Number of clients (for data splitting)")
    
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
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        env_short = args.env_name.replace('-', '_').replace('/', '_')
        args.output_dir = f"./model/policy/{env_short}/sac"

    if args.metrics_dir is None:
        env_short = args.env_name.replace('-', '_').replace('/', '_')
        args.output_dir = f"./metrics/{env_short}/sac"
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print(f"\nLoading D4RL dataset: {args.env_name}...")
    try:
        transition_datasets, env = load_d4rl_data(
            env_name=args.env_name,
            n_clients=args.num_clients
        )
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
        
        metrics = trainer.train_one_round()
        metrics['round'] = round_num
        history.append(metrics)
        
        print(f"\n  [Round {round_num}] Metrics:")
        print(f"    Actor Loss: {metrics['train/loss/actor']:.4f}")
        print(f"    Critic Loss: {metrics['train/loss/critic']:.4f}")
        print(f"    Q Value: {metrics['train/q_value']:.4f}")
        if 'eval/return' in metrics:
            print(f"    Eval Return: {metrics['eval/return']:.4f}")
        
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


