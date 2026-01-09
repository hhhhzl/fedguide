"""
Run Centralized PPO baseline for D4RL environments (reacher, maze2d, antmaze, flow).

This script trains a central PPO agent on D4RL environments using on-policy rollouts.
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

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from fedguide.baselines.ppo.agent import PPOAgent
from fedguide.baselines.ppo.trainer import CentralPPOTrainer
from fedguide.datasets.base import TransitionDataset
from fedguide.utils.seeds import set_all_seeds


def main():
    parser = argparse.ArgumentParser(description="Centralized PPO for D4RL environments")
    
    # Environment args
    parser.add_argument("--env_name", type=str, required=True,
                       help="D4RL environment name (e.g., 'reacher-medium-v2', 'maze2d-umaze-v1')")
    parser.add_argument("--num_clients", type=int, default=1,
                       help="Number of clients (for compatibility, not used in on-policy)")
    
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
    
    # Evaluation args
    parser.add_argument("--eval_episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    parser.add_argument("--eval_stochastic_samples", type=int, default=64,
                       help="Number of action samples per state for stochastic evaluation")
    
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
        env_short = args.env_name.replace('-', '_').replace('/', '_')
        args.output_dir = f"./model/policy/{env_short}/ppo"

    if args.metrics_dir is None:
        env_short = args.env_name.replace('-', '_').replace('/', '_')
        args.metrics_dir = f"./metrics/{env_short}/ppo"
    
    # Set render save directory
    if args.render_eval and args.render_save_dir is None:
        env_short = args.env_name.replace('-', '_').replace('/', '_')
        args.render_save_dir = f"./videos/{env_short}/ppo"
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    # Create environment (for on-policy training)
    print(f"\nCreating D4RL environment: {args.env_name}...")
    try:
        import d4rl
        env = gym.make(args.env_name)
        # Ensure environment is also seeded
        set_all_seeds(args.seed, env)
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise
    
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
    
    # Create dummy datasets for compatibility (not used in on-policy)
    dummy_datasets = [TransitionDataset([])]
    
    # Create PPO agent
    print(f"\nCreating PPO agent...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
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
        action_low=action_low,
        action_high=action_high,
    )
    print("Agent created successfully")
    
    # Create trainer
    print(f"\nCreating centralized trainer...")
    trainer = CentralPPOTrainer(
        agent=agent,
        datasets=dummy_datasets,  # Not used in on-policy, but kept for compatibility
        env=env,
        steps_per_round=args.steps_per_round,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        eval_episodes=args.eval_episodes,
        eval_stochastic_samples=args.eval_stochastic_samples,
        device=device,
        render_eval=args.render_eval,
        render_mode=args.render_mode,
        render_save_dir=args.render_save_dir,
        render_every_n_rounds=args.render_every_n_rounds,
        render_episodes=args.render_episodes,
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

