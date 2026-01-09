"""
Run Centralized PPO for Reacher with client heterogeneity from metadata.json.

This script loads client configurations from metadata.json and trains PPO
on different Reacher environments with client-specific configurations using on-policy rollouts.
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
import torch
import gymnasium as gym
from typing import List, Dict, Any

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from fedguide.baselines.ppo.agent import PPOAgent
from fedguide.baselines.ppo.trainer import CentralPPOTrainer
from fedguide.datasets.base import TransitionDataset
from fedguide.utils.seeds import set_all_seeds
from fedguide.envs.reacher import CustomizedReacherEnv
from gymnasium.wrappers import TimeLimit


def load_reacher_metadata(metadata_path: str):
    """Load reacher metadata.json file."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Centralized PPO for Reacher with client heterogeneity"
    )
    
    # Environment args
    parser.add_argument("--metadata_path", type=str, required=True,
                       help="Path to reacher metadata.json file")
    parser.add_argument("--num_clients", type=int, default=None,
                       help="Number of clients to use (selects first N or random N if --random_select_clients). If None, uses all clients.")
    parser.add_argument("--random_select_clients", action="store_true",
                       help="If set, randomly select num_clients from metadata.json; otherwise select first num_clients")
    
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
    
    # Policy logprob collection
    parser.add_argument("--collect_logprob", action="store_true",
                       help="Collect policy logprob distribution on grid")
    parser.add_argument("--logprob_grid_size", type=int, default=200,
                       help="Grid size for logprob collection")
    parser.add_argument("--logprob_bounds", type=float, nargs=2, default=[-1.5, 1.5],
                       help="Bounds for logprob grid")
    
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
    
    # Set random seeds
    print(f"Setting random seed: {args.seed}")
    set_all_seeds(args.seed)
    
    # Load metadata
    print(f"\nLoading metadata from {args.metadata_path}...")
    metadata = load_reacher_metadata(args.metadata_path)
    all_client_configs = metadata.get("clients", [])
    total_clients = len(all_client_configs)
    print(f"Loaded {total_clients} client configurations from metadata")
    print(f"Hetero type: {metadata.get('hetero_type', 'unknown')}")
    
    # Select clients based on num_clients and random_select_clients
    if args.num_clients is not None and args.num_clients < total_clients:
        if args.random_select_clients:
            # Randomly select num_clients
            rng = np.random.RandomState(args.seed)
            selected_indices = rng.choice(total_clients, size=args.num_clients, replace=False)
            selected_indices = sorted(selected_indices)  # Sort for reproducibility
            client_configs = [all_client_configs[i] for i in selected_indices]
            print(f"Randomly selected {args.num_clients} clients (indices: {selected_indices})")
        else:
            # Select first num_clients
            client_configs = all_client_configs[:args.num_clients]
            print(f"Selected first {args.num_clients} clients")
    else:
        # Use all clients
        client_configs = all_client_configs
        if args.num_clients is not None:
            print(f"Requested {args.num_clients} clients, but only {total_clients} available. Using all {total_clients} clients.")
    
    n_clients = len(client_configs)
    print(f"Using {n_clients} clients for training")
    
    # For on-policy PPO, we use the first client's environment for training
    # (or we could cycle through clients, but for simplicity use first one)
    first_client_config = client_configs[0]
    variant = first_client_config.get('variant', 'medium-v2')
    
    # Create training environment with first client's configuration
    print(f"\nCreating training environment (using first client config: variant={variant})...")
    train_env = TimeLimit(
        CustomizedReacherEnv(
            qpos_high_low=first_client_config["qpos_high_low"],
            action_noise=np.array(first_client_config["action_noise"]),
            reward_scale=first_client_config["reward_scale"],
            angle_noise=first_client_config["angle_noise"],
            variant=variant
        ),
        max_episode_steps=50
    )
    set_all_seeds(args.seed, train_env)
    
    # Create evaluation environments for all selected clients
    eval_envs = []
    for i, client_config in enumerate(client_configs):
        variant = client_config.get('variant', 'medium-v2')
        eval_env = TimeLimit(
            CustomizedReacherEnv(
                qpos_high_low=client_config["qpos_high_low"],
                action_noise=np.array(client_config["action_noise"]),
                reward_scale=client_config["reward_scale"],
                angle_noise=client_config["angle_noise"],
                variant=variant
            ),
            max_episode_steps=50
        )
        set_all_seeds(args.seed, eval_env)
        eval_envs.append(eval_env)
    
    # Use first eval env for dimensions
    eval_env = eval_envs[0]
    obs_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    
    # Get action bounds
    if hasattr(eval_env.action_space, 'low') and hasattr(eval_env.action_space, 'high'):
        action_low = eval_env.action_space.low
        action_high = eval_env.action_space.high
    else:
        action_low = None
        action_high = None
    
    print(f"\nEnvironment dimensions:")
    print(f"  Observation: {obs_dim}")
    print(f"  Action: {action_dim}")
    if action_low is not None and action_high is not None:
        print(f"  Action bounds: [{action_low}, {action_high}]")
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\nUsing device: {device}")
    
    # Set output directories
    if args.output_dir is None:
        args.output_dir = f"./model/policy/reacher/ppo"
    if args.metrics_dir is None:
        args.metrics_dir = f"./metrics/reacher/ppo"
    
    # Set render save directory
    if args.render_eval and args.render_save_dir is None:
        args.render_save_dir = f"./videos/reacher/ppo"
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    if args.render_eval and args.render_save_dir:
        os.makedirs(args.render_save_dir, exist_ok=True)
    
    # Create dummy datasets for compatibility (not used in on-policy)
    dummy_datasets = [TransitionDataset([])]
    
    # Create agent
    print(f"\nCreating PPO agent...")
    agent = PPOAgent(
        state_dim=obs_dim,
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
    
    # Create trainer (use training environment)
    print(f"\nCreating centralized trainer...")
    trainer = CentralPPOTrainer(
        agent=agent,
        datasets=dummy_datasets,  # Not used in on-policy, but kept for compatibility
        env=train_env,  # Use training environment for rollouts
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
    print(f"\n{'='*60}")
    print(f"Starting PPO Training")
    print(f"{'='*60}")
    print(f"Rounds: {args.rounds}")
    print(f"Steps per round: {args.steps_per_round}")
    print(f"Update epochs: {args.update_epochs}")
    print(f"Minibatch size: {args.minibatch_size if args.minibatch_size is not None else 'full batch'}")
    print(f"Learnable std: {args.learnable_std}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"{'='*60}\n")
    
    history = []
    
    for round_num in range(1, args.rounds + 1):
        # Train one round
        metrics = trainer.train_one_round(round_num=round_num)
        metrics['round'] = round_num
        history.append(metrics)
        
        # Collect policy logprob distribution (for visualization)
        if args.collect_logprob and (round_num % args.save_every == 0 or round_num == args.rounds or round_num == 1):
            try:
                print(f"  [Round {round_num}] Computing policy logprob distribution on grid...", flush=True)
                # For reacher, action_dim is 2, so we can visualize directly
                action_dims = None if action_dim == 2 else [0, 1]
                policy_metrics = trainer.evaluate_policy_logprob_on_grid(
                    grid_size=args.logprob_grid_size,
                    bounds=tuple(args.logprob_bounds),
                    action_dims=action_dims
                )
                if policy_metrics is not None:
                    # Store in metrics (convert to list for JSON serialization)
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
        
        # Print progress
        if round_num % 10 == 0 or round_num == 1:
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{args.rounds}")
            print(f"{'='*60}")
            print(f"  Train Loss (Actor): {metrics.get('train/loss/actor', 'N/A'):.4f}")
            print(f"  Train Loss (Critic): {metrics.get('train/loss/critic', 'N/A'):.4f}")
            print(f"  Entropy: {metrics.get('train/entropy', 'N/A'):.4f}")
            print(f"  Returns mean: {metrics.get('train/returns_mean', 'N/A'):.4f}")
            print(f"  V mean: {metrics.get('train/V_mean', 'N/A'):.4f}")
            if 'eval/return' in metrics:
                print(f"  Eval Return (deterministic): {metrics.get('eval/return', 'N/A'):.2f}")
            if 'eval/return_stochastic_mean' in metrics:
                print(f"  Eval Return (stochastic mean): {metrics.get('eval/return_stochastic_mean', 'N/A'):.2f}")
                print(f"  Eval Return (stochastic max): {metrics.get('eval/return_stochastic_max', 'N/A'):.2f}")
        
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
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final model and metrics
    final_model_path = os.path.join(args.output_dir, "final_model.pkl")
    with open(final_model_path, 'wb') as f:
        pickle.dump({
            'agent_state': agent.actor.state_dict(),
            'args': vars(args),
        }, f)
    print(f"\nSaved final model: {final_model_path}")
    
    metrics_path = os.path.join(args.metrics_dir, "training_history.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump({
            'history': history,
            'args': vars(args),
            'final_metrics': history[-1] if history else {},
        }, f)
    print(f"Saved training history: {metrics_path}")
    
    # Cleanup
    train_env.close()
    for env in eval_envs:
        env.close()
    
    print(f"\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()

