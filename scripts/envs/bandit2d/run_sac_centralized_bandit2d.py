"""
Run Centralized SAC baseline for 2D Bandit environment.

This script trains a central SAC agent on mixed data from multiple clients.
No federated aggregation is performed - pure centralized training.
"""

import argparse
import os
import sys
import pickle
import json
import numpy as np
import torch
from typing import List

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from fedguide.baselines.sac.agent import SACAgent
from fedguide.baselines.sac.trainer import CentralSACTrainer
from fedguide.envs.bandit2d import Bandit2D
from fedguide.datasets.base import TransitionDataset, TrajectoryDataset
from fedguide.utils.seeds import set_all_seeds
from scripts.generate_data.generate_bandit2d_data import generate_bandit2d_datasets


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
    parser = argparse.ArgumentParser(description="Centralized SAC for Bandit2D")
    
    # Data args
    parser.add_argument("--num_clients", type=int, default=4,
                       help="Number of clients")
    parser.add_argument("--data_dir", type=str, default="data/bandit2d",
                       help="Directory containing bandit2d data")
    
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
                       help="Temperature parameter (entropy regularization)")
    
    # Environment args
    parser.add_argument("--K", type=int, default=4,
                       help="Number of peaks in bandit")
    parser.add_argument("--sigma", type=float, default=0.2,
                       help="Standard deviation for reward function")
    
    # Evaluation args
    parser.add_argument("--eval_episodes", type=int, default=50,
                       help="Number of episodes for evaluation (increased for more stable evaluation)")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default=f"./model/policy/bandit2d/sac",
                       help="Directory to save results")
    parser.add_argument("--metrics_dir", type=str, default=f"./metrics/bandit2d/sac",
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
    
    # Create SAC agent
    print(f"\nCreating SAC agent...")
    print(f"  State dim: 2, Action dim: 2")
    print(f"  Hidden dim: {args.hidden_dim}, LR: {args.lr}")
    print(f"  Device: {device}")
    import sys
    sys.stdout.flush()
    
    print("Step 1: Creating networks...")
    sys.stdout.flush()
    
    print("Step 2: Creating agent instance...")
    sys.stdout.flush()
    
    try:
        agent = SACAgent(
            state_dim=2,
            action_dim=2,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
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
        import sys
        sys.stdout.flush()
        
        # Train one round
        print(f"  [Round {round_num}] Starting training...", flush=True)
        metrics = trainer.train_one_round()
        metrics['round'] = round_num
        
        # Store history
        history.append(metrics)
        
        # Print progress (always print, not just every 10 rounds)
        print(f"\n  [Round {round_num}] Training completed:")
        print(f"    Loss: {metrics['loss']:.4f}")
        print(f"    Actor Loss: {metrics['train/loss/actor']:.4f}")
        print(f"    Critic Loss: {metrics['train/loss/critic']:.4f}")
        print(f"    Q Value: {metrics['train/q_value']:.4f}")
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
        print(f"  Actor Loss: {final_metrics['train/loss/actor']:.4f}")
        print(f"  Critic Loss: {final_metrics['train/loss/critic']:.4f}")
        print(f"  Q Value: {final_metrics['train/q_value']:.4f}")
        if 'eval/return' in final_metrics:
            print(f"  Eval Return: {final_metrics['eval/return']:.4f}")
    
    return history


if __name__ == "__main__":
    main()

