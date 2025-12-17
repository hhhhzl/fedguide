"""
Centralized SAC Trainer Implementation

This module implements a centralized trainer that merges data from multiple clients
into a single replay buffer and trains a central SAC agent.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
from collections import deque

from .agent import SACAgent
from fedguide.datasets.base import TransitionDataset


class CentralSACTrainer:
    """
    Centralized SAC Trainer that learns from multiple clients' data.
    
    This trainer:
    1. Merges data from multiple client datasets into a single replay buffer
    2. Trains a central SAC agent on the mixed data
    3. Supports evaluation on the environment
    4. Collects training metrics
    
    No federated aggregation is performed - this is pure centralized training.
    """
    
    def __init__(
        self,
        agent: SACAgent,
        datasets: List[TransitionDataset],
        env: Any,
        batch_size: int = 256,
        update_steps: int = 1000,
        gamma: float = 0.99,
        eval_episodes: int = 10,
        device: Optional[str] = None,
    ):
        """
        Initialize centralized SAC trainer.
        
        Args:
            agent: SAC agent to train
            datasets: List of TransitionDataset objects from different clients
            env: Environment for evaluation (gymnasium.Env)
            batch_size: Batch size for training
            update_steps: Number of update steps per training round
            gamma: Discount factor (should match agent's gamma)
            eval_episodes: Number of episodes for evaluation
            device: Device to run on
        """
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.gamma = gamma
        self.eval_episodes = eval_episodes
        
        # Set device
        if device is None:
            self.device = agent.device
        else:
            self.device = torch.device(device)
            self.agent.to(self.device)
        
        # Build replay buffer from all client datasets
        self.replay_buffer = self._build_replay_buffer(datasets)
        
        # Track data statistics
        self.client_data_sizes = [len(ds) for ds in datasets]
        self.num_clients = len(datasets)
        self.total_transitions = len(self.replay_buffer['states'])
        
        # Evaluation tracking
        self.eval_returns = deque(maxlen=100)
        
        print(f"[CentralSACTrainer] Initialized with {self.num_clients} clients")
        print(f"[CentralSACTrainer] Total transitions: {self.total_transitions}")
        print(f"[CentralSACTrainer] Client data sizes: {self.client_data_sizes}")
    
    def _build_replay_buffer(self, datasets: List[TransitionDataset]) -> Dict[str, np.ndarray]:
        """
        Merge multiple client datasets into a single replay buffer.
        
        Args:
            datasets: List of TransitionDataset objects
        
        Returns:
            Dictionary with keys: 'states', 'actions', 'rewards', 'next_states', 'dones'
        """
        buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }
        
        # Merge all transitions from all clients
        for client_id, dataset in enumerate(datasets):
            for i in range(len(dataset)):
                sample = dataset[i]
                buffer['states'].append(sample['s'].numpy())
                buffer['actions'].append(sample['a'].numpy())
                buffer['rewards'].append(sample['r'].item())
                buffer['next_states'].append(sample['s_'].numpy())
                buffer['dones'].append(sample['d'].item())
        
        # Convert to numpy arrays
        buffer = {k: np.array(v, dtype=np.float32) for k, v in buffer.items()}
        
        return buffer
    
    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch from replay buffer.
        
        Returns:
            Dictionary with batched tensors ready for agent.update()
        """
        # Random sampling with replacement
        indices = np.random.choice(
            len(self.replay_buffer['states']),
            size=self.batch_size,
            replace=True
        )
        
        batch = {
            's': torch.FloatTensor(self.replay_buffer['states'][indices]).to(self.device),
            'a': torch.FloatTensor(self.replay_buffer['actions'][indices]).to(self.device),
            'r': torch.FloatTensor(self.replay_buffer['rewards'][indices]).to(self.device),
            's_next': torch.FloatTensor(self.replay_buffer['next_states'][indices]).to(self.device),
            'done': torch.FloatTensor(self.replay_buffer['dones'][indices]).to(self.device),
        }
        
        return batch
    
    def _evaluate(self) -> float:
        """
        Evaluate the current policy on the environment.
        
        Returns:
            Average episode return over eval_episodes
        """
        returns = []
        
        for _ in range(self.eval_episodes):
            state, _ = self.env.reset()
            total_return = 0.0
            done = False
            step_count = 0
            max_steps = 1000  # Safety limit
            
            while not done and step_count < max_steps:
                # Get action from agent (deterministic for evaluation)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, _ = self.agent.act(state_tensor, eval=True)
                action_np = action.cpu().numpy()[0]
                
                # Step environment
                next_state, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated
                total_return += reward
                state = next_state
                step_count += 1
            
            returns.append(total_return)
        
        avg_return = np.mean(returns)
        self.eval_returns.append(avg_return)
        return avg_return
    
    def train_one_round(self) -> Dict[str, float]:
        """
        Train for one round: perform multiple update steps on replay buffer.
        
        Returns:
            Dictionary of training metrics
        """
        # Initialize metrics
        metrics = {
            'loss': 0.0,
            'train/loss/actor': 0.0,
            'train/loss/critic': 0.0,
            'train/q_value': 0.0,
            'train/q_value_min': 0.0,
            'train/buffer_size': self.total_transitions,
        }

        # Perform multiple update steps
        for step in range(self.update_steps):
            # Sample batch from replay buffer
            batch = self._sample_batch()
            # Update agent
            actor_loss, critic_loss, q_values = self.agent.update(batch)
            # Accumulate metrics
            metrics['loss'] += (actor_loss + critic_loss)
            metrics['train/loss/actor'] += actor_loss
            metrics['train/loss/critic'] += critic_loss
            metrics['train/q_value'] += q_values.mean().item()
            metrics['train/q_value_min'] += q_values.min().item()
        
        # Average metrics over update steps
        for key in [
            'loss',
            'train/loss/actor',
            'train/loss/critic',
            'train/q_value',
            'train/q_value_min'
        ]:
            metrics[key] /= self.update_steps
        
        # Add data statistics
        metrics['data/num_clients'] = self.num_clients
        metrics['data/total_transitions'] = self.total_transitions
        metrics['data/client_sizes'] = self.client_data_sizes
        
        # Evaluation (if enabled)
        if self.eval_episodes > 0:
            eval_return = self._evaluate()
            metrics['eval/return'] = eval_return
            
            # Compute statistics over recent evaluations
            if len(self.eval_returns) > 1:
                metrics['eval/return/mean'] = np.mean(self.eval_returns)
                metrics['eval/return/std'] = np.std(self.eval_returns)
        
        return metrics
    
    def save_eval(self, cid: str, round_num: int) -> bool:
        """
        Placeholder for compatibility with FedRLClient interface.
        
        Args:
            cid: Client ID (not used in centralized training)
            round_num: Round number
        
        Returns:
            True (always succeeds)
        """
        return True
    
    @property
    def n_steps(self) -> int:
        """Number of steps used in training (for compatibility)."""
        return self.update_steps * self.batch_size

