"""
FedKL Trainer Implementation

This module implements the PPO trainer with dual KL divergence penalties
for federated reinforcement learning.
"""

import torch
import numpy as np
from typing import Any
from collections import deque


class RolloutBuffer:
    """Buffer for storing trajectories."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.values,
            self.log_probs,
            self.dones,
        )


class FedKLTrainer:
    """
    FedKL Trainer implementing PPO with KL divergence penalty.
    
    The trainer adds two types of KL penalties:
    1. Global penalty: KL divergence from global policy (lambda_global)
    2. Local penalty: KL divergence from policy at start of round (lambda_local)
    """
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        lambda_global: float = 0.1,
        lambda_local: float = 0.05,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.agent = agent
        self.env = env
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.buffer = RolloutBuffer()
        
        # Track metrics
        self.episode_rewards = deque(maxlen=10)
        self.episode_lengths = deque(maxlen=10)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Local policy snapshot (for local KL penalty)
        self.local_policy_snapshot = None
        self.local_log_std_snapshot = None
    
    def collect_rollouts(self) -> int:
        """Collect n_steps of experience."""
        self.buffer.clear()
        state, _ = self.env.reset()
        steps_collected = 0
        
        while steps_collected < self.n_steps:
            # Get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                mean = self.agent.policy(state_tensor)
                std = torch.exp(self.agent.log_std)
                dist = torch.distributions.Normal(mean, std)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor).sum(dim=-1)
                value = self.agent.value(state_tensor)
            
            action = action_tensor.cpu().numpy()[0]
            
            # Step environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.buffer.add(
                state, action, reward, 
                value.item(), log_prob.item(), done
            )
            
            # Update metrics
            self.current_episode_reward += reward
            self.current_episode_length += 1
            steps_collected += 1
            
            # Handle episode end
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.current_episode_reward = 0
                self.current_episode_length = 0
                state, _ = self.env.reset()
            else:
                state = next_state
        
        return steps_collected
    
    def compute_returns_and_advantages(self):
        """Compute returns and GAE advantages."""
        states, actions, rewards, values, log_probs, dones = self.buffer.get()
        
        returns = []
        advantages = []
        
        # Get final value estimate
        state_tensor = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.agent.value(state_tensor).item()
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value if not dones[t] else 0
            else:
                next_value_t = values[t + 1] if not dones[t] else 0
            
            delta = rewards[t] + self.gamma * next_value_t - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return returns, advantages
    
    def save_local_policy_snapshot(self):
        """Save current policy as local snapshot for local KL penalty."""
        self.local_policy_snapshot = {
            k: v.clone().detach() 
            for k, v in self.agent.policy.state_dict().items()
        }
        self.local_log_std_snapshot = self.agent.log_std.clone().detach()
    
    def compute_local_kl(self, states: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from local policy snapshot."""
        if self.local_policy_snapshot is None:
            return torch.tensor(0.0, device=self.device)
        
        # Current policy
        mean_current = self.agent.policy(states)
        std_current = torch.exp(self.agent.log_std)
        
        # Load snapshot temporarily
        original_state = self.agent.policy.state_dict()
        self.agent.policy.load_state_dict(self.local_policy_snapshot)
        
        with torch.no_grad():
            mean_snapshot = self.agent.policy(states)
            std_snapshot = torch.exp(self.local_log_std_snapshot)
        
        # Restore current policy
        self.agent.policy.load_state_dict(original_state)
        
        # KL divergence
        var_current = std_current.pow(2)
        var_snapshot = std_snapshot.pow(2)
        
        kl = (
            torch.log(std_snapshot / std_current)
            + (var_current + (mean_current - mean_snapshot).pow(2)) / (2 * var_snapshot)
            - 0.5
        )
        
        return kl.sum(dim=-1).mean()
    
    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        """Update policy using PPO with KL penalties."""
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of updates
        total_loss = 0
        num_updates = 0
        
        for _ in range(self.update_epochs):
            # Mini-batch updates
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Policy forward
                mean = self.agent.policy(batch_states)
                std = torch.exp(self.agent.log_std)
                dist = torch.distributions.Normal(mean, std)
                
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.agent.value(batch_states).squeeze()
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                
                # KL penalties
                kl_global = self.agent.compute_kl_divergence(batch_states)
                kl_local = self.compute_local_kl(batch_states)
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.lambda_global * kl_global  # Global KL penalty
                    + self.lambda_local * kl_local    # Local KL penalty
                )
                
                # Optimize
                self.agent.policy_optimizer.zero_grad()
                self.agent.value_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.agent.policy.parameters()) + [self.agent.log_std],
                    self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.agent.value.parameters(),
                    self.max_grad_norm
                )
                
                self.agent.policy_optimizer.step()
                self.agent.value_optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        return total_loss / max(num_updates, 1)
    
    def train_one_round(self) -> float:
        """Train for one federated round."""
        # Save local policy snapshot at start of round
        self.save_local_policy_snapshot()
        
        # Collect rollouts
        self.collect_rollouts()
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        
        # Get data from buffer
        states, actions, _, _, log_probs, _ = self.buffer.get()
        
        # Update policy
        loss = self.update_policy(states, actions, log_probs, advantages, returns)
        
        return loss
    
    def save_eval(self, cid: str, round_num: int) -> bool:
        """Evaluate and save (placeholder for compatibility)."""
        return True
    
    @property
    def return_(self):
        """Average episode return."""
        return np.mean(self.episode_rewards) if self.episode_rewards else 0.0
    
    @property
    def episode_len(self):
        """Average episode length."""
        return np.mean(self.episode_lengths) if self.episode_lengths else 0.0
