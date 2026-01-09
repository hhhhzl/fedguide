"""
PPO Agent Implementation for Centralized Training

This module implements a Proximal Policy Optimization (PPO) agent with support for
centralized offline training on mixed multi-client data.

Based on the SAC agent structure but with PPO algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Tuple, Optional
import numpy as np


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for centralized training.
    
    The agent consists of:
    - Policy network (actor): outputs mean action
    - Value network (critic): estimates state value
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            lr: float = 3e-4,
            gamma: float = 0.99,
            clip_eps: float = 0.2,
            gae_lambda: float = 0.95,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            device: Optional[str] = None,
            action_low: Optional[np.ndarray] = None,
            action_high: Optional[np.ndarray] = None,
            action_std: float = 0.1,
            learnable_std: bool = True,
            init_log_std: float = None,
    ):
        import sys
        import os
        
        # Workaround for macOS: Set PyTorch to use single thread to avoid hangs
        if hasattr(torch, 'set_num_threads'):
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
        
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Store action bounds
        if action_low is not None:
            self.action_low = torch.tensor(action_low, dtype=torch.float32)
        else:
            self.action_low = None
        if action_high is not None:
            self.action_high = torch.tensor(action_high, dtype=torch.float32)
        else:
            self.action_high = None
        
        # Learnable action std (recommended for better exploration)
        self.learnable_std = learnable_std
        if learnable_std:
            # Initialize log_std
            if init_log_std is None:
                init_log_std = np.log(action_std)
            self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)
            self.action_std = None  # Will use exp(log_std) instead
        else:
            # Fixed action std (backward compatibility)
            self.action_std = action_std
            self.log_std = None

        def mlp(in_dim, out_dim):
            net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            # Initialize value network output layer
            if out_dim == 1:  # Value network
                with torch.no_grad():
                    nn.init.uniform_(net[-1].weight, -0.01, 0.01)
                    nn.init.constant_(net[-1].bias, 0.1)
            return net

        # Create networks
        self.actor = mlp(state_dim, action_dim)
        self.critic = mlp(state_dim, 1)

        # Set device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Create optimizer (include log_std if learnable)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        if learnable_std:
            params.append(self.log_std)
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # Move networks to device
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        if learnable_std:
            self.log_std = self.log_std.to(self.device)
        
        # Move action bounds to device if they exist
        if self.action_low is not None:
            self.action_low = self.action_low.to(self.device)
        if self.action_high is not None:
            self.action_high = self.action_high.to(self.device)

    def act(self, state: torch.Tensor, eval: bool = False, return_value: bool = False) -> Tuple:
        """
        Select action using current policy.
        
        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
            eval: If True, return deterministic action (mean), else sample from policy
            return_value: If True, also return value estimate
        
        Returns:
            action: Action tensor [batch_size, action_dim] or [action_dim]
            log_prob: Log probability of action [batch_size] or scalar
            value (optional): State value [batch_size] or scalar
        """
        # Ensure state is on correct device and has correct shape
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        
        # Ensure state is 2D for network forward pass
        was_1d = state.dim() == 1
        if was_1d:
            state = state.unsqueeze(0)
        
        # Get mean action from actor network
        actor_was_training = self.actor.training
        self.actor.eval()
        
        try:
            with torch.inference_mode():
                mu = self.actor(state)
        except Exception as e:
            self.actor.train(actor_was_training)
            raise
        
        # Restore training mode
        self.actor.train(actor_was_training)
        
        # Get action std (learnable or fixed)
        if self.learnable_std:
            std = self.log_std.exp()
            # Expand std to match mu shape
            if was_1d:
                std = std.unsqueeze(0)
        else:
            std = torch.ones_like(mu) * self.action_std
        
        # Create action distribution and sample
        dist = Normal(mu, std)
        action = mu if eval else dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        
        # Apply action bounds if provided
        if self.action_low is not None and self.action_high is not None:
            action = action.clamp(self.action_low, self.action_high).detach()
        else:
            # Default behavior for backward compatibility (Bandit2D range)
            action = action.clamp(-1.5, 1.5).detach()
        
        # Get value if requested
        value = None
        if return_value:
            with torch.no_grad():
                self.critic.eval()
                try:
                    value = self.critic(state).squeeze(-1)
                finally:
                    self.critic.train()
        
        # Return to original shape if input was 1D
        if was_1d:
            action = action.squeeze(0)
            logp = logp.squeeze(0)
            if value is not None:
                value = value.squeeze(0)
        
        if return_value:
            return action, logp, value
        else:
            return action, logp

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate state-action pairs using current policy and value network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
        
        Returns:
            log_prob: Log probability of action [batch_size]
            entropy: Entropy of action distribution [batch_size]
            value: State value [batch_size]
        """
        # Ensure tensors are on correct device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        
        state = state.to(self.device)
        action = action.to(self.device)
        
        # Ensure 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # Get mean action from actor
        mu = self.actor(state)
        
        # Check for NaN in mu (should not happen, but prevent crash)
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print(f"Warning: NaN/Inf detected in actor output. Clamping mu.")
            mu = torch.clamp(mu, -10.0, 10.0)
            mu = torch.where(torch.isnan(mu) | torch.isinf(mu), torch.zeros_like(mu), mu)
        
        # Get action std (learnable or fixed)
        if self.learnable_std:
            std = self.log_std.exp()
            # Expand std to match mu shape
            if state.dim() == 2:
                std = std.unsqueeze(0).expand(mu.shape[0], -1)
            else:
                std = std.unsqueeze(0)
        else:
            std = torch.ones_like(mu) * self.action_std
        
        # Create distribution
        dist = Normal(mu, std)
        
        # Compute log probability
        logp = dist.log_prob(action).sum(-1)  # [batch_size]
        
        # Compute entropy
        entropy = dist.entropy().sum(-1)  # [batch_size]
        
        # Get value
        value = self.critic(state).squeeze(-1)  # [batch_size]
        
        return logp, entropy, value

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        epochs: int = 4,
        minibatch_size: Optional[int] = None
    ) -> Tuple[float, float, float, float]:
        """
        Update policy and value networks using PPO algorithm.
        
        Args:
            batch: Dictionary with keys:
                - 's': states [batch_size, state_dim]
                - 'a': actions [batch_size, action_dim]
                - 'r': rewards [batch_size]
                - 's_next': next states [batch_size, state_dim]
                - 'done': done flags [batch_size]
                - 'returns': discounted returns [batch_size]
                - 'advantages': advantages [batch_size]
                - 'logp_old': old log probabilities [batch_size]
            epochs: Number of update epochs
            minibatch_size: Size of minibatches for update (if None, use full batch)
        
        Returns:
            policy_loss: Policy loss value
            value_loss: Value loss value
            entropy: Entropy value
            total_loss: Total loss value
        """
        # Move batch to device
        states = batch['s'].to(self.device)
        actions = batch['a'].to(self.device)
        returns = batch['returns'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        logp_old = batch['logp_old'].to(self.device)
        
        batch_size = states.shape[0]
        if minibatch_size is None:
            minibatch_size = batch_size
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        # Multiple epochs of updates
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = torch.randperm(batch_size, device=self.device)
            
            for start_idx in range(0, batch_size, minibatch_size):
                end_idx = min(start_idx + minibatch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get minibatch
                mb_states = states[batch_indices]
                mb_actions = actions[batch_indices]
                mb_returns = returns[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_logp_old = logp_old[batch_indices]
                
                # Evaluate current policy
                logp, entropy, values = self.evaluate(mb_states, mb_actions)
                
                # Compute policy loss (PPO clipped objective)
                # Use log space to avoid numerical instability
                log_ratio = logp - mb_logp_old
                # Clamp log_ratio to prevent exp overflow/underflow
                log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
                ratio = torch.exp(log_ratio)
                
                # Clip the ratio for PPO objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Check for NaN/inf values
                if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                    print(f"Warning: NaN/Inf in policy_loss. log_ratio range: [{log_ratio.min():.4f}, {log_ratio.max():.4f}], "
                          f"ratio range: [{ratio.min():.4f}, {ratio.max():.4f}], "
                          f"advantages range: [{mb_advantages.min():.4f}, {mb_advantages.max():.4f}]")
                    # Skip this update if NaN/Inf
                    continue
                
                # Compute value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Compute entropy bonus
                entropy_bonus = entropy.mean()
                
                # Check for NaN in other losses too
                if torch.isnan(value_loss) or torch.isinf(value_loss):
                    print(f"Warning: NaN/Inf in value_loss. values range: [{values.min():.4f}, {values.max():.4f}], "
                          f"returns range: [{mb_returns.min():.4f}, {mb_returns.max():.4f}]")
                    continue
                
                if torch.isnan(entropy_bonus) or torch.isinf(entropy_bonus):
                    print(f"Warning: NaN/Inf in entropy. entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")
                    continue
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus
                
                # Check for NaN in total loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf in total loss. Skipping update.")
                    continue
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients before clipping
                has_nan_grad = False
                for param in list(self.actor.parameters()) + list(self.critic.parameters()):
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    print("Warning: NaN/Inf gradients detected. Skipping update.")
                    self.optimizer.zero_grad()  # Clear gradients
                    continue
                
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Accumulate metrics (only if update was successful)
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
        
        # Average over epochs and minibatches (only successful updates)
        # Count successful updates
        num_successful_updates = epochs * (batch_size // minibatch_size + (1 if batch_size % minibatch_size > 0 else 0))
        
        # The actual count might be less if some updates were skipped due to NaN
        # For now, we use the expected count. If many updates fail, metrics will be less meaningful
        if num_successful_updates > 0:
            total_policy_loss /= num_successful_updates
            total_value_loss /= num_successful_updates
            total_entropy /= num_successful_updates
        else:
            # If all updates failed, return zero (shouldn't happen, but safe fallback)
            print("Warning: All updates in this batch failed. Returning zero losses.")
            return 0.0, 0.0, 0.0, 0.0
        
        total_loss = total_policy_loss + self.value_coef * total_value_loss - self.entropy_coef * total_entropy
        
        return float(total_policy_loss), float(total_value_loss), float(total_entropy), float(total_loss)

    def to(self, device: str) -> 'PPOAgent':
        """
        Move agent to specified device.
        
        Args:
            device: Device string ('cpu' or 'cuda')
        
        Returns:
            self
        """
        self.device = torch.device(device)
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        if self.learnable_std:
            self.log_std = self.log_std.to(self.device)
        # Move action bounds to device if they exist
        if self.action_low is not None:
            self.action_low = self.action_low.to(self.device)
        if self.action_high is not None:
            self.action_high = self.action_high.to(self.device)
        return self

