"""
SAC Agent Implementation for Centralized Training

This module implements a Soft Actor-Critic (SAC) agent with support for
centralized offline training on mixed multi-client data.

Based on the original sac_agent.py but with enhanced interface for centralized training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Tuple, Optional
import numpy as np


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent for centralized training.
    
    The agent consists of:
    - Actor (policy network): outputs mean action
    - Two Q-networks (critics): Q1 and Q2 for double Q-learning
    - Target Q-networks: soft-updated copies of Q-networks
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            lr: float = 3e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            alpha: float = 0.2,
            device: Optional[str] = None,
            action_low: Optional[np.ndarray] = None,
            action_high: Optional[np.ndarray] = None,
            action_std: float = 0.1,
    ):
        import sys
        import os
        
        # Workaround for macOS: Set PyTorch to use single thread to avoid hangs
        # This is a known issue on macOS with certain PyTorch versions
        if hasattr(torch, 'set_num_threads'):
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
        
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        
        # Store action bounds and std (with backward compatibility)
        if action_low is not None:
            self.action_low = torch.tensor(action_low, dtype=torch.float32)
        else:
            self.action_low = None
        if action_high is not None:
            self.action_high = torch.tensor(action_high, dtype=torch.float32)
        else:
            self.action_high = None
        self.action_std = action_std

        def mlp(in_dim, out_dim):
            net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            # Initialize Q-network output layer
            # For rewards in [0, 1], we want Q-values to start near the mean reward (~0.18)
            # This helps Q-values learn in the correct direction
            if out_dim == 1:  # Q-network
                with torch.no_grad():
                    # Initialize with small weights
                    nn.init.uniform_(net[-1].weight, -0.01, 0.01)
                    # Initialize bias to a small positive value (around mean reward)
                    # This ensures Q-values start positive and can learn upward
                    nn.init.constant_(net[-1].bias, 0.1)
            return net

        # Create networks
        self.actor = mlp(state_dim, action_dim)
        self.q1 = mlp(state_dim + action_dim, 1)
        self.q2 = mlp(state_dim + action_dim, 1)
        self.q1_target = mlp(state_dim + action_dim, 1)
        self.q2_target = mlp(state_dim + action_dim, 1)

        # Set device (avoid checking cuda.is_available() on macOS to prevent hangs)
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Create optimizers before moving to device (safer)
        actor_params = list(self.actor.parameters())
        critic_params = list(self.q1.parameters()) + list(self.q2.parameters())
        self.opt_actor = torch.optim.Adam(actor_params, lr=lr)
        self.opt_critic = torch.optim.Adam(critic_params, lr=lr)

        # Move networks to device
        self.actor = self.actor.to(self.device)
        self.q1 = self.q1.to(self.device)
        self.q2 = self.q2.to(self.device)
        self.q1_target = self.q1_target.to(self.device)
        self.q2_target = self.q2_target.to(self.device)
        
        # Move action bounds to device if they exist
        if self.action_low is not None:
            self.action_low = self.action_low.to(self.device)
        if self.action_high is not None:
            self.action_high = self.action_high.to(self.device)
        
        # Mark that target networks need initialization (will be done during first update)
        self._target_initialized = False

    def act(self, state: torch.Tensor, eval: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.
        
        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
            eval: If True, return deterministic action (mean), else sample from policy
        
        Returns:
            action: Action tensor [batch_size, action_dim] or [action_dim]
            log_prob: Log probability of action [batch_size] or scalar
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
        # Workaround for macOS: Set to eval mode temporarily and use inference_mode
        # This avoids hangs on macOS while still allowing gradient flow when needed
        actor_was_training = self.actor.training
        self.actor.eval()
        
        try:
            # Use inference_mode for macOS stability (only disables gradient tracking, not computation)
            # For actor updates, we'll recompute logp with gradients in update() method
            with torch.inference_mode():
                mu = self.actor(state)
        except Exception as e:
            # Restore training mode before raising
            self.actor.train(actor_was_training)
            raise
        
        # Restore training mode
        self.actor.train(actor_was_training)
        
        # Create action distribution and sample
        dist = Normal(mu, torch.ones_like(mu) * self.action_std)
        action = mu if eval else dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        
        # Apply action bounds if provided, otherwise use default (backward compatibility)
        if self.action_low is not None and self.action_high is not None:
            action = action.clamp(self.action_low, self.action_high).detach()
        else:
            # Default behavior for backward compatibility (Bandit2D range)
            action = action.clamp(-1.5, 1.5).detach()
        
        # Return to original shape if input was 1D
        if was_1d:
            action = action.squeeze(0)
            logp = logp.squeeze(0)
        
        return action, logp

    def update(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float, torch.Tensor]:
        """
        Update actor and critic networks using SAC algorithm.
        
        Args:
            batch: Dictionary with keys:
                - 's': states [batch_size, state_dim]
                - 'a': actions [batch_size, action_dim]
                - 'r': rewards [batch_size]
                - 's_next': next states [batch_size, state_dim]
                - 'done': done flags [batch_size]
        
        Returns:
            actor_loss: Actor loss value
            critic_loss: Critic loss value
            q_values: Q-values from Q1 [batch_size]
        """
        import sys
        
        # Move batch to device
        s = batch['s'].to(self.device)
        a = batch['a'].to(self.device)
        r = batch['r'].to(self.device)
        s2 = batch['s_next'].to(self.device)
        done = batch['done'].to(self.device)

        # Critic update
        # Compute target Q-values in no_grad context
        with torch.no_grad():
            # Check if all transitions are terminal (common in bandit environments)
            # For terminal states (done=1), Q-target should equal reward
            # For non-terminal states, Q-target = r + gamma * (min(Q_next) - alpha * logp_next)
            all_done = done.all().item() if done.numel() > 0 else True
            
            if all_done:
                # All transitions are terminal: Q-target = reward
                # This is correct for bandit environments where each episode is one step
                # Ensure q_target has shape [batch_size, 1] to match Q network output
                q_target = r.unsqueeze(-1) if r.dim() == 1 else r
            else:
                # Some transitions are non-terminal: need to compute next Q-values
                # Get next action from policy (for target Q-value computation)
                next_a, logp_next = self.act(s2)
                
                # Ensure next_a has correct shape for concatenation
                if next_a.dim() == 1:
                    next_a = next_a.unsqueeze(0)
                
                # Compute next Q-values using target networks
                sa_next = torch.cat([s2, next_a], dim=-1)
                if not hasattr(self, '_target_initialized') or not self._target_initialized:
                    # First update: use main networks (targets not initialized yet)
                    q1t = self.q1(sa_next)
                    q2t = self.q2(sa_next)
                else:
                    # Subsequent updates: use target networks
                    q1t = self.q1_target(sa_next)
                    q2t = self.q2_target(sa_next)
                q_next = torch.min(q1t, q2t)  # Keep shape [batch_size, 1]
                # Ensure dimensions match: r [batch_size], done [batch_size], q_next [batch_size, 1], logp_next [batch_size]
                q_target = r.unsqueeze(-1) + self.gamma * (1 - done.unsqueeze(-1)) * (q_next - self.alpha * logp_next.unsqueeze(-1))

        # Compute current Q-values
        sa = torch.cat([s, a], dim=-1)
        q1v = self.q1(sa)  # Shape: [batch_size, 1]
        q2v = self.q2(sa)  # Shape: [batch_size, 1]
        # q_target should now have shape [batch_size, 1] to match q1v and q2v
        loss_critic = F.mse_loss(q1v, q_target) + F.mse_loss(q2v, q_target)

        # Update critic
        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        # Actor update
        # For actor update, we need gradients, so recompute action and logp without inference_mode
        # Get action from actor (with gradients enabled)
        mu = self.actor(s)
        
        # Create distribution and sample (with gradients)
        dist = Normal(mu, torch.ones_like(mu) * self.action_std)
        new_a = dist.rsample()
        logp = dist.log_prob(new_a).sum(-1)  # Shape: [batch_size]
        
        # Apply action bounds if provided, otherwise use default (backward compatibility)
        if self.action_low is not None and self.action_high is not None:
            new_a = new_a.clamp(self.action_low, self.action_high)
        else:
            # Default behavior for backward compatibility (Bandit2D range)
            new_a = new_a.clamp(-1.5, 1.5)
        
        # Ensure new_a has correct shape for concatenation
        if new_a.dim() == 1:
            new_a = new_a.unsqueeze(0)
        sa_new = torch.cat([s, new_a], dim=-1)
        q1v = self.q1(sa_new)  # Shape: [batch_size, 1]
        q2v = self.q2(sa_new)  # Shape: [batch_size, 1]
        q_val = torch.min(q1v, q2v).squeeze(-1)  # Shape: [batch_size]
        # Actor loss: maximize (Q - alpha * logp), so minimize (alpha * logp - Q)
        loss_actor = (self.alpha * logp - q_val).mean()

        # Update actor
        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        # Soft target update
        if not hasattr(self, '_target_initialized') or not self._target_initialized:
            self._target_initialized = True
        
        # Soft update target networks
        with torch.no_grad():
            for param, target in zip(self.q1.parameters(), self.q1_target.parameters()):
                target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
            for param, target in zip(self.q2.parameters(), self.q2_target.parameters()):
                target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)

        return float(loss_actor.item()), float(loss_critic.item()), q1v.detach().squeeze()

    def to(self, device: str) -> 'SACAgent':
        """
        Move agent to specified device.
        
        Args:
            device: Device string ('cpu' or 'cuda')
        
        Returns:
            self
        """
        self.device = torch.device(device)
        self.actor = self.actor.to(self.device)
        self.q1 = self.q1.to(self.device)
        self.q2 = self.q2.to(self.device)
        self.q1_target = self.q1_target.to(self.device)
        self.q2_target = self.q2_target.to(self.device)
        # Move action bounds to device if they exist
        if self.action_low is not None:
            self.action_low = self.action_low.to(self.device)
        if self.action_high is not None:
            self.action_high = self.action_high.to(self.device)
        return self
