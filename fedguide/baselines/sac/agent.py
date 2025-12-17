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
    ):
        self.gamma, self.tau, self.alpha = gamma, tau, alpha

        def mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )

        self.actor = mlp(state_dim, action_dim)
        self.q1 = mlp(state_dim + action_dim, 1)
        self.q2 = mlp(state_dim + action_dim, 1)
        self.q1_target = mlp(state_dim + action_dim, 1)
        self.q2_target = mlp(state_dim + action_dim, 1)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr
        )

        # Set device (allow override, default to cuda if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Move networks to device (matching original pattern)
        self.actor.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)
        # Also move target networks (original doesn't, but we should)
        self.q1_target.to(self.device)
        self.q2_target.to(self.device)

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
        mu = self.actor(state)
        dist = Normal(mu, torch.ones_like(mu) * 0.1)
        action = mu if eval else dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        return action.clamp(-1.5, 1.5).detach(), logp

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
        s = batch['s'].to(self.device)
        a = batch['a'].to(self.device)
        r = batch['r'].to(self.device)
        s2 = batch['s_next'].to(self.device)
        done = batch['done'].to(self.device)

        with torch.no_grad():
            next_a, logp_next = self.act(s2)
            q1t = self.q1_target(torch.cat([s2, next_a], dim=-1))
            q2t = self.q2_target(torch.cat([s2, next_a], dim=-1))
            q_target = r + self.gamma * (1 - done) * (torch.min(q1t, q2t) - self.alpha * logp_next)

        q1v = self.q1(torch.cat([s, a], dim=-1))
        q2v = self.q2(torch.cat([s, a], dim=-1))
        loss_critic = F.mse_loss(q1v, q_target) + F.mse_loss(q2v, q_target)

        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        # Actor update
        new_a, logp = self.act(s)
        q1v = self.q1(torch.cat([s, new_a], dim=-1))
        q2v = self.q2(torch.cat([s, new_a], dim=-1))
        q_val = torch.min(q1v, q2v)
        loss_actor = (self.alpha * logp - q_val).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        # Soft target update
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
        return self
