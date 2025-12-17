"""
FedKL Agent Implementation

This module implements the FedKL agent with KL divergence computation
between local and global policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Any, Iterable


def _to_device(module_or_tensor, device):
    if hasattr(module_or_tensor, "to"):
        return module_or_tensor.to(device)
    return module_or_tensor


class PolicyNetwork(nn.Module):
    """Policy network for continuous action spaces."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_dim) or (state_dim,)
        
        Returns:
            mean: (batch_size, action_dim)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        # Clamp to valid range for Bandit2D
        mean = torch.clamp(mean, -1.5, 1.5)
        return mean


class ValueNetwork(nn.Module):
    """Value network for critic."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value


class FedKLAgent(nn.Module):
    """
    FedKL Agent implementing KL divergence penalty for federated RL.
    
    The agent maintains a local policy and a reference to the global policy.
    It penalizes divergence from the global policy during local training.
    
    Key differences from FedGuide:
    - No prior or guidance models
    - Only policy and value networks (value remains local, policy is aggregated)
    - KL divergence computed against global policy snapshot
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
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device or "cpu")
        
        # Local policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))
        
        # Value network (remains local, not aggregated)
        self.value_fn = ValueNetwork(state_dim, hidden_dim).to(self.device)
        
        # Global policy (reference) - updated from server
        self.global_policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.global_log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))
        
        # Copy initial parameters to global policy
        self.global_policy.load_state_dict(self.policy.state_dict())
        self.global_log_std.data = self.log_std.data.clone()
        
        # Optimizer
        self.lr = lr
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + [self.log_std] + list(self.value_fn.parameters()),
            lr=lr
        )
        
        # Clamp log_std to reasonable range
        self.log_std.data.clamp_(-5.0, 2.0)
    
    # ========= Distribution / Evaluate (similar to FedGuide) =========
    def _dist(self, state: torch.Tensor):
        """Create distribution from policy output."""
        mu = self.policy(state)
        std = self.log_std.exp().clamp(min=1e-6)
        return torch.distributions.Normal(mu, std), mu
    
    def evaluate(self, state, action):
        """Evaluate log prob, entropy, value for given state-action pairs."""
        state = state.to(self.device).float()
        action = action.to(self.device).float()
        dist, mu = self._dist(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_fn(state).squeeze(-1)
        return log_prob, entropy, value, mu
    
    @torch.no_grad()
    def select_action(self, state, deterministic=False):
        """Select action using current policy."""
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        dist, mu = self._dist(state)
        action = mu if deterministic else dist.sample()
        action = torch.clamp(action, -1.5, 1.5)  #only for Bandit2D
        logp = dist.log_prob(action).sum(dim=-1)
        value = self.value_fn(state).squeeze(-1)
        
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()
    
    def compute_kl_divergence(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between local policy and global policy.
        KL(π_local || π_global) for Gaussian policies.
        """
        # Local policy distribution
        mean_local = self.policy(states)
        std_local = torch.exp(self.log_std).clone()
        
        # Global policy distribution (detached)
        with torch.no_grad():
            mean_global = self.global_policy(states)
            std_global = torch.exp(self.global_log_std).clone()
        
        # KL divergence for diagonal Gaussian
        var_local = std_local.pow(2) + 1e-8
        var_global = std_global.pow(2) + 1e-8
        
        kl = (
            torch.log(std_global / std_local)
            + (var_local + (mean_local - mean_global).pow(2)) / (2 * var_global)
            - 0.5
        )
        
        return kl.sum(dim=-1).mean()
    
    def update_global_policy(self):
        """Update the global policy reference from current local policy."""
        self.global_policy.load_state_dict(self.policy.state_dict())
        self.global_log_std.data = self.log_std.data.clone()
    
    # ========= Update (similar to FedGuide.update) =========
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        epochs: int = 1,
        minibatch_size: Optional[int] = None,
        lambda_global: float = 0.1,
        lambda_local: float = 0.05,
    ) -> Dict[str, float]:
        """
        Update policy using PPO with KL penalties.
        
        Args:
            batch: Dict with keys 's', 'a', 'old_logp', 'ret', 'adv'
            epochs: Number of update epochs
            minibatch_size: Size of minibatches
            lambda_global: Weight for KL(π || π_global)
            lambda_local: Weight for KL(π || π_local_snapshot)
        """
        s = batch["s"].to(self.device).float()
        a = batch["a"].to(self.device).float()
        old_logp = batch["old_logp"].to(self.device).float()
        ret = batch["ret"].to(self.device).float()
        adv = batch["adv"].to(self.device).float()
        
        # Normalize advantages (PPO standard)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        
        N = s.size(0)
        if not minibatch_size or minibatch_size <= 0:
            minibatch_size = N
        
        # Track metrics
        last_policy_loss = last_value_loss = 0.0
        last_kl_global = last_kl_local = 0.0
        last_entropy = 0.0
        clip_frac = 0.0
        approx_kl = 0.0
        
        for _ in range(epochs):
            perm = torch.randperm(N, device=self.device)
            for st in range(0, N, minibatch_size):
                idx = perm[st: st + minibatch_size]
                mb_s, mb_a, mb_old_logp, mb_ret, mb_adv = s[idx], a[idx], old_logp[idx], ret[idx], adv[idx]
                
                # Policy/value/entropy under current policy
                logp, entropy, value, mu = self.evaluate(mb_s, mb_a)
                
                # PPO clipped objective
                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(value, mb_ret)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # KL penalties
                kl_global = self.compute_kl_divergence(mb_s)
                
                # Local KL (if lambda_local > 0, compute KL from policy at start of round)
                kl_local = torch.tensor(0.0, device=self.device)
                if lambda_local > 0.0:
                    kl_local = (mb_old_logp - logp).mean().abs()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                    + lambda_global * kl_global
                    + lambda_local * kl_local
                )
                
                # Optimize
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + [self.log_std] + list(self.value_fn.parameters()),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Track metrics
                last_policy_loss = float(policy_loss.detach().cpu())
                last_value_loss = float(value_loss.detach().cpu())
                last_kl_global = float(kl_global.detach().cpu())
                last_kl_local = float(kl_local.detach().cpu())
                last_entropy = float(entropy.mean().detach().cpu())
                approx_kl = float((mb_old_logp - logp).mean().abs().detach().cpu())
                clip_frac = float(((ratio - 1.0).abs() > self.clip_eps).float().mean().detach().cpu())
        
        return {
            "loss/total": last_policy_loss + self.vf_coef * last_value_loss,
            "loss/policy": last_policy_loss,
            "loss/value": last_value_loss,
            "loss/kl_global": last_kl_global,
            "loss/kl_local": last_kl_local,
            "entropy": last_entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }
    
    # ========= Federated Aggregation (similar to FedGuide) =========
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters for federated aggregation."""
        # Only aggregate policy (not value network)
        return {
            "policy": {k: v.detach().cpu() for k, v in self.policy.state_dict().items()},
            "log_std": self.log_std.detach().cpu(),
        }
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set parameters from federated aggregation."""
        if "policy" in parameters:
            self.policy.load_state_dict(
                {k: v.to(self.device) for k, v in parameters["policy"].items()},
                strict=False
            )
        if "log_std" in parameters:
            self.log_std.data = parameters["log_std"].to(self.device).clone()
        
        # IMPORTANT: Update global policy reference after receiving new parameters
        # This ensures KL divergence is computed against the updated global policy
        self.update_global_policy()
    
    def parameters_iter(self) -> Iterable[torch.nn.Parameter]:
        """Iterator over all trainable parameters."""
        return list(self.policy.parameters()) + [self.log_std] + list(self.value_fn.parameters())
    
    def to(self, device: str):
        """Move agent to device."""
        self.device = torch.device(device)
        self.policy = self.policy.to(self.device)
        self.value_fn = self.value_fn.to(self.device)
        self.global_policy = self.global_policy.to(self.device)
        self.log_std.data = self.log_std.data.to(self.device)
        self.global_log_std.data = self.global_log_std.data.to(self.device)
        return self