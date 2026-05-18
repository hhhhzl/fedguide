"""
FedMomentum Agent Implementation

This module implements the FedMomentum agent with policy gradient computation support.
Based on PPO/FedKL agent structure but with gradient extraction capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Any, Tuple
from torch.distributions import Normal
import copy


def _to_device(module_or_tensor, device):
    if hasattr(module_or_tensor, "to"):
        return module_or_tensor.to(device)
    return module_or_tensor


_DEFAULT_ACTION_LOW: float = -1.0
_DEFAULT_ACTION_HIGH: float = 1.0


class PolicyNetwork(nn.Module):
    """Policy network for continuous action spaces.

    Tanh hidden activations match the paper's HalfCheetah setup and the
    repo's BC pretrain checkpoints.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        activation: str = "tanh",
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self._act = torch.tanh if activation == "tanh" else F.relu
        
    def forward(self, state):
        """Forward pass."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        x = self._act(self.fc1(state))
        x = self._act(self.fc2(x))
        # Output the raw mean. Bounding the mean before sampling clips
        # gradients on saturated outputs (the previous Bandit2D-only
        # `clamp(-1.5, 1.5)` here broke training on Reacher / HalfCheetah).
        # Action clipping is applied at sampling time using the env's actual
        # action_space bounds, not here.
        mean = self.mean(x)
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


class FedMomentumAgent(nn.Module):
    """
    FedMomentum Agent with policy gradient computation support.
    
    The agent maintains:
    - Policy network (aggregated): outputs mean action
    - Value network (local): estimates state value
    - Support for gradient extraction for momentum-based aggregation
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
        action_low: Optional[float] = None,
        action_high: Optional[float] = None,
        init_log_std: float = 0.0,
        bc_ckpt_path: Optional[str] = None,
        bc_blend_alpha: float = 1.0,
        policy_activation: str = "tanh",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        # Bounds used to clip *sampled actions* (NOT the policy mean) at
        # sampling time. Defaults to [-1, 1] which matches MuJoCo MuJoCo
        # cont-control envs (Reacher / HalfCheetah / Hopper / Walker / Ant).
        self.action_low = float(action_low) if action_low is not None else _DEFAULT_ACTION_LOW
        self.action_high = float(action_high) if action_high is not None else _DEFAULT_ACTION_HIGH
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device or "cpu")
        
        # Policy network (will be aggregated)
        self.policy = PolicyNetwork(
            state_dim, action_dim, hidden_dim, activation=policy_activation
        ).to(self.device)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(init_log_std), device=self.device)
        )

        # Optional BC warm-start from scripts/envs/*/_bc_pretrain.py checkpoints.
        # Checkpoints store a Sequential policy with keys 0/2/4; map them into
        # fc1/fc2/mean and blend with random init when alpha < 1.
        if bc_ckpt_path is not None:
            try:
                bc = torch.load(bc_ckpt_path, map_location=self.device, weights_only=False)
                pol_sd = bc.get("policy", bc) if isinstance(bc, dict) else bc
                a = max(0.0, min(1.0, float(bc_blend_alpha)))

                def _blend(dst: torch.Tensor, src: torch.Tensor) -> None:
                    if src.shape != dst.shape:
                        return
                    src_t = src.to(self.device, dtype=dst.dtype)
                    dst.data.mul_(1.0 - a).add_(src_t, alpha=a)

                if isinstance(pol_sd, dict):
                    # Sequential BC checkpoint layout.
                    if "0.weight" in pol_sd:
                        _blend(self.policy.fc1.weight, pol_sd["0.weight"])
                        _blend(self.policy.fc1.bias, pol_sd["0.bias"])
                        _blend(self.policy.fc2.weight, pol_sd["2.weight"])
                        _blend(self.policy.fc2.bias, pol_sd["2.bias"])
                        _blend(self.policy.mean.weight, pol_sd["4.weight"])
                        _blend(self.policy.mean.bias, pol_sd["4.bias"])
                    # Native FedMomentum/FedKL-style layout.
                    elif "fc1.weight" in pol_sd:
                        _blend(self.policy.fc1.weight, pol_sd["fc1.weight"])
                        _blend(self.policy.fc1.bias, pol_sd["fc1.bias"])
                        _blend(self.policy.fc2.weight, pol_sd["fc2.weight"])
                        _blend(self.policy.fc2.bias, pol_sd["fc2.bias"])
                        _blend(self.policy.mean.weight, pol_sd["mean.weight"])
                        _blend(self.policy.mean.bias, pol_sd["mean.bias"])

                if isinstance(bc, dict) and "log_std" in bc and bc["log_std"] is not None:
                    ls = bc["log_std"]
                    if isinstance(ls, torch.Tensor) and ls.shape == self.log_std.shape:
                        _blend(self.log_std, ls)
                print(f"[FedMomentumAgent] BC warm-start ← {bc_ckpt_path} (blend α={a:.2f})")
            except Exception as e:
                print(f"[FedMomentumAgent] BC warm-start FAILED ({bc_ckpt_path}): {e}")
        
        # Value network (remains local, not aggregated)
        self.value_fn = ValueNetwork(state_dim, hidden_dim).to(self.device)
        
        # Optimizer
        self.lr = lr
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + [self.log_std] + list(self.value_fn.parameters()),
            lr=lr
        )
        
        # Clamp log_std to reasonable range
        self.log_std.data.clamp_(-5.0, 2.0)
    
    # ========= Distribution / Evaluate =========
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
        # Store and train against the same action actually applied to the env.
        # The previous code stored a clipped action but kept the old log-prob of
        # the unclipped sample, making PPO/SVRPG ratios inconsistent.
        action = torch.clamp(action, self.action_low, self.action_high)
        logp = dist.log_prob(action).sum(dim=-1)
        value = self.value_fn(state).squeeze(-1)
        
        # Return numpy arrays
        if state.dim() == 2 and state.shape[0] == 1:
            action = action.squeeze(0)
            logp = logp.squeeze(0)
            value = value.squeeze(0)
        
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()
    
    def compute_policy_gradient(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_logps: Optional[torch.Tensor] = None,
        use_clipped: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute policy gradient ascent direction: ∇_θ log π_θ(a|s) * A(s,a)
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            advantages: Batch of advantages [batch_size]
            old_logps: Old log probabilities for PPO clipping [batch_size]
            use_clipped: If True, use PPO clipped objective
        
        Returns:
            Dictionary of policy gradients (keyed by parameter name)
        """
        # Move to device
        states = states.to(self.device).float()
        actions = actions.to(self.device).float()
        advantages = advantages.to(self.device).detach()  # Detach advantages
        
        # Compute current log probabilities
        logps, _, _, _ = self.evaluate(states, actions)
        
        # Compute policy loss
        if use_clipped and old_logps is not None:
            # PPO clipped objective
            old_logps = old_logps.to(self.device).detach()
            ratio = torch.exp(logps - old_logps)
            ratio = torch.clamp(ratio, -10.0, 10.0)  # Prevent exp overflow
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
        else:
            # Vanilla policy gradient
            policy_loss = -(logps * advantages).mean()
        
        # Compute gradients (but don't update parameters yet)
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Extract ascent directions in the same key order as get_parameters()
        # / Flower flat list. `policy_loss = -J`, so autograd gives -∇J; the
        # server applies θ ← θ + λg, therefore return +∇J.
        # (sorted policy state_dict keys, then log_std) — required for server aggregation.
        policy_grad = {}
        params_by_name = {n: p for n, p in self.policy.named_parameters()}
        for key in sorted(params_by_name.keys()):
            param = params_by_name[key]
            g = param.grad if param.grad is not None else torch.zeros_like(param.data)
            policy_grad[f"policy.{key}"] = (-g).clone()
        g_ls = self.log_std.grad if self.log_std.grad is not None else torch.zeros_like(self.log_std.data)
        policy_grad["log_std"] = (-g_ls).clone()
        
        # Clear gradients (we'll compute them again during actual update)
        self.optimizer.zero_grad()
        
        return policy_grad
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        epochs: int = 4,
        minibatch_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Update policy and value networks using PPO algorithm.
        
        Args:
            batch: Dictionary with keys:
                - 's': states [batch_size, state_dim]
                - 'a': actions [batch_size, action_dim]
                - 'old_logp': old log probabilities [batch_size]
                - 'ret': returns [batch_size]
                - 'adv': advantages [batch_size]
            epochs: Number of update epochs
            minibatch_size: Size of minibatches for update
        
        Returns:
            Dictionary with loss metrics
        """
        states = batch['s'].to(self.device)
        actions = batch['a'].to(self.device)
        returns = batch['ret'].to(self.device)
        advantages = batch['adv'].to(self.device)
        old_logps = batch.get('old_logp', None)
        if old_logps is not None:
            old_logps = old_logps.to(self.device)
        
        batch_size = states.shape[0]
        if minibatch_size is None:
            minibatch_size = batch_size
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
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
                mb_old_logps = old_logps[batch_indices] if old_logps is not None else None
                
                # Evaluate current policy (evaluate returns log_prob, entropy, value, mu)
                logps, entropy, values, _mu = self.evaluate(mb_states, mb_actions)
                
                # Compute policy loss (PPO clipped objective)
                if mb_old_logps is not None:
                    log_ratio = logps - mb_old_logps
                    log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
                    ratio = torch.exp(log_ratio)
                    
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                else:
                    policy_loss = -(logps * mb_advantages).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Compute entropy bonus
                entropy_bonus = entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + [self.log_std] + list(self.value_fn.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
                num_updates += 1
        
        # Average over updates
        if num_updates > 0:
            avg_policy_loss = total_policy_loss / num_updates
            avg_value_loss = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates
        else:
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            avg_entropy = 0.0
        
        return {
            "loss": avg_policy_loss + self.vf_coef * avg_value_loss - self.ent_coef * avg_entropy,
            "loss/policy": avg_policy_loss,
            "loss/value": avg_value_loss,
            "loss/entropy": avg_entropy,
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for federated aggregation.
        
        Returns:
            Dictionary with policy parameters
        """
        return {
            "policy": {k: v.detach().cpu() for k, v in self.policy.state_dict().items()},
            "log_std": self.log_std.detach().cpu(),
        }
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Set parameters from federated aggregation.
        
        Args:
            parameters: Dictionary with policy parameters
        """
        if "policy" in parameters:
            self.policy.load_state_dict(
                {k: v.to(self.device) for k, v in parameters["policy"].items()},
                strict=False
            )
        
        if "log_std" in parameters:
            self.log_std.data = parameters["log_std"].to(self.device)
    
    def rebuild_optimizer(self):
        """Recreate optimizer after parameter aggregation."""
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + [self.log_std] + list(self.value_fn.parameters()),
            lr=self.lr
        )
    
    def to(self, device: str):
        """Move agent to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.policy = self.policy.to(device)
        self.value_fn = self.value_fn.to(device)
        self.log_std = self.log_std.to(device)
        return self
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            "policy": self.policy.state_dict(),
            "log_std": self.log_std.data,
            "value_fn": self.value_fn.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        if "policy" in state_dict:
            self.policy.load_state_dict(state_dict["policy"])
        if "log_std" in state_dict:
            self.log_std.data = state_dict["log_std"]
        if "value_fn" in state_dict:
            self.value_fn.load_state_dict(state_dict["value_fn"])
