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
    """Policy network for continuous action spaces.

    Activation defaults to Tanh so the architecture matches BC pretrain
    checkpoints (`scripts/envs/*/_bc_pretrain.py` uses Tanh). The previous
    ReLU + `clamp(mean, -1.5, 1.5)` was a Bandit2D-only setup that prevented
    direct BC weight loading.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 activation: str = "tanh"):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self._act = torch.tanh if activation == "tanh" else F.relu

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = self._act(self.fc1(state))
        x = self._act(self.fc2(x))
        return self.mean(x)


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
        init_log_std: float = 0.0,
        bc_ckpt_path: Optional[str] = None,
        bc_blend_alpha: float = 1.0,
        policy_activation: str = "tanh",
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
        self.init_log_std = init_log_std

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device or "cpu")

        # Local policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, activation=policy_activation).to(self.device)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std, device=self.device))

        # Optional BC warm-start (load Linear weights from `scripts/envs/.../_bc_pretrain.py`
        # checkpoint into our PolicyNetwork's fc1, fc2, mean). Architecture must match
        # the BC pretrain script's MLP (Linear(state,256) → Tanh → Linear(256,256) →
        # Tanh → Linear(256,act)). Also overwrites log_std with the BC-pretrained value.
        if bc_ckpt_path is not None:
            try:
                bc = torch.load(bc_ckpt_path, map_location=self.device, weights_only=False)
                pol_sd = bc["policy"]
                a = float(bc_blend_alpha)
                a = max(0.0, min(1.0, a))
                # Blend: w ← a * w_BC + (1-a) * w_init. a=1 → pure BC; a=0 → keep random init.
                # Sequential keys 0.weight/0.bias (fc1), 2.weight/2.bias (fc2), 4.weight/4.bias (mean)
                def _blend(dst, src):
                    src = src.to(self.device, dtype=dst.dtype)
                    dst.data.mul_(1.0 - a).add_(src, alpha=a)
                _blend(self.policy.fc1.weight, pol_sd["0.weight"])
                _blend(self.policy.fc1.bias,   pol_sd["0.bias"])
                _blend(self.policy.fc2.weight, pol_sd["2.weight"])
                _blend(self.policy.fc2.bias,   pol_sd["2.bias"])
                _blend(self.policy.mean.weight, pol_sd["4.weight"])
                _blend(self.policy.mean.bias,   pol_sd["4.bias"])
                if "log_std" in bc and bc["log_std"] is not None:
                    ls = bc["log_std"]
                    if isinstance(ls, torch.Tensor):
                        _blend(self.log_std, ls)
                print(f"[FedKLAgent] BC warm-start ← {bc_ckpt_path} (blend α={a:.2f})")
            except Exception as e:
                print(f"[FedKLAgent] BC warm-start FAILED ({bc_ckpt_path}): {e}")
        
        # Value network (remains local, not aggregated)
        self.value_fn = ValueNetwork(state_dim, hidden_dim).to(self.device)
        
        # Global policy (reference) - updated from server
        self.global_policy = PolicyNetwork(state_dim, action_dim, hidden_dim, activation=policy_activation).to(self.device)
        self.global_log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))
        
        # Copy initial parameters to global policy
        self.global_policy.load_state_dict(self.policy.state_dict())
        self.global_log_std.data = self.log_std.data.clone()
        
        # For log_std annealing (bandit-friendly): decay std over rounds
        self.log_std_anneal_target = -2.0  # std ≈ 0.14 by end
        self.log_std_anneal_rounds = 40
        self.log_std_anneal_enabled = False  # set via anneal_log_std(round, enabled=True)
        
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
    
    def compute_local_kl_from_snapshot(
        self, 
        states: torch.Tensor, 
        snapshot_state_dict: Dict[str, torch.Tensor], 
        snapshot_log_std: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from local policy snapshot.
        KL(π_current || π_snapshot) for Gaussian policies.
        
        Args:
            states: Batch of states
            snapshot_state_dict: State dict of the policy snapshot
            snapshot_log_std: Log std parameter of the snapshot
            
        Returns:
            Mean KL divergence
        """
        if snapshot_state_dict is None:
            return torch.tensor(0.0, device=self.device)
        
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        # Current policy distribution
        mean_current = self.policy(states)
        std_current = torch.exp(self.log_std)
        
        # Create temporary policy for snapshot (avoids in-place mutation)
        # Infer hidden_dim from state_dict (fc1 weight shape: [hidden_dim, state_dim])
        if 'fc1.weight' in snapshot_state_dict:
            hidden_dim = snapshot_state_dict['fc1.weight'].shape[0]
        else:
            # Fallback: use the same hidden_dim as current policy (from fc1.out_features)
            hidden_dim = self.policy.fc1.out_features
        
        temp_policy = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim=hidden_dim)
        temp_policy.load_state_dict(snapshot_state_dict)
        temp_policy.to(self.device)
        temp_policy.eval()
        
        with torch.no_grad():
            mean_snapshot = temp_policy(states)
            std_snapshot = torch.exp(snapshot_log_std.to(self.device))
        
        # KL divergence for diagonal Gaussian: KL(π_current || π_snapshot)
        var_current = std_current.pow(2) + 1e-8
        var_snapshot = std_snapshot.pow(2) + 1e-8
        
        kl = (
            torch.log(std_snapshot / std_current)
            + (var_current + (mean_current - mean_snapshot).pow(2)) / (2.0 * var_snapshot)
            - 0.5
        )
        
        return kl.sum(dim=-1).mean()
    
    def update_global_policy(self):
        """Update the global policy reference from current local policy."""
        self.global_policy.load_state_dict(self.policy.state_dict())
        self.global_log_std.data = self.log_std.data.clone()
    
    def anneal_log_std(self, server_round: int, target: float = -2.0, decay_rounds: int = 40):
        """Linearly decay log_std from init_log_std to target over decay_rounds (bandit-friendly)."""
        if decay_rounds <= 0:
            return
        progress = min(1.0, server_round / decay_rounds)
        new_val = self.init_log_std + (target - self.init_log_std) * progress
        new_val = max(-5.0, min(2.0, new_val))
        self.log_std.data.fill_(new_val)
        self.global_log_std.data = self.log_std.data.clone()
    
    # ========= Update (similar to FedGuide.update) =========
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        epochs: int = 1,
        minibatch_size: Optional[int] = None,
        lambda_global: float = 0.1,
        lambda_local: float = 0.05,
        local_snapshot_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        local_snapshot_log_std: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update policy using PPO with KL penalties.
        
        Args:
            batch: Dict with keys 's', 'a', 'old_logp', 'ret', 'adv'
            epochs: Number of update epochs
            minibatch_size: Size of minibatches
            lambda_global: Weight for KL(π || π_global)
            lambda_local: Weight for KL(π || π_local_snapshot)
            local_snapshot_state_dict: State dict of policy at start of round
            local_snapshot_log_std: Log std of policy at start of round
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
                if lambda_local > 0.0 and local_snapshot_state_dict is not None and local_snapshot_log_std is not None:
                    kl_local = self.compute_local_kl_from_snapshot(
                        mb_s,
                        local_snapshot_state_dict,
                        local_snapshot_log_std
                    )
                
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