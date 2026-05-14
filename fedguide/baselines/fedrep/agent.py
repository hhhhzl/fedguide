"""
FedRep Agent Implementation

FedRep splits the policy network into:
- Shared encoder (aggregated across clients)
- Client-specific head (stays local)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Any, Iterable


class PolicyEncoder(nn.Module):
    """Shared encoder that gets aggregated in FedRep."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, state):
        """Forward pass through encoder."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x


class PolicyHead(nn.Module):
    """Client-specific head that stays local."""

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.mean = nn.Linear(hidden_dim, action_dim)

    def forward(self, encoded_state):
        # NB: do NOT clamp the mean here. A pre-`Normal()` clamp kills
        # gradients on saturated outputs (the previous Bandit2D-only
        # `clamp(-1.5, 1.5)` broke Reacher / HalfCheetah). Action clipping
        # is applied at sampling time using env action bounds.
        return self.mean(encoded_state)


class ValueNetwork(nn.Module):
    """Value network (stays local, not aggregated)."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """Forward pass through value network."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value


class FedRepAgent(nn.Module):
    """
    FedRep Agent with split architecture.
    
    - Encoder: aggregated across clients
    - Head: stays local
    - Value: stays local
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
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = float(action_low) if action_low is not None else -1.0
        self.action_high = float(action_high) if action_high is not None else 1.0
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device or "cpu")

        # Split architecture: Encoder (shared, FedAvg'd) + Head (local).
        self.encoder = PolicyEncoder(state_dim, hidden_dim).to(self.device)
        self.head = PolicyHead(hidden_dim, action_dim).to(self.device)
        # log_std stays local (not in `get_parameters`).
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))

        # Value network (stays local)
        self.value_fn = ValueNetwork(state_dim, hidden_dim).to(self.device)

        # Two optimizers — one per FedRep phase. This lets us cleanly
        # alternate "head/personal phase" and "rep/encoder phase" without
        # toggling requires_grad in a hot loop.
        self.lr = lr
        self.optimizer_head = torch.optim.Adam(
            list(self.head.parameters())
            + [self.log_std]
            + list(self.value_fn.parameters()),
            lr=lr,
        )
        self.optimizer_rep = torch.optim.Adam(
            list(self.encoder.parameters()),
            lr=lr,
        )
        # Backward-compat: leave `self.optimizer` as the head+value optimizer
        # (some legacy callers may still reference it). FedRep two-phase
        # callers should use `update(..., phase="head"/"rep")` instead.
        self.optimizer = self.optimizer_head

        # Clamp log_std to reasonable range
        self.log_std.data.clamp_(-5.0, 2.0)
    
    def _dist(self, state: torch.Tensor):
        """Create distribution from policy output."""
        encoded = self.encoder(state)
        mu = self.head(encoded)
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
        # log_prob computed on unclamped sample so gradients still flow.
        logp = dist.log_prob(action).sum(dim=-1)
        action = torch.clamp(action, self.action_low, self.action_high)
        value = self.value_fn(state).squeeze(-1)
        
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        epochs: int = 1,
        minibatch_size: Optional[int] = None,
        phase: str = "both",
    ) -> Dict[str, float]:
        """
        FedRep local update with explicit two-phase support.

        FedRep (Collins et al. 2021) alternates:
          - phase="head": freeze the shared encoder, update the personalized
            head + log_std + value network for E_h epochs.
          - phase="rep":  freeze head + log_std + value, update only the
            shared encoder for E_r epochs.

        ``phase="both"`` (default) preserves the legacy single-phase
        behaviour for callers that still expect it; the FedRep trainer
        should call this twice per round, once with each phase.

        Args:
            batch: Dict with keys 's', 'a', 'old_logp', 'ret', 'adv'
            epochs: Number of update epochs (E_h or E_r depending on phase).
            minibatch_size: Size of minibatches.
            phase: ``"head"``, ``"rep"``, or ``"both"``.
        """
        s = batch["s"].to(self.device).float()
        a = batch["a"].to(self.device).float()
        old_logp = batch["old_logp"].to(self.device).float()
        ret = batch["ret"].to(self.device).float()
        adv = batch["adv"].to(self.device).float()

        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        N = s.size(0)
        if not minibatch_size or minibatch_size <= 0:
            minibatch_size = N

        encoder_params = list(self.encoder.parameters())
        head_params = list(self.head.parameters()) + [self.log_std] + list(self.value_fn.parameters())

        if phase == "head":
            for p in encoder_params:
                p.requires_grad = False
            for p in head_params:
                p.requires_grad = True
            optimizer = self.optimizer_head
            clip_params = head_params
        elif phase == "rep":
            for p in encoder_params:
                p.requires_grad = True
            for p in head_params:
                p.requires_grad = False
            optimizer = self.optimizer_rep
            clip_params = encoder_params
        else:  # "both" — legacy single-phase
            for p in encoder_params + head_params:
                p.requires_grad = True
            optimizer = self.optimizer  # = optimizer_head (head+value)
            clip_params = encoder_params + head_params

        last_policy_loss = last_value_loss = 0.0
        last_entropy = 0.0
        clip_frac = 0.0
        approx_kl = 0.0

        for _ in range(epochs):
            perm = torch.randperm(N, device=self.device)
            for st in range(0, N, minibatch_size):
                idx = perm[st: st + minibatch_size]
                mb_s, mb_a, mb_old_logp, mb_ret, mb_adv = s[idx], a[idx], old_logp[idx], ret[idx], adv[idx]

                logp, entropy, value, mu = self.evaluate(mb_s, mb_a)

                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, mb_ret)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                # Restore zero_grad on BOTH optimizers since cross-phase
                # gradients can leak otherwise.
                self.optimizer_head.zero_grad(set_to_none=True)
                self.optimizer_rep.zero_grad(set_to_none=True)
                loss.backward()

                nn.utils.clip_grad_norm_(clip_params, self.max_grad_norm)
                optimizer.step()

                last_policy_loss = float(policy_loss.detach().cpu())
                last_value_loss = float(value_loss.detach().cpu())
                last_entropy = float(entropy.mean().detach().cpu())
                approx_kl = float((mb_old_logp - logp).mean().abs().detach().cpu())
                clip_frac = float(((ratio - 1.0).abs() > self.clip_eps).float().mean().detach().cpu())

        # Re-enable grads for everyone so the next phase's freeze starts clean.
        for p in encoder_params + head_params:
            p.requires_grad = True

        return {
            "loss/total": last_policy_loss + self.vf_coef * last_value_loss,
            "loss/policy": last_policy_loss,
            "loss/value": last_value_loss,
            "entropy": last_entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "phase": 0 if phase == "head" else (1 if phase == "rep" else 2),
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters for federated aggregation - ONLY ENCODER."""
        return {
            "encoder": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
        }
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set parameters from federated aggregation - ONLY ENCODER."""
        if "encoder" in parameters:
            self.encoder.load_state_dict(
                {k: v.to(self.device) for k, v in parameters["encoder"].items()},
                strict=False
            )
    
    def parameters_iter(self) -> Iterable[torch.nn.Parameter]:
        """Iterator over all trainable parameters."""
        return (
            list(self.encoder.parameters()) + 
            list(self.head.parameters()) + 
            [self.log_std] + 
            list(self.value_fn.parameters())
        )
    
    def to(self, device: str):
        """Move agent to device."""
        self.device = torch.device(device)
        self.encoder = self.encoder.to(self.device)
        self.head = self.head.to(self.device)
        self.value_fn = self.value_fn.to(self.device)
        self.log_std.data = self.log_std.data.to(self.device)
        return self

