# a2c_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class A2CAgent:
    """Advantage Actor-Critic agent (plug-and-play replacement for PPOAgent)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,      # kept for drop-in compatibility; unused
        gae_lambda: float = 0.95,   # kept for drop-in compatibility; unused
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        init_std: float = 0.1,
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.init_std = init_std

        # Actor (mean of Gaussian) and Critic
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.value.to(self.device)

        self.rebuild_optimizer()

    # ----------------------------------------------------------
    # Optimizer util (kept to match PPOAgent interface)
    # ----------------------------------------------------------
    def rebuild_optimizer(self):
        """Recreate optimizer to rebind new parameters."""
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=self.lr,
        )

    # ----------------------------------------------------------
    # Action interface
    # ----------------------------------------------------------
    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False):
        """
        Args:
            state: shape (..., state_dim) on the correct device
        Returns:
            action: numpy array
            logp:   log-prob tensor on device (kept for trainer usage)
        """
        mu = self.policy(state)
        std = torch.ones_like(mu) * self.init_std
        dist = Normal(mu, std)
        action = mu if deterministic else dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        return action.detach().cpu().numpy(), logp.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """
        Returns:
            logp, entropy, value
        """
        mu = self.policy(state)
        std = torch.ones_like(mu) * self.init_std
        dist = Normal(mu, std)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        v = self.value(state).squeeze(-1)
        return logp, entropy, v

    # ----------------------------------------------------------
    # Update from on-policy batch (keeps same tuple layout)
    # ----------------------------------------------------------
    def update(self, batch):
        """
        batch = (states, actions, logps_old, returns, advs)
        We ignore logps_old (no clipping in A2C) but keep the signature
        to stay drop-in compatible with LocalTrainer.
        """
        states, actions, _logps_old, returns, advs = batch

        # Evaluate current policy
        logps, entropy, values = self.evaluate(states, actions)

        # A2C losses
        policy_loss = -(advs.detach() * logps).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        return float(loss.detach().cpu().item())

    # ----------------------------------------------------------
    # Critic accessor (used by LocalTrainer)
    # ----------------------------------------------------------
    def value_fn(self, state: torch.Tensor):
        return self.value(state)
