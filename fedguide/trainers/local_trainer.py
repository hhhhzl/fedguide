import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
import logging
import math


class LocalTrainer:
    """
    Local Trainer for FedGuide:
    - Performs rollout and on-policy policy gradient update.
    - Includes KL-local regularization (trust region)
    - Includes KL-guide regularization (prior consistency)
    """

    def __init__(
        self,
        agent,
        env,
        prior: Optional[nn.Module] = None,
        lambda_local: float = 0.1,
        lambda_guide: float = 0.1,
        gamma: float = 0.99,
        n_steps: int = 200,
        device: Optional[str] = None,
        grad_clip: float = 1.0,
    ):
        self.agent = agent
        self.env = env
        self.prior = prior
        self.lambda_local = lambda_local
        self.lambda_guide = lambda_guide
        self.gamma = gamma
        self.n_steps = n_steps
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.agent.policy.to(self.device)
        if self.prior is not None and hasattr(self.prior, "to"):
            self.prior.to(self.device)

        self._last_log_probs = None
        self.logger = logging.getLogger("FedGuide.LocalTrainer")

    # ----------------------------------------------------------
    # Rollout
    # ----------------------------------------------------------
    @torch.no_grad()
    def rollout(self):
        """Collect on-policy trajectory from the environment."""
        states, actions, logps, rewards = [], [], [], []
        s, _ = self.env.reset() if isinstance(self.env.reset(), tuple) else (self.env.reset(), None)

        for _ in range(self.n_steps):
            s_tensor = torch.as_tensor(s, dtype=torch.float32, device=self.device)
            a, lp = self.agent.act(s_tensor)
            s2, r, done, *_ = self.env.step(a)
            states.append(s_tensor)
            actions.append(torch.as_tensor(a, dtype=torch.float32, device=self.device))
            logps.append(lp)
            rewards.append(torch.as_tensor(r, dtype=torch.float32, device=self.device))
            s = self.env.reset()[0] if done else s2

        states = torch.stack(states)
        actions = torch.stack(actions)
        logps = torch.stack(logps)
        rewards = torch.stack(rewards)

        returns = self._discounted_returns(rewards)
        return states, actions, logps, returns

    def _discounted_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute discounted return sequence."""
        returns = torch.zeros_like(rewards)
        running = torch.tensor(0.0, device=rewards.device)
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return returns

    # ----------------------------------------------------------
    # Loss computation
    # ----------------------------------------------------------
    def compute_loss(self, states, actions, logps, returns):
        values = self.agent.value_fn(states).squeeze(-1)
        advs = (returns - values).detach()
        pg_loss = -(advs * logps).mean()
        value_loss = F.mse_loss(values, returns)

        kl_local = torch.tensor(0.0, device=self.device)
        if self._last_log_probs is not None and len(self._last_log_probs) == len(logps):
            kl_local = (self._last_log_probs - logps).mean()

        kl_guide = torch.tensor(0.0, device=self.device)
        if self.prior is not None:
            guide_logp = self.prior.log_prob(actions, states)
            kl_guide = (guide_logp - logps).mean()

        loss = (
            pg_loss + 0.5 * value_loss
            + self.lambda_local * kl_local
            + self.lambda_guide * kl_guide
        )
        return loss, pg_loss, value_loss, kl_local, kl_guide

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    def train_one_round(self):
        self.agent.policy.train()
        states, actions, logps_old, returns = self.rollout()
        with torch.no_grad():
            values = self.agent.value_fn(states).squeeze(-1)
            advs = (returns - values)
        batch = (states, actions.float(), logps_old, returns, advs)
        loss = self.agent.update(batch)
        self._last_log_probs = logps_old
        return loss
