import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, gae_lambda=0.95):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda

        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )

        self.lr = lr
        self.rebuild_optimizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.value.to(self.device)

    def rebuild_optimizer(self):
        """Recreate optimizer to rebind new parameters."""
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=self.lr
        )

    def act(self, state, deterministic=False):
        mu = self.policy(state)
        dist = Normal(mu, torch.ones_like(mu) * 0.1)
        action = mu if deterministic else dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        return action.detach().cpu().numpy(), logp.detach()

    def evaluate(self, state, action):
        mu = self.policy(state)
        dist = Normal(mu, torch.ones_like(mu) * 0.1)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        v = self.value(state).squeeze(-1)
        return logp, entropy, v

    def update(self, batch):
        states, actions, logps_old, returns, advs = batch
        logps, entropy, values = self.evaluate(states, actions)

        ratio = torch.exp(logps - logps_old)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def value_fn(self, state):
        return self.value(state)
