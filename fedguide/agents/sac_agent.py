import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma, self.tau, self.alpha = gamma, tau, alpha

        def mlp(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim))

        self.actor = mlp(state_dim, action_dim)
        self.q1 = mlp(state_dim + action_dim, 1)
        self.q2 = mlp(state_dim + action_dim, 1)
        self.q1_target = mlp(state_dim + action_dim, 1)
        self.q2_target = mlp(state_dim + action_dim, 1)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)

    def act(self, state, eval=False):
        mu = self.actor(state)
        dist = Normal(mu, torch.ones_like(mu) * 0.1)
        action = mu if eval else dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1).detach(), logp

    def update(self, replay):
        s, a, r, s2, done = replay
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

        # actor update
        new_a, logp = self.act(s)
        q1v = self.q1(torch.cat([s, new_a], dim=-1))
        q2v = self.q2(torch.cat([s, new_a], dim=-1))
        q_val = torch.min(q1v, q2v)
        loss_actor = (self.alpha * logp - q_val).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        # soft target update
        for param, target in zip(self.q1.parameters(), self.q1_target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)
        for param, target in zip(self.q2.parameters(), self.q2_target.parameters()):
            target.data.copy_(self.tau * param.data + (1 - self.tau) * target.data)

        return float(loss_actor.item() + loss_critic.item())
