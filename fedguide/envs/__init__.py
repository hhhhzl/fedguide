import torch
from abc import ABC, abstractmethod
from typing import Tuple


class BaseAgent(ABC):
    """Agent wrapper: policy + optimizer + sampling."""

    def __init__(
            self,
            policy,
            lr=1e-3,
            **kwargs
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.momentum = None

    def act(self, state):
        dist = self.policy(torch.as_tensor(state, dtype=torch.float32))
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
