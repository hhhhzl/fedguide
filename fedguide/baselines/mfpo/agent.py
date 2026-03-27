"""
MFPO (Momentum-assisted Federated Policy Optimization) — 1:1 port of
MFPO-INFOCOM24/code/agent/worker_continuous.py and worker_discrete.py (CartPole path).

Reference: "A Framework for Federated Reinforcement Learning with Interaction
and Communication Efficiency" (INFOCOM 2024).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# ---------------------------------------------------------------------------
# MFPO util.py (minimal)
# ---------------------------------------------------------------------------


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def _reset_env(env) -> np.ndarray:
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env, action) -> Tuple[np.ndarray, float, bool]:
    out = env.step(action)
    if len(out) == 5:
        state, reward, terminated, truncated, _ = out
        return state, float(reward), bool(terminated or truncated)
    state, reward, done, _ = out
    return state, float(reward), bool(done)


# ---------------------------------------------------------------------------
# Continuous worker (MFPO worker_continuous.py)
# ---------------------------------------------------------------------------


class policy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(policy, self).__init__()
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
        self.network = nn.Sequential(
            init_(nn.Linear(in_dim, 256)),
            nn.Tanh(),
            init_(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.output = init_(nn.Linear(256, out_dim))
        self.output_ = init_(nn.Linear(256, out_dim))

    def forward(self, state):
        s = self.network(state)
        mu = self.output(s)
        sigma = self.output_(s)
        return mu, sigma


class critic(nn.Module):
    def __init__(self, in_dim):
        super(critic, self).__init__()
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
        self.network = nn.Sequential(
            init_(nn.Linear(in_dim, 256)),
            nn.Tanh(),
            init_(nn.Linear(256, 256)),
            nn.Tanh(),
            init_(nn.Linear(256, 1)),
        )

    def forward(self, x):
        c = self.network(x)
        return c


class MFPOContinuousWorker(nn.Module):
    """1:1 logic from MFPO Worker_continuous."""

    def __init__(
        self,
        env,
        method_conf: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.method_conf = method_conf
        self.fault_type = method_conf["fault_type"]
        self.env = env
        dev = device or "cpu"
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        self.learning_rate_a = method_conf["learning_rate_a"]
        self.learning_rate_c = method_conf["learning_rate_c"]
        self.max_step = 1000
        self.gamma = method_conf["gamma"]
        self.c = method_conf["c"]
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.network = policy(self.observation_space, self.action_space)
        self.old_network = policy(self.observation_space, self.action_space)
        self.critic = critic(self.observation_space)
        self.target = critic(self.observation_space)

        self.to(self.device)

        self.optimizer_new = torch.optim.Adam(
            self.network.network.parameters(),
            lr=self.learning_rate_a,
            eps=method_conf["eps"],
            weight_decay=1e-6,
        )
        self.optimizer_old = torch.optim.Adam(
            self.old_network.network.parameters(),
            lr=self.learning_rate_a,
            eps=method_conf["eps"],
            weight_decay=1e-6,
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate_c,
            eps=method_conf["eps"],
            weight_decay=1e-6,
        )
        self.lr_scheduler_new = optim.lr_scheduler.ExponentialLR(
            self.optimizer_new, method_conf["decay_rate"]
        )
        self.lr_scheduler_old = optim.lr_scheduler.ExponentialLR(
            self.optimizer_old, method_conf["decay_rate"]
        )
        self.lr_scheduler_critic = optim.lr_scheduler.ExponentialLR(
            self.optimizer_critic, method_conf["decay_rate"]
        )
        self.pi = Variable(torch.FloatTensor([math.pi]).to(self.device))

    def normal(self, x, mu, sigma_sq):
        a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
        b = 1 / (2 * sigma_sq * self.pi.expand_as(sigma_sq)).sqrt()
        return a * b

    def gen_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        mu, sigma = self.network(state)
        sigma = F.softplus(sigma)
        eps = torch.randn(mu.size(), device=self.device)
        action = (mu + sigma.sqrt() * Variable(eps)).clamp(
            -self.max_action, self.max_action
        ).data
        prob = self.normal(action, mu, sigma)
        log_prob = prob.log()
        return action, log_prob

    def gen_action_prob(self, state, action):
        state = torch.from_numpy(state).float().to(self.device)
        mu, sigma = self.old_network(state)
        sigma = F.softplus(sigma)
        prob = self.normal(action.to(self.device), mu, sigma)
        log_prob = prob.log()
        return log_prob

    def gen_action_prob_new(self, state, action):
        state = torch.from_numpy(state).float().to(self.device)
        mu, sigma = self.network(state)
        sigma = F.softplus(sigma)
        prob = self.normal(action.to(self.device), mu, sigma)
        log_prob = prob.log()
        return log_prob

    def gen_critic(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        v = self.critic(state)
        return v

    def collect_trajectory(self, batch_size):
        state_batch = []
        action_batch = []
        action_prob_batch = []
        batch_weights: List[float] = []
        critic_batch = []
        state_prime_batch = []
        r_batch = []
        for _ in range(batch_size):
            state = _reset_env(self.env)
            reward, done = 0, False
            reward_batch = []
            step = 0
            while True:
                step += 1
                action, action_prob = self.gen_action(state)

                state_batch.append(state)
                v = self.gen_critic(state)
                critic_batch.append(v)
                state, reward, done = _step_env(self.env, action.detach().cpu().numpy())
                reward_batch.append(reward)
                r_batch.append(reward)
                action_batch.append(action)
                state_prime_batch.append(state)
                action_prob_batch.append(action_prob)

                if done:
                    returns = []
                    R = 0
                    for r in reward_batch[::-1]:
                        R = r + self.gamma * R
                        returns.insert(0, R)
                    returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

                    advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                    batch_weights.extend(advantage.detach().cpu().tolist())
                    break

        batch_weights = torch.as_tensor(batch_weights, dtype=torch.float32, device=self.device)

        state_prime_batch = torch.from_numpy(
            np.stack(state_prime_batch, axis=0).astype(np.float32, copy=False)
        ).to(self.device)
        r_batch = torch.as_tensor(r_batch, dtype=torch.float32, device=self.device)

        return (
            batch_weights,
            state_batch,
            action_batch,
            action_prob_batch,
            critic_batch,
            state_prime_batch,
            r_batch,
        )

    def train(self, batch_size, step):
        (
            returns,
            state_batch,
            action_batch,
            action_prob_batch,
            critic_batch,
            state_prime_batch,
            reward_batch,
        ) = self.collect_trajectory(batch_size)

        critic_prime = self.target(state_prime_batch)
        q = self.gamma * critic_prime.detach()
        q = q.squeeze()
        q = q + reward_batch.detach()

        loss = F.mse_loss(q, torch.stack(critic_batch).squeeze(-1))
        self.optimizer_critic.zero_grad()

        returns = returns.unsqueeze(-1).repeat(1, self.action_space)
        advantage = returns - torch.stack(critic_batch).detach()

        grad = [item.grad for item in self.network.parameters()]

        self.optimizer_new.zero_grad()
        batch_loss = -(torch.stack(action_prob_batch) * advantage).mean()
        batch_loss = batch_loss + loss
        loss_total = float(batch_loss.detach().item())
        batch_loss.backward()
        self.optimizer_critic.step()

        old_logp = []
        for idx, _ in enumerate(state_batch):
            action_prob = self.gen_action_prob(state_batch[idx], action_batch[idx])
            old_logp.append(action_prob)
        old_logp = torch.stack(old_logp)

        ratios = torch.exp(old_logp.detach() - torch.stack(action_prob_batch).detach())

        loss_old = -(old_logp * advantage * ratios).mean()
        self.optimizer_old.zero_grad()
        loss_old.backward()

        grad_old = [item.grad for item in self.old_network.parameters()]

        if grad[0] is not None:
            for idx, item in enumerate(self.network.parameters()):
                item.grad = item.grad + (1 - self.c * self.learning_rate_a**2) * (
                    grad[idx] - grad_old[idx]
                )

        grad = [item.grad for item in self.network.parameters()]

        self.old_model = copy.deepcopy(self.network)

        self.optimizer_new.step()

        if step > self.method_conf["decay_start_iter_id"]:
            self.lr_scheduler_new.step()
            self.lr_scheduler_old.step()
            self.lr_scheduler_critic.step()

        metrics = {
            "loss": loss_total,
            "loss/critic": float(loss.detach().item()),
            "loss/policy_combined": loss_total,
        }
        return grad, metrics

    def test(self, i):
        sum_r = 0
        for _ in range(10):
            state = _reset_env(self.env)
            while True:
                action, _ = self.gen_action(state)
                state, reward, done = _step_env(self.env, action.detach().cpu().numpy())
                sum_r += reward
                if done:
                    break
        return sum_r / 10


# ---------------------------------------------------------------------------
# Discrete CartPole (MFPO worker_discrete.py — CartPole branch only)
# ---------------------------------------------------------------------------


class MFPODiscreteCartPoleWorker(nn.Module):
    """1:1 logic from MFPO Worker_discrete for CartPole-v1."""

    def __init__(
        self,
        env,
        method_conf: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.method_conf = method_conf
        self.fault_type = method_conf["fault_type"]
        self.env = env
        dev = device or "cpu"
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        self.learning_rate_a = method_conf["learning_rate_a"]
        self.learning_rate_c = method_conf["learning_rate_c"]
        self.max_step = 1000
        self.gamma = method_conf["gamma"]
        self.c = method_conf["c"]
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.network = nn.Sequential(
            nn.Linear(self.observation_space, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        self.old_model = nn.Sequential(
            nn.Linear(self.observation_space, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.observation_space, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.target = nn.Sequential(
            nn.Linear(self.observation_space, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.my_softmax = nn.Softmax(dim=-1)

        self.to(self.device)

        self.optimizer_new = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate_a,
            eps=method_conf["eps"],
            weight_decay=1e-6,
        )
        self.optimizer_old = torch.optim.Adam(
            self.old_model.parameters(),
            lr=self.learning_rate_a,
            eps=method_conf["eps"],
            weight_decay=1e-6,
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate_c,
            eps=method_conf["eps"],
            weight_decay=1e-6,
        )
        self.lr_scheduler_new = optim.lr_scheduler.ExponentialLR(
            self.optimizer_new, method_conf["decay_rate"]
        )
        self.lr_scheduler_old = optim.lr_scheduler.ExponentialLR(
            self.optimizer_old, method_conf["decay_rate"]
        )
        self.lr_scheduler_critic = optim.lr_scheduler.ExponentialLR(
            self.optimizer_critic, method_conf["decay_rate"]
        )

    def gen_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action_prob = self.my_softmax(self.network(state))
        action_prob = action_prob.detach().cpu().numpy()
        action = np.random.choice(self.action_space, 1, p=action_prob)[0]
        return action, action_prob[action]

    def gen_action_prob(self, state, action):
        state = torch.from_numpy(state).float().to(self.device)
        action_prob = self.my_softmax(self.old_model(state))
        prob = action_prob[action]
        return prob

    def gen_action_prob_new(self, state, action):
        state = torch.from_numpy(state).float().to(self.device)
        action_prob = self.my_softmax(self.network(state))
        prob = action_prob[action]
        return prob

    def gen_critic(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        v = self.critic(state)
        return v

    def collect_trajectory(self, batch_size):
        state_batch = []
        action_batch = []
        action_prob_batch = []
        batch_weights: List[float] = []
        critic_batch = []
        state_prime_batch = []
        r_batch = []

        for _ in range(batch_size):
            state = _reset_env(self.env)
            reward, done = 0, False
            reward_batch = []
            step = 0
            while True:
                step += 1
                action, action_prob = self.gen_action(state)
                state_batch.append(state)
                v = self.gen_critic(state)
                critic_batch.append(v)
                state, reward, done = _step_env(self.env, action)
                reward_batch.append(reward)
                r_batch.append(reward)
                action_batch.append(action)
                state_prime_batch.append(state)
                action_prob_batch.append(action_prob)

                if done or step >= self.max_step:
                    returns = []
                    R = 0
                    for r in reward_batch[::-1]:
                        R = r + self.gamma * R
                        returns.insert(0, R)
                    returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

                    advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                    batch_weights.extend(advantage.detach().cpu().tolist())
                    break

        batch_weights = torch.as_tensor(batch_weights, dtype=torch.float32, device=self.device)
        action_prob_batch = torch.as_tensor(action_prob_batch, dtype=torch.float32, device=self.device)

        critic_batch = torch.stack(critic_batch).squeeze(-1)
        state_prime_batch = torch.from_numpy(
            np.stack(state_prime_batch, axis=0).astype(np.float32, copy=False)
        ).to(self.device)
        r_batch = torch.as_tensor(r_batch, dtype=torch.float32, device=self.device)

        return (
            batch_weights,
            state_batch,
            action_batch,
            action_prob_batch,
            critic_batch,
            state_prime_batch,
            r_batch,
        )

    def train(self, batch_size, step):
        (
            returns,
            state_batch,
            action_batch,
            action_prob_batch,
            critic_batch,
            state_prime_batch,
            reward_batch,
        ) = self.collect_trajectory(batch_size)

        new_logp = []
        for idx, _ in enumerate(state_batch):
            action_prob = self.gen_action_prob_new(state_batch[idx], action_batch[idx])
            new_logp.append(action_prob)
        new_logp = torch.stack(new_logp)

        critic_prime = self.target(state_prime_batch)
        q = self.gamma * critic_prime
        q = q.squeeze()
        q = q + reward_batch

        loss = F.mse_loss(q, critic_batch)
        loss_critic_f = float(loss.detach().item())
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        advantage = returns - critic_batch

        batch_loss = -(torch.log(new_logp) * advantage).mean()
        loss_policy_f = float(batch_loss.detach().item())
        batch_loss.backward()

        old_logp = []
        for idx, _ in enumerate(state_batch):
            action_prob = self.gen_action_prob(state_batch[idx], action_batch[idx])
            old_logp.append(action_prob)
        old_logp = torch.stack(old_logp)

        ratios = torch.exp(torch.log(old_logp.detach()) - torch.log(action_prob_batch.detach()))

        loss_old = -(torch.log(old_logp) * advantage * ratios).mean()
        loss_old_f = float(loss_old.detach().item())
        self.optimizer_old.zero_grad()
        loss_old.backward()

        grad_old = [item.grad for item in self.old_model.parameters()]

        for idx, item in enumerate(self.network.parameters()):
            item.grad = item.grad - (1 - self.c * self.learning_rate_a**2) * grad_old[idx]

        grad = [item.grad for item in self.network.parameters()]

        self.old_model = copy.deepcopy(self.network)

        self.optimizer_new.step()

        if step > self.method_conf["decay_start_iter_id"]:
            self.lr_scheduler_new.step()
            self.lr_scheduler_old.step()
            self.lr_scheduler_critic.step()

        metrics = {
            "loss": loss_critic_f + loss_policy_f + loss_old_f,
            "loss/critic": loss_critic_f,
            "loss/policy": loss_policy_f + loss_old_f,
        }
        return grad, metrics

    def test(self, i):
        sum_r = 0
        for _ in range(10):
            state = _reset_env(self.env)
            while True:
                action, _ = self.gen_action(state)
                state, reward, done = _step_env(self.env, action)
                sum_r += reward
                if done:
                    break
        return sum_r / 10


# ---------------------------------------------------------------------------
# Federated wrapper (aggregates network / critic / target like MFPO server)
# ---------------------------------------------------------------------------


class MFPOAgent(nn.Module):
    """
    Wraps MFPO worker; exposes get_parameters / set_parameters for Flower.
    Server averages all three; client replaces only `average_type` module (MFPO server.py).
    """

    def __init__(
        self,
        worker: nn.Module,
        average_type: str,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.worker = worker
        self.average_type = average_type
        dev = device or "cpu"
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

    def get_parameters(self) -> Dict[str, Any]:
        w = self.worker
        return {
            "critic": {k: v.detach().cpu() for k, v in w.critic.state_dict().items()},
            "network": {k: v.detach().cpu() for k, v in w.network.state_dict().items()},
            "target": {k: v.detach().cpu() for k, v in w.target.state_dict().items()},
        }

    def set_parameters(self, parameters: Dict[str, Any]):
        w = self.worker
        at = self.average_type
        if at == "target" and "target" in parameters:
            w.target.load_state_dict(
                {k: v.to(self.device) for k, v in parameters["target"].items()}, strict=False
            )
        elif at == "network" and "network" in parameters:
            w.network.load_state_dict(
                {k: v.to(self.device) for k, v in parameters["network"].items()}, strict=False
            )
        elif at == "critic" and "critic" in parameters:
            w.critic.load_state_dict(
                {k: v.to(self.device) for k, v in parameters["critic"].items()}, strict=False
            )
        self.rebuild_optimizer()

    def rebuild_optimizer(self):
        """Recreate optimizers after partial weight sync (matches MFPO init)."""
        w = self.worker
        mc = w.method_conf
        if isinstance(w, MFPOContinuousWorker):
            w.optimizer_new = torch.optim.Adam(
                w.network.network.parameters(),
                lr=w.learning_rate_a,
                eps=mc["eps"],
                weight_decay=1e-6,
            )
            w.optimizer_old = torch.optim.Adam(
                w.old_network.network.parameters(),
                lr=w.learning_rate_a,
                eps=mc["eps"],
                weight_decay=1e-6,
            )
            w.optimizer_critic = torch.optim.Adam(
                w.critic.parameters(),
                lr=w.learning_rate_c,
                eps=mc["eps"],
                weight_decay=1e-6,
            )
        else:
            w.optimizer_new = torch.optim.Adam(
                w.network.parameters(),
                lr=w.learning_rate_a,
                eps=mc["eps"],
                weight_decay=1e-6,
            )
            w.optimizer_old = torch.optim.Adam(
                w.old_model.parameters(),
                lr=w.learning_rate_a,
                eps=mc["eps"],
                weight_decay=1e-6,
            )
            w.optimizer_critic = torch.optim.Adam(
                w.critic.parameters(),
                lr=w.learning_rate_c,
                eps=mc["eps"],
                weight_decay=1e-6,
            )
        w.lr_scheduler_new = optim.lr_scheduler.ExponentialLR(w.optimizer_new, mc["decay_rate"])
        w.lr_scheduler_old = optim.lr_scheduler.ExponentialLR(w.optimizer_old, mc["decay_rate"])
        w.lr_scheduler_critic = optim.lr_scheduler.ExponentialLR(
            w.optimizer_critic, mc["decay_rate"]
        )

    def to(self, device: Union[str, torch.device]):
        self.device = torch.device(device)
        self.worker.to(self.device)
        if isinstance(self.worker, MFPOContinuousWorker):
            self.worker.pi = Variable(torch.FloatTensor([math.pi]).to(self.device))
        return self
