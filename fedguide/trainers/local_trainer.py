import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
import logging
import math
import os, json, numpy as np
from datetime import datetime


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32)


def _unpack_reset(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]  # (obs, info)
    return out


def _unpack_step(out):
    if len(out) == 4:
        obs, rew, done, info = out
        terminated, truncated = done, False
    else:
        obs, rew, terminated, truncated, info = out
    return obs, rew, bool(terminated), bool(truncated), (info or {})


def _gate_indices(env):
    grid = env.grid
    mid = env.size // 2
    row = grid[mid]
    open_cols = np.where(row == 1)[0]
    if len(open_cols) == 0:
        return mid, 0, 0
    return mid, int(open_cols.min()), int(open_cols.max())


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

    def eval_episode(self, max_steps=400, deterministic=True):
        obs = _unpack_reset(self.env)
        s = _to_numpy(obs).astype(np.float32)
        traj = [s.copy()]

        mid, jmin, jmax = _gate_indices(self.env)
        passed_gate = False
        reached_goal = False

        prev_i, prev_j = int(np.clip(s[0], 0, self.env.size-1)), int(np.clip(s[1], 0, self.env.size-1))

        for _ in range(max_steps):
            a, _ = self.agent.act(torch.as_tensor(s, dtype=torch.float32, device=self.device),
                                  deterministic=deterministic)
            step_out = self.env.step(_to_numpy(a))
            obs, rew, terminated, truncated, info = _unpack_step(step_out)
            s = _to_numpy(obs).astype(np.float32)
            traj.append(s.copy())

            i, j = int(np.clip(s[0], 0, self.env.size-1)), int(np.clip(s[1], 0, self.env.size-1))
            if not passed_gate:
                cross_mid = (prev_i < mid and i >= mid) or (prev_i > mid and i <= mid)
                if cross_mid and (jmin <= j <= jmax):
                    passed_gate = True

            if terminated or truncated:
                reached_goal = True if (terminated and rew > 0) else reached_goal
                break

            prev_i, prev_j = i, j

        return np.asarray(traj, dtype=np.float32), bool(passed_gate), bool(reached_goal)

    def save_eval(self, cid: str, rnd: int, outdir="/tmp/fedguide"):
        traj, passed_gate, reached_goal = self.eval_episode()
        d = os.path.join(outdir, f"client_{cid}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"round_{rnd}_traj.npy"), traj)
        meta = {"round": int(rnd), "passed_gate": bool(passed_gate), "reached_goal": bool(reached_goal)}
        with open(os.path.join(d, f"round_{rnd}_meta.json"), "w") as f:
            json.dump(meta, f)
        return passed_gate
