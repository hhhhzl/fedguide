"""
FedSVRPG-M (arxiv:2405.19499) — strict local loop matching Algorithm 1 and Eq. (4).

One trajectory per local iteration k; REINFORCE-style policy gradient; importance
sampling weight w(τ|θ_{r-1}, θ_{r,k}); server update θ_{r+1} = θ_r + λ u_{r+1} with
u_{r+1} = (1/(η N K)) Σ_i Δ_r^(i), Δ = θ_{r,K} - θ_r.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from fedguide.utils.federated_render import maybe_save_federated_eval_video


def _discounted_returns_from_t(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    T = rewards.shape[0]
    G = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    for t in range(T):
        for h in range(t, T):
            G[t] = G[t] + (gamma ** (h - t)) * rewards[h]
    return G


def _policy_grad_dict_from_trajectory(
    agent: Any,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
) -> Dict[str, torch.Tensor]:
    device = agent.device
    states = states.to(device).float()
    actions = actions.to(device).float()
    rewards = rewards.to(device).float()
    G = _discounted_returns_from_t(rewards, gamma).detach()

    logps, _, _, _ = agent.evaluate(states, actions)
    loss = -(G * logps).sum()

    agent.optimizer.zero_grad()
    loss.backward()

    params_by_name = {n: p for n, p in agent.policy.named_parameters()}
    out: Dict[str, torch.Tensor] = {}
    for key in sorted(params_by_name.keys()):
        p = params_by_name[key]
        g = p.grad if p.grad is not None else torch.zeros_like(p.data)
        out[f"policy.{key}"] = g.detach().clone()
    g_ls = agent.log_std.grad if agent.log_std.grad is not None else torch.zeros_like(agent.log_std.data)
    out["log_std"] = g_ls.detach().clone()
    agent.optimizer.zero_grad()
    return out


def _log_prob_sum(agent: Any, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    logps, _, _, _ = agent.evaluate(states.to(agent.device).float(), actions.to(agent.device).float())
    return logps.sum()


def _flat_to_policy_payload(agent: Any, flat: List[np.ndarray]) -> Dict[str, Any]:
    """Rebuild policy state_dict + log_std from Flower flat list order."""
    idx = 0
    sd: Dict[str, torch.Tensor] = {}
    for key in sorted(agent.policy.state_dict().keys()):
        sd[key] = torch.tensor(flat[idx], dtype=torch.float32, device=agent.device)
        idx += 1
    log_std = torch.tensor(flat[idx], dtype=torch.float32, device=agent.device)
    return {"policy": sd, "log_std": log_std}


def _flat_to_grad_dict(agent: Any, flat: List[np.ndarray]) -> Dict[str, torch.Tensor]:
    gd: Dict[str, torch.Tensor] = {}
    idx = 0
    for key in sorted(agent.policy.state_dict().keys()):
        gd[f"policy.{key}"] = torch.tensor(flat[idx], dtype=torch.float32, device=agent.device)
        idx += 1
    gd["log_std"] = torch.tensor(flat[idx], dtype=torch.float32, device=agent.device)
    return gd


def _apply_parameter_step(
    agent: Any, grad_step: Dict[str, torch.Tensor], eta: float
) -> None:
    with torch.no_grad():
        params_by_name = {n: p for n, p in agent.policy.named_parameters()}
        for key in sorted(params_by_name.keys()):
            params_by_name[key].add_(eta * grad_step[f"policy.{key}"])
        agent.log_std.add_(eta * grad_step["log_std"])


class FedSVRPGMStrictTrainer:
    """
    Local training for one communication round (one client):
      θ_{r,0} = θ_r; for k=0..K-1: sample τ, compute u_{r,k} (Eq. 4), θ ← θ + η u_{r,k};
      Δ = θ_{r,K} - θ_r.
    """

    def __init__(
        self,
        agent: Any,
        env: Any,
        device: Optional[str] = None,
        gamma: float = 0.99,
        eta: float = 0.01,
        beta: float = 0.2,
        local_steps_k: int = 5,
        max_horizon: int = 500,
        eval_episodes: int = 1,
        is_w_clip: Tuple[float, float] = (1e-8, 1e4),
        render_eval: bool = False,
        render_mode: str = "video",
        render_save_dir: Optional[str] = None,
        render_every_n_rounds: int = 10,
        render_episodes: int = 5,
        render_client_tag: str = "0",
    ):
        self.agent = agent
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eta = eta
        self.beta = beta
        self.local_steps_k = local_steps_k
        self.max_horizon = max_horizon
        self.eval_episodes = eval_episodes
        self.is_w_clip = is_w_clip
        self.server_round = 0
        self.render_eval = render_eval
        self.render_mode = render_mode
        self.render_save_dir = render_save_dir
        self.render_every_n_rounds = render_every_n_rounds
        self.render_episodes = render_episodes
        self.render_client_tag = render_client_tag
        self.n_steps = int(local_steps_k * max_horizon)

        reset_result = self.env.reset()
        self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        self.last_actions: Optional[np.ndarray] = None

    def set_server_round(self, rnd: int) -> None:
        self.server_round = int(rnd)

    def _sample_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_list, act_list, rew_list = [], [], []
        obs = self._obs
        for _ in range(self.max_horizon):
            a, _lp, _v = self.agent.select_action(obs, deterministic=False)
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            obs_t = torch.tensor(obs, dtype=torch.float32)
            act_t = torch.tensor(a, dtype=torch.float32)
            next_obs, r, terminated, truncated, _ = self.env.step(a)
            done = bool(terminated) or bool(truncated)
            obs_list.append(obs_t)
            act_list.append(act_t)
            rew_list.append(float(r))
            obs = next_obs
            self._obs = obs
            if done:
                rr = self.env.reset()
                self._obs = rr[0] if isinstance(rr, tuple) else rr
                break
        if not obs_list:
            obs_list.append(torch.tensor(self._obs, dtype=torch.float32))
            act_list.append(torch.zeros(self.agent.action_dim, dtype=torch.float32))
            rew_list.append(0.0)
        states = torch.stack(obs_list)
        actions = torch.stack(act_list)
        rewards = torch.tensor(rew_list, dtype=torch.float32, device=self.agent.device)
        return states, actions, rewards

    def _save_policy_snapshot(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        snap = {k: v.clone().detach() for k, v in self.agent.policy.state_dict().items()}
        ls = self.agent.log_std.data.clone().detach()
        return snap, ls

    def _load_policy_snapshot(self, snap: Dict[str, torch.Tensor], ls: torch.Tensor) -> None:
        self.agent.policy.load_state_dict(snap)
        self.agent.log_std.data.copy_(ls)

    def train_one_round(self, fed: Dict[str, Any]) -> Tuple[Dict[str, float], List[np.ndarray]]:
        """
        fed keys:
          u_r_flat: list of numpy (server u_r for Eq. 4)
          theta_prev_flat: list of numpy (global θ_{r-1} for IS and g(τ|θ_{r-1}))
        """
        t0 = time.time()
        u_r = _flat_to_grad_dict(self.agent, fed["u_r_flat"])
        tp = _flat_to_policy_payload(self.agent, fed["theta_prev_flat"])

        theta_start_snap, theta_start_ls = self._save_policy_snapshot()

        train_ep_returns: List[float] = []
        for _ in range(self.local_steps_k):
            states, actions, rewards = self._sample_trajectory()
            if states.shape[0] == 0:
                continue
            train_ep_returns.append(float(rewards.sum().item()))

            g_ck = _policy_grad_dict_from_trajectory(
                self.agent, states, actions, rewards, self.gamma
            )
            logp_ck = _log_prob_sum(self.agent, states, actions)

            snap_cur, ls_cur = self._save_policy_snapshot()
            self.agent.policy.load_state_dict(tp["policy"])
            self.agent.log_std.data.copy_(tp["log_std"])
            try:
                g_r1 = _policy_grad_dict_from_trajectory(
                    self.agent, states, actions, rewards, self.gamma
                )
                logp_old = _log_prob_sum(self.agent, states, actions)
            finally:
                self._load_policy_snapshot(snap_cur, ls_cur)

            log_w = (logp_old - logp_ck).detach().clamp(-40.0, 40.0)
            w = torch.exp(log_w).item()
            lo, hi = self.is_w_clip
            w = max(lo, min(hi, w))

            u_step: Dict[str, torch.Tensor] = {}
            for key in g_ck.keys():
                u_step[key] = self.beta * g_ck[key] + (1.0 - self.beta) * (
                    u_r[key] + g_ck[key] - w * g_r1[key]
                )

            _apply_parameter_step(self.agent, u_step, self.eta)

        theta_end_snap, theta_end_ls = self._save_policy_snapshot()

        delta_flat: List[np.ndarray] = []
        for k in sorted(theta_start_snap.keys()):
            delta_flat.append((theta_end_snap[k] - theta_start_snap[k]).detach().cpu().numpy())
        delta_flat.append((theta_end_ls - theta_start_ls).detach().cpu().numpy())

        self._load_policy_snapshot(theta_end_snap, theta_end_ls)

        train_mean = (
            float(sum(train_ep_returns) / max(1, len(train_ep_returns)))
            if train_ep_returns
            else 0.0
        )
        # Surrogate loss for logging (maximize return ⇔ minimize negative return)
        loss_surrogate = -train_mean

        dur = max(time.time() - t0, 1e-8)
        # No eval/return here: that belongs to Flower evaluate() with global θ only.
        out: Dict[str, float] = {
            "loss": float(loss_surrogate),
            "train/return": float(train_mean),
            "time/sec_per_round": float(dur),
        }
        self.n_steps = int(self.local_steps_k * self.max_horizon)

        maybe_save_federated_eval_video(
            self.env,
            server_round=self.server_round,
            render_eval=self.render_eval,
            render_mode=self.render_mode,
            render_save_dir=self.render_save_dir,
            render_every_n_rounds=self.render_every_n_rounds,
            render_episodes=self.render_episodes,
            eval_episodes=self.eval_episodes,
            client_tag=self.render_client_tag,
            act_fn=self._policy_action_for_render,
        )

        return out, delta_flat

    def _policy_action_for_render(self, obs: Any) -> Any:
        a, _, _ = self.agent.select_action(obs, deterministic=True)
        return np.asarray(a, dtype=np.float32).reshape(-1)

    def _eval_episode(self) -> float:
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        ep_ret = 0.0
        done = False
        while not done:
            a, _, _ = self.agent.select_action(obs, deterministic=True)
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            obs, r, terminated, truncated, _ = self.env.step(a)
            done = bool(terminated) or bool(truncated)
            ep_ret += float(r)
        return ep_ret

    def save_eval(self, cid: str, rnd: int, outdir: str = "./results/fedmomentum") -> bool:
        return True
