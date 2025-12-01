import time
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np


class FedguideTrainer:
    def __init__(
        self,
        agent,
        env,
        device: Optional[str] = None,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 4,
        minibatch_size: int = 256,
        lambda_local: float = 0.0,
        lambda_guide: float = 0.0,
        online_guidance: bool = False,
        online_prior: bool = False,
        eval_episodes: int = 1,
        writer: Optional[Any] = None,
    ):
        self.agent = agent
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.lambda_local = lambda_local
        self.lambda_guide = lambda_guide
        self.online_guidance = online_guidance
        self.online_prior = online_prior
        self.eval_episodes = eval_episodes
        self.writer = writer

        self._obs = self.env.reset()
        self.last_actions = None  # Store last rollout actions for metrics collection

    # ---------------- Rollout + GAE ----------------
    def _rollout(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        for _ in range(self.n_steps):
            a, logp, v = self.agent.select_action(self._obs, deterministic=False)
            a = np.asarray(a)[0] if isinstance(a, (list, np.ndarray)) and np.asarray(a).ndim > 1 else a
            next_obs, r, d, _trunc, _info = self.env.step(a)

            obs_buf.append(torch.tensor(self._obs, dtype=torch.float32))
            act_buf.append(torch.tensor(a, dtype=torch.float32))
            logp_buf.append(torch.tensor(logp, dtype=torch.float32).reshape(()))
            rew_buf.append(torch.tensor(r, dtype=torch.float32).reshape(()))
            val_buf.append(torch.tensor(v, dtype=torch.float32).reshape(()))
            done_buf.append(torch.tensor(d, dtype=torch.float32).reshape(()))

            self._obs = next_obs
            if d:
                self._obs = self.env.reset()

        states = torch.stack(obs_buf)
        actions = torch.stack(act_buf)
        logps_old = torch.stack(logp_buf)
        rewards = torch.stack(rew_buf)
        values = torch.stack(val_buf)
        dones = torch.stack(done_buf)

        with torch.no_grad():
            _, _, last_v = self.agent.select_action(self._obs, deterministic=True)
            last_v = torch.tensor(last_v, dtype=torch.float32).reshape(())

        adv, ret = self._gae(rewards, values, dones, last_v)
        extras = {
            "adv": adv,
            "r": rewards,
            "done": dones,
            "s_next": torch.vstack([states[1:], torch.tensor(self._obs, dtype=torch.float32).unsqueeze(0)]),
            "ep_return": rewards.sum(),
        }
        return states, actions, logps_old, ret, extras

    def _gae(self, rews, vals, dones, last_v):
        T = len(rews)
        adv = torch.zeros(T)
        ret = torch.zeros(T)
        gae = 0.0
        next_v = last_v
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rews[t] + self.gamma * next_v * mask - vals[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            adv[t] = gae
            ret[t] = adv[t] + vals[t]
            next_v = vals[t]
        return adv, ret

    # ---------------- One local round ----------------
    def train_one_round(self) -> Dict[str, float]:
        t0 = time.time()
        states, actions, logps_old, returns, extras = self._rollout()
        
        # Store actions for metrics collection
        self.last_actions = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions

        batch = {
            "s": states,
            "a": actions,
            "old_logp": logps_old,
            "ret": returns,
            "adv": extras["adv"],
            "r": extras["r"],
            "s_next": extras["s_next"],
            "done": extras["done"],
        }

        logs = self.agent.update(
            batch,
            epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
            lambda_local=self.lambda_local,
            lambda_guide=self.lambda_guide,
        )

        if self.online_guidance and hasattr(self.agent, "online_guidance_step"):
            self.agent.online_guidance_step(batch)
        if self.online_prior and hasattr(self.agent, "online_prior_step"):
            self.agent.online_prior_step(batch)

        eval_ret = 0.0
        for _ in range(self.eval_episodes):
            eval_ret += self._eval_episode()
        eval_ret /= max(1, self.eval_episodes)

        dur = max(time.time() - t0, 1e-8)
        out = {
            "train/return": float(extras["r"].sum().item()),
            "eval/return": float(eval_ret),
            "time/sec_per_round": float(dur),
        }
        if isinstance(logs, dict):
            out.update({f"train/{k}": float(v) for k, v in logs.items()})

        if self.writer is not None:
            try:
                for k, v in out.items():
                    self.writer.log({k: v})
            except Exception:
                pass

        return out

    def _eval_episode(self) -> float:
        obs = self.env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            a, _, _ = self.agent.select_action(obs, deterministic=True)
            a = np.asarray(a)[0] if isinstance(a, (list, np.ndarray)) and np.asarray(a).ndim > 1 else a
            obs, r, done, _trunc, _info = self.env.step(a)
            ep_ret += r
        return ep_ret