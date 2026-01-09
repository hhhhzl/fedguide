"""
FedRep Trainer Implementation

Uses standard PPO training without KL penalties.
"""

import torch
import numpy as np
from typing import Any, Tuple, Dict, Optional
import time


class FedRepTrainer:
    """FedRep Trainer implementing standard PPO."""
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        device: Optional[str] = None,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        eval_episodes: int = 1,
        writer: Optional[Any] = None,
    ):
        self.agent = agent
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.eval_episodes = eval_episodes
        self.writer = writer
        
        # Current observation
        reset_result = self.env.reset()
        self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        # Store last rollout actions for metrics collection
        self.last_actions = None
    
    def _rollout(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Collect rollout data."""
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        
        for _ in range(self.n_steps):
            # Get action from agent
            a, logp, v = self.agent.select_action(self._obs, deterministic=False)
            a = np.asarray(a)[0] if isinstance(a, (list, np.ndarray)) and np.asarray(a).ndim > 1 else a
            
            # Step environment
            next_obs, r, d, _trunc, _info = self.env.step(a)
            
            # Store transition
            obs_buf.append(torch.tensor(self._obs, dtype=torch.float32))
            act_buf.append(torch.tensor(a, dtype=torch.float32))
            logp_buf.append(torch.tensor(logp, dtype=torch.float32).reshape(()))
            rew_buf.append(torch.tensor(r, dtype=torch.float32).reshape(()))
            val_buf.append(torch.tensor(v, dtype=torch.float32).reshape(()))
            done_buf.append(torch.tensor(d, dtype=torch.float32).reshape(()))
            
            # Update observation
            self._obs = next_obs
            if d:
                reset_result = self.env.reset()
                self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        # Stack into tensors
        states = torch.stack(obs_buf)
        actions = torch.stack(act_buf)
        logps_old = torch.stack(logp_buf)
        rewards = torch.stack(rew_buf)
        values = torch.stack(val_buf)
        dones = torch.stack(done_buf)
        
        # Get last value for GAE
        with torch.no_grad():
            _, _, last_v = self.agent.select_action(self._obs, deterministic=True)
            last_v = torch.tensor(last_v, dtype=torch.float32).reshape(())
        
        # Compute advantages and returns
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
        """Generalized Advantage Estimation."""
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
    
    def train_one_round(self) -> Dict[str, float]:
        """Train for one federated round."""
        t0 = time.time()
        
        # Collect rollouts
        states, actions, logps_old, returns, extras = self._rollout()
        
        # Store actions for metrics collection
        self.last_actions = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        
        # Prepare batch
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
        
        # Update policy with standard PPO (no KL penalties)
        logs = self.agent.update(
            batch,
            epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
        )
        
        # Evaluation
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
        
        # Log to writer if available
        if self.writer is not None:
            try:
                for k, v in out.items():
                    self.writer.log({k: v})
            except Exception:
                pass
        
        return out
    
    def _eval_episode(self) -> float:
        """Evaluate policy for one episode."""
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        ep_ret = 0.0
        done = False
        while not done:
            a, _, _ = self.agent.select_action(obs, deterministic=True)
            a = np.asarray(a)[0] if isinstance(a, (list, np.ndarray)) and np.asarray(a).ndim > 1 else a
            obs, r, done, _trunc, _info = self.env.step(a)
            ep_ret += r
        return ep_ret
    
    def save_eval(self, cid: str, rnd: int, outdir = "./results/fedrep") -> bool:
        """Save evaluation trajectory and metadata."""
        import os
        import json
        
        # Run evaluation episode and collect trajectory
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        traj = [obs.copy() if hasattr(obs, 'copy') else np.array(obs)]
        ep_ret = 0.0
        done = False
        
        while not done:
            a, _, _ = self.agent.select_action(obs, deterministic=True)
            a = np.asarray(a)[0] if isinstance(a, (list, np.ndarray)) and np.asarray(a).ndim > 1 else a
            obs, r, done, _trunc, _info = self.env.step(a)
            ep_ret += r
            traj.append(obs.copy() if hasattr(obs, 'copy') else np.array(obs))
        
        # Save trajectory and metadata
        d = os.path.join(outdir, f"client_{cid}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"round_{rnd}_traj.npy"), np.asarray(traj, dtype=np.float32))
        meta = {
            "round": int(rnd),
            "ep_return": float(ep_ret),
            "ep_length": len(traj),
        }
        with open(os.path.join(d, f"round_{rnd}_meta.json"), "w") as f:
            json.dump(meta, f)
        
        return True
    
    @property
    def return_(self):
        """Average episode return."""
        return 0.0
    
    @property
    def episode_len(self):
        """Average episode length."""
        return 1.0

