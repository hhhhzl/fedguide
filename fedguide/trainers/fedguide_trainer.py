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
        lambda_guide_anneal: bool = False,
        lambda_guide_decay_rounds: int = 40,
        online_guidance: bool = False,
        online_prior: bool = False,
        eval_episodes: int = 1,
        writer: Optional[Any] = None,
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

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.lambda_local = lambda_local
        self.lambda_guide = lambda_guide
        self.lambda_guide_anneal = lambda_guide_anneal
        self.lambda_guide_decay_rounds = lambda_guide_decay_rounds
        self.server_round = 0
        self.online_guidance = online_guidance
        self.online_prior = online_prior
        self.eval_episodes = eval_episodes
        self.writer = writer
        self.render_eval = render_eval
        self.render_mode = render_mode
        self.render_save_dir = render_save_dir
        self.render_every_n_rounds = render_every_n_rounds
        self.render_episodes = render_episodes
        self.render_client_tag = render_client_tag

        # Initialize current obs once; rollouts continue across rounds for non-bandit envs.
        reset_result = self.env.reset()
        self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        self.last_actions = None  # Store last rollout actions for metrics collection

    def set_server_round(self, rnd: int):
        """Set current server round for lambda_guide annealing.
        Note: do NOT reset env here — that destroys rollout continuity for continuous tasks.
        """
        self.server_round = int(rnd)
        if hasattr(self.agent, "anneal_log_std") and getattr(self.agent, "log_std_anneal", False):
            self.agent.anneal_log_std(
                self.server_round,
                target=getattr(self.agent, "log_std_anneal_target", -2.0),
                decay_rounds=getattr(self.agent, "log_std_anneal_rounds", 40),
            )

    # ---------------- Rollout + GAE ----------------
    def _rollout(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        for _ in range(self.n_steps):
            a, logp, v = self.agent.select_action(self._obs, deterministic=False)
            a = np.asarray(a)[0] if isinstance(a, (list, np.ndarray)) and np.asarray(a).ndim > 1 else a
            next_obs, r, terminated, truncated, _info = self.env.step(a)
            d = bool(terminated) or bool(truncated)

            obs_buf.append(torch.tensor(self._obs, dtype=torch.float32))
            act_buf.append(torch.tensor(a, dtype=torch.float32))
            logp_buf.append(torch.tensor(logp, dtype=torch.float32).reshape(()))
            rew_buf.append(torch.tensor(r, dtype=torch.float32).reshape(()))
            val_buf.append(torch.tensor(v, dtype=torch.float32).reshape(()))
            done_buf.append(torch.tensor(float(d), dtype=torch.float32).reshape(()))

            self._obs = next_obs
            if d:
                reset_result = self.env.reset()
                self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

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

        # Anneal lambda_guide: decay from lambda_guide to 0 over lambda_guide_decay_rounds
        lambda_guide_eff = self.lambda_guide
        if self.lambda_guide_anneal and self.lambda_guide_decay_rounds > 0:
            progress = min(1.0, self.server_round / self.lambda_guide_decay_rounds)
            lambda_guide_eff = self.lambda_guide * (1.0 - progress)

        logs = self.agent.update(
            batch,
            epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
            lambda_local=self.lambda_local,
            lambda_guide=lambda_guide_eff,
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

        from fedguide.utils.federated_render import maybe_save_federated_eval_video

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

        return out

    def _policy_action_for_render(self, obs: Any) -> Any:
        a, _, _ = self.agent.select_action(obs, deterministic=True)
        a = np.asarray(a)[0] if isinstance(a, (list, np.ndarray)) and np.asarray(a).ndim > 1 else a
        return a

    def _eval_episode(self) -> float:
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        ep_ret = 0.0
        done = False
        while not done:
            a, _, _ = self.agent.select_action(obs, deterministic=True)
            a = np.asarray(a)[0] if isinstance(a, (list, np.ndarray)) and np.asarray(a).ndim > 1 else a
            obs, r, terminated, truncated, _info = self.env.step(a)
            done = bool(terminated) or bool(truncated)
            ep_ret += r
        return ep_ret

    def save_eval(self, cid: str, rnd: int, outdir="./results/fedguide") -> bool:
        """Save evaluation trajectory and metadata.
        
        For Bandit2D environment, this is a simplified version that doesn't
        track passed_gate or reached_goal (those are maze-specific concepts).
        """
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
            obs, r, terminated, truncated, _info = self.env.step(a)
            done = bool(terminated) or bool(truncated)
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
        
        # Return True to indicate success (for compatibility with base.py interface)
        return True