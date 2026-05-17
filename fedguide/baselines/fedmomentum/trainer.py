"""
FedMomentum trainer (PPO rollout + SVRPG-style gradient for uplink).

arxiv:2405.19499 (FedSVRPG-M) uses local VR momentum (Eq. 4) and importance weights;
this codebase uses a practical PPO rollout + reference-policy VR for the
policy_gradient sent to the server. The server applies θ ← θ + λ·ḡ (see server.py).
"""

import torch
import numpy as np
from typing import Any, Tuple, Dict, Optional
from collections import deque
import time
import copy
import json
import os


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clone tensors before temporary policy swaps.

    ``nn.Module.state_dict()`` returns tensors tied to module storage; using it
    directly as a snapshot means a later ``load_state_dict`` can mutate the
    supposed "original" values.
    """
    return {k: v.detach().clone() for k, v in state_dict.items()}


class SVRPGTrainer:
    """
    SVRPG (Stochastic Variance Reduced Policy Gradient) Trainer.
    
    Uses reference policy snapshots to reduce variance in gradient estimation.
    Key features:
    - Periodic reference policy updates
    - Variance-reduced gradient computation
    - On-policy trajectory collection
    """
    
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
        update_epochs: int = 4,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        eval_episodes: int = 1,
        # SVRPG-specific parameters
        reference_update_freq: int = 5,  # Update reference policy every K rounds (paper: 5-10)
        use_svrpg: bool = True,  # If False, use vanilla policy gradient
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
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.eval_episodes = eval_episodes
        self.server_round = 0
        self.render_eval = render_eval
        self.render_mode = render_mode
        self.render_save_dir = render_save_dir
        self.render_every_n_rounds = render_every_n_rounds
        self.render_episodes = render_episodes
        self.render_client_tag = render_client_tag

        # SVRPG-specific
        self.reference_update_freq = reference_update_freq
        self.use_svrpg = use_svrpg
        
        # Reference policy (snapshot)
        self.reference_policy = None
        self.reference_log_std = None
        
        # Smooth reference gradient (accumulated over reference period)
        self.smooth_reference_gradient = None
        self.reference_gradient_count = 0
        
        # Current round counter (for reference update)
        self.round_count = 0
        
        self.writer = writer
        
        # Current observation
        reset_result = self.env.reset()
        self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        # Store last rollout actions for metrics collection
        self.last_actions = None
        
        # Store last computed policy gradient (for client to return)
        self.last_policy_gradient = None

    def set_server_round(self, rnd: int):
        self.server_round = int(rnd)
    
    # ---------------- Rollout + GAE ----------------
    def _rollout(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Collect rollout data using current policy."""
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        
        for _ in range(self.n_steps):
            # Get action from agent
            a, logp, v = self.agent.select_action(self._obs, deterministic=False)
            a = np.asarray(a)[0] if isinstance(a, (list, np.ndarray)) and np.asarray(a).ndim > 1 else a
            
            # Step environment (Gymnasium: terminated, truncated — both end the episode)
            next_obs, r, terminated, truncated, _info = self.env.step(a)
            d = bool(terminated) or bool(truncated)
            
            # Store transition
            obs_buf.append(torch.tensor(self._obs, dtype=torch.float32))
            act_buf.append(torch.tensor(a, dtype=torch.float32))
            logp_buf.append(torch.tensor(logp, dtype=torch.float32).reshape(()))
            rew_buf.append(torch.tensor(r, dtype=torch.float32).reshape(()))
            val_buf.append(torch.tensor(v, dtype=torch.float32).reshape(()))
            done_buf.append(torch.tensor(float(d), dtype=torch.float32).reshape(()))
            
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
    
    # ---------------- Reference Policy Management ----------------
    def update_reference_policy(self):
        """Update reference policy snapshot."""
        # Deep copy current policy state
        self.reference_policy = {
            k: v.clone().detach() 
            for k, v in self.agent.policy.state_dict().items()
        }
        self.reference_log_std = self.agent.log_std.clone().detach()
        
        # Reset smooth reference gradient accumulator
        self.smooth_reference_gradient = None
        self.reference_gradient_count = 0
        
        print(f"[SVRPG] Updated reference policy (round {self.round_count})")
    
    def _compute_reference_log_probs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities using reference policy.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
        
        Returns:
            Log probabilities [batch_size]
        """
        if self.reference_policy is None:
            # No reference policy yet, return zeros (will be initialized in first update)
            return torch.zeros(states.shape[0], device=self.device)
        
        # Temporarily swap to reference policy
        original_state_dict = _clone_state_dict(self.agent.policy.state_dict())
        original_log_std = self.agent.log_std.data.clone()
        
        # Load reference policy
        self.agent.policy.load_state_dict(self.reference_policy)
        self.agent.log_std.data.copy_(self.reference_log_std.to(self.agent.log_std.device))
        
        # Compute log probabilities with reference policy
        with torch.no_grad():
            logps_ref, _, _, _ = self.agent.evaluate(states, actions)
        
        # Restore original policy
        self.agent.policy.load_state_dict(original_state_dict)
        self.agent.log_std.data.copy_(original_log_std.to(self.agent.log_std.device))
        
        return logps_ref
    
    def _compute_reference_advantages(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Recompute advantages using reference policy.
        
        This is a simplified version - in full SVRPG, we would re-evaluate
        the entire trajectory with the reference policy. For efficiency,
        we approximate by using current value estimates.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            rewards: Batch of rewards [batch_size]
            dones: Batch of done flags [batch_size]
            values: Optional value estimates (if None, recompute)
        
        Returns:
            Approximate advantages [batch_size]
        """
        # For efficiency, use current value estimates
        # In full SVRPG, would re-evaluate trajectory with reference policy
        if values is None:
            _, _, values, _ = self.agent.evaluate(states, actions)
            values = values.detach()
        
        # Compute simple advantages (TD error)
        advantages = rewards - values
        # Note: This is a simplified version. Full SVRPG would recompute GAE with reference policy
        
        return advantages
    
    # ---------------- SVRPG Gradient Computation ----------------
    def _compute_svrpg_gradient(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_logps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SVRPG gradient with variance reduction.
        
        SVRPG gradient formula:
        g_svrpg = g_current - g_reference + g_smooth_reference
        
        where:
        - g_current: gradient with current policy
        - g_reference: gradient with reference policy (computed on same trajectory)
        - g_smooth_reference: smoothed gradient with reference policy (accumulated)
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            advantages: Advantages [batch_size]
            old_logps: Old log probabilities [batch_size]
        
        Returns:
            Dictionary of policy gradients
        """
        if not self.use_svrpg or self.reference_policy is None:
            # Fallback to vanilla policy gradient
            return self.agent.compute_policy_gradient(
                states, actions, advantages, old_logps, use_clipped=True
            )
        
        # 1. Compute current policy gradient
        current_grad = self.agent.compute_policy_gradient(
            states, actions, advantages, old_logps, use_clipped=True
        )
        
        # 2. Compute reference policy gradient
        # Recompute advantages with reference policy (simplified)
        ref_logps = self._compute_reference_log_probs(states, actions)
        
        # For reference gradient, use same advantages (approximation)
        # In full SVRPG, would recompute advantages with reference value function
        ref_advantages = advantages.detach()  # Simplified: use current advantages
        
        # Temporarily swap to reference policy to compute gradient
        original_state_dict = _clone_state_dict(self.agent.policy.state_dict())
        original_log_std = self.agent.log_std.data.clone()
        
        # Load reference policy
        self.agent.policy.load_state_dict(self.reference_policy)
        self.agent.log_std.data.copy_(self.reference_log_std.to(self.agent.log_std.device))
        
        # Compute reference gradient (without clipping, for variance reduction)
        reference_grad = self.agent.compute_policy_gradient(
            states, actions, ref_advantages, ref_logps, use_clipped=False
        )
        
        # Restore original policy
        self.agent.policy.load_state_dict(original_state_dict)
        self.agent.log_std.data.copy_(original_log_std.to(self.agent.log_std.device))
        
        # 3. Update smooth reference gradient (moving average)
        if self.smooth_reference_gradient is None:
            self.smooth_reference_gradient = reference_grad.copy()
        else:
            # Accumulate reference gradient
            alpha = 1.0 / (self.reference_gradient_count + 1)
            for key in reference_grad.keys():
                if key in self.smooth_reference_gradient:
                    self.smooth_reference_gradient[key] = (
                        (1 - alpha) * self.smooth_reference_gradient[key] +
                        alpha * reference_grad[key]
                    )
                else:
                    self.smooth_reference_gradient[key] = reference_grad[key]
        
        self.reference_gradient_count += 1
        
        # 4. Compute SVRPG gradient: g_current - g_reference + g_smooth_reference
        svrpg_grad = {}
        for key in current_grad.keys():
            ref_term = reference_grad.get(key, torch.zeros_like(current_grad[key]))
            smooth_ref_term = self.smooth_reference_gradient.get(
                key, torch.zeros_like(current_grad[key])
            )
            svrpg_grad[key] = current_grad[key] - ref_term + smooth_ref_term
        
        return svrpg_grad
    
    # ---------------- Training ----------------
    def train_one_round(self) -> Dict[str, float]:
        """
        Train for one federated round.
        
        Returns:
            Dictionary with training metrics
        """
        t0 = time.time()
        
        # Update reference policy periodically
        if self.round_count % self.reference_update_freq == 0:
            self.update_reference_policy()
        
        self.round_count += 1
        
        # Collect rollouts
        states, actions, logps_old, returns, extras = self._rollout()
        
        # Store actions for metrics collection
        self.last_actions = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        
        # Normalize advantages (for stability)
        advantages = extras["adv"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute SVRPG gradient (for server aggregation)
        policy_gradient = self._compute_svrpg_gradient(
            states, actions, advantages, logps_old
        )
        
        # Store gradient for client to return
        self.last_policy_gradient = policy_gradient
        
        # Also perform local update (optional, can be disabled if only gradients are needed)
        # For now, we'll still do local update for value function
        batch = {
            "s": states,
            "a": actions,
            "old_logp": logps_old,
            "ret": returns,
            "adv": advantages,
        }
        
        # Update agent (updates both policy and value, but we'll extract policy gradient separately)
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
    
    def get_policy_gradient(self) -> Dict[str, torch.Tensor]:
        """
        Get last computed policy gradient (for client to return to server).
        
        Returns:
            Dictionary of policy gradients (will be converted to numpy for transmission)
        """
        if self.last_policy_gradient is None:
            # Return zero gradients if not computed yet
            # This should not happen if train_one_round was called
            policy_grad = {}
            for name, param in self.agent.policy.named_parameters():
                policy_grad[f"policy.{name}"] = torch.zeros_like(param)
            policy_grad["log_std"] = torch.zeros_like(self.agent.log_std)
            return policy_grad
        
        return self.last_policy_gradient

    def state_dict(self) -> Dict[str, Any]:
        """Persist trainer-side SVRPG state across Flower VCE client rebuilds."""
        smooth = None
        if self.smooth_reference_gradient is not None:
            smooth = {
                k: v.detach().cpu().clone()
                for k, v in self.smooth_reference_gradient.items()
            }
        return {
            "round_count": int(self.round_count),
            "reference_policy": (
                {k: v.detach().cpu().clone() for k, v in self.reference_policy.items()}
                if self.reference_policy is not None
                else None
            ),
            "reference_log_std": (
                self.reference_log_std.detach().cpu().clone()
                if self.reference_log_std is not None
                else None
            ),
            "smooth_reference_gradient": smooth,
            "reference_gradient_count": int(self.reference_gradient_count),
            "obs": np.asarray(self._obs, dtype=np.float32),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore trainer-side SVRPG state.

        The current policy is intentionally not restored here; the server's
        broadcast parameters remain authoritative at the start of each round.
        """
        if not state:
            return
        self.round_count = int(state.get("round_count", self.round_count))
        ref = state.get("reference_policy")
        self.reference_policy = (
            {k: v.detach().clone() for k, v in ref.items()}
            if isinstance(ref, dict)
            else None
        )
        ref_log_std = state.get("reference_log_std")
        self.reference_log_std = (
            ref_log_std.detach().clone().to(self.agent.device)
            if isinstance(ref_log_std, torch.Tensor)
            else None
        )
        smooth = state.get("smooth_reference_gradient")
        self.smooth_reference_gradient = (
            {k: v.detach().clone().to(self.agent.device) for k, v in smooth.items()}
            if isinstance(smooth, dict)
            else None
        )
        self.reference_gradient_count = int(
            state.get("reference_gradient_count", self.reference_gradient_count)
        )
        if "obs" in state and state["obs"] is not None:
            self._obs = np.asarray(state["obs"], dtype=np.float32)
    
    def _eval_episode(self) -> float:
        """Evaluate policy for one episode."""
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
    
    # ---------------- Compatibility Methods ----------------
    def save_eval(self, cid: str, rnd: int, outdir: str = "./results/fedmomentum") -> bool:
        """Save evaluation trajectory and metadata."""
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
        
        return True
    
    @property
    def return_(self):
        """Average episode return (for compatibility)."""
        return 0.0
    
    @property
    def episode_len(self):
        """Average episode length (for compatibility)."""
        return 1.0


class HAPGTrainer(SVRPGTrainer):
    """
    HAPG (Hessian-Aware Policy Gradient) Trainer.
    
    Uses Hessian/Fisher Information Matrix to improve gradient estimation.
    Based on the paper: "Momentum for the Win: Collaborative Federated Reinforcement Learning across Heterogeneous Environments"
    
    Algorithm:
    1. Compute vanilla policy gradient g_vanilla
    2. Compute/approximate Hessian H (or Fisher Information Matrix)
    3. Apply Hessian correction: g_hapg = (I + α * H)^{-1} * g_vanilla
    
    Key features:
    - Hessian approximation using Fisher Information Matrix (FIM)
    - Diagonal approximation for computational efficiency
    - Optional momentum integration with SVRPG
    """
    
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
        update_epochs: int = 4,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        eval_episodes: int = 1,
        # HAPG-specific parameters
        hessian_alpha: float = 0.1,  # Scaling factor for Hessian correction (paper: 0.1-1.0)
        use_diagonal_approx: bool = True,  # Use diagonal approximation for efficiency
        fisher_update_freq: int = 1,  # Update Fisher matrix every K rounds
        use_fisher_info: bool = True,  # Use Fisher Information Matrix (FIM) instead of true Hessian
        # SVRPG parameters (HAPG can be combined with SVRPG)
        reference_update_freq: int = 5,
        use_svrpg: bool = False,  # Can combine with SVRPG
        writer: Optional[Any] = None,
        render_eval: bool = False,
        render_mode: str = "video",
        render_save_dir: Optional[str] = None,
        render_every_n_rounds: int = 10,
        render_episodes: int = 5,
        render_client_tag: str = "0",
    ):
        # Initialize parent SVRPG trainer
        super().__init__(
            agent=agent,
            env=env,
            device=device,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
            eval_episodes=eval_episodes,
            reference_update_freq=reference_update_freq,
            use_svrpg=use_svrpg,
            writer=writer,
            render_eval=render_eval,
            render_mode=render_mode,
            render_save_dir=render_save_dir,
            render_every_n_rounds=render_every_n_rounds,
            render_episodes=render_episodes,
            render_client_tag=render_client_tag,
        )
        
        # HAPG-specific parameters
        self.hessian_alpha = hessian_alpha
        self.use_diagonal_approx = use_diagonal_approx
        self.fisher_update_freq = fisher_update_freq
        self.use_fisher_info = use_fisher_info
        
        # Fisher Information Matrix (diagonal approximation)
        self.fisher_diagonal = None  # Dictionary of diagonal elements
        self.fisher_update_count = 0
        
    def _compute_fisher_information_matrix(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix (FIM) diagonal approximation.
        
        Fisher Information Matrix: F = E[∇log π(a|s) * ∇log π(a|s)^T]
        
        For computational efficiency, we use diagonal approximation:
        F_diag = E[(∇log π(a|s))^2]
        
        We compute per-sample gradients and average the squared gradients.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
        
        Returns:
            Dictionary of Fisher diagonal elements (keyed by parameter name)
        """
        states = states.to(self.device).float()
        actions = actions.to(self.device).float()
        batch_size = states.shape[0]
        
        # Initialize Fisher diagonal accumulator
        fisher_diag = {}
        
        # Compute per-sample gradients for more accurate Fisher estimate
        # For efficiency, we can compute on a subset or use batch gradient with correction
        # Here we use batch gradient as approximation (scaled appropriately)
        
        # Compute log probabilities
        logps, _, _, _ = self.agent.evaluate(states, actions)
        
        # Compute gradients of log probability w.r.t. policy parameters
        self.agent.optimizer.zero_grad()
        
        # Sum log probabilities over batch (for gradient computation)
        logp_sum = logps.sum()
        
        # Compute gradients
        logp_sum.backward()
        
        # Extract Fisher diagonal: F_diag = E[(∇log π)^2]
        # For batch gradient: F ≈ (1/N) * (∇Σ log π)^2 / N
        # More accurately: F ≈ (1/N) * Σ(∇log π_i)^2, but we approximate with batch gradient
        # Correction factor: batch_grad = (1/N) * Σ grad_i, so F ≈ N * (batch_grad)^2
        # However, for simplicity and stability, we use (batch_grad)^2 directly
        
        for name, param in self.agent.policy.named_parameters():
            if param.grad is not None:
                # Fisher diagonal = (∇log π)^2 (using batch gradient as approximation)
                # This is an approximation; true Fisher requires per-sample gradients
                fisher_diag[f"policy.{name}"] = (param.grad ** 2).clone().detach()
            else:
                fisher_diag[f"policy.{name}"] = torch.zeros_like(param).detach()
        
        # Log std Fisher diagonal
        if self.agent.log_std.grad is not None:
            fisher_diag["log_std"] = (self.agent.log_std.grad ** 2).clone().detach()
        else:
            fisher_diag["log_std"] = torch.zeros_like(self.agent.log_std).detach()
        
        # Clear gradients
        self.agent.optimizer.zero_grad()
        
        # Add small epsilon for numerical stability
        for key in fisher_diag.keys():
            fisher_diag[key] = fisher_diag[key] + 1e-8
        
        return fisher_diag
    
    def _update_fisher_matrix(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update Fisher Information Matrix (using moving average).
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
        """
        # Compute current Fisher diagonal
        current_fisher = self._compute_fisher_information_matrix(states, actions)
        
        if self.fisher_diagonal is None:
            # Initialize Fisher diagonal (deep copy to avoid references)
            self.fisher_diagonal = {}
            for key, value in current_fisher.items():
                if isinstance(value, torch.Tensor):
                    self.fisher_diagonal[key] = value.clone().detach()
                else:
                    self.fisher_diagonal[key] = value
        else:
            # Moving average update
            alpha = 1.0 / (self.fisher_update_count + 1)
            for key in current_fisher.keys():
                if key in self.fisher_diagonal:
                    # Exponential moving average
                    if isinstance(current_fisher[key], torch.Tensor):
                        # Ensure shapes match
                        current_val = current_fisher[key]
                        stored_val = self.fisher_diagonal[key]
                        
                        if current_val.shape == stored_val.shape:
                            self.fisher_diagonal[key] = (
                                (1 - alpha) * stored_val +
                                alpha * current_val
                            ).detach()
                        else:
                            # Shape mismatch: use current value
                            self.fisher_diagonal[key] = current_val.clone().detach()
                    else:
                        # Non-tensor: simple average
                        self.fisher_diagonal[key] = (
                            (1 - alpha) * self.fisher_diagonal[key] +
                            alpha * current_fisher[key]
                        )
                else:
                    # New parameter: initialize
                    if isinstance(current_fisher[key], torch.Tensor):
                        self.fisher_diagonal[key] = current_fisher[key].clone().detach()
                    else:
                        self.fisher_diagonal[key] = current_fisher[key]
        
        self.fisher_update_count += 1
    
    def _apply_hessian_correction(
        self,
        gradient: Dict[str, torch.Tensor],
        hessian: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Hessian correction to gradient.
        
        HAPG gradient formula: g_hapg = (I + α * H)^{-1} * g_vanilla
        
        For diagonal approximation:
        g_hapg[i] = g_vanilla[i] / (1 + α * H[i,i])
        
        Args:
            gradient: Vanilla policy gradient dictionary
            hessian: Hessian/Fisher diagonal dictionary (if None, use stored Fisher)
        
        Returns:
            Hessian-corrected gradient dictionary
        """
        if hessian is None:
            hessian = self.fisher_diagonal
        
        if hessian is None:
            # No Hessian available, return original gradient
            print("[HAPG] Warning: No Hessian available, returning vanilla gradient")
            return gradient
        
        corrected_grad = {}
        for key in gradient.keys():
            if key in hessian:
                # Diagonal approximation: g_corrected = g / (1 + α * H_diag)
                hessian_diag = hessian[key]
                grad = gradient[key]
                
                # Ensure hessian_diag has same shape as gradient
                if isinstance(hessian_diag, torch.Tensor):
                    # Reshape/broadcast to match gradient shape
                    if hessian_diag.shape != grad.shape:
                        if hessian_diag.numel() == 1:
                            # Scalar: broadcast to full shape
                            hessian_diag = hessian_diag.expand_as(grad)
                        elif hessian_diag.numel() == grad.numel():
                            # Same number of elements: reshape
                            hessian_diag = hessian_diag.view_as(grad)
                        else:
                            # Shape mismatch: use original gradient for this parameter
                            print(f"[HAPG] Warning: Shape mismatch for {key}, using original gradient")
                            corrected_grad[key] = grad
                            continue
                    
                    # Apply correction with numerical stability
                    # Clamp hessian to prevent overflow
                    hessian_diag = torch.clamp(hessian_diag, max=1e6)
                    denominator = 1.0 + self.hessian_alpha * hessian_diag
                    denominator = torch.clamp(denominator, min=1e-8)  # Prevent division by zero
                    
                    corrected_grad[key] = grad / denominator
                else:
                    # Non-tensor hessian (shouldn't happen, but handle gracefully)
                    corrected_grad[key] = grad
            else:
                # No Hessian for this parameter, use original gradient
                corrected_grad[key] = gradient[key]
        
        return corrected_grad
    
    def _compute_hapg_gradient(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_logps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute HAPG gradient with Hessian correction.
        
        Algorithm:
        1. Compute vanilla policy gradient (or SVRPG gradient if use_svrpg=True)
        2. Update Fisher Information Matrix (if needed)
        3. Apply Hessian correction
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            advantages: Advantages [batch_size]
            old_logps: Old log probabilities [batch_size]
        
        Returns:
            Dictionary of HAPG-corrected policy gradients
        """
        # 1. Compute base gradient (vanilla or SVRPG)
        if self.use_svrpg and self.reference_policy is not None:
            # Use SVRPG gradient as base
            base_gradient = super()._compute_svrpg_gradient(
                states, actions, advantages, old_logps
            )
        else:
            # Use vanilla policy gradient
            base_gradient = self.agent.compute_policy_gradient(
                states, actions, advantages, old_logps, use_clipped=True
            )
        
        # 2. Apply Hessian correction (Fisher matrix should be updated in train_one_round before this call)
        hapg_gradient = self._apply_hessian_correction(base_gradient)
        
        return hapg_gradient
    
    def train_one_round(self) -> Dict[str, float]:
        """
        Train for one federated round with HAPG.
        
        Returns:
            Dictionary with training metrics
        """
        t0 = time.time()
        
        # Update reference policy periodically (if using SVRPG)
        if self.use_svrpg and self.round_count % self.reference_update_freq == 0:
            self.update_reference_policy()
        
        self.round_count += 1
        
        # Collect rollouts
        states, actions, logps_old, returns, extras = self._rollout()
        
        # Store actions for metrics collection
        self.last_actions = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        
        # Normalize advantages (for stability)
        advantages = extras["adv"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update Fisher Information Matrix periodically (before computing gradient)
        if self.round_count % self.fisher_update_freq == 0:
            self._update_fisher_matrix(states, actions)
        
        # Compute HAPG gradient (with Hessian correction)
        policy_gradient = self._compute_hapg_gradient(
            states, actions, advantages, logps_old
        )
        
        # Store gradient for client to return
        self.last_policy_gradient = policy_gradient
        
        # Also perform local update (for value function and policy)
        batch = {
            "s": states,
            "a": actions,
            "old_logp": logps_old,
            "ret": returns,
            "adv": advantages,
        }
        
        # Update agent (updates both policy and value)
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
            "fisher_update_count": float(self.fisher_update_count),
        }
        if isinstance(logs, dict):
            out.update({f"train/{k}": float(v) for k, v in logs.items()})
        
        # Log Fisher matrix norm for monitoring
        if self.fisher_diagonal is not None:
            fisher_norm = sum(
                fisher_val.norm().item() if isinstance(fisher_val, torch.Tensor) else fisher_val
                for fisher_val in self.fisher_diagonal.values()
            )
            out["train/fisher_norm"] = fisher_norm
        
        # Log to writer if available
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
