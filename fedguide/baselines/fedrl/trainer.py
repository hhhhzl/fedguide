"""
FedRL Trainer Implementation

This module implements DQN and DDPG trainers for federated reinforcement learning.
Based on FedRL paper: "Federated Reinforcement Learning with Environment Heterogeneity" (AISTATS 2022)
"""

import random
import os
import json
import time
import numpy as np
import torch
from collections import deque
from typing import Dict, Optional, Any, Tuple, List


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    Simple experience replay buffer for off-policy algorithms.
    
    Stores (state, action, reward, next_state, done) tuples.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()


# ============================================================================
# DQN Trainer
# ============================================================================

class DQNTrainer:
    """
    DQN Trainer for federated reinforcement learning.
    
    Implements off-policy training with experience replay buffer.
    Based on FedRL/deep/DeepRLAlgo.py double_DQN function.
    """
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        device: Optional[str] = None,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.01,
        batch_size: int = 16,
        replay_size: int = 1000,
        sync_interval: int = 10,
        merge_interval: int = 16,  # Number of steps to train locally (E in FedRL)
        eval_episodes: int = 1,
        replay_initial: int = None,  # Minimum buffer size before training (default: 2 * batch_size)
        render_eval: bool = False,
        render_mode: str = "video",
        render_save_dir: Optional[str] = None,
        render_every_n_rounds: int = 10,
        render_episodes: int = 5,
        render_client_tag: str = "0",
    ):
        """
        Initialize DQN trainer.
        
        Args:
            agent: DQNAgent instance
            env: Gym environment
            device: Device to use ('cpu', 'cuda', or 'auto')
            gamma: Discount factor
            epsilon: Initial epsilon for epsilon-greedy
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            batch_size: Batch size for updates
            replay_size: Replay buffer capacity
            sync_interval: Steps between target network sync
            merge_interval: Number of steps to collect and train per round (E in FedRL)
            eval_episodes: Number of episodes for evaluation
            replay_initial: Minimum buffer size before training (default: 2 * batch_size)
        """
        self.agent = agent
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.sync_interval = sync_interval
        self.merge_interval = merge_interval
        self.eval_episodes = eval_episodes
        self.replay_initial = replay_initial if replay_initial is not None else 2 * batch_size
        self.server_round = 0
        self.render_eval = render_eval
        self.render_mode = render_mode
        self.render_save_dir = render_save_dir
        self.render_every_n_rounds = render_every_n_rounds
        self.render_episodes = render_episodes
        self.render_client_tag = render_client_tag

        # Set agent epsilon
        self.agent.epsilon = epsilon
        self.agent.epsilon_decay = epsilon_decay
        self.agent.epsilon_min = epsilon_min
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_size)
        
        # Current observation
        reset_result = self.env.reset()
        self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        # Training statistics
        self.n_steps = 0
        self.last_actions = None  # Store last collected actions for metrics
        
        # Episode tracking
        self._episode_reward = 0.0
        self._episode_length = 0

    def set_server_round(self, rnd: int):
        self.server_round = int(rnd)
    
    def _collect_experiences(self, num_steps: int):
        """
        Collect experiences and store in replay buffer.
        
        Args:
            num_steps: Number of steps to collect
        """
        collected_actions = []
        
        for _ in range(num_steps):
            # Select action using epsilon-greedy
            action = self.agent.select_action(self._obs, deterministic=False)
            collected_actions.append(action)
            
            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            
            # Store in replay buffer
            self.replay_buffer.push(
                state=np.copy(self._obs),
                action=action,
                reward=reward,
                next_state=np.copy(next_obs) if not done else None,
                done=done
            )
            
            # Update observation
            self._obs = next_obs
            self._episode_reward += reward
            self._episode_length += 1
            self.n_steps += 1
            
            # Reset if done
            if done:
                reset_result = self.env.reset()
                self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                self._episode_reward = 0.0
                self._episode_length = 0
        
        # Store actions for metrics collection
        self.last_actions = np.array(collected_actions) if collected_actions else None
    
    def _update_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one update step from replay buffer.
        
        Returns:
            Dictionary with loss and metrics, or None if buffer too small
        """
        if len(self.replay_buffer) < self.replay_initial:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Update agent
        metrics = self.agent.update(batch)
        
        return metrics
    
    def train_one_round(self) -> Dict[str, float]:
        """
        Train for one federated round (merge_interval steps).
        
        This corresponds to training for E steps in FedRL terminology.
        
        Returns:
            Dictionary with training metrics
        """
        t0 = time.time()
        
        # Collect experiences for merge_interval steps
        self._collect_experiences(self.merge_interval)
        
        # Perform multiple update steps (if buffer is large enough)
        total_loss = 0.0
        num_updates = 0
        
        # Update multiple times based on collected experiences
        # Each update uses a different batch from replay buffer
        for _ in range(self.merge_interval):
            metrics = self._update_step()
            if metrics is not None:
                total_loss += metrics.get("loss", 0.0)
                num_updates += 1
        
        avg_loss = total_loss / max(num_updates, 1)
        
        # Evaluation
        eval_ret = 0.0
        for _ in range(self.eval_episodes):
            eval_ret += self._eval_episode()
        eval_ret /= max(1, self.eval_episodes)
        
        dur = max(time.time() - t0, 1e-8)

        out = {
            "loss": avg_loss,
            "train/return": float(self._episode_reward),  # Last episode reward
            "eval/return": float(eval_ret),
            "train/episode_length": float(self._episode_length),
            "train/epsilon": float(self.agent.epsilon),
            "train/num_updates": float(num_updates),
            "train/buffer_size": float(len(self.replay_buffer)),
            "time/sec_per_round": float(dur),
        }

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
            act_fn=lambda o: self.agent.select_action(o, deterministic=True),
        )

        return out
    
    def _eval_episode(self) -> float:
        """Evaluate policy for one episode (deterministic)."""
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        ep_ret = 0.0
        done = False
        
        while not done:
            action = self.agent.select_action(obs, deterministic=True)
            obs, r, done, truncated, _ = self.env.step(action)
            done = done or truncated
            ep_ret += r
        
        return ep_ret
    
    def save_eval(self, cid: str, rnd: int, outdir: str = "./results/fedrl_dqn") -> bool:
        """
        Save evaluation trajectory and metadata.
        
        Args:
            cid: Client ID
            rnd: Round number
            outdir: Output directory
        
        Returns:
            True if successful
        """
        # Run evaluation episode and collect trajectory
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        traj = [obs.copy() if hasattr(obs, 'copy') else np.array(obs)]
        ep_ret = 0.0
        done = False
        
        while not done:
            action = self.agent.select_action(obs, deterministic=True)
            obs, r, done, truncated, _ = self.env.step(action)
            done = done or truncated
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
        return self._episode_reward
    
    @property
    def episode_len(self):
        """Average episode length (for compatibility)."""
        return self._episode_length


# ============================================================================
# DDPG Trainer
# ============================================================================

class DDPGTrainer:
    """
    DDPG Trainer for federated reinforcement learning.
    
    Implements off-policy training with experience replay buffer.
    Based on FedRL/deep/DeepRLAlgo.py DDPG_TRAIN function.
    """
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        device: Optional[str] = None,
        gamma: float = 0.99,
        batch_size: int = 64,
        replay_size: int = 100000,
        replay_initial: int = 1000,  # Minimum buffer size before training
        merge_interval: int = 1000,  # Number of steps to train locally (E in FedRL)
        tau: float = 0.001,  # Soft update coefficient (for target networks)
        eval_episodes: int = 1,
        add_noise: bool = True,  # Whether to add exploration noise
        replay_persist_across_rounds: bool = False,  # False: match FedRL DDPG_TRAIN fresh buffer each round
        render_eval: bool = False,
        render_mode: str = "video",
        render_save_dir: Optional[str] = None,
        render_every_n_rounds: int = 10,
        render_episodes: int = 5,
        render_client_tag: str = "0",
    ):
        """
        Initialize DDPG trainer.
        
        Args:
            agent: DDPGAgent instance
            env: Gym environment
            device: Device to use ('cpu', 'cuda', or 'auto')
            gamma: Discount factor
            batch_size: Batch size for updates
            replay_size: Replay buffer capacity
            replay_initial: Minimum buffer size before training
            merge_interval: Number of steps to collect and train per round (E in FedRL)
            tau: Soft update coefficient for target networks
            eval_episodes: Number of episodes for evaluation
            add_noise: Whether to add exploration noise during training
            replay_persist_across_rounds: If False, clear replay at each federated round (FedRL-style).
        """
        self.agent = agent
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.replay_initial = replay_initial
        self.merge_interval = merge_interval
        self.tau = tau
        self.eval_episodes = eval_episodes
        self.add_noise = add_noise
        self.replay_persist_across_rounds = replay_persist_across_rounds
        self.server_round = 0
        self.render_eval = render_eval
        self.render_mode = render_mode
        self.render_save_dir = render_save_dir
        self.render_every_n_rounds = render_every_n_rounds
        self.render_episodes = render_episodes
        self.render_client_tag = render_client_tag

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_size)
        
        # Current observation
        reset_result = self.env.reset()
        self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        # Training statistics
        self.n_steps = 0
        self.last_actions = None  # Store last collected actions for metrics
        
        # Episode tracking
        self._episode_reward = 0.0
        self._episode_length = 0

    def set_server_round(self, rnd: int):
        self.server_round = int(rnd)

    def _update_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one update step from replay buffer.
        
        Returns:
            Dictionary with losses and metrics, or None if buffer too small
        """
        if len(self.replay_buffer) < self.replay_initial:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Update agent (handles both actor and critic)
        metrics = self.agent.update(batch)
        
        return metrics
    
    def train_one_round(self) -> Dict[str, float]:
        """
        Train for one federated round (merge_interval env steps).
        
        Matches FedRL/deep/DeepRLAlgo.DDPG_TRAIN: one environment step then (if buffer
        is warm) one gradient step per iteration — not collect-all-then-update-all.
        """
        t0 = time.time()

        if not self.replay_persist_across_rounds:
            self.replay_buffer.clear()
        if hasattr(self.agent, "reset_ou_noise"):
            self.agent.reset_ou_noise()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        num_updates = 0
        collected_actions: List[np.ndarray] = []

        for _ in range(self.merge_interval):
            action = self.agent.select_action(
                self._obs,
                deterministic=not self.add_noise,
                add_noise=self.add_noise,
            )
            collected_actions.append(action)

            next_obs, reward, done, truncated, _ = self.env.step(action)
            done = done or truncated

            self.replay_buffer.push(
                state=np.copy(self._obs),
                action=action,
                reward=reward,
                next_state=np.copy(next_obs) if not done else None,
                done=done,
            )

            self._obs = next_obs
            self._episode_reward += reward
            self._episode_length += 1
            self.n_steps += 1

            if done:
                reset_result = self.env.reset()
                self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                self._episode_reward = 0.0
                self._episode_length = 0
                if hasattr(self.agent, "reset_ou_noise"):
                    self.agent.reset_ou_noise()

            if len(self.replay_buffer) >= self.replay_initial:
                metrics = self._update_step()
                if metrics is not None:
                    total_actor_loss += metrics.get("loss/actor", 0.0)
                    total_critic_loss += metrics.get("loss/critic", 0.0)
                    num_updates += 1

        self.last_actions = np.array(collected_actions) if collected_actions else None

        avg_actor_loss = total_actor_loss / max(num_updates, 1)
        avg_critic_loss = total_critic_loss / max(num_updates, 1)
        avg_total_loss = avg_actor_loss + avg_critic_loss
        
        # Evaluation
        eval_ret = 0.0
        for _ in range(self.eval_episodes):
            eval_ret += self._eval_episode()
        eval_ret /= max(1, self.eval_episodes)
        
        dur = max(time.time() - t0, 1e-8)

        out = {
            "loss": avg_total_loss,
            "loss/actor": avg_actor_loss,
            "loss/critic": avg_critic_loss,
            "train/return": float(self._episode_reward),  # Last episode reward
            "eval/return": float(eval_ret),
            "train/episode_length": float(self._episode_length),
            "train/num_updates": float(num_updates),
            "train/buffer_size": float(len(self.replay_buffer)),
            "time/sec_per_round": float(dur),
        }

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
            act_fn=lambda o: self.agent.select_action(o, deterministic=True, add_noise=False),
        )

        return out
    
    def _eval_episode(self) -> float:
        """Evaluate policy for one episode (deterministic)."""
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        ep_ret = 0.0
        done = False
        
        while not done:
            action = self.agent.select_action(obs, deterministic=True, add_noise=False)
            obs, r, done, truncated, _ = self.env.step(action)
            done = done or truncated
            ep_ret += r
        
        return ep_ret
    
    def save_eval(self, cid: str, rnd: int, outdir: str = "./results/fedrl_ddpg") -> bool:
        """
        Save evaluation trajectory and metadata.
        
        Args:
            cid: Client ID
            rnd: Round number
            outdir: Output directory
        
        Returns:
            True if successful
        """
        # Run evaluation episode and collect trajectory
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        traj = [obs.copy() if hasattr(obs, 'copy') else np.array(obs)]
        ep_ret = 0.0
        done = False
        
        while not done:
            action = self.agent.select_action(obs, deterministic=True, add_noise=False)
            obs, r, done, truncated, _ = self.env.step(action)
            done = done or truncated
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
        return self._episode_reward
    
    @property
    def episode_len(self):
        """Average episode length (for compatibility)."""
        return self._episode_length

