"""
Centralized SAC Trainer Implementation

This module implements a centralized trainer that merges data from multiple clients
into a single replay buffer and trains a central SAC agent.
"""

import numpy as np
import torch
from torch.distributions import Normal
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import os

from .agent import SACAgent
from fedguide.datasets.base import TransitionDataset


class CentralSACTrainer:
    """
    Centralized SAC Trainer that learns from multiple clients' data.
    
    This trainer:
    1. Merges data from multiple client datasets into a single replay buffer
    2. Trains a central SAC agent on the mixed data
    3. Supports evaluation on the environment
    4. Collects training metrics
    
    No federated aggregation is performed - this is pure centralized training.
    """
    
    def __init__(
        self,
        agent: SACAgent,
        datasets: List[TransitionDataset],
        env: Any,
        batch_size: int = 256,
        update_steps: int = 1000,
        gamma: float = 0.99,
        eval_episodes: int = 10,
        eval_stochastic_samples: int = 64,
        device: Optional[str] = None,
        render_eval: bool = False,
        render_mode: str = "video",
        render_save_dir: Optional[str] = None,
        render_every_n_rounds: int = 10,
        render_episodes: int = 1,
    ):
        """
        Initialize centralized SAC trainer.
        
        Args:
            agent: SAC agent to train
            datasets: List of TransitionDataset objects from different clients
            env: Environment for evaluation (gymnasium.Env)
            batch_size: Batch size for training
            update_steps: Number of update steps per training round
            gamma: Discount factor (should match agent's gamma)
            eval_episodes: Number of episodes for evaluation
            eval_stochastic_samples: Number of action samples per state for stochastic evaluation
            device: Device to run on
            render_eval: Whether to render evaluation episodes
            render_mode: Rendering mode ("human", "rgb_array", or "video")
            render_save_dir: Directory to save rendered videos (if render_mode="video")
            render_every_n_rounds: Render every N rounds (0 = only last round, -1 = all rounds)
            render_episodes: Number of episodes to render per round
        """
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.gamma = gamma
        self.eval_episodes = eval_episodes
        self.eval_stochastic_samples = eval_stochastic_samples
        self.render_eval = render_eval
        self.render_mode = render_mode
        self.render_save_dir = render_save_dir
        self.render_every_n_rounds = render_every_n_rounds
        self.render_episodes = render_episodes
        
        # Set device
        if device is None:
            self.device = agent.device
        else:
            self.device = torch.device(device)
            self.agent.to(self.device)
        
        # Build replay buffer from all client datasets
        self.replay_buffer = self._build_replay_buffer(datasets)
        
        # Track data statistics
        self.client_data_sizes = [len(ds) for ds in datasets]
        self.num_clients = len(datasets)
        self.total_transitions = len(self.replay_buffer['states'])
        
        # Evaluation tracking
        self.eval_returns = deque(maxlen=100)
        self.eval_returns_smoothed = deque(maxlen=100)  # For smoothed evaluation
        self.eval_returns_stochastic = deque(maxlen=100)  # For stochastic evaluation (mean)
        self.eval_returns_stochastic_smoothed = deque(maxlen=100)  # Smoothed stochastic eval (mean)
        self.eval_returns_stochastic_max = deque(maxlen=100)  # For stochastic evaluation (max)
        self.eval_returns_stochastic_max_smoothed = deque(maxlen=100)  # Smoothed stochastic eval (max)
        
        # Get action bounds from environment (for evaluation)
        if hasattr(env, 'action_space') and hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
        else:
            # Default bounds (for backward compatibility with Bandit2D)
            self.action_low = None
            self.action_high = None
        
        print(f"[CentralSACTrainer] Initialized with {self.num_clients} clients")
        print(f"[CentralSACTrainer] Total transitions: {self.total_transitions}")
        print(f"[CentralSACTrainer] Client data sizes: {self.client_data_sizes}")
        if self.render_eval:
            print(f"[CentralSACTrainer] Rendering enabled: mode={self.render_mode}, save_dir={self.render_save_dir}")
        if self.action_low is not None and self.action_high is not None:
            print(f"[CentralSACTrainer] Action bounds: [{self.action_low}, {self.action_high}]")
    
    def _build_replay_buffer(self, datasets: List[TransitionDataset]) -> Dict[str, np.ndarray]:
        """
        Merge multiple client datasets into a single replay buffer.
        
        Args:
            datasets: List of TransitionDataset objects
        
        Returns:
            Dictionary with keys: 'states', 'actions', 'rewards', 'next_states', 'dones'
        """
        buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }
        
        # Merge all transitions from all clients
        for client_id, dataset in enumerate(datasets):
            for i in range(len(dataset)):
                sample = dataset[i]
                buffer['states'].append(sample['s'].numpy())
                buffer['actions'].append(sample['a'].numpy())
                buffer['rewards'].append(sample['r'].item())
                buffer['next_states'].append(sample['s_'].numpy())
                buffer['dones'].append(sample['d'].item())
        
        # Convert to numpy arrays
        buffer = {k: np.array(v, dtype=np.float32) for k, v in buffer.items()}
        
        return buffer
    
    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch from replay buffer.
        
        Returns:
            Dictionary with batched tensors ready for agent.update()
        """
        # Random sampling with replacement
        indices = np.random.choice(
            len(self.replay_buffer['states']),
            size=self.batch_size,
            replace=True
        )
        
        batch = {
            's': torch.FloatTensor(self.replay_buffer['states'][indices]).to(self.device),
            'a': torch.FloatTensor(self.replay_buffer['actions'][indices]).to(self.device),
            'r': torch.FloatTensor(self.replay_buffer['rewards'][indices]).to(self.device),
            's_next': torch.FloatTensor(self.replay_buffer['next_states'][indices]).to(self.device),
            'done': torch.FloatTensor(self.replay_buffer['dones'][indices]).to(self.device),
        }
        
        return batch
    
    def _evaluate(self) -> float:
        """
        Evaluate the current policy on the environment.
        
        Returns:
            Average episode return over eval_episodes
        """
        returns = []
        
        for _ in range(self.eval_episodes):
            state, _ = self.env.reset()
            total_return = 0.0
            done = False
            step_count = 0
            max_steps = 1000  # Safety limit
            
            while not done and step_count < max_steps:
                # Get action from agent (deterministic for evaluation)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, _ = self.agent.act(state_tensor, eval=True)
                
                # Handle both 1D and 2D action outputs
                if isinstance(action, torch.Tensor):
                    action_np = action.detach().cpu().numpy()
                    # Remove batch dimension if present
                    if action_np.ndim > 1:
                        action_np = action_np[0]
                    elif action_np.ndim == 0:
                        action_np = np.array([action_np])
                else:
                    action_np = np.array(action)
                    if action_np.ndim > 1:
                        action_np = action_np[0]
                
                # Ensure action is in valid range for environment
                if self.action_low is not None and self.action_high is not None:
                    action_np = np.clip(action_np, self.action_low, self.action_high)
                else:
                    # Fallback for backward compatibility (Bandit2D range)
                    action_np = np.clip(action_np, -1.5, 1.5)
                
                # Step environment
                next_state, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated
                total_return += reward
                state = next_state
                step_count += 1
                
                # For bandit environments, always done after one step
                if done:
                    break
            
            returns.append(total_return)
        
        # Debug: Print evaluation statistics occasionally
        if len(returns) > 0 and np.mean(returns) == 0.0:
            # Only print once to avoid spam
            if not hasattr(self, '_eval_warned'):
                print(f"Warning: All eval returns are 0. Action range: [{action_np.min() if len(action_np) > 0 else 'N/A'}, {action_np.max() if len(action_np) > 0 else 'N/A'}]")
                self._eval_warned = True
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        self.eval_returns.append(avg_return)
        
        # Apply exponential moving average to smooth evaluation results
        # This reduces variance and makes the evaluation curve smoother
        if len(self.eval_returns) > 1:
            alpha = 0.3  # Smoothing factor (0.3 means 30% weight on new value, 70% on previous)
            smoothed = alpha * avg_return + (1 - alpha) * self.eval_returns_smoothed[-1] if len(self.eval_returns_smoothed) > 0 else avg_return
            self.eval_returns_smoothed.append(smoothed)
            return smoothed
        else:
            self.eval_returns_smoothed.append(avg_return)
            return avg_return
    
    def _evaluate_with_render(self, round_num: int) -> Tuple[float, Optional[str]]:
        """
        Evaluate with rendering support.
        
        Args:
            round_num: Current training round number
            
        Returns:
            avg_return: Average episode return
            video_path: Path to saved video (if rendered), None otherwise
        """
        returns = []
        frames = []  # For video rendering
        
        # Check if we should render this round
        should_render = False
        if self.render_eval:
            if self.render_every_n_rounds == -1:
                # Render all rounds
                should_render = True
            elif self.render_every_n_rounds == 0:
                # Only render last round (handled in training script)
                # For now, we'll render if explicitly requested (round_num will be checked externally)
                should_render = False
            else:
                # Render every N rounds (including first round)
                should_render = (round_num % self.render_every_n_rounds == 0 or round_num == 1)
        
        for ep_idx in range(self.eval_episodes):
            state, _ = self.env.reset()
            total_return = 0.0
            done = False
            step_count = 0
            max_steps = 1000
            
            # Collect frames only for the first render_episodes episodes
            episode_frames = [] if (should_render and ep_idx < self.render_episodes) else None
            
            while not done and step_count < max_steps:
                # Get action from agent (deterministic for evaluation)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, _ = self.agent.act(state_tensor, eval=True)
                
                # Handle both 1D and 2D action outputs
                if isinstance(action, torch.Tensor):
                    action_np = action.detach().cpu().numpy()
                    # Remove batch dimension if present
                    if action_np.ndim > 1:
                        action_np = action_np[0]
                    elif action_np.ndim == 0:
                        action_np = np.array([action_np])
                else:
                    action_np = np.array(action)
                    if action_np.ndim > 1:
                        action_np = action_np[0]
                
                # Ensure action is in valid range for environment
                if self.action_low is not None and self.action_high is not None:
                    action_np = np.clip(action_np, self.action_low, self.action_high)
                else:
                    # Fallback for backward compatibility (Bandit2D range)
                    action_np = np.clip(action_np, -1.5, 1.5)
                
                # Render if needed
                if episode_frames is not None:
                    try:
                        if self.render_mode in ["rgb_array", "video"]:
                            frame = self.env.render(mode="rgb_array")
                            if frame is not None:
                                episode_frames.append(frame)
                        elif self.render_mode == "human":
                            self.env.render()
                    except Exception as e:
                        # Some environments may not support rendering
                        # Silently skip if rendering fails
                        pass
                
                # Step environment
                next_state, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated
                total_return += reward
                state = next_state
                step_count += 1
                
                # For bandit environments, always done after one step
                if done:
                    break
            
            returns.append(total_return)
            if episode_frames:
                frames.extend(episode_frames)
        
        # Save video if frames were collected
        video_path = None
        if frames and self.render_save_dir and should_render and self.render_mode == "video":
            try:
                import imageio
                os.makedirs(self.render_save_dir, exist_ok=True)
                video_path = os.path.join(self.render_save_dir, f"round_{round_num:04d}.mp4")
                imageio.mimsave(video_path, frames, fps=30)
                print(f"  [Rendering] Saved evaluation video to {video_path}")
            except ImportError:
                print(f"  [Rendering] Warning: imageio not installed. Cannot save video. Install with: pip install imageio")
            except Exception as e:
                print(f"  [Rendering] Warning: Failed to save video: {e}")
        
        avg_return = np.mean(returns)
        self.eval_returns.append(avg_return)
        
        # Apply exponential moving average to smooth evaluation results
        if len(self.eval_returns) > 1:
            alpha = 0.3
            smoothed = alpha * avg_return + (1 - alpha) * self.eval_returns_smoothed[-1] if len(self.eval_returns_smoothed) > 0 else avg_return
            self.eval_returns_smoothed.append(smoothed)
            return smoothed, video_path
        else:
            self.eval_returns_smoothed.append(avg_return)
            return avg_return, video_path
    
    def _evaluate_stochastic(self) -> Dict[str, float]:
        """
        Evaluate using stochastic action sampling (multiple samples per state).
        This is useful for multi-modal policies where deterministic eval fails.
        
        For each episode, we sample multiple actions from the policy and compute:
        - max_return: Best return across all samples (best case performance)
        - mean_return: Average return across all samples (expected performance)
        
        Returns:
            Dictionary with 'max' and 'mean' returns
        """
        max_returns = []
        mean_returns = []
        
        for _ in range(self.eval_episodes):
            state, _ = self.env.reset()
            episode_returns = []
            
            # Sample multiple actions for the same state
            for _ in range(self.eval_stochastic_samples):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                # Use eval=False to sample from policy (stochastic)
                action, _ = self.agent.act(state_tensor, eval=False)
                
                # Convert to numpy
                if isinstance(action, torch.Tensor):
                    action_np = action.detach().cpu().numpy()
                    # Remove batch dimension if present
                    if action_np.ndim > 1:
                        action_np = action_np[0]
                    elif action_np.ndim == 0:
                        action_np = np.array([action_np])
                else:
                    action_np = np.array(action)
                    if action_np.ndim > 1:
                        action_np = action_np[0]
                
                # Ensure action is in valid range for environment
                if self.action_low is not None and self.action_high is not None:
                    action_np = np.clip(action_np, self.action_low, self.action_high)
                else:
                    # Fallback for backward compatibility (Bandit2D range)
                    action_np = np.clip(action_np, -1.5, 1.5)
                
                # Compute reward
                # For bandit environments, we can use compute_reward if available
                # Otherwise, step the environment and reset for next sample
                if hasattr(self.env, 'compute_reward'):
                    reward = self.env.compute_reward(action_np)
                else:
                    # Step environment (for bandit, this is one-step)
                    next_state, reward, terminated, truncated, _ = self.env.step(action_np)
                    # Reset environment to original state for next sample
                    # This is safe for bandit environments (single-step)
                    try:
                        self.env.state = state.copy()
                    except (AttributeError, TypeError):
                        # If we can't directly set state, reset the environment
                        # This will change the state, but for bandit it's acceptable
                        # since each step is independent
                        pass
                
                episode_returns.append(reward)
            
            # Record max and mean returns for this episode
            max_returns.append(max(episode_returns))
            mean_returns.append(np.mean(episode_returns))
        
        # Compute statistics
        avg_max_return = np.mean(max_returns)
        avg_mean_return = np.mean(mean_returns)
        
        # Track returns
        self.eval_returns_stochastic.append(avg_mean_return)
        self.eval_returns_stochastic_max.append(avg_max_return)
        
        # Apply exponential moving average
        alpha = 0.3
        if len(self.eval_returns_stochastic) > 1:
            smoothed = alpha * avg_mean_return + (1 - alpha) * self.eval_returns_stochastic_smoothed[-1]
            smoothed_max = alpha * avg_max_return + (1 - alpha) * self.eval_returns_stochastic_max_smoothed[-1]
        else:
            smoothed = avg_mean_return
            smoothed_max = avg_max_return
        
        self.eval_returns_stochastic_smoothed.append(smoothed)
        self.eval_returns_stochastic_max_smoothed.append(smoothed_max)
        
        return {
            'max': avg_max_return,
            'mean': avg_mean_return,
            'max_smoothed': smoothed_max,
            'mean_smoothed': smoothed,
        }
    
    def train_one_round(self, round_num: int = 0) -> Dict[str, float]:
        """
        Train for one round: perform multiple update steps on replay buffer.
        
        Args:
            round_num: Current training round number (for rendering and logging)
        
        Returns:
            Dictionary of training metrics
        """
        import sys
        
        # Initialize metrics
        metrics = {
            'loss': 0.0,
            'train/loss/actor': 0.0,
            'train/loss/critic': 0.0,
            'train/q_value': 0.0,
            'train/q_value_min': 0.0,
            'train/buffer_size': self.total_transitions,
        }

        # Perform multiple update steps
        print_interval = max(1, self.update_steps // 10)  # Print 10 times per round
        
        for step in range(self.update_steps):
            # Sample batch from replay buffer
            batch = self._sample_batch()
            
            # Update agent
            actor_loss, critic_loss, q_values = self.agent.update(batch)
            
            # Accumulate metrics
            metrics['loss'] += (actor_loss + critic_loss)
            metrics['train/loss/actor'] += actor_loss
            metrics['train/loss/critic'] += critic_loss
            metrics['train/q_value'] += q_values.mean().item()
            metrics['train/q_value_min'] += q_values.min().item()
            
            # Print progress during training
            if (step + 1) % print_interval == 0 or step == 0:
                print(f"      [Training] Step {step+1}/{self.update_steps}: "
                      f"actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}, "
                      f"q_value={q_values.mean().item():.4f}", flush=True)
        
        # Average metrics over update steps
        for key in [
            'loss',
            'train/loss/actor',
            'train/loss/critic',
            'train/q_value',
            'train/q_value_min'
        ]:
            metrics[key] /= self.update_steps
        
        # Add data statistics
        metrics['data/num_clients'] = self.num_clients
        metrics['data/total_transitions'] = self.total_transitions
        metrics['data/client_sizes'] = self.client_data_sizes
        
        # Evaluation (if enabled)
        if self.eval_episodes > 0:
            # Deterministic evaluation (with optional rendering)
            if self.render_eval:
                eval_return, video_path = self._evaluate_with_render(round_num)
                if video_path:
                    metrics['eval/video_path'] = video_path
            else:
                eval_return = self._evaluate()
            metrics['eval/return'] = eval_return
            metrics['eval/return_det'] = eval_return  # Explicitly mark as deterministic
            
            # Compute statistics over recent evaluations
            if len(self.eval_returns) > 1:
                metrics['eval/return/mean'] = np.mean(self.eval_returns)
                metrics['eval/return/std'] = np.std(self.eval_returns)
            
            # Stochastic evaluation (for multi-modal policies)
            stochastic_eval = self._evaluate_stochastic()
            metrics['eval/return_stochastic_max'] = stochastic_eval['max']
            metrics['eval/return_stochastic_mean'] = stochastic_eval['mean']
            metrics['eval/return_stochastic_max_smoothed'] = stochastic_eval['max_smoothed']
            metrics['eval/return_stochastic_mean_smoothed'] = stochastic_eval['mean_smoothed']
            
            # Compute statistics over recent stochastic evaluations
            if len(self.eval_returns_stochastic) > 1:
                metrics['eval/return_stochastic/mean'] = np.mean(self.eval_returns_stochastic)
                metrics['eval/return_stochastic/std'] = np.std(self.eval_returns_stochastic)
        
        return metrics
    
    def evaluate_policy_logprob_on_grid(
        self,
        grid_size: int = 200,
        bounds: Optional[tuple] = None,
        action_dims: Optional[List[int]] = None,
        state_sample: Optional[torch.Tensor] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Evaluate policy log probability on a 2D grid for visualization.
        
        For 2D action spaces: directly visualize the full action space.
        For high-dimensional action spaces: visualize marginal distribution of first two dimensions.
        
        Args:
            grid_size: Size of evaluation grid (grid_size x grid_size)
            bounds: Tuple of (min, max) for action space bounds. If None, infer from environment.
            action_dims: List of two action dimensions to visualize [dim1, dim2]. 
                        If None, use [0, 1] for high-dim or all dims for 2D.
            state_sample: Sample state tensor for high-dim environments. If None, use zero state.
            
        Returns:
            Dictionary with:
                - 'policy_logprob': Policy log probability on grid [grid_size, grid_size]
                - 'policy_density': Policy density (normalized) on grid [grid_size, grid_size]
                - 'X': X coordinates of grid [grid_size, grid_size]
                - 'Y': Y coordinates of grid [grid_size, grid_size]
                - 'action_dims': Which action dimensions were visualized
            Returns None if action space is not suitable for 2D visualization.
        """
        # Get action space information
        action_dim = self.agent.actor.out_features if hasattr(self.agent.actor, 'out_features') else None
        if action_dim is None:
            # Try to infer from agent
            try:
                # Create a dummy input to infer output dimension
                dummy_state = torch.zeros(1, self.agent.actor[0].in_features, device=self.device)
                with torch.no_grad():
                    dummy_action = self.agent.actor(dummy_state)
                    action_dim = dummy_action.shape[-1]
            except:
                print("Warning: Could not infer action dimension. Skipping logprob visualization.")
                return None
        
        # Determine which dimensions to visualize
        if action_dims is None:
            if action_dim == 2:
                action_dims = [0, 1]  # Full 2D space
            elif action_dim > 2:
                action_dims = [0, 1]  # Marginal distribution of first two dims
                print(f"Warning: Action space is {action_dim}D. Visualizing marginal distribution of dimensions {action_dims}.")
            else:
                print(f"Warning: Action space is {action_dim}D. Cannot visualize 2D grid.")
                return None
        
        # Get bounds
        if bounds is None:
            if self.action_low is not None and self.action_high is not None:
                # Use environment bounds for the selected dimensions
                bounds = (
                    float(self.action_low[action_dims[0]]),
                    float(self.action_high[action_dims[0]]),
                    float(self.action_low[action_dims[1]]),
                    float(self.action_high[action_dims[1]])
                )
            else:
                # Default bounds (for backward compatibility with Bandit2D)
                bounds = (-1.5, 1.5, -1.5, 1.5)
        
        # Handle both (min, max) and (min_x, max_x, min_y, max_y) formats
        if len(bounds) == 2:
            min_val, max_val = bounds
            bounds = (min_val, max_val, min_val, max_val)
        
        min_x, max_x, min_y, max_y = bounds
        
        # Create evaluation grid for selected dimensions
        xs = np.linspace(min_x, max_x, grid_size)
        ys = np.linspace(min_y, max_y, grid_size)
        X, Y = np.meshgrid(xs, ys)
        grid_points_2d = np.stack([X.ravel(), Y.ravel()], axis=-1)  # [N, 2]
        
        # For high-dim action spaces, create full action vectors
        if action_dim > 2:
            # Use mean action for other dimensions (or zero if not available)
            if state_sample is None:
                # Create a zero state (will need actual state for proper evaluation)
                # Try to infer state dimension from actor network
                try:
                    # Get first layer's input dimension
                    if hasattr(self.agent.actor, '__getitem__'):
                        state_dim = self.agent.actor[0].in_features
                    elif hasattr(self.agent.actor, 'in_features'):
                        state_dim = self.agent.actor.in_features
                    else:
                        # Try to infer by creating a dummy input
                        dummy = torch.zeros(1, 2, device=self.device)
                        with torch.no_grad():
                            _ = self.agent.actor(dummy)
                        state_dim = 2  # Default fallback
                except:
                    state_dim = 2  # Default fallback for Bandit2D
                state_sample = torch.zeros(1, state_dim, device=self.device)
            
            # Expand to batch size
            batch_size = grid_points_2d.shape[0]
            states = state_sample.repeat(batch_size, 1)
            
            # Create full action vectors: set selected dims to grid values, others to mean
            grid_actions = torch.zeros(batch_size, action_dim, device=self.device)
            grid_actions[:, action_dims[0]] = torch.tensor(grid_points_2d[:, 0], dtype=torch.float32, device=self.device)
            grid_actions[:, action_dims[1]] = torch.tensor(grid_points_2d[:, 1], dtype=torch.float32, device=self.device)
            
            # For other dimensions, use mean action from policy
            with torch.no_grad():
                self.agent.actor.eval()
                try:
                    mu_full = self.agent.actor(states)
                    # Fill in mean values for non-visualized dimensions
                    for dim in range(action_dim):
                        if dim not in action_dims:
                            grid_actions[:, dim] = mu_full[:, dim]
                finally:
                    self.agent.actor.train()
        else:
            # For 2D action space, state = action (Bandit2D case)
            states = torch.tensor(grid_points_2d, dtype=torch.float32, device=self.device)
            grid_actions = states
        
        # Compute policy log probability
        with torch.no_grad():
            # Get mean action from actor
            self.agent.actor.eval()
            try:
                mu = self.agent.actor(states)
            finally:
                self.agent.actor.train()
            
            # Create distribution
            dist = Normal(mu, torch.ones_like(mu) * self.agent.action_std)
            
            # Compute log prob at grid_actions
            # For marginal distribution, we only care about the selected dimensions
            if action_dim > 2:
                # Compute full log prob, but we're visualizing marginal of selected dims
                policy_logp = dist.log_prob(grid_actions).sum(dim=-1)
            else:
                # Full 2D space
                policy_logp = dist.log_prob(grid_actions).sum(dim=-1)
            
            # Convert to numpy
            policy_logp = policy_logp.cpu().numpy()
            
            # Reshape to grid
            policy_logp_grid = policy_logp.reshape(X.shape)
            
            # Convert to density (normalized) - this is what we'll visualize
            policy_logp_flat = policy_logp_grid.ravel()
            policy_logp_flat = policy_logp_flat - policy_logp_flat.max()  # Avoid overflow
            policy_density = np.exp(policy_logp_flat).reshape(X.shape)
        
        return {
            'policy_logprob': policy_logp_grid,
            'policy_density': policy_density,
            'X': X,
            'Y': Y,
            'action_dims': action_dims,
            'action_dim': action_dim,
        }
    
    def save_eval(self, cid: str, round_num: int) -> bool:
        """
        Placeholder for compatibility with FedRLClient interface.
        
        Args:
            cid: Client ID (not used in centralized training)
            round_num: Round number
            
        Returns:
            True (always succeeds)
        """
        return True
    
    @property
    def n_steps(self) -> int:
        """Number of steps used in training (for compatibility)."""
        return self.update_steps * self.batch_size

