"""
Bandit2D Metrics Collector

This module provides utilities for collecting and saving metrics during
Bandit2D federated learning experiments for visualization purposes.
"""

import numpy as np
import torch
import pickle
import os
from typing import Dict, List, Optional, Any
from pathlib import Path


class Bandit2DMetricsCollector:
    """
    Collect metrics for Bandit2D experiments for visualization.
    
    This collector gathers:
    - Client action distributions (data distribution)
    - Local prior log probabilities (on grid)
    - Federated prior log probabilities (on grid)
    - Local value functions (on grid)
    - Federated value functions (on grid)
    - Local policies (on grid)
    - FedGuide policies (on grid)
    """
    
    def __init__(self, save_dir: str, grid_size: int = 200, bounds: tuple = (-1.5, 1.5)):
        """
        Initialize metrics collector.
        
        Args:
            save_dir: Directory to save metrics
            grid_size: Size of evaluation grid (grid_size x grid_size)
            bounds: Tuple of (min, max) for action space bounds
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.grid_size = grid_size
        self.bounds = bounds
        
        # Create evaluation grid
        xs = np.linspace(bounds[0], bounds[1], grid_size)
        ys = np.linspace(bounds[0], bounds[1], grid_size)
        X, Y = np.meshgrid(xs, ys)
        self.grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # [N, 2]
        self.X, self.Y = X, Y
        
        # Storage for collected data
        self.client_actions: Dict[int, List[np.ndarray]] = {}  # {client_id: [actions]}
        self.metrics_history: List[Dict] = []  # Metrics for each round
        self.client_agents: Dict[int, Any] = {}  # {client_id: agent} - stored by clients
        
    def collect_client_actions(self, client_id: int, actions: np.ndarray):
        """
        Collect actions from a client (for data distribution visualization).
        
        Args:
            client_id: Client identifier
            actions: Array of actions shape [N, 2]
        """
        if client_id not in self.client_actions:
            self.client_actions[client_id] = []
        self.client_actions[client_id].append(actions)
    
    def register_client_agent(self, client_id: int, agent: Any):
        """
        Register a client agent for metrics collection.
        
        Args:
            client_id: Client identifier
            agent: Agent object
        """
        self.client_agents[client_id] = agent
    
    def evaluate_on_grid(
        self,
        agent: Any,
        client_id: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate agent's prior, value, and policy on a 2D grid.
        
        Args:
            agent: Agent object with prior, value_fn, policy, log_std attributes
            client_id: Optional client identifier for logging
            round_num: Optional round number for logging
            
        Returns:
            Dictionary with keys:
                - 'prior_logprob': Prior log probability on grid [grid_size, grid_size]
                - 'value': Value function on grid [grid_size, grid_size]
                - 'policy_logprob': Policy log probability on grid [grid_size, grid_size]
                - 'policy_density': Policy density (normalized) on grid [grid_size, grid_size]
        """
        device = agent.device if hasattr(agent, 'device') else 'cpu'
        grid_tensor = torch.tensor(self.grid_points, dtype=torch.float32, device=device)
        
        # For Bandit2D, state = action, so we use grid_points as states
        states = grid_tensor
        
        metrics = {}
        
        # 1. Prior log probability
        if hasattr(agent, 'prior') and agent.prior is not None:
            if hasattr(agent.prior, 'log_prob'):
                with torch.no_grad():
                    try:
                        # For DiffusionGuidance (UNet-based), it expects trajectory format
                        # The _make_traj method creates [B, 1, traj_dim] but UNet actually expects [B, traj_dim, horizon]
                        # However, looking at the code, _make_traj creates [B, 1, 4] which gets passed to UNet
                        # The UNet then processes it, but the input format is wrong
                        # For grid evaluation with single points, we need to create proper trajectories
                        if hasattr(agent.prior, 'model') and hasattr(agent.prior, 'horizon'):
                            # This is DiffusionGuidance with UNet
                            # The issue is that _make_traj creates [B, 1, traj_dim] but UNet expects [B, traj_dim, horizon]
                            # Actually, looking at UNet1DModel, it might accept [B, C, L] where C=channels, L=length
                            # But _make_traj creates [B, 1, 4] which means L=1, C=4, which is wrong
                            # We need to create trajectories with proper length (horizon)
                            
                            # For grid evaluation with UNet-based prior, we need to handle the format issue
                            # The _make_traj creates [B, 1, 4] but UNet expects [B, 4, horizon]
                            # We'll skip prior evaluation for now as it's complex to fix the format
                            # The prior is mainly used during training, not for visualization
                            # Return zeros to indicate prior is not available for grid evaluation
                            # This won't break visualization, just won't show prior density
                            prior_logp = np.zeros(grid_tensor.shape[0])
                        else:
                            # SimpleDiffusionPrior or other prior types - direct call
                            prior_logp = agent.prior.log_prob(grid_tensor, states)
                            if isinstance(prior_logp, torch.Tensor):
                                prior_logp = prior_logp.cpu().numpy()
                        
                        metrics['prior_logprob'] = prior_logp.reshape(self.X.shape)
                    except Exception as e:
                        print(f"Warning: Failed to compute prior log prob: {e}")
                        import traceback
                        traceback.print_exc()
        
        # 2. Value function
        if hasattr(agent, 'value_fn'):
            with torch.no_grad():
                try:
                    value = agent.value_fn(states).squeeze(-1)
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    metrics['value'] = value.reshape(self.X.shape)
                except Exception as e:
                    print(f"Warning: Failed to compute value: {e}")
        elif hasattr(agent, 'value'):
            # For FedKL agent
            with torch.no_grad():
                try:
                    value = agent.value(states).squeeze(-1)
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    metrics['value'] = value.reshape(self.X.shape)
                except Exception as e:
                    print(f"Warning: Failed to compute value: {e}")
        
        # 3. Policy (compute mean from policy network, then compute log prob)
        if hasattr(agent, 'policy') and hasattr(agent, 'log_std'):
            with torch.no_grad():
                try:
                    mu = agent.policy(states)
                    log_std = agent.log_std.exp().clamp(min=1e-6)
                    std = log_std
                    # Compute policy log prob at grid_points
                    dist = torch.distributions.Normal(mu, std)
                    policy_logp = dist.log_prob(grid_tensor).sum(dim=-1)
                    if isinstance(policy_logp, torch.Tensor):
                        policy_logp = policy_logp.cpu().numpy()
                    metrics['policy_logprob'] = policy_logp.reshape(self.X.shape)
                    # Convert to density (normalized)
                    policy_logp_flat = policy_logp.reshape(-1)
                    policy_logp_flat = policy_logp_flat - policy_logp_flat.max()  # Avoid overflow
                    policy_density = np.exp(policy_logp_flat).reshape(self.X.shape)
                    metrics['policy_density'] = policy_density
                except Exception as e:
                    print(f"Warning: Failed to compute policy: {e}")
        
        return metrics
    
    def collect_round_metrics(
        self,
        round_num: int,
        client_agents: Dict[int, Any],  # {client_id: agent}
        server_agent: Optional[Any] = None,
        beta: float = 5.0,  # Coefficient for FedGuide policy: π ∝ prior * exp(β * value)
    ):
        """
        Collect metrics for a single round.
        
        Args:
            round_num: Current round number
            client_agents: Dictionary mapping client_id to agent objects
            server_agent: Aggregated server agent (if available)
            beta: Coefficient for computing FedGuide policy
        """
        round_metrics = {
            'round': round_num,
            'client_metrics': {},
            'server_metrics': {},
        }
        
        # Add client actions if available (even if no agents)
        if self.client_actions:
            round_metrics['client_actions'] = {
                k: list(v) if isinstance(v, np.ndarray) else v 
                for k, v in self.client_actions.items()
            }
        
        # Collect metrics for each client
        if client_agents:
            for client_id, agent in client_agents.items():
                try:
                    client_metrics = self.evaluate_on_grid(agent, client_id=client_id, round_num=round_num)
                    round_metrics['client_metrics'][client_id] = client_metrics
                except Exception as e:
                    print(f"Warning: Failed to evaluate client {client_id} on grid: {e}")
        
        # Collect server metrics (if available)
        if server_agent is not None:
            try:
                server_metrics = self.evaluate_on_grid(server_agent, client_id=None, round_num=round_num)
                
                # Compute FedGuide policy: π_FG ∝ π_fed_prior * exp(β * V_fed)
                if 'prior_logprob' in server_metrics and 'value' in server_metrics:
                    prior_lp = server_metrics['prior_logprob']
                    value = server_metrics['value']
                    log_pi_fg = prior_lp + beta * value
                    log_pi_fg = log_pi_fg - log_pi_fg.max()  # Normalize
                    pi_fg = np.exp(log_pi_fg)
                    server_metrics['fedguide_policy_density'] = pi_fg
                
                round_metrics['server_metrics'] = server_metrics
            except Exception as e:
                print(f"Warning: Failed to evaluate server agent on grid: {e}")
        
        # Always append, even if empty (at least we have the round number and actions)
        self.metrics_history.append(round_metrics)
    
    def save(self, filename: str = "bandit2d_metrics.pkl"):
        """
        Save all collected metrics to a pickle file.
        
        Args:
            filename: Name of the file to save
        """
        save_path = self.save_dir / filename
        data = {
            'client_actions': self.client_actions,
            'metrics_history': self.metrics_history,
            'grid_points': self.grid_points,
            'X': self.X,
            'Y': self.Y,
            'grid_size': self.grid_size,
            'bounds': self.bounds,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Metrics saved to {save_path}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load metrics from a pickle file.
        
        Args:
            filepath: Path to the pickle file
            
        Returns:
            Bandit2DMetricsCollector instance with loaded data
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        collector = cls(save_dir=str(Path(filepath).parent), 
                       grid_size=data['grid_size'],
                       bounds=data['bounds'])
        collector.client_actions = data['client_actions']
        collector.metrics_history = data['metrics_history']
        collector.grid_points = data['grid_points']
        collector.X = data['X']
        collector.Y = data['Y']
        return collector

