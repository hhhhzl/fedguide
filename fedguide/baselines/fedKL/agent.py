
"""
FedKL Agent Implementation

This module implements the FedKL agent with KL divergence computation
between local and global policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


def _to_device(module_or_tensor, device):
    if hasattr(module_or_tensor, "to"):
        return module_or_tensor.to(device)
    return module_or_tensor


class PolicyNetwork(nn.Module):
    """Policy network for continuous action spaces."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))  #tanh stuff
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        return mean


class ValueNetwork(nn.Module):
    """Value network for critic."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value


class FedKLAgent:
    """
    FedKL Agent implementing KL divergence penalty for federated RL.
    
    The agent maintains a local policy and a reference to the global policy.
    It penalizes divergence from the global policy during local training.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Local policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))
        
        # Value network
        self.value = ValueNetwork(state_dim, hidden_dim).to(self.device)
        
        # Global policy (reference) - updated from server
        self.global_policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.global_log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))
        
        # Copy initial parameters to global policy
        self.global_policy.load_state_dict(self.policy.state_dict())
        self.global_log_std.data = self.log_std.data.clone()
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + [self.log_std], lr=lr
        )
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean = self.policy(state_tensor)
            
            if deterministic:
                return mean.cpu().numpy()[0]
            
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return action.cpu().numpy()[0]
    
    def compute_kl_divergence(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between local policy and global policy.
        KL(π_local || π_global) for Gaussian policies.
        """
        # Local policy distribution
        mean_local = self.policy(states)
        std_local = torch.exp(self.log_std)
        
        # Global policy distribution
        with torch.no_grad():
            mean_global = self.global_policy(states)
            std_global = torch.exp(self.global_log_std)
        
        # KL divergence for diagonal Gaussian
        var_local = std_local.pow(2)
        var_global = std_global.pow(2)
        
        kl = (
            torch.log(std_global / std_local)
            + (var_local + (mean_local - mean_global).pow(2)) / (2 * var_global)
            - 0.5
        )
        
        return kl.sum(dim=-1).mean()
    
    def update_global_policy(self):
        """Update the global policy reference from current local policy."""
        self.global_policy.load_state_dict(self.policy.state_dict())
        self.global_log_std.data = self.log_std.data.clone()
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get parameters for federated aggregation."""
        params = {}
        
        # Policy parameters
        for name, param in self.policy.named_parameters():
            params[f"policy.{name}"] = param.detach().cpu().numpy()
        params["log_std"] = self.log_std.detach().cpu().numpy()
        
        # Value parameters (optional, can be kept local)
        for name, param in self.value.named_parameters():
            params[f"value.{name}"] = param.detach().cpu().numpy()
        
        return params
    
def set_parameters(self, params: Dict[str, np.ndarray]):
    """Set parameters from federated aggregation."""
    # Update policy network
    policy_state = {}
    for name, param in self.policy.named_parameters():
        key = f"policy.{name}"
        if key in params:
            policy_state[name] = torch.FloatTensor(params[key]).to(self.device)
    
    # Apply policy parameters
    for name, param in self.policy.named_parameters():
        if name in policy_state:
            param.data.copy_(policy_state[name])
    
    # Update log_std
    if "log_std" in params:
        self.log_std.data.copy_(
            torch.FloatTensor(params["log_std"]).to(self.device)
        )
    
    # Update value network (if provided, though typically not aggregated)
    value_state = {}
    for name, param in self.value.named_parameters():
        key = f"value.{name}"
        if key in params:
            value_state[name] = torch.FloatTensor(params[key]).to(self.device)
    
    for name, param in self.value.named_parameters():
        if name in value_state:
            param.data.copy_(value_state[name])
    
    # IMPORTANT: Update global policy reference after receiving new parameters
    # This ensures KL divergence is computed against the updated global policy
    self.update_global_policy()

    
    def to(self, device: str):
        """Move agent to device."""
        self.device = device
        self.policy = self.policy.to(device)
        self.value = self.value.to(device)
        self.global_policy = self.global_policy.to(device)
        self.log_std.data = self.log_std.data.to(device)
        self.global_log_std.data = self.global_log_std.data.to(device)
        return self

