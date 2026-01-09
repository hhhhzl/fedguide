"""
FedRL Agent Implementation

This module implements DQN and DDPG agents for federated reinforcement learning.
Based on FedRL paper: "Federated Reinforcement Learning with Environment Heterogeneity" (AISTATS 2022)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Any, Tuple, List


# ============================================================================
# Parameter Aggregation Utilities (from FedRL/deep/DQNAvg.py)
# ============================================================================

def net_para_add(para_a: Dict[str, torch.Tensor], para_b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Add two parameter dictionaries element-wise.
    
    Args:
        para_a: First parameter dictionary
        para_b: Second parameter dictionary (must have same keys as para_a)
    
    Returns:
        New dictionary with added parameters
    """
    para_c = copy.deepcopy(para_a)
    for key in para_c:
        para_c[key] = para_a[key] + para_b[key]
    return para_c


def net_para_scale(para_a: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    """
    Scale all parameters in a dictionary by a scalar.
    
    Args:
        para_a: Parameter dictionary
        scale: Scaling factor
    
    Returns:
        New dictionary with scaled parameters
    """
    para_c = copy.deepcopy(para_a)
    for key in para_c:
        para_c[key] = para_a[key] * scale
    return para_c


# ============================================================================
# DQN Networks and Agent
# ============================================================================

class DQNNetwork(nn.Module):
    """
    DQN Network for discrete action spaces.
    
    Architecture: MLP(obs_size -> hidden_size -> n_actions)
    Based on FedRL/deep/DeepRLAlgo.py MLP_Q_Net
    """
    
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.network_type = 'Q'  # For compatibility with FedRL
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: return Q-values for each action."""
        return self.net(x.float())


class DQNAgent(nn.Module):
    """
    DQN Agent for discrete action spaces.
    
    Implements Double DQN with epsilon-greedy exploration.
    Supports federated aggregation of Q-network parameters.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.01,
        sync_interval: int = 10,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.sync_interval = sync_interval
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device or "cpu")
        
        # Q-networks: main and target
        self.q_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Step counter for target network sync
        self.update_step = 0
    
    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (numpy array)
            deterministic: If True, always select best action (for evaluation)
        
        Returns:
            Selected action (int)
        """
        if deterministic or np.random.random() > self.epsilon:
            # Exploit: select best action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        else:
            # Explore: random action
            action = np.random.randint(0, self.action_dim)
        
        return action
    
    def update(self, batch: List[Tuple]) -> Dict[str, float]:
        """
        Update Q-network using batch of experiences.
        
        Args:
            batch: List of (state, action, reward, next_state, done) tuples
                  where next_state may be None if done=True
        
        Returns:
            Dictionary with loss and metrics
        """
        # Unpack batch - handle None next_states (terminal states)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Handle next_states: use current state if next_state is None (terminal)
        next_states_list = []
        for i, e in enumerate(batch):
            if e[3] is None or dones[i]:
                next_states_list.append(e[0])  # Use current state as placeholder
            else:
                next_states_list.append(e[3])
        next_states = torch.FloatTensor(next_states_list).to(self.device)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states)
            next_q_value = next_q_values.max(dim=1)[0]
            next_q_value[dones] = 0.0  # Zero out terminal states
            target_q = rewards + self.gamma * next_q_value
        
        # Compute current Q-values
        current_q_values = self.q_net(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_step += 1
        if self.update_step % self.sync_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for federated aggregation.
        
        Returns:
            Dictionary with Q-network parameters
        """
        return {
            "q_net": {k: v.detach().cpu() for k, v in self.q_net.state_dict().items()},
        }
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Set parameters from federated aggregation.
        
        Args:
            parameters: Dictionary with Q-network parameters
        """
        if "q_net" in parameters:
            self.q_net.load_state_dict(
                {k: v.to(self.device) for k, v in parameters["q_net"].items()},
                strict=False
            )
            # Also update target network
            self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def rebuild_optimizer(self):
        """Recreate optimizer after parameter aggregation."""
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
    
    def to(self, device: str):
        """Move agent to device."""
        self.device = torch.device(device)
        self.q_net = self.q_net.to(self.device)
        self.target_q_net = self.target_q_net.to(self.device)
        return self
    
    def parameters_iter(self):
        """Iterator over all trainable parameters."""
        return list(self.q_net.parameters())
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "epsilon": self.epsilon,
            "update_step": self.update_step,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        self.q_net.load_state_dict(state_dict["q_net"])
        self.target_q_net.load_state_dict(state_dict["target_q_net"])
        self.epsilon = state_dict.get("epsilon", 1.0)
        self.update_step = state_dict.get("update_step", 0)


# ============================================================================
# DDPG Networks and Agent
# ============================================================================

class DDPGActor(nn.Module):
    """
    DDPG Actor Network for continuous action spaces.
    
    Architecture: MLP(obs_size -> 400 -> 300 -> action_dim) -> Tanh
    Based on FedRL/deep/DeepRLAlgo.py DDPGActor
    """
    
    def __init__(self, obs_size: int, action_size: int, threshold: float = 2.0):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.network_type = 'DP'  # For compatibility with FedRL
        self.rescale = threshold
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_size),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: return action in [-threshold, threshold]."""
        return self.rescale * self.net(x.float())


class DDPGCritic(nn.Module):
    """
    DDPG Critic Network for continuous action spaces.
    
    Architecture: 
    - obs_net: MLP(obs_size -> 400) -> ReLU
    - out_net: MLP(400+action_size -> 300 -> 1)
    Based on FedRL/deep/DeepRLAlgo.py DDPGCritic
    """
    
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )
        
        self.out_net = nn.Sequential(
            nn.Linear(400 + action_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
    
    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: return Q-value for state-action pair.
        
        Args:
            x: State tensor (batch_size, obs_size)
            a: Action tensor (batch_size, action_size)
        
        Returns:
            Q-value tensor (batch_size, 1)
        """
        obs = self.obs_net(x.float())
        return self.out_net(torch.cat([obs, a.float()], dim=1))


class DDPGAgent(nn.Module):
    """
    DDPG Agent for continuous action spaces.
    
    Implements Deep Deterministic Policy Gradient with actor-critic architecture.
    Supports federated aggregation of actor (and optionally critic) parameters.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.001,  # Soft update coefficient for target networks
        threshold: float = 2.0,  # Action clipping threshold
        device: Optional[str] = None,
        aggregate_critic: bool = False,  # Whether to aggregate critic parameters
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.threshold = threshold
        self.aggregate_critic = aggregate_critic
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device or "cpu")
        
        # Actor networks: main and target
        self.actor = DDPGActor(state_dim, action_dim, threshold).to(self.device)
        self.target_actor = DDPGActor(state_dim, action_dim, threshold).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()
        
        # Critic networks: main and target
        self.critic = DDPGCritic(state_dim, action_dim).to(self.device)
        self.target_critic = DDPGCritic(state_dim, action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        
        # Optimizers
        self.lr = lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
    
    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = True, add_noise: bool = False) -> np.ndarray:
        """
        Select action using actor network.
        
        Args:
            state: Current state (numpy array)
            deterministic: If True, use deterministic policy (for evaluation)
            add_noise: If True, add exploration noise (for training)
        
        Returns:
            Selected action (numpy array)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor)
        
        if add_noise and not deterministic:
            # Add OU noise for exploration (simplified version)
            noise = torch.randn_like(action) * 0.1
            action = action + noise
        
        action = action.clamp(-self.threshold, self.threshold)
        return action.squeeze(0).cpu().numpy()
    
    def update(self, batch: List[Tuple]) -> Dict[str, float]:
        """
        Update actor and critic networks using batch of experiences.
        
        Args:
            batch: List of (state, action, reward, next_state, done) tuples
                  where next_state may be None if done=True
        
        Returns:
            Dictionary with losses and metrics
        """
        # Unpack batch - handle None next_states (terminal states)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device).unsqueeze(1)
        
        # Handle next_states: use current state if next_state is None (terminal)
        next_states_list = []
        dones_list = []
        for e in batch:
            if e[3] is None or e[4]:
                next_states_list.append(e[0])  # Use current state as placeholder
                dones_list.append(True)
            else:
                next_states_list.append(e[3])
                dones_list.append(e[4])
        
        next_states = torch.FloatTensor(next_states_list).to(self.device)
        dones = torch.BoolTensor(dones_list).to(self.device).unsqueeze(1)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q = self.target_critic(next_states, next_actions)
            next_q[dones] = 0.0  # Zero out terminal states
            target_q = rewards + self.gamma * next_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        policy_actions = self.actor(states)
        actor_loss = -self.critic(states, policy_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_actor, self.actor, self.tau)
        self._soft_update(self.target_critic, self.critic, self.tau)
        
        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/total": actor_loss.item() + critic_loss.item(),
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for federated aggregation.
        
        Returns:
            Dictionary with actor (and optionally critic) parameters
        """
        params = {
            "actor": {k: v.detach().cpu() for k, v in self.actor.state_dict().items()},
        }
        if self.aggregate_critic:
            params["critic"] = {k: v.detach().cpu() for k, v in self.critic.state_dict().items()}
        return params
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Set parameters from federated aggregation.
        
        Args:
            parameters: Dictionary with actor (and optionally critic) parameters
        """
        if "actor" in parameters:
            self.actor.load_state_dict(
                {k: v.to(self.device) for k, v in parameters["actor"].items()},
                strict=False
            )
            # Also update target actor
            self.target_actor.load_state_dict(self.actor.state_dict())
        
        if "critic" in parameters and self.aggregate_critic:
            self.critic.load_state_dict(
                {k: v.to(self.device) for k, v in parameters["critic"].items()},
                strict=False
            )
            # Also update target critic
            self.target_critic.load_state_dict(self.critic.state_dict())
    
    def rebuild_optimizer(self):
        """Recreate optimizers after parameter aggregation."""
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        if self.aggregate_critic:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    
    def to(self, device: str):
        """Move agent to device."""
        self.device = torch.device(device)
        self.actor = self.actor.to(self.device)
        self.target_actor = self.target_actor.to(self.device)
        self.critic = self.critic.to(self.device)
        self.target_critic = self.target_critic.to(self.device)
        return self
    
    def parameters_iter(self):
        """Iterator over all trainable parameters."""
        params = list(self.actor.parameters())
        if self.aggregate_critic:
            params.extend(list(self.critic.parameters()))
        return params
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        self.actor.load_state_dict(state_dict["actor"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])

