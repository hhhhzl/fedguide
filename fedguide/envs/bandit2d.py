import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Bandit2D(gym.Env):
    """
    2D Bandit Environment for Federated Learning Toy Experiment
    
    - Action = State: a = (x, y) ∈ [-1.5, 1.5]²
    - Global reward: K peaks on unit circle
    - R(a) = max_{i=1..K} exp(-||a - μ_i||² / (2σ²))
    
    Each client i only sees data near μ_i (client heterogeneity).
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, K=4, sigma=0.2, seed=None, preferred_peak: int = None):
        """
        Args:
            K: Number of peaks (default: 4)
            sigma: Standard deviation for reward function (default: 0.2)
            seed: Random seed
            preferred_peak: If set, client-specific heterogeneity: only this peak gets full reward.
                          Other peaks get weight 0.1. Use for federated training where each
                          client sees data near one peak. None = global (all peaks equal).
        """
        super().__init__()
        self.K = K  # number of peaks
        self.sigma = sigma  # σ in the reward formula
        self.preferred_peak = preferred_peak  # None = global, int = client-specific
        
        # Place K peaks on unit circle
        angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
        self.mu = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
        
        # Peak weights for heterogeneous reward: client i prefers peak i (others get 0.1 for gradient)
        if preferred_peak is not None:
            self.peak_weights = np.ones(K, dtype=np.float32) * 0.1
            self.peak_weights[preferred_peak % K] = 1.0
            # Peak locations: 0=(1,0), 1=(0,1), 2=(-1,0), 3=(0,-1)
        else:
            self.peak_weights = np.ones(K, dtype=np.float32)
        
        # Action = State space: [-1.5, 1.5]²
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.5, high=1.5, shape=(2,), dtype=np.float32
        )
        
        self.state = None
        self._seed = seed
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed or self._seed)
        # Start at random position
        self.state = self.action_space.sample()
        return self.state.copy(), {}
    
    def step(self, action):
        # Clip action to valid range
        action = np.clip(action, -1.5, 1.5)
        self.state = action.copy()
        
        # Compute reward: R(a) = max_i w_i * exp(-||a - μ_i||² / (2σ²))
        # With preferred_peak, w_i=1 for client's peak, 0.1 for others (heterogeneity)
        distances = np.linalg.norm(self.state - self.mu, axis=1)
        rewards = self.peak_weights * np.exp(-distances**2 / (2 * self.sigma**2))
        reward = float(np.max(rewards))
        
        # Bandit: always done after one step
        done = True
        terminated = True
        truncated = False
        return self.state.copy(), reward, terminated, truncated, {}
    
    def get_peak_locations(self):
        """Return the peak locations μ_i for visualization."""
        return self.mu.copy()
    
    def compute_reward(self, action):
        """Compute reward for a given action (for data generation)."""
        action = np.clip(action, -1.5, 1.5)
        distances = np.linalg.norm(action - self.mu, axis=1)
        rewards = self.peak_weights * np.exp(-distances**2 / (2 * self.sigma**2))
        return float(np.max(rewards))

