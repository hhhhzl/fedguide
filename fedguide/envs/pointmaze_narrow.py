import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PointMazeNarrow(gym.Env):
    """2D maze with a narrow passage; reward=+1 when reaching goal region."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, size: int = 10, passage_width: int = 1):
        """
        Args:
            size: Maze grid size.
            passage_width: Integer width of the passage (>=1).
        """
        super().__init__()
        assert isinstance(passage_width, int) and passage_width >= 1, \
            "passage_width must be an integer >= 1"

        self.size = size
        self.goal = np.array([size - 1, size - 1], dtype=np.float32)
        self.state = np.zeros(2, dtype=np.float32)
        self.observation_space = spaces.Box(0, size, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

        # 1 = free, 0 = wall
        self.grid = np.ones((size, size), dtype=np.int32)
        mid = size // 2
        half = passage_width // 2
        # Ensure at least one free cell along the center
        self.grid[mid, :mid - half] = 0
        self.grid[mid, mid + half + 1:] = 0

    # ----------------------------------------------------------
    # Core gym API
    # ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.5, 0.5], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        s_new = self.state + 0.5 * action
        s_new = np.clip(s_new, 0, self.size - 1)

        # Collision check
        if self._is_blocked(s_new):
            s_new = self.state
        self.state = s_new

        reward = 1.0 if np.linalg.norm(s_new - self.goal) < 0.5 else 0.0
        done = reward > 0
        return s_new, reward, done, False, {}

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def _is_blocked(self, s):
        i, j = np.clip(s.astype(int), 0, self.size - 1)
        return self.grid[i, j] == 0

    def render(self):
        grid = self.grid.copy()
        i, j = self.state.astype(int)
        grid[i, j] = 8
        print(grid[::-1])
