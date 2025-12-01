"""Base dataset classes."""
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Dataset for trajectory data (observations, actions)."""
    def __init__(self, observations, actions):
        self.obs = np.asarray(observations, dtype=np.float32)
        self.acts = np.asarray(actions, dtype=np.float32)

    def __len__(self): 
        return len(self.obs)

    def __getitem__(self, i):
        return np.concatenate([self.obs[i], self.acts[i]], axis=-1)


class TransitionDataset(Dataset):
    """Dataset class for transitions."""
    def __init__(self, trajs):
        self.data = []
        for tr in trajs:
            s, a, r, s_next, d = tr["s"], tr["a"], tr["r"], tr["s_next"], tr["d"]
            T = len(a)  # actions define number of transitions (there are t-1 actions for t states)
            for t in range(T):
                self.data.append((s[t], a[t], r[t], s_next[t], d[t]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, a, r, s_next, d = self.data[idx]
        return {
            "s": torch.tensor(s, dtype=torch.float32),
            "a": torch.tensor(a, dtype=torch.float32),
            "r": torch.tensor(r, dtype=torch.float32),
            "s_": torch.tensor(s_next, dtype=torch.float32),
            "d": torch.tensor(d, dtype=torch.float32),
        }


class TrajectoryWindowDataset(Dataset):
    """
    Yields fixed-length windows of (state sequence, action sequence).

    Each item:
      - s: [H, obs_dim]
      - a: [H, act_dim]
    """
    def __init__(self, trajs, horizon=32, stride=16):
        self.windows = []
        self.horizon = horizon

        for tr in trajs:
            s = tr["s"]          # [T_s, obs_dim]
            a = tr["a"]          # [T_a, act_dim]
            T = a.shape[0]       # number of transitions
            s = s[:T]            # align states with actions

            if T < horizon:
                continue  # skip very short trajectories, or pad if you prefer

            for t in range(0, T - horizon + 1, stride):
                s_win = s[t:t + horizon]       # [H, obs_dim]
                a_win = a[t:t + horizon]       # [H, act_dim]
                self.windows.append((s_win, a_win))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s_win, a_win = self.windows[idx]
        return {
            "s": torch.tensor(s_win, dtype=torch.float32),   # [H, obs_dim]
            "a": torch.tensor(a_win, dtype=torch.float32),   # [H, act_dim]
        }

