from diffusers import DDPMScheduler, UNet1DModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleDiffusionPrior(nn.Module):
    """
    Lightweight diffusion-inspired guidance prior for low-dimensional RL trajectories.
    No UNet; uses MLP + Gaussian noise to simulate diffusion reconstruction.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, timesteps=1000):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.timesteps = timesteps

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # ----------------------------------------------------------
    # log_prob proxy
    # ----------------------------------------------------------
    def log_prob(self, actions, states):
        """Compute diffusion-style log-prob approximation."""
        if actions is None or states is None:
            return torch.zeros(states.shape[0], device=self.device)

        # unify shape
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        traj = torch.cat([states, actions], dim=-1).to(self.device)

        # simulate Gaussian diffusion step
        noise = torch.randn_like(traj)
        t = torch.randint(0, self.timesteps, (traj.shape[0],), device=self.device).float().unsqueeze(-1)
        noisy = traj + 0.1 * (t / self.timesteps) * noise

        # reconstruct
        h = self.encoder(noisy)
        recon = self.decoder(h)
        recon_error = (recon - traj).pow(2).mean(dim=-1)
        return -recon_error  # proxy for log-prob

    # ----------------------------------------------------------
    # training
    # ----------------------------------------------------------
    def update(self, states, actions, lr=1e-4):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        traj = torch.cat([states, actions], dim=-1).to(self.device)

        noise = torch.randn_like(traj)
        t = torch.randint(0, self.timesteps, (traj.shape[0],), device=self.device).float().unsqueeze(-1)
        noisy = traj + 0.1 * (t / self.timesteps) * noise

        h = self.encoder(noisy)
        recon = self.decoder(h)
        loss = F.mse_loss(recon, traj)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())


class DiffusionGuidance(nn.Module):
    """
    Diffusion-based trajectory guidance prior.
    Models trajectories (s,a) pairs as continuous 1D time series.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64, timesteps=1000, horizon=32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.traj_dim = state_dim + action_dim
        self.horizon = horizon

        # Two-stage UNet required by diffusers
        self.model = UNet1DModel(
            sample_size=horizon,
            in_channels=self.traj_dim,
            out_channels=self.traj_dim,
            layers_per_block=2,
            block_out_channels=(hidden_dim, hidden_dim, hidden_dim),
            down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D"),
            up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D"),
        )
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _make_traj(self, states, actions):
        """Concatenate (s,a) correctly regardless of action shape."""
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        traj = torch.cat([states, actions], dim=-1)  # [B, s+a]
        traj = traj.unsqueeze(1)  # [B,1,s+a]
        return traj

    @torch.no_grad()
    def log_prob(self, actions, states):
        """
        Compute a 'guidance score' approximating log-prob of (s,a) under diffusion prior.
        Returns a differentiable tensor [B].
        """
        if actions is None or states is None:
            return torch.zeros(states.shape[0], device=self.device)

        traj = self._make_traj(states, actions)
        noise = torch.randn_like(traj)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (traj.shape[0],), device=self.device
        ).long()

        noisy = self.noise_scheduler.add_noise(traj, noise, timesteps)
        pred = self.model(noisy, timesteps).sample
        recon_error = (pred - noise).pow(2).mean(dim=[1, 2])
        # convert to pseudo log-prob (larger = better alignment)
        return -recon_error

    def update(self, states, actions, lr=1e-4):
        self.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        traj = self._make_traj(states, actions)
        noise = torch.randn_like(traj)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (traj.shape[0],), device=self.device
        ).long()

        noisy = self.noise_scheduler.add_noise(traj, noise, timesteps)
        pred = self.model(noisy, timesteps).sample
        loss = (pred - noise).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())