from diffusers import DDPMScheduler, UNet1DModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianBehaviorPrior(nn.Module):
    """Closed-form 2D Gaussian behavior prior with the same interface as
    SimpleDiffusionPrior. Designed for low-dim toy environments (Bandit2D)
    where SimpleDiffusionPrior's autoencoder-style log_prob proxy fails to
    rank action density correctly (it ranked the origin above the training
    peak, which destroyed bonus-PG-driven FedGuide variants).

    State_dict keys are deliberately picked to start with "head_" so they
    are always selected by FedguideAgent._init_prior_adapt_params() (the
    "head" tag triggers prior_adapt aggregation), keeping OT-MoE on the
    Gaussian parameters as if they were a small adapter head.

    Parameters
    ----------
    state_dim, action_dim : int
        Dimensionality of state/action vectors. Total density is over the
        action ``a`` only — states are accepted to match the interface but
        the marginal Gaussian on ``a`` does not depend on ``s`` here. This
        matches Bandit2D where the optimal action does not depend on state.
    hidden_dim, timesteps : int
        Accepted for ctor-signature compatibility with SimpleDiffusionPrior;
        unused by the Gaussian density.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, timesteps=1000):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        # parameters use the "head" tag so prior_adapt aggregation picks them up.
        self.head_mu = nn.Parameter(torch.zeros(self.action_dim))
        self.head_log_sigma = nn.Parameter(torch.zeros(self.action_dim))  # σ=1 init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _to_dev(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device).float()

    def log_prob(self, actions, states=None):
        if actions is None:
            B = 0 if states is None else states.shape[0]
            return torch.zeros(B, device=self.device)
        a = self._to_dev(actions)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        mu = self.head_mu
        log_sigma = self.head_log_sigma.clamp(-5.0, 2.0)
        sigma = log_sigma.exp()
        # Diagonal Gaussian log pdf, summed over action dims.
        var = sigma.pow(2) + 1e-8
        logp = -0.5 * (((a - mu) ** 2) / var + 2 * log_sigma + np.log(2 * np.pi))
        return logp.sum(dim=-1)

    @torch.no_grad()
    def sample(self, states=None, batch_size=1, num_steps=100, noise_scale=0.1):
        if states is not None:
            if states.dim() == 1:
                states = states.unsqueeze(0)
            B = int(states.shape[0])
        else:
            B = int(batch_size)
        sigma = self.head_log_sigma.clamp(-5.0, 2.0).exp()
        eps = torch.randn(B, self.action_dim, device=self.device)
        return self.head_mu + sigma * eps

    def update(self, states, actions, lr=1e-4):
        """Closed-form MLE refit on the supplied minibatch (no NN training)."""
        a = self._to_dev(actions)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        mu = a.mean(dim=0)
        sigma = a.std(dim=0).clamp(min=1e-3)
        with torch.no_grad():
            self.head_mu.data = mu.detach()
            self.head_log_sigma.data = sigma.log().detach()
        # Return -log-likelihood for parity with SimpleDiffusionPrior.update.
        return float(-self.log_prob(a).mean().item())

    def fit(self, actions: torch.Tensor):
        """Fit μ, log σ to the entire offline action set in closed form."""
        a = self._to_dev(actions)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        mu = a.mean(dim=0)
        sigma = a.std(dim=0).clamp(min=1e-3)
        with torch.no_grad():
            self.head_mu.data = mu.detach()
            self.head_log_sigma.data = sigma.log().detach()


class GaussianMixtureBehaviorPrior(nn.Module):
    """A density-space mixture of Bandit2D Gaussian behavior priors.

    OT routing supplies one weight per expert.  Keeping the experts as mixture
    components is essential here: averaging the component means would collapse
    four symmetric modes to a single Gaussian at the origin.
    """

    def __init__(self, component_mu, component_log_sigma, weights):
        super().__init__()
        mu = torch.as_tensor(component_mu, dtype=torch.float32)
        log_sigma = torch.as_tensor(component_log_sigma, dtype=torch.float32)
        mix_weights = torch.as_tensor(weights, dtype=torch.float32).flatten()
        if mu.ndim != 2 or log_sigma.shape != mu.shape:
            raise ValueError("Gaussian-mixture components must have shape [M, action_dim]")
        if mix_weights.numel() != mu.shape[0]:
            raise ValueError("Gaussian-mixture weights must have one entry per component")
        mix_weights = mix_weights.clamp_min(0.0)
        mix_weights = mix_weights / mix_weights.sum().clamp_min(1e-12)
        self.register_buffer("component_mu", mu)
        self.register_buffer("component_log_sigma", log_sigma)
        self.register_buffer("log_weights", mix_weights.clamp_min(1e-12).log())

    @property
    def device(self):
        return self.component_mu.device

    def log_prob(self, actions, states=None):
        del states
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if a.ndim == 1:
            a = a.unsqueeze(0)
        log_sigma = self.component_log_sigma.clamp(-5.0, 2.0)
        var = log_sigma.exp().pow(2) + 1e-8
        diff = a.unsqueeze(1) - self.component_mu.unsqueeze(0)
        component_logp = -0.5 * (
            diff.pow(2) / var.unsqueeze(0)
            + 2.0 * log_sigma.unsqueeze(0)
            + np.log(2 * np.pi)
        ).sum(dim=-1)
        return torch.logsumexp(component_logp + self.log_weights.unsqueeze(0), dim=1)

    @torch.no_grad()
    def sample(self, states=None, batch_size=1, num_steps=100, noise_scale=0.1):
        del num_steps, noise_scale
        count = int(states.shape[0]) if states is not None else int(batch_size)
        component = torch.distributions.Categorical(logits=self.log_weights).sample((count,))
        mu = self.component_mu[component]
        sigma = self.component_log_sigma[component].clamp(-5.0, 2.0).exp()
        return mu + sigma * torch.randn_like(mu)


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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
    # sampling
    # ----------------------------------------------------------
    @torch.no_grad()
    def sample(self, states=None, batch_size=1, num_steps=100, noise_scale=0.1):
        """
        Sample actions using diffusion reverse process.
        
        Args:
            states: [B, state_dim] state tensor (optional, if None samples from scratch)
            batch_size: Batch size when states is None
            num_steps: Number of denoising steps
            noise_scale: Noise scale factor (should match training, default 0.1)
        
        Returns:
            actions: [B, action_dim] sampled actions
        """
        self.eval()
        
        if states is None:
            # Sample from scratch: use specified batch_size
            pass
        else:
            if states.dim() == 1:
                states = states.unsqueeze(0)
            batch_size = states.shape[0]
        
        # Initialize trajectory with noise
        traj_dim = self.state_dim + self.action_dim
        if states is None:
            # Sample from scratch: start with pure noise
            x = torch.randn(batch_size, traj_dim, device=self.device)
        else:
            # Start with given states and random actions
            actions_init = torch.randn(batch_size, self.action_dim, device=self.device)
            x = torch.cat([states, actions_init], dim=-1)
        
        # Reverse diffusion: iteratively denoise
        for step in range(num_steps):
            # Compute noise level: start from high noise, decrease to zero
            t_ratio = 1.0 - (step / num_steps)  # Goes from 1.0 to 0.0
            
            # Add noise at current level (simulating forward process)
            noise = torch.randn_like(x)
            noisy = x + noise_scale * t_ratio * noise
            
            # Reconstruct using the model
            h = self.encoder(noisy)
            recon = self.decoder(h)
            
            # Denoise: move towards reconstruction
            # Use larger steps early, smaller steps later for stability
            step_size = 0.2 * (0.5 + t_ratio)  # Starts at 0.3, decreases to 0.1
            x = x - step_size * (x - recon)
            
            # Clamp to reasonable range to prevent divergence
            x = torch.clamp(x, -2.0, 2.0)
        
        # Final refinement: use reconstruction directly
        h = self.encoder(x)
        recon = self.decoder(h)
        x = 0.3 * x + 0.7 * recon
        
        # Extract action part
        actions = x[:, self.state_dim:]
        return actions

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

        # Handle single-step case (bandit2d, D4RL): [B, dim]
        # Format for UNet1D: [B, C, T] where C=traj_dim, T=horizon
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        
        # Concatenate states and actions: [B, state_dim + action_dim]
        traj = torch.cat([states, actions], dim=-1).to(self.device)  # [B, traj_dim]
        
        # Expand to horizon dimension for UNet1D: [B, traj_dim, horizon]
        traj = traj.unsqueeze(-1).repeat(1, 1, self.horizon)  # [B, traj_dim, horizon]
        
        # Generate noise and timesteps
        noise = torch.randn_like(traj)  # [B, traj_dim, horizon]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (traj.shape[0],), device=self.device
        ).long()

        # Add noise
        noisy = self.noise_scheduler.add_noise(traj, noise, timesteps)
        
        # Predict noise
        pred = self.model(noisy, timesteps).sample  # [B, traj_dim, horizon]
        
        # Compute reconstruction error (only on action part, average over horizon)
        pred_noise_on_a = pred[:, -self.action_dim:, :].mean(dim=-1)  # [B, action_dim]
        noise_on_a = noise[:, -self.action_dim:, :].mean(dim=-1)  # [B, action_dim]
        recon_error = (pred_noise_on_a - noise_on_a).pow(2).mean(dim=-1)  # [B]
        
        # convert to pseudo log-prob (larger = better alignment)
        return -recon_error

    def update(self, states, actions, lr=1e-4):
        self.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Handle single-step case (bandit2d, D4RL): [B, dim]
        # Format for UNet1D: [B, C, T] where C=traj_dim, T=horizon
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        
        # Concatenate states and actions: [B, state_dim + action_dim]
        traj = torch.cat([states, actions], dim=-1).to(self.device)  # [B, traj_dim]
        
        # Expand to horizon dimension for UNet1D: [B, traj_dim, horizon]
        traj = traj.unsqueeze(-1).repeat(1, 1, self.horizon)  # [B, traj_dim, horizon]
        
        noise = torch.randn_like(traj)  # [B, traj_dim, horizon]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (traj.shape[0],), device=self.device
        ).long()

        noisy = self.noise_scheduler.add_noise(traj, noise, timesteps)
        pred = self.model(noisy, timesteps).sample  # [B, traj_dim, horizon]
        
        # Compute loss (only on action part, average over horizon, like in pretrain)
        pred_noise_on_a = pred[:, -self.action_dim:, :].mean(dim=-1)  # [B, action_dim]
        noise_on_a = noise[:, -self.action_dim:, :].mean(dim=-1)  # [B, action_dim]
        loss = (pred_noise_on_a - noise_on_a).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())
