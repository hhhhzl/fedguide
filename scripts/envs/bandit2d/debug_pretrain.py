import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from fedguide.envs.bandit2d import Bandit2D
from fedguide.guidance.diffusion_prior import SimpleDiffusionPrior


def sample_mixture(env, batch_size=1024, local_radius=0.3, n_clients=None):
    """
    Sample data from sector-shaped regions (like generate_bandit2d_datasets).
    Each sample is randomly assigned to one of K peaks, then sampled from
    the corresponding sector region.
    """
    peaks = env.get_peak_locations()
    K = peaks.shape[0]
    
    # Calculate angles for peaks
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    
    # Determine angle span (use n_clients if provided, otherwise use K)
    if n_clients is None:
        n_clients = K
    angle_span = 2 * np.pi / n_clients
    
    # Radius range
    r_min = 1.0 - local_radius
    r_max = 1.0 + local_radius
    
    actions = []
    for _ in range(batch_size):
        # Randomly select a peak
        peak_idx = np.random.randint(0, K)
        angle_center = angles[peak_idx]
        
        # Define sector region around this peak
        theta_min = angle_center - angle_span / 2.0
        theta_max = angle_center + angle_span / 2.0
        
        # Sample from sector: uniform in radius and angle
        u = np.random.rand()
        r = np.sqrt((r_max ** 2 - r_min ** 2) * u + r_min ** 2)
        theta = np.random.uniform(theta_min, theta_max)
        
        # Convert to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        action = np.array([x, y])
        action = np.clip(action, -1.5, 1.5)
        actions.append(action)
    
    a = np.array(actions, dtype=np.float32)
    a = torch.tensor(a, dtype=torch.float32)
    s = torch.zeros_like(a)
    return s, a


def train_simple_prior(
    steps=20000,
    batch_size=512,
    lr=1e-4,
    hidden_dim=256,
    timesteps=1000,
    device="cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    env = Bandit2D(K=4, sigma=0.2)
    prior = SimpleDiffusionPrior(
        state_dim=2,
        action_dim=2,
        hidden_dim=hidden_dim,
        timesteps=timesteps,
    ).to(device)

    opt = torch.optim.Adam(prior.parameters(), lr=lr)

    for step in range(steps):
        s, a = sample_mixture(env, batch_size=batch_size, local_radius=0.3)
        s = s.to(device)
        a = a.to(device)

        lp = prior.log_prob(a, s)
        loss = -lp.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 1000 == 0:
            with torch.no_grad():
                s_data, a_data = sample_mixture(env, batch_size=4096, local_radius=0.3)
                s_rand = torch.zeros_like(s_data)
                a_rand = torch.empty_like(a_data).uniform_(-1.5, 1.5)

                lp_data = prior.log_prob(a_data.to(device), s_data.to(device))
                lp_rand = prior.log_prob(a_rand.to(device), s_rand.to(device))

                print(
                    f"step {step+1:5d}  "
                    f"loss={loss.item():.4f}  "
                    f"E[logπ(data)]={lp_data.mean().item():.4f}  "
                    f"E[logπ(rand)]={lp_rand.mean().item():.4f}  "
                    f"Δ={lp_data.mean().item()-lp_rand.mean().item():.4f}"
                )

    ckpt_path = Path("simple_prior_bandit2d.pth")
    torch.save(prior.state_dict(), ckpt_path)
    print("saved to", ckpt_path)
    return prior, env


def sample_from_prior(
    prior,
    n_samples=8000,
    num_steps=50,
    device="cuda",
):
    """Sample actions from prior using diffusion reverse process."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    prior = prior.to(device)
    prior.eval()

    print(f"Sampling {n_samples} actions using diffusion reverse process ({num_steps} steps)...")
    
    # For bandit2d, state = action, so we sample the full trajectory from scratch
    batch_size = 256
    all_actions = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - i)
            
            # Sample batch of actions from scratch
            batch_actions = prior.sample(None, batch_size=current_batch, num_steps=num_steps, noise_scale=0.1)
            all_actions.append(batch_actions.cpu().numpy())
    
    actions = np.concatenate(all_actions, axis=0)
    actions = np.clip(actions, -1.5, 1.5)
    
    print(f"Sampled {len(actions)} actions, range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    return actions


def visualize_simple_prior(
    prior,
    env,
    n_data=8000,
    n_samples=8000,
    save_path="simple_prior_bandit2d_vis.png",
    device="cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    prior = prior.to(device)
    prior.eval()

    # Sample offline data
    s_data, a_data = sample_mixture(env, batch_size=n_data, local_radius=0.3)
    a_np = a_data.numpy()

    # Sample from prior using diffusion
    print(f"Sampling {n_samples} actions from prior...")
    prior_samples = sample_from_prior(prior, n_samples=n_samples, num_steps=50, device=device)
    
    peaks = env.get_peak_locations()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot: offline data
    ax0 = axes[0]
    hb = ax0.hexbin(
        a_np[:, 0],
        a_np[:, 1],
        gridsize=60,
        extent=(-1.5, 1.5, -1.5, 1.5),
        cmap="viridis",
    )
    ax0.set_title("Synthetic offline data (mixture)", fontsize=12)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_aspect("equal")
    cb0 = fig.colorbar(hb, ax=ax0)
    cb0.set_label("Density (hexbin count)")
    ax0.scatter(
        peaks[:, 0],
        peaks[:, 1],
        c="red",
        marker="*",
        s=120,
        edgecolors="white",
        linewidths=1,
        zorder=10,
    )

    # Right plot: prior samples
    ax1 = axes[1]
    hb = ax1.hexbin(
        prior_samples[:, 0],
        prior_samples[:, 1],
        gridsize=60,
        extent=(-1.5, 1.5, -1.5, 1.5),
        cmap="viridis",
    )
    ax1.set_title("SimpleDiffusionPrior samples (s = a)", fontsize=12)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    cb1 = fig.colorbar(hb, ax=ax1)
    cb1.set_label("Density (hexbin count)")
    ax1.scatter(
        peaks[:, 0],
        peaks[:, 1],
        c="red",
        marker="*",
        s=120,
        edgecolors="white",
        linewidths=1,
        zorder=10,
    )

    plt.suptitle("Bandit2D: Mixture data vs SimpleDiffusionPrior", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print("figure saved to", save_path)
    plt.show()


if __name__ == "__main__":
    prior, env = train_simple_prior()
    visualize_simple_prior(prior, env)