"""
Aggregate and visualize multiple client prior models on Bandit2D,
using sampling + histogram instead of log_prob heatmaps.

- Load client priors
- FedAvg aggregate to get global prior
- Sample actions from aggregated prior via DDPM reverse process
- Sample synthetic offline data (mixture of Gaussians around peaks)
- Plot data vs prior samples as 2D hexbin density maps
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler

from fedguide.guidance.diffusion_prior import DiffusionGuidance, SimpleDiffusionPrior
from fedguide.envs.bandit2d import Bandit2D
from fedguide.fed.fedguide.aggregator import ot_moe_aggregate


def load_prior_model(
        ckpt_path,
        state_dim=2,
        action_dim=2,
        hidden_dim=256,
        timesteps=1000,
        horizon=64,
        device="cuda",
):
    """
    Load prior model from checkpoint, supporting both SimpleDiffusionPrior and DiffusionGuidance.
    Automatically detects the model type from checkpoint format.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    torch_path = os.path.join(ckpt_path, "torch_prior.pth")
    if not os.path.isfile(torch_path):
        if os.path.isfile(ckpt_path):
            torch_path = ckpt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} or {torch_path}")

    sd = torch.load(torch_path, map_location="cpu")

    # Try to detect model type from checkpoint
    is_simple_prior = False
    if isinstance(sd, dict):
        # SimpleDiffusionPrior format: has "prior" key or "state_dim" but no "unet"
        if "prior" in sd or ("state_dim" in sd and "unet" not in sd):
            is_simple_prior = True
        # Also check if it's a direct state dict with encoder/decoder keys
        elif not ("unet" in sd or "scheduler_config" in sd):
            # Check if keys suggest SimpleDiffusionPrior structure
            if any("encoder" in k or "decoder" in k for k in sd.keys()):
                is_simple_prior = True

    if is_simple_prior:
        # Load SimpleDiffusionPrior
        if isinstance(sd, dict) and "prior" in sd:
            state_dim = sd.get("state_dim", state_dim)
            action_dim = sd.get("action_dim", action_dim)
            hidden_dim = sd.get("hidden_dim", hidden_dim)
            timesteps = sd.get("timesteps", timesteps)
            prior = SimpleDiffusionPrior(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                timesteps=timesteps
            ).to(device)
            prior.load_state_dict(sd["prior"], strict=False)
        elif isinstance(sd, dict) and "state_dim" in sd:
            state_dim = sd.get("state_dim", state_dim)
            action_dim = sd.get("action_dim", action_dim)
            hidden_dim = sd.get("hidden_dim", hidden_dim)
            timesteps = sd.get("timesteps", timesteps)
            prior = SimpleDiffusionPrior(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                timesteps=timesteps
            ).to(device)
            # Try loading as full state dict
            prior.load_state_dict(sd, strict=False)
        else:
            # Direct state dict
            prior = SimpleDiffusionPrior(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                timesteps=timesteps
            ).to(device)
            prior.load_state_dict(sd, strict=False)
        print(f"Loaded SimpleDiffusionPrior from {torch_path}")
    else:
        # Load DiffusionGuidance
        prior = DiffusionGuidance(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            timesteps=timesteps,
            horizon=horizon,
        ).to(device)

        # Try diffusers format first
        if os.path.isdir(ckpt_path):
            try:
                prior.model = prior.model.from_pretrained(ckpt_path)
                prior.noise_scheduler = prior.noise_scheduler.from_pretrained(ckpt_path)
                print(f"Loaded DiffusionGuidance from diffusers format: {ckpt_path}")
                prior.eval()
                return prior
            except Exception as e:
                print(f"Failed to load from diffusers format: {e}")

        # Load from torch checkpoint
        if isinstance(sd, dict):
            if "unet" in sd:
                prior.model.load_state_dict(sd["unet"], strict=False)
                if "scheduler_config" in sd:
                    prior.noise_scheduler.config = sd["scheduler_config"]
            else:
                prior.load_state_dict(sd, strict=False)
        else:
            prior.model.load_state_dict(sd, strict=False)
        print(f"Loaded DiffusionGuidance from {torch_path}")

    prior.eval()
    return prior


def _l2_cost_fn(a: list, b: list) -> float:
    """OT cost: sum of L2 distance between parameter lists."""
    return float(sum(np.sum((aa - bb) ** 2) for aa, bb in zip(a, b)))


def aggregate_priors(priors, method="fedavg", num_experts=1):
    """
    Aggregate multiple prior models using FedAvg or OT-MoE.
    Supports both SimpleDiffusionPrior and DiffusionGuidance.
    Creates a new prior instance instead of reusing priors[0].
    
    Args:
        priors: List of prior models to aggregate
        method: "fedavg" or "ot_moe"
        num_experts: Number of experts for OT-MoE (default: 1)
    """
    if len(priors) == 0:
        raise ValueError("No priors to aggregate")

    if len(priors) == 1:
        # Create a copy instead of returning the original
        prior = priors[0]
        if isinstance(prior, SimpleDiffusionPrior):
            # Get hidden_dim from encoder's first Linear layer
            hidden_dim = prior.encoder[0].out_features
            aggregated = SimpleDiffusionPrior(
                state_dim=prior.state_dim,
                action_dim=prior.action_dim,
                hidden_dim=hidden_dim,
                timesteps=prior.timesteps
            ).to(prior.device)
        else:
            hidden_dim = prior.model.config.block_out_channels[0] if hasattr(prior.model, 'config') else 64
            aggregated = DiffusionGuidance(
                state_dim=prior.state_dim,
                action_dim=prior.action_dim,
                hidden_dim=hidden_dim,
                timesteps=prior.noise_scheduler.config.num_train_timesteps,
                horizon=prior.horizon
            ).to(prior.device)
        aggregated.load_state_dict(prior.state_dict())
        aggregated.eval()
        return aggregated

    # Get device and type from first prior
    device = priors[0].device
    is_simple_prior = isinstance(priors[0], SimpleDiffusionPrior)

    # Get dimensions from first prior
    if is_simple_prior:
        state_dim = priors[0].state_dim
        action_dim = priors[0].action_dim
        # Get hidden_dim from encoder's first Linear layer
        hidden_dim = priors[0].encoder[0].out_features
        timesteps = priors[0].timesteps
    else:
        state_dim = priors[0].state_dim
        action_dim = priors[0].action_dim
        hidden_dim = priors[0].model.config.block_out_channels[0] if hasattr(priors[0].model, 'config') else 64
        timesteps = priors[0].noise_scheduler.config.num_train_timesteps
        horizon = priors[0].horizon

    # Collect all parameters
    all_params = []
    for prior in priors:
        # Ensure all priors are of the same type
        if isinstance(prior, SimpleDiffusionPrior) != is_simple_prior:
            raise ValueError("Cannot aggregate different prior types (SimpleDiffusionPrior vs DiffusionGuidance)")

        if is_simple_prior:
            params = list(prior.parameters())
        else:
            params = list(prior.model.parameters())
        all_params.append([p.detach().cpu().clone().numpy() for p in params])

    # Aggregate parameters based on method
    num_layers = len(all_params[0])
    
    if method == "ot_moe":
        # OT-MoE aggregation
        # Convert to numpy arrays for OT-MoE
        client_params_np = all_params  # Already numpy arrays
        
        # For single expert case, we need to ensure all clients contribute
        # Hungarian algorithm with 1 expert may only match 1 client, so we use uniform weights
        # This ensures global sampling capability (can sample from all sectors)
        if num_experts == 1:
            # Use uniform weights for all clients to ensure global distribution
            # This is similar to FedAvg but goes through OT-MoE framework
            N = len(client_params_np)
            uniform_weights = np.ones(N) / N  # Equal weights for all clients
            
            # Aggregate with uniform combination of all clients
            aggregated_params = []
            for layer_idx in range(num_layers):
                weighted_sum = np.zeros_like(client_params_np[0][layer_idx])
                for i in range(N):
                    weighted_sum += uniform_weights[i] * client_params_np[i][layer_idx]
                aggregated_params.append(torch.from_numpy(weighted_sum))
        else:
            # Multiple experts: use standard OT-MoE matching
            expert_params = []
            for _ in range(num_experts):
                expert = []
                for layer_idx in range(num_layers):
                    layer_params = [client_params_np[i][layer_idx] for i in range(len(priors))]
                    stacked = np.stack(layer_params, axis=0)
                    expert.append(stacked.mean(axis=0))
                expert_params.append(expert)
            
            # Run OT-MoE aggregation
            aggregated_params_np = ot_moe_aggregate(
                client_params=client_params_np,
                expert_params=expert_params,
                cost_fn=_l2_cost_fn
            )
            
            # For multiple experts, average all experts to get global prior
            # This ensures we can sample from all sectors
            num_experts_result = len(aggregated_params_np)
            if num_experts_result > 1:
                # Average all experts
                averaged_expert = []
                for layer_idx in range(num_layers):
                    expert_layers = [expert[layer_idx] for expert in aggregated_params_np]
                    stacked = np.stack(expert_layers, axis=0)
                    averaged_expert.append(stacked.mean(axis=0))
                aggregated_params = [torch.from_numpy(p) for p in averaged_expert]
            else:
                aggregated_params = [torch.from_numpy(p) for p in aggregated_params_np[0]]
    else:
        # FedAvg aggregation
        averaged_params = []
        for layer_idx in range(num_layers):
            # all_params is already numpy arrays, convert to torch for stacking
            layer_params = [torch.from_numpy(client_params[layer_idx]) for client_params in all_params]
            stacked = torch.stack(layer_params, dim=0)
            averaged = stacked.mean(dim=0)
            averaged_params.append(averaged)
        aggregated_params = averaged_params

    # Create new aggregated prior instance
    if is_simple_prior:
        aggregated = SimpleDiffusionPrior(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            timesteps=timesteps
        ).to(device)
    else:
        aggregated = DiffusionGuidance(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            timesteps=timesteps,
            horizon=horizon
        ).to(device)

    # Load aggregated parameters
    with torch.no_grad():
        if is_simple_prior:
            for param, agg_param in zip(aggregated.parameters(), aggregated_params):
                param.data.copy_(agg_param.to(device))
        else:
            for param, agg_param in zip(aggregated.model.parameters(), aggregated_params):
                param.data.copy_(agg_param.to(device))

    aggregated.eval()
    return aggregated


def sample_prior_actions(
        prior,
        n_samples=8000,
        num_inference_steps=500,
        device="cuda",
        use_mixture_sampling=False,
        client_priors=None,
):
    """
    Sample actions from prior model.
    For DiffusionGuidance: uses DDPM reverse sampling.
    For SimpleDiffusionPrior: uses rejection sampling based on log_prob.
    
    Args:
        use_mixture_sampling: If True and client_priors provided, sample from each client prior and mix
        client_priors: List of client prior models for mixture sampling
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Use mixture sampling if requested and client priors are provided
    if use_mixture_sampling and client_priors is not None and len(client_priors) > 0:
        print(f"Using mixture sampling from {len(client_priors)} client priors...")
        n_samples_per_client = n_samples // len(client_priors)
        remaining = n_samples % len(client_priors)
        
        all_actions = []
        for idx, client_prior in enumerate(client_priors):
            current_n = n_samples_per_client + (1 if idx < remaining else 0)
            client_samples = sample_prior_actions(
                client_prior,
                n_samples=current_n,
                num_inference_steps=num_inference_steps,
                device=device,
                use_mixture_sampling=False,
                client_priors=None,
            )
            all_actions.append(client_samples)
        
        actions = np.concatenate(all_actions, axis=0)
        # Shuffle to mix the samples
        np.random.shuffle(actions)
        actions = np.clip(actions, -1.5, 1.5)
        print(f"Mixture sampled {len(actions)} actions, range: [{actions.min():.3f}, {actions.max():.3f}]")
        return actions
    
    prior = prior.to(device)
    prior.eval()

    if isinstance(prior, SimpleDiffusionPrior):
        # For SimpleDiffusionPrior, use sample() method (like debug_pretrain.py)
        # Sample from scratch using None (not s=0, which is only for log_prob evaluation)
        print(f"Sampling {n_samples} actions using diffusion reverse process ({num_inference_steps} steps)...")
        
        # For bandit2d, state = action, so we sample the full trajectory from scratch
        batch_size = 256
        all_actions = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                current_batch = min(batch_size, n_samples - i)
                
                # Sample batch of actions from scratch (like debug_pretrain.py)
                batch_actions = prior.sample(None, batch_size=current_batch, num_steps=num_inference_steps, noise_scale=0.1)
                all_actions.append(batch_actions.cpu().numpy())
        
        actions = np.concatenate(all_actions, axis=0)
        actions = np.clip(actions, -1.5, 1.5)
        
        print(f"Sampled {len(actions)} actions, range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        return actions

    else:
        # DiffusionGuidance: use DDPM reverse sampling
        prior.model.eval()

        traj_dim = prior.traj_dim
        horizon = prior.horizon

        scheduler = DDPMScheduler.from_config(prior.noise_scheduler.config)
        scheduler.set_timesteps(num_inference_steps, device=device)

        x = torch.randn((n_samples, traj_dim, horizon), device=device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                out = prior.model(x, t)
                eps = out.sample if hasattr(out, "sample") else out
                step = scheduler.step(eps, t, x)
                x = step.prev_sample

        traj0 = x.mean(dim=-1)  # [N, traj_dim]
        actions = traj0[:, -prior.action_dim:].cpu().numpy()
        actions = np.clip(actions, -1.5, 1.5)
        return actions


def sample_synthetic_bandit2d_data(env, n_samples=5000, local_radius=0.3, n_clients=None, overlap_factor=1.33):
    """
    Sample data from sector-shaped regions (like generate_bandit2d_datasets and debug_pretrain.py).
    Each sample is randomly assigned to one of K peaks, then sampled from
    the corresponding sector region.
    
    Args:
        overlap_factor: Factor to create overlap between adjacent sectors (default 1.33 = 30% overlap)
                        For 50% overlap, use 1.5
    """
    peaks = env.get_peak_locations()
    K = peaks.shape[0]
    
    # Calculate angles for peaks
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    
    # Determine angle span with overlap
    # Use overlap_factor to create overlapping sectors
    # angle_span = 2π / K * overlap_factor (e.g., 1.33 for 30% overlap, 1.5 for 50% overlap)
    if n_clients is None:
        n_clients = K
    angle_span = 2 * np.pi / K * overlap_factor
    
    # Radius range
    r_min = 1.0 - local_radius
    r_max = 1.0 + local_radius
    
    data_actions = []
    for _ in range(n_samples):
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
        data_actions.append(action)
    
    return np.stack(data_actions, axis=0)


def visualize_data_vs_prior(
        data_actions,
        prior_actions,
        env,
        output_path=None,
):
    # Calculate vmin and vmax from offline data to unify scale
    fig_temp = plt.figure()
    ax_temp = fig_temp.add_subplot(111)
    hb_temp = ax_temp.hexbin(
        data_actions[:, 0],
        data_actions[:, 1],
        gridsize=60,
        extent=(-1.5, 1.5, -1.5, 1.5),
    )
    counts = hb_temp.get_array()
    if counts is not None:
        valid_counts = counts[~np.isnan(counts)]
        vmin = 0
        vmax = max(valid_counts) if len(valid_counts) > 0 else 1
    else:
        vmin = 0
        vmax = 1
    plt.close(fig_temp)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    titles = ["Synthetic offline data (Sector-shaped)", "Samples from aggregated prior"]
    actions_list = [data_actions, prior_actions]

    for ax, pts, title in zip(axes, actions_list, titles):
        hb = ax.hexbin(
            pts[:, 0],
            pts[:, 1],
            gridsize=60,
            extent=(-1.5, 1.5, -1.5, 1.5),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,  # Use unified scale based on offline data
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        ax.set_aspect("equal")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Density (hexbin count)")

        peaks = env.get_peak_locations()
        ax.scatter(
            peaks[:, 0],
            peaks[:, 1],
            c="red",
            marker="*",
            s=150,
            edgecolors="white",
            linewidths=1,
            zorder=10,
        )

    plt.suptitle("Bandit2D: Offline data vs Aggregated Prior Samples", fontsize=14, y=0.98)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def visualize_data_vs_priors(
        data_actions,
        prior_actions_fedavg,
        prior_actions_otmoe,
        env,
        output_path=None,
):
    """Visualize offline data vs both FedAvg and OT-MoE aggregated priors."""
    # Calculate vmin and vmax from offline data to unify scale
    fig_temp = plt.figure()
    ax_temp = fig_temp.add_subplot(111)
    hb_temp = ax_temp.hexbin(
        data_actions[:, 0],
        data_actions[:, 1],
        gridsize=60,
        extent=(-1.5, 1.5, -1.5, 1.5),
    )
    counts = hb_temp.get_array()
    if counts is not None:
        valid_counts = counts[~np.isnan(counts)]
        vmin = 0
        vmax = max(valid_counts) if len(valid_counts) > 0 else 1
    else:
        vmin = 0
        vmax = 1
    plt.close(fig_temp)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    titles = ["Synthetic offline data (Sector-shaped)", "FedAvg aggregated prior", "OT-MoE aggregated prior"]
    actions_list = [data_actions, prior_actions_fedavg, prior_actions_otmoe]

    for ax, pts, title in zip(axes, actions_list, titles):
        hb = ax.hexbin(
            pts[:, 0],
            pts[:, 1],
            gridsize=60,
            extent=(-1.5, 1.5, -1.5, 1.5),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,  # Use unified scale based on offline data
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        ax.set_aspect("equal")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Density (hexbin count)")

        peaks = env.get_peak_locations()
        ax.scatter(
            peaks[:, 0],
            peaks[:, 1],
            c="red",
            marker="*",
            s=150,
            edgecolors="white",
            linewidths=1,
            zorder=10,
        )

    plt.suptitle("Bandit2D: Offline data vs Aggregated Priors (FedAvg vs OT-MoE)", fontsize=14, y=0.98)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def visualize_multiple_priors(
        data_actions,
        prior_samples_dict,
        env,
        output_path=None,
):
    # Calculate vmin and vmax from offline data to unify scale
    fig_temp = plt.figure()
    ax_temp = fig_temp.add_subplot(111)
    hb_temp = ax_temp.hexbin(
        data_actions[:, 0],
        data_actions[:, 1],
        gridsize=60,
        extent=(-1.5, 1.5, -1.5, 1.5),
    )
    counts = hb_temp.get_array()
    if counts is not None:
        valid_counts = counts[~np.isnan(counts)]
        vmin = 0
        vmax = max(valid_counts) if len(valid_counts) > 0 else 1
    else:
        vmin = 0
        vmax = 1
    plt.close(fig_temp)

    n_priors = len(prior_samples_dict)
    n_cols = min(4, n_priors + 1)  # +1 for data
    n_rows = int(np.ceil((n_priors + 1) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    ax0 = axes[0]
    hb = ax0.hexbin(
        data_actions[:, 0],
        data_actions[:, 1],
        gridsize=60,
        extent=(-1.5, 1.5, -1.5, 1.5),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,  # Use unified scale based on offline data
    )
    ax0.set_title("Synthetic offline data", fontsize=12)
    ax0.set_xlabel("x", fontsize=10)
    ax0.set_ylabel("y", fontsize=10)
    ax0.set_aspect("equal")
    cb = fig.colorbar(hb, ax=ax0)
    cb.set_label("Density (hexbin count)")
    peaks = env.get_peak_locations()
    ax0.scatter(
        peaks[:, 0],
        peaks[:, 1],
        c="red",
        marker="*",
        s=150,
        edgecolors="white",
        linewidths=1,
        zorder=10,
    )

    for idx, (name, pts) in enumerate(prior_samples_dict.items(), start=1):
        ax = axes[idx]
        hb = ax.hexbin(
            pts[:, 0],
            pts[:, 1],
            gridsize=60,
            extent=(-1.5, 1.5, -1.5, 1.5),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,  # Use unified scale based on offline data
        )
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        ax.set_aspect("equal")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Density (hexbin count)")
        ax.scatter(
            peaks[:, 0],
            peaks[:, 1],
            c="red",
            marker="*",
            s=150,
            edgecolors="white",
            linewidths=1,
            zorder=10,
        )

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Bandit2D: Offline data vs Client / Aggregated Priors", fontsize=14, y=0.98)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Multi-prior figure saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize multiple client prior models (sampling-based)")
    parser.add_argument(
        "--base_path",
        type=str,
        default="./model/models_prior/Bandit2D",
    )
    parser.add_argument(
        "--client_ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="final",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8000,
        help="Number of samples for data/prior",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="DDPM reverse steps for sampling",
    )
    parser.add_argument(
        "--local_radius",
        type=float,
        default=0.3,
        help="Local radius for synthetic data around peaks",
    )
    parser.add_argument(
        "--show_clients",
        action="store_true",
        help="Also visualize each client's prior samples",
    )

    args = parser.parse_args()

    print(f"Loading priors from {args.base_path}.")
    priors = []
    loaded_client_ids = []
    for client_id in args.client_ids:
        ckpt_path = os.path.join(args.base_path, f"client_{client_id}", args.ckpt_dir)
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found, skipping client {client_id}")
            continue
        try:
            prior = load_prior_model(
                ckpt_path,
                state_dim=2,
                action_dim=2,
                hidden_dim=args.hidden_dim,
                timesteps=args.timesteps,
                horizon=args.horizon,
                device=args.device,
            )
            priors.append(prior)
            loaded_client_ids.append(client_id)
            print(f"Loaded prior for client {client_id}")
        except Exception as e:
            print(f"Failed to load prior for client {client_id}: {e}")
            continue

    if len(priors) == 0:
        raise ValueError("No priors loaded!")

    print(f"Loaded {len(priors)} prior models")

    env = Bandit2D(K=4, sigma=0.2)

    print(f"Sampling {args.n_samples} synthetic offline data points.")
    # Use number of loaded clients to match training data distribution
    data_actions = sample_synthetic_bandit2d_data(
        env, n_samples=args.n_samples, local_radius=args.local_radius, n_clients=len(loaded_client_ids)
    )

    # Aggregate using both FedAvg and OT-MoE
    print("Aggregating priors using FedAvg.")
    aggregated_prior_fedavg = aggregate_priors(priors, method="fedavg")
    
    print(f"Aggregating priors using OT-MoE with {len(priors)} experts.")
    aggregated_prior_otmoe = aggregate_priors(priors, method="ot_moe", num_experts=len(priors))

    # Sample from both aggregated priors using mixture sampling
    # This ensures global sampling (covering all sectors) instead of collapsing to center
    print(f"Sampling {args.n_samples} actions from FedAvg aggregated prior (using mixture sampling from clients).")
    prior_actions_fedavg = sample_prior_actions(
        aggregated_prior_fedavg,
        n_samples=args.n_samples,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
        use_mixture_sampling=True,  # Use mixture sampling for global distribution
        client_priors=priors,
    )

    print(f"Sampling {args.n_samples} actions from OT-MoE aggregated prior (using mixture sampling from clients).")
    prior_actions_otmoe = sample_prior_actions(
        aggregated_prior_otmoe,
        n_samples=args.n_samples,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
        use_mixture_sampling=True,  # Use mixture sampling for global distribution
        client_priors=priors,
    )

    print("Generating sampling-based visualization (data vs priors).")
    visualize_data_vs_priors(
        data_actions,
        prior_actions_fedavg,
        prior_actions_otmoe,
        env,
        output_path=args.output_path,
    )

    if args.show_clients:
        prior_samples_dict = {}
        prior_samples_dict["FedAvg aggregated"] = prior_actions_fedavg
        prior_samples_dict["OT-MoE aggregated"] = prior_actions_otmoe
        for cid, prior in zip(loaded_client_ids, priors):
            print(f"Sampling {args.n_samples} actions from client {cid} prior.")
            client_samples = sample_prior_actions(
                prior,
                n_samples=args.n_samples,
                num_inference_steps=args.num_inference_steps,
                device=args.device,
            )
            prior_samples_dict[f"Client {cid} prior"] = client_samples

        print("Generating per-client prior visualization.")
        multi_out = None
        if args.output_path is not None:
            root, ext = os.path.splitext(args.output_path)
            multi_out = root + "_clients" + ext

        visualize_multiple_priors(
            data_actions,
            prior_samples_dict,
            env,
            output_path=multi_out,
        )

    print("Done!")


if __name__ == "__main__":
    main()
