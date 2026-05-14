"""
Pretrain script for 2D Bandit environment.
Trains SimpleDiffusionPrior (prior) and optionally SDICE_Critic (guidance).
"""
import argparse
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Add project root and scripts/envs/bandit2d to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
sys.path.insert(0, _project_root)
sys.path.insert(0, _script_dir)

from fedguide.guidance.diffusion_prior import SimpleDiffusionPrior, GaussianBehaviorPrior
from fedguide.guidance.model import SDICE_Critic
from fedguide.envs.bandit2d import Bandit2D


class _TrajDataset(Dataset):
    """Minimal trajectory dataset (obs, actions)."""
    def __init__(self, obs, acts):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.acts = np.asarray(acts, dtype=np.float32)
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, i):
        return np.concatenate([self.obs[i], self.acts[i]], axis=-1).astype(np.float32)


def generate_bandit2d_datasets(K=4, n_clients=4, samples_per_client=1000,
                               sigma=0.2, local_radius=0.3, seed=42, overlap_factor=1.33):
    """Generate Bandit2D client datasets (inlined to avoid mujoco/reacher imports)."""
    import numpy as _np
    _np.random.seed(seed)
    angles = _np.linspace(0, 2 * _np.pi, K, endpoint=False)
    mu = _np.array([[_np.cos(a), _np.sin(a)] for a in angles])
    r_min, r_max = 1.0 - local_radius, 1.0 + local_radius
    datasets = []
    for client_id in range(n_clients):
        angle_center = angles[client_id % K]
        angle_span = 2 * _np.pi / K * overlap_factor
        theta_min = angle_center - angle_span / 2.0
        theta_max = angle_center + angle_span / 2.0
        obs_list, act_list = [], []
        for _ in range(samples_per_client):
            u = _np.random.rand()
            r = _np.sqrt((r_max**2 - r_min**2) * u + r_min**2)
            theta = _np.random.uniform(theta_min, theta_max)
            x, y = r * _np.cos(theta), r * _np.sin(theta)
            action = _np.clip(_np.array([x, y], dtype=_np.float32), -1.5, 1.5)
            obs_list.append(action)
            act_list.append(action)
        datasets.append(_TrajDataset(obs_list, act_list))
        print(f"Client {client_id}: {len(datasets[-1])} samples")
    return datasets, mu


class Bandit2DTransitionDataset(Dataset):
    """Dataset yielding (s, a, r, s_next, d) for SDICE training. Bandit: s=a, r=reward(a), d=1."""
    def __init__(self, traj_dataset, env: Bandit2D):
        self.obs = traj_dataset.obs
        self.acts = traj_dataset.acts
        self.env = env

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        s = self.obs[i]
        a = self.acts[i]
        r = self.env.compute_reward(a)
        s_next = a.copy()  # terminal: same for bandit
        d = 1.0
        return {
            "s": torch.tensor(s, dtype=torch.float32),
            "a": torch.tensor(a, dtype=torch.float32),
            "r": torch.tensor(r, dtype=torch.float32),
            "s_": torch.tensor(s_next, dtype=torch.float32),
            "d": torch.tensor(d, dtype=torch.float32),
        }


def _train_guidance_one_epoch(sdice, loader, device, do_update_v0=True, do_update_wt=True):
    """Train SDICE guidance for one epoch."""
    for batch in loader:
        s = batch["s"].to(device)
        a = batch["a"].to(device)
        r = batch["r"].to(device)
        s_ = batch["s_"].to(device)
        d = batch["d"].to(device)
        if do_update_v0 and hasattr(sdice, "update_v0"):
            sdice.update_v0({"s": s, "a": a, "r": r, "s_": s_, "d": d})
        if do_update_wt and hasattr(sdice, "update_wt"):
            sdice.update_wt({"s": s, "a": a})


def main():
    parser = argparse.ArgumentParser()
    
    # Dataset args
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--samples_per_client", type=int, default=1000)
    parser.add_argument("--K", type=int, default=4, help="Number of peaks")
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--local_radius", type=float, default=0.3)
    parser.add_argument("--overlap_factor", type=float, default=1.33,
                        help="Overlap factor for sectors (1.33 = 30%% overlap, 1.5 = 50%% overlap)")
    parser.add_argument("--seed", type=int, default=42)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    # Prior training args
    parser.add_argument('--num_train_timesteps', type=int, default=1000)
    parser.add_argument('--dg_hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_behavior_epochs', type=int, default=1500)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Guidance args (SDICE_Critic)
    parser.add_argument('--guidance_mode', type=str, default="interleave",
                       choices=["off", "warmup", "interleave"],
                       help="off=prior only; interleave=train prior+guidance; warmup=guidance after prior")
    parser.add_argument('--prior_type', type=str, default="simple",
                       choices=["simple", "gaussian"],
                       help="simple=SimpleDiffusionPrior (autoencoder, legacy/buggy density); gaussian=GaussianBehaviorPrior (closed-form 2D Gaussian fit)")
    parser.add_argument('--save_root', type=str, default="./model/models_prior",
                       help="Root dir for saved prior/guidance ckpts (so gaussian variant doesn't overwrite simple).")
    parser.add_argument('--guidance_warmup_epochs', type=int, default=100)
    parser.add_argument('--guidance_scale_init', type=float, default=0.0)
    parser.add_argument('--guidance_scale_target', type=float, default=1.0)
    parser.add_argument('--guidance_scale_warmup_epochs', type=int, default=50)
    parser.add_argument('--guidance_interval', type=int, default=5)
    parser.add_argument('--guidance_epochs_per_call', type=int, default=1)
    
    # SDICE
    parser.add_argument('--q_ensemble_num', type=int, default=0)
    parser.add_argument('--value_lr', type=float, default=1e-4)
    parser.add_argument('--wt_lr', type=float, default=1e-4)
    parser.add_argument('--use_lr_schedule', type=int, default=0)
    parser.add_argument('--min_value_lr', type=float, default=1e-5)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=256)
    
    args = parser.parse_args()
    
    # Set env name for pretrain (used for saving models)
    args.env = "Bandit2D"
    
    # Generate datasets
    print("Generating datasets...")
    datasets, mu = generate_bandit2d_datasets(
        K=args.K,
        n_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        sigma=args.sigma,
        local_radius=args.local_radius,
        seed=args.seed,
        overlap_factor=args.overlap_factor
    )
    
    print(f"\nStarting pretrain for {len(datasets)} clients...")
    
    # Get dimensions from environment
    bandit_env = Bandit2D(K=args.K, sigma=args.sigma, seed=args.seed)
    obs_dim = int(bandit_env.observation_space.shape[0])
    act_dim = int(bandit_env.action_space.shape[0])
    
    # Disable wandb for pretrain (SDICE uses it optionally)
    os.environ.setdefault("WANDB_MODE", "disabled")
    
    # Pretrain each client
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    for client_id in range(args.num_clients):
        print(f"\n[Pretrain] Client {client_id} | guidance_mode={args.guidance_mode} | prior_type={args.prior_type}")

        # Closed-form Gaussian prior path: fit μ, σ on the client's offline
        # actions, save, optionally still train the SDICE_Critic guidance.
        if args.prior_type == "gaussian":
            prior = GaussianBehaviorPrior(
                state_dim=obs_dim, action_dim=act_dim,
            ).to(device)
            actions_all = torch.tensor(datasets[client_id].acts, dtype=torch.float32, device=device)
            prior.fit(actions_all)
            mu_np = prior.head_mu.detach().cpu().numpy()
            sg_np = prior.head_log_sigma.detach().exp().cpu().numpy()
            print(f"  [Gaussian fit] μ={mu_np.tolist()}  σ={sg_np.tolist()}")

            sdice = None
            if args.guidance_mode in ("warmup", "interleave"):
                class _C: pass
                c = _C()
                c.device = device
                c.q_ensemble_num = args.q_ensemble_num
                c.value_lr = args.value_lr
                c.wt_lr = args.wt_lr
                c.weight_decay = args.weight_decay
                c.use_lr_schedule = 0
                c.train_epoch = 1
                c.min_value_lr = args.min_value_lr
                c.M = args.M
                c.alpha = args.alpha
                c.hidden_dim = args.hidden_dim
                sdice = SDICE_Critic(adim=act_dim, sdim=obs_dim, args=c).to(device)
                sdice.guidance_scale = args.guidance_scale_target
                guidance_ds = Bandit2DTransitionDataset(datasets[client_id], bandit_env)
                guidance_loader = DataLoader(guidance_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
                # Train SDICE for guidance_warmup_epochs over the offline buffer.
                n_warm = max(args.guidance_warmup_epochs, 100)
                for e in range(1, n_warm + 1):
                    _train_guidance_one_epoch(sdice, guidance_loader, device, do_update_v0=True, do_update_wt=True)
                print(f"  [SDICE warmup] {n_warm} epochs done")

            save_dir = os.path.join(args.save_root, args.env, f"client_{client_id}")
            final_dir = os.path.join(save_dir, "final")
            os.makedirs(final_dir, exist_ok=True)
            torch.save({
                "prior": prior.state_dict(),
                "state_dim": obs_dim,
                "action_dim": act_dim,
                "prior_type": "gaussian",
            }, os.path.join(final_dir, "torch_prior.pth"))
            if sdice is not None:
                torch.save(sdice.state_dict(), os.path.join(final_dir, "guidance_sdice.pth"))
                print(f"[Client {client_id}] Saved Gaussian prior + guidance to {final_dir}")
            else:
                print(f"[Client {client_id}] Saved Gaussian prior (no guidance) to {final_dir}")
            continue

        # Create SimpleDiffusionPrior
        prior = SimpleDiffusionPrior(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=args.dg_hidden_dim,
            timesteps=args.num_train_timesteps
        ).to(device)
        
        # Create dataloader for prior (obs, act concatenated)
        loader = DataLoader(datasets[client_id], batch_size=args.batch_size, shuffle=True, drop_last=True)
        
        # Create SDICE guidance if needed
        sdice = None
        if args.guidance_mode in ("warmup", "interleave"):
            class _C:
                pass
            c = _C()
            c.device = device
            c.q_ensemble_num = args.q_ensemble_num
            c.value_lr = args.value_lr
            c.wt_lr = args.wt_lr
            c.weight_decay = args.weight_decay
            c.use_lr_schedule = 0
            c.train_epoch = 1
            c.min_value_lr = args.min_value_lr
            c.M = args.M
            c.alpha = args.alpha
            c.hidden_dim = args.hidden_dim
            sdice = SDICE_Critic(adim=act_dim, sdim=obs_dim, args=c).to(device)
            sdice.guidance_scale = args.guidance_scale_init
            # Guidance dataset with rewards for SDICE
            guidance_ds = Bandit2DTransitionDataset(datasets[client_id], bandit_env)
            guidance_loader = DataLoader(guidance_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        
        # Create optimizer for prior
        optimizer = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Save directory
        save_dir = os.path.join("./model/models_prior", args.env, f"client_{client_id}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop: prior + optional guidance (interleave)
        for epoch in range(1, args.n_behavior_epochs + 1):
            prior.train()
            total_loss = 0.0
            n_batches = 0
            
            for batch in loader:
                if isinstance(batch, dict):
                    a = batch.get("a", batch.get("actions")).to(device)
                else:
                    x = batch.to(device)
                    a = x[:, obs_dim:obs_dim + act_dim]
                if a.dim() == 1:
                    a = a.unsqueeze(-1)
                s = torch.zeros_like(a)
                optimizer.zero_grad()
                lp = prior.log_prob(a, s)
                loss = -lp.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(1, n_batches)
            
            # Interleave: train guidance every guidance_interval epochs
            if args.guidance_mode == "interleave" and sdice is not None:
                if epoch % args.guidance_interval == 0:
                    scale_done = min(epoch, args.guidance_scale_warmup_epochs) if args.guidance_scale_warmup_epochs > 0 else epoch
                    if args.guidance_scale_warmup_epochs > 0:
                        ratio = min(1.0, scale_done / args.guidance_scale_warmup_epochs)
                        sdice.guidance_scale = args.guidance_scale_init + (args.guidance_scale_target - args.guidance_scale_init) * ratio
                    for _ in range(args.guidance_epochs_per_call):
                        _train_guidance_one_epoch(sdice, guidance_loader, device, do_update_v0=True, do_update_wt=True)
            
            if epoch % 10 == 0:
                msg = f"[Client {client_id}] epoch {epoch}/{args.n_behavior_epochs} | prior_loss={avg_loss:.6f}"
                if sdice is not None:
                    msg += f" | guidance_scale={sdice.guidance_scale:.3f}"
                print(msg)
            
            if (epoch % args.save_interval == 0) or (epoch == args.n_behavior_epochs):
                ckpt_dir = os.path.join(save_dir, f"ckpt_epoch{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                prior.eval()
                torch.save({
                    "prior": prior.state_dict(),
                    "state_dim": obs_dim,
                    "action_dim": act_dim,
                    "hidden_dim": args.dg_hidden_dim,
                    "timesteps": args.num_train_timesteps,
                }, os.path.join(ckpt_dir, "torch_prior.pth"))
                if sdice is not None and args.guidance_mode == "interleave":
                    torch.save(sdice.state_dict(), os.path.join(ckpt_dir, "guidance_sdice.pth"))
                prior.train()
        
        # Guidance warmup (after prior) if mode=warmup
        if args.guidance_mode == "warmup" and sdice is not None and args.guidance_warmup_epochs > 0:
            print(f"[Client {client_id}] Guidance warmup: {args.guidance_warmup_epochs} epochs")
            for e in range(1, args.guidance_warmup_epochs + 1):
                ratio = min(1.0, e / max(1, args.guidance_scale_warmup_epochs))
                sdice.guidance_scale = args.guidance_scale_init + (args.guidance_scale_target - args.guidance_scale_init) * ratio
                _train_guidance_one_epoch(sdice, guidance_loader, device, do_update_v0=True, do_update_wt=True)
            print(f"  warmup done | guidance_scale={sdice.guidance_scale:.3f}")
        
        # Save final model
        final_dir = os.path.join(save_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        prior.eval()
        torch.save({
            "prior": prior.state_dict(),
            "state_dim": obs_dim,
            "action_dim": act_dim,
            "hidden_dim": args.dg_hidden_dim,
            "timesteps": args.num_train_timesteps,
        }, os.path.join(final_dir, "torch_prior.pth"))
        if sdice is not None:
            torch.save(sdice.state_dict(), os.path.join(final_dir, "guidance_sdice.pth"))
            print(f"[Client {client_id}] Saved prior + guidance to {final_dir}")
        else:
            print(f"[Client {client_id}] Saved prior only to {final_dir}")
    
    print("\n[Pretrain] All clients finished.")


if __name__ == "__main__":
    main()

