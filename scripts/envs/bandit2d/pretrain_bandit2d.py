"""
Pretrain script for 2D Bandit environment.
"""
import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fedguide.guidance.diffusion_prior import SimpleDiffusionPrior
from generate_bandit2d_data import generate_bandit2d_datasets


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
    
    # Guidance args
    parser.add_argument('--guidance_mode', type=str, default="off", 
                       choices=["off", "warmup", "interleave"])
    parser.add_argument('--guidance_warmup_epochs', type=int, default=0)
    parser.add_argument('--guidance_scale_init', type=float, default=0.0)
    parser.add_argument('--guidance_scale_target', type=float, default=1.0)
    parser.add_argument('--guidance_scale_warmup_epochs', type=int, default=0)
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
    from fedguide.envs.bandit2d import Bandit2D
    env = Bandit2D(K=args.K, sigma=args.sigma, seed=args.seed)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    
    # Pretrain each client
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    for client_id in range(args.num_clients):
        print(f"\n[Pretrain] Client {client_id}")
        
        # Create SimpleDiffusionPrior
        prior = SimpleDiffusionPrior(
            state_dim=obs_dim,
            action_dim=act_dim,
            hidden_dim=args.dg_hidden_dim,
            timesteps=args.num_train_timesteps
        ).to(device)
        
        # Create dataloader
        loader = DataLoader(datasets[client_id], batch_size=args.batch_size, shuffle=True, drop_last=True)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Save directory
        save_dir = os.path.join("./model/models_prior", args.env, f"client_{client_id}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        print(f"[Client {client_id}] Start prior pretrain...")
        for epoch in range(1, args.n_behavior_epochs + 1):
            prior.train()
            total_loss = 0.0
            n_batches = 0
            
            for batch in loader:
                # Handle batch format
                if isinstance(batch, dict):
                    a = batch.get("a", batch.get("actions")).to(device)
                else:
                    x = batch.to(device)
                    # For bandit2d, obs and action are the same, use only action part
                    a = x[:, obs_dim:obs_dim + act_dim]
                
                if a.dim() == 1:
                    a = a.unsqueeze(-1)
                s = torch.zeros_like(a)
                
                optimizer.zero_grad()
                
                # Use log_prob and negative log-likelihood loss (like debug_pretrain.py)
                lp = prior.log_prob(a, s)
                loss = -lp.mean()
                
                # Backward
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(1, n_batches)
            
            if epoch % 10 == 0:
                print(f"[Client {client_id}] epoch {epoch}/{args.n_behavior_epochs} | prior_loss={avg_loss:.6f}")
            
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
                prior.train()
        
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
        print(f"[Client {client_id}] Saved final model to {final_dir}")
    
    print("\n[Pretrain] All clients finished.")


if __name__ == "__main__":
    main()

