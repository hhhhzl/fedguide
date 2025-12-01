"""
Pretrain script for 2D Bandit environment.
"""
import argparse
import sys
import os
import torch

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fedguide.guidance.pretrain import pretrain_one_client
from generate_bandit2d_data import generate_bandit2d_datasets


def main():
    parser = argparse.ArgumentParser()
    
    # Dataset args
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--samples_per_client", type=int, default=1000)
    parser.add_argument("--K", type=int, default=4, help="Number of peaks")
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--local_radius", type=float, default=0.3)
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
    
    # Trajectory args (for window dataset, not used for bandit but needed for pretrain)
    # Note: traj_horizon must be >= 64 for UNet1D with 3 downsampling blocks to work properly
    parser.add_argument('--traj_horizon', type=int, default=64)
    parser.add_argument('--traj_stride', type=int, default=1)
    
    args = parser.parse_args()
    
    # Set env name for pretrain (used for saving models)
    args.env = "Bandit2D"
    args.using_minari = "False"  # Use TrajectoryDataset format
    
    # Generate datasets
    print("Generating datasets...")
    datasets, mu = generate_bandit2d_datasets(
        K=args.K,
        n_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        sigma=args.sigma,
        local_radius=args.local_radius,
        seed=args.seed
    )
    
    print(f"\nStarting pretrain for {len(datasets)} clients...")
    
    # Pretrain each client
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No GPU detected. Running sequentially on CPU.")
        for client_id in range(args.num_clients):
            args.device = "cpu"
            print(f"\n[Pretrain] Client {client_id}")
            pretrain_one_client(args, client_id, datasets[client_id])
    else:
        # Use GPU if available
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        
        from fedguide.guidance.pretrain import _worker
        
        # Distribute clients across GPUs
        parts = [[] for _ in range(n_gpus)]
        for k, cid in enumerate(range(args.num_clients)):
            parts[k % n_gpus].append(cid)
        
        for gid, cids in enumerate(parts):
            print(f"[FedGuide] GPU {gid} <- clients {cids}")
        
        procs = []
        for gpu_id, client_indices in enumerate(parts):
            if not client_indices:
                continue
            p = mp.Process(target=_worker, args=(gpu_id, args, datasets, client_indices))
            p.start()
            procs.append(p)
        
        for p in procs:
            p.join()
    
    print("\n[Pretrain] All clients finished.")


if __name__ == "__main__":
    main()

