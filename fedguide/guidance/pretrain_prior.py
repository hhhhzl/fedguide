"""
FedGuide: Multi-Client Diffusion Prior Pretraining
"""
import os
import argparse
import gymnasium as gym
import d4rl
import torch
import math
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from accelerate import Accelerator
from diffusers import UNet1DModel, DDPMScheduler

# fedguide imports
from fedguide.utils.datasets import _make_d4rl_datasets
from fedguide.utils.herero import build_hetero_config


def _worker(rank, args, datasets, client_indices):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    for i in client_indices:
        args.device = device
        print(f"\n[GPU {rank}] >>> Start training client {i}")
        pretrain_one_client(args, i, datasets[i])
        torch.cuda.empty_cache()
        print(f"[GPU {rank}] <<< Finished client {i}\n")


# ======================================================
# PRETRAIN ONE CLIENT (Hugging Face diffusers, keep datasets as-is)
# ======================================================
def pretrain_one_client(args, client_id, dataset):
    # ---------------- paths & device ----------------
    save_dir = f"./model/models_prior/{args.env}/client_{client_id}"
    os.makedirs(save_dir, exist_ok=True)

    device = getattr(args, "device", "cuda")
    batch_size = getattr(args, "batch_size", 256)
    lr = getattr(args, "lr", 1e-4)
    max_epochs = int(getattr(args, "n_behavior_epochs", 2000))
    save_interval = int(getattr(args, "save_interval", 200))
    num_train_timesteps = int(getattr(args, "num_train_timesteps", 1000))

    # ---------------- dims ----------------
    _env = gym.make(args.env)
    obs_dim = int(_env.observation_space.shape[0])
    act_dim = int(_env.action_space.shape[0])
    _env.close()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---------------- model & scheduler ----------------
    in_channels = obs_dim + act_dim
    unet = UNet1DModel(
        sample_size=1,  # L=1
        in_channels=in_channels,  # C_in = obs+act
        out_channels=in_channels,  #
        block_out_channels=(128, 128, 256),
        down_block_types=("DownBlock1D", "DownBlock1D"),
        up_block_types=("UpBlock1D", "UpBlock1D"),
        time_embedding_type="positional",
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    accelerator = Accelerator()
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)
    noise_scheduler.set_format("pt")

    # ---------------- helpers ----------------
    def _split_sa(batch_tensor_or_dict):
        if isinstance(batch_tensor_or_dict, dict):
            if "s" in batch_tensor_or_dict and "a" in batch_tensor_or_dict:
                s = batch_tensor_or_dict["s"]
                a = batch_tensor_or_dict["a"]
            elif "observations" in batch_tensor_or_dict and "actions" in batch_tensor_or_dict:
                s = batch_tensor_or_dict["observations"]
                a = batch_tensor_or_dict["actions"]
            else:
                raise ValueError("Dataset dict must contain ('s','a') or ('observations','actions').")
        else:
            x = batch_tensor_or_dict
            s, a = x[:, :obs_dim], x[:, obs_dim:obs_dim + act_dim]
        return s.to(accelerator.device), a.to(accelerator.device)

    # ---------------- train ----------------
    global_epoch = 0
    unet.train()
    while global_epoch < max_epochs:
        avg_loss = 0.0
        n_it = 0
        for batch in loader:
            s, a = _split_sa(batch)  # [B, obs_dim], [B, act_dim]
            noise = torch.randn_like(a)
            timesteps = torch.randint(0, num_train_timesteps,(a.shape[0],), device=a.device).long()
            noisy_a = noise_scheduler.add_noise(a, noise, timesteps)

            # 2) UNet input：[s, a_noisy] → (B, C, L=1)
            x = torch.cat([s, noisy_a], dim=-1).unsqueeze(-1)  # [B, obs+act, 1]

            # 3) predict noise
            model_pred = unet(x, timesteps).sample.squeeze(-1)  # [B, obs+act]

            # 4) MSE
            pred_noise_on_a = model_pred[:, -act_dim:]
            loss = torch.mean((pred_noise_on_a - noise) ** 2)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            avg_loss += loss.item()
            n_it += 1

        global_epoch += 1
        if accelerator.is_main_process:
            print(f"[Client {client_id}] epoch {global_epoch}/{max_epochs} | loss={avg_loss / max(1, n_it):.6f}")
            if (global_epoch % save_interval == 0) or (global_epoch >= max_epochs):
                ckpt_dir = os.path.join(save_dir, f"ckpt_epoch{global_epoch}")
                unet.eval()
                unet.save_pretrained(ckpt_dir)
                noise_scheduler.save_pretrained(ckpt_dir)
                unet.train()

    # final
    if accelerator.is_main_process:
        final_dir = os.path.join(save_dir, "final")
        unet.eval()
        unet.save_pretrained(final_dir)
        noise_scheduler.save_pretrained(final_dir)
    accelerator.wait_for_everyone()


# ======================================================
# MULTI-CLIENT ENTRY
# ======================================================
def main(args):
    os.makedirs(f"./model/models_prior/{args.env}", exist_ok=True)

    # ---- 1. Prepare heterogeneity meta ----
    if "reacher" in args.env:
        build_hetero_config(
            env_name='reacher',
            num_clients=args.num_clients,
            hetero_type="both"
        )

    datasets = _make_d4rl_datasets(
        env_group=args.env_group,
        n_clients=args.num_clients,
        hetero_modes=tuple(args.hetero_modes),
        save_json=f"./configs/clients/{args.env_group}_d4rl.json"
    )

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPU available for multi-client training!")

    n_clients = len(datasets)
    print(f"[FedGuide] {n_clients} clients on {n_gpus} GPUs.")

    clients_per_gpu = math.ceil(n_clients / n_gpus)
    assignments = []
    for g in range(n_gpus):
        start = g * clients_per_gpu
        end = min((g + 1) * clients_per_gpu, n_clients)
        if start < end:
            assignments.append(list(range(start, end)))

    # === batch：run n_gpus clients at the same time ===
    for batch_id, batch_indices in enumerate(range(0, n_clients, n_gpus)):
        current = list(range(batch_indices, min(batch_indices + n_gpus, n_clients)))
        print(f"\n Batch {batch_id + 1}: training clients {current} on {n_gpus} GPUs...")
        mp.spawn(
            _worker,
            nprocs=min(n_gpus, len(current)),
            args=(args, datasets, current),
        )
    print("All clients finished multi-GPU batched training.")


# ======================================================
# ARGUMENT PARSING
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="walker2d-medium-replay-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--device_num", default=0, type=int)
    parser.add_argument('--actor_load_path', type=str, default=None)
    parser.add_argument('--inference_sample', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--use_lr_schedule', type=int, default=0)
    parser.add_argument('--min_value_lr', type=float, default=1e-4)
    # Add federated options
    parser.add_argument('--num_clients', type=float, default=4)
    parser.add_argument('--lr', type=float, default=4)
    parser.add_argument('--batch_size', type=float, default=256)
    parser.add_argument('--n_behavior_epochs', type=float, default=2000)
    parser.add_argument('--save_interval', type=float, default=200)
    parser.add_argument('--horizon', type=float, default=20)
    args = parser.parse_args()
    args.env_group = args.env.split("-")[0] if "-" in args.env else "reacher"
    args.hetero_modes = ["task", "state_region", "dyn_shift"]
    main(args)
