import os
import argparse
import gymnasium as gym
# import d4rl
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from fedguide.guidance.diffusion_prior import DiffusionGuidance
from fedguide.guidance.model import SDICE_Critic
import torch.multiprocessing as mp


def _worker(rank, args, datasets, client_indices):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    for i in client_indices:
        args.device = device
        print(f"\n[GPU {rank}] >>> Start training client {i}")
        pretrain_one_client(args, i, datasets[i])
        torch.cuda.empty_cache()
        print(f"[GPU {rank}] <<< Finished client {i}\n")


def _split_batch(batch, obs_dim, act_dim, device):
    if isinstance(batch, dict):
        s = batch.get("s", batch.get("observations"))
        a = batch.get("a", batch.get("actions"))
        r = batch.get("r", batch.get("rewards", None))
        s_next = batch.get("s_", batch.get("next_observations", None))
        d = batch.get("d", batch.get("dones", None))
        s = s.to(device)
        a = a.to(device)
        r = None if r is None else r.to(device)
        s_next = None if s_next is None else s_next.to(device)
        d = None if d is None else d.to(device)
    else:
        x = batch.to(device)
        s, a = x[:, :obs_dim], x[:, obs_dim:obs_dim + act_dim]
        r = s_next = d = None
    return s, a, r, s_next, d


def train_prior_one_epoch(unet, noise_scheduler, loader, accelerator, optimizer, act_dim, horizon=8):
    unet.train()
    total, n = 0.0, 0

    for batch in loader:
        if isinstance(batch, dict):
            s = batch.get("s", batch.get("observations")).to(accelerator.device)
            a = batch.get("a", batch.get("actions")).to(accelerator.device)
        else:
            x = batch.to(accelerator.device)
            # Fallback: flat [B, obs+act] – D4RL case
            s = x[:, :-act_dim]
            a = x[:, -act_dim:]

        # --------- CASE 1: trajectory windows [B, H, dim] ---------
        if s.dim() == 3 and a.dim() == 3:
            B, H, _ = s.shape

            noise = torch.randn_like(a)  # [B, H, act_dim]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=a.device,
                dtype=torch.long,
            )

            noisy_a = noise_scheduler.add_noise(a, noise, timesteps.view(B, 1, 1))

            # concatenate along feature dim, then permute to [B, C, T]
            x = torch.cat([s, noisy_a], dim=-1)      # [B, H, obs+act]
            x = x.permute(0, 2, 1)                   # [B, C=obs+act, T=H]

            out = unet(x, timesteps)
            # diffusers UNet1DModel returns either tensor or object; be safe
            model_pred = out.sample if hasattr(out, "sample") else out

            # bring back to [B, H, obs+act]
            model_pred = model_pred.permute(0, 2, 1)        # [B, H, obs+act]
            pred_noise_on_a = model_pred[:, :, -act_dim:]   # [B, H, act_dim]

            loss = torch.mean((pred_noise_on_a - noise) ** 2)

        # --------- CASE 2: single-step [B, dim] (legacy / D4RL) ---------
        else:
            noise = torch.randn_like(a)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (a.shape[0],),
                device=a.device,
                dtype=torch.long,
            )
            noisy_a = noise_scheduler.add_noise(a, noise, timesteps)

            # Use horizon parameter (required for UNet1D padding in single-step case)
            x = torch.cat([s, noisy_a], dim=-1)  # [B, obs+act]
            x = x.unsqueeze(-1).repeat(1, 1, horizon)  # [B, obs+act, horizon]
            
            out = unet(x, timesteps)
            model_pred = out.sample if hasattr(out, "sample") else out  # [B, obs+act, horizon]
            pred_noise_on_a = model_pred[:, -act_dim:, :].mean(dim=-1)  # [B, act_dim]
            loss = torch.mean((pred_noise_on_a - noise) ** 2)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def _linear_warmup_scale(epoch, total_epochs, init_val, target_val):
    if total_epochs <= 0:
        return target_val
    ratio = min(1.0, float(epoch) / float(total_epochs))
    return init_val + (target_val - init_val) * ratio


def train_guidance_one_epoch(
        sdice: SDICE_Critic,
        loader,
        device,
        obs_dim,
        act_dim,
        do_update_v0=True,
        do_update_wt=True
):
    """
    reference to main_diffusion_dice：if batch has r/s_next/d, v0 + wt；otherwise wt only
    """
    for batch in loader:
        s, a, r, s_next, d = _split_batch(batch, obs_dim, act_dim, device)
        if do_update_v0 and (r is not None and s_next is not None and d is not None) and hasattr(sdice, "update_v0"):
            sdice.update_v0({"s": s, "a": a, "r": r, "s_": s_next, "d": d})
        if do_update_wt and hasattr(sdice, "update_wt"):
            sdice.update_wt({"s": s, "a": a})

def pretrain_one_client(args, client_id, dataset):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -------------------
    # Minari case (maze)
    # -------------------
    if "True" in args.using_minari:
        # using maze minari to get dims
        sample = dataset[0]
        # s: [H, obs_dim], a: [H, act_dim]
        obs_dim = sample["s"].shape[-1]
        act_dim = sample["a"].shape[-1]
    else:
        try:
            if args.env.lower() in ["bandit2d", "bandit_2d", "2dbandit"]:
                from fedguide.envs.bandit2d import Bandit2D
                _env = Bandit2D(K=4, sigma=0.2, seed=0)
                _env.reset(seed=0)
                obs_dim = int(_env.observation_space.shape[0])
                act_dim = int(_env.action_space.shape[0])
            else:
                # Try gym.make for standard environments
                _env = gym.make(args.env)
                obs_dim = int(_env.observation_space.shape[0])
                act_dim = int(_env.action_space.shape[0])
                _env.close()
        except Exception as e:
            raise RuntimeError(f"Failed to create environment '{args.env}': {e}")

    # dataloader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # prior
    prior = DiffusionGuidance(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=args.dg_hidden_dim,
        timesteps=args.num_train_timesteps,
        horizon=args.traj_horizon,
    ).to(device)
    unet = prior.model
    noise_scheduler = prior.noise_scheduler

    # Older code used set_format, but not all diffusers versions have this
    if hasattr(noise_scheduler, "set_format"):
        noise_scheduler.set_format("pt")

    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)

    # guidance: SDICE
    sdice = None
    if args.guidance_mode in ("warmup", "interleave"):
        class _C: pass

        c = _C()
        c.device = accelerator.device
        c.q_ensemble_num = args.q_ensemble_num
        c.value_lr = args.value_lr
        c.wt_lr = args.wt_lr
        c.weight_decay = args.weight_decay
        c.use_lr_schedule = args.use_lr_schedule
        c.train_epoch = 1
        c.min_value_lr = args.min_value_lr
        c.M = args.M
        c.alpha = args.alpha
        c.hidden_dim = args.hidden_dim
        sdice = SDICE_Critic(adim=act_dim, sdim=obs_dim, args=c).to(accelerator.device)
        sdice.guidance_scale = args.guidance_scale_init

    # save
    save_dir = os.path.join("./model/models_prior", args.env, f"client_{client_id}")
    os.makedirs(save_dir, exist_ok=True)

    # ====== train prior + guidance：off / warmup / interleave  ======
    print(f"[Client {client_id}] Start prior pretrain: mode={args.guidance_mode}")
    for epoch in range(1, args.n_behavior_epochs + 1):
        prior_loss = train_prior_one_epoch(unet, noise_scheduler, loader, accelerator, optimizer, act_dim, horizon=args.traj_horizon)
        if args.guidance_mode == "interleave" and sdice is not None:
            if epoch % args.guidance_interval == 0:
                if args.guidance_scale_warmup_epochs > 0:
                    done_steps = min(epoch, args.guidance_scale_warmup_epochs)
                    sdice.guidance_scale = _linear_warmup_scale(
                        done_steps, args.guidance_scale_warmup_epochs,
                        args.guidance_scale_init, args.guidance_scale_target
                    )
                for _ in range(args.guidance_epochs_per_call):
                    train_guidance_one_epoch(
                        sdice, loader, accelerator.device, obs_dim, act_dim,
                        do_update_v0=True, do_update_wt=True
                    )

        if accelerator.is_main_process:
            if epoch % 10 == 0:
                print(f"[Client {client_id}] epoch {epoch}/{args.n_behavior_epochs} | prior_loss={prior_loss:.6f}")
            if (epoch % args.save_interval == 0) or (epoch == args.n_behavior_epochs):
                ckpt_dir = os.path.join(save_dir, f"ckpt_epoch{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                unet.eval()
                unet.save_pretrained(ckpt_dir)
                noise_scheduler.save_pretrained(ckpt_dir)
                torch.save({
                    "unet": unet.state_dict(),
                    "scheduler_config": noise_scheduler.config,
                }, os.path.join(ckpt_dir, "torch_prior.pth"))
                if sdice is not None and args.guidance_mode == "interleave":
                    torch.save(sdice.state_dict(), os.path.join(ckpt_dir, "guidance_sdice.pth"))
                unet.train()

    # ====== guidance warmup mode (after prior) ======
    if args.guidance_mode == "warmup" and sdice is not None and args.guidance_warmup_epochs > 0:
        print(f"[Client {client_id}] Guidance warmup AFTER prior: {args.guidance_warmup_epochs} epochs")
        for e in range(1, args.guidance_warmup_epochs + 1):
            sdice.guidance_scale = _linear_warmup_scale(
                e, max(1, args.guidance_scale_warmup_epochs),
                args.guidance_scale_init, args.guidance_scale_target
            )
            train_guidance_one_epoch(
                sdice, loader, accelerator.device, obs_dim, act_dim,
                do_update_v0=True, do_update_wt=True
            )
            if e % max(1, args.guidance_warmup_epochs // 5) == 0:
                print(f"  warmup {e}/{args.guidance_warmup_epochs} | guidance_scale={sdice.guidance_scale:.3f}")

        g_path = os.path.join(save_dir, "guidance_warmup.pth")
        torch.save(sdice.state_dict(), g_path)
        print(f"[Client {client_id}] Saved guidance warmup -> {g_path}")

    if accelerator.is_main_process:
        final_dir = os.path.join(save_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unet.eval()
        unet.save_pretrained(final_dir)
        noise_scheduler.save_pretrained(final_dir)
        torch.save({
            "unet": unet.state_dict(),
            "scheduler_config": noise_scheduler.config,
        }, os.path.join(final_dir, "torch_prior.pth"))
        if sdice is not None:
            torch.save(sdice.state_dict(), os.path.join(final_dir, "guidance_sdice.pth"))


# ========== multi-clients ==========
def main(args):
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Using Minari Maze datasets
    using_minari = "True" in args.using_minari

    # Unified dataset loader
    from fedguide.datasets import make_datasets, DatasetType, build_hetero_config

    if using_minari:
        print("Using Minari dataset loader")
        args.env_group = "maze"
        datasets = make_datasets(
            dataset_type=DatasetType.MINARI,
            n_clients=args.num_clients,
            dataset_id=args.minari_dataset_id,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
            horizon=args.traj_horizon,
            stride=args.traj_stride,
        )
    else:
        # D4RL Loader
        print("Using D4RL dataset loader")

        # Applying Reacher
        if "reacher" in args.env and not using_minari:
            build_hetero_config(
                env_name='reacher',
                num_clients=args.num_clients,
                hetero_type="both"
            )

        args.env_group = args.env.split("-")[0]
        args.hetero_modes = getattr(args, "hetero_modes", ["task", "state_region", "dyn_shift"])

        datasets = make_datasets(
            dataset_type=DatasetType.D4RL,
            n_clients=args.num_clients,
            env_group=args.env_group,
            hetero_modes=tuple(args.hetero_modes),
            save_json=f"./configs/clients/{args.env_group}_d4rl.json"
        )

    n_clients = len(datasets)
    n_gpus = torch.cuda.device_count()
    print(f"[FedGuide] n_clients={n_clients}, n_gpus={n_gpus}")

    # ===== No GPU found：CPU for loop =====
    if n_gpus == 0:
        print("[FedGuide] No GPU detected. Running sequentially on CPU.")
        for cid in range(n_clients):
            args.device = "cpu"
            pretrain_one_client(args, cid, datasets[cid])
        return

    # ===== GPU =====
    parts = [[] for _ in range(n_gpus)]
    for k, cid in enumerate(range(n_clients)):
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
    print("[FedGuide] All clients finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="walker2d-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # minari
    parser.add_argument("--using_minari", type=str, default="True")
    parser.add_argument("--minari_dataset_id", type=str, default="D4RL/pointmaze/medium-v2")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--traj_horizon", type=int, default=64)
    parser.add_argument("--traj_stride", type=int, default=16)

    # prior
    parser.add_argument('--num_train_timesteps', type=int, default=1000) #1000
    parser.add_argument('--dg_hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_behavior_epochs', type=int, default=1500) # 1500
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # guidance
    parser.add_argument('--guidance_mode', type=str, default="off", choices=["off", "warmup", "interleave"])

    # warmup
    parser.add_argument('--guidance_warmup_epochs', type=int, default=0)
    parser.add_argument('--guidance_scale_init', type=float, default=0.0)
    parser.add_argument('--guidance_scale_target', type=float, default=1.0)
    parser.add_argument('--guidance_scale_warmup_epochs', type=int, default=0)

    # interleave
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
    main(args)