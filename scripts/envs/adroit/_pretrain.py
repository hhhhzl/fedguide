"""Pretrain DiffusionGuidance + SDICE_Critic for federated D4RL Adroit.

Each client = one (task, quality) D4RL Adroit pair. We load the corresponding
HDF5 dataset directly (bypassing d4rl's qlearning_dataset assertion) and
pretrain the diffusion prior + SDICE critic on it.

Outputs:
    ./model/models_prior/Adroit/client_{i}/final/torch_prior.pth
    ./model/models_prior/Adroit/client_{i}/final/guidance_sdice.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from fedguide.guidance.diffusion_prior import DiffusionGuidance
from fedguide.guidance.model import SDICE_Critic


class _SADataset(Dataset):
    def __init__(self, s, a):
        self.s = torch.as_tensor(np.asarray(s), dtype=torch.float32)
        self.a = torch.as_tensor(np.asarray(a), dtype=torch.float32)

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, i):
        return torch.cat([self.s[i], self.a[i]], dim=-1)


class _TransitionDataset(Dataset):
    def __init__(self, s, a, r, s_next, d):
        self.s = torch.as_tensor(np.asarray(s), dtype=torch.float32)
        self.a = torch.as_tensor(np.asarray(a), dtype=torch.float32)
        self.r = torch.as_tensor(np.asarray(r), dtype=torch.float32)
        self.s_next = torch.as_tensor(np.asarray(s_next), dtype=torch.float32)
        self.d = torch.as_tensor(np.asarray(d), dtype=torch.float32)

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, i):
        return {
            "s": self.s[i], "a": self.a[i], "r": self.r[i],
            "s_": self.s_next[i], "d": self.d[i],
        }


def _collect_d4rl_dataset(variant: str, max_size=None, seed: int = 0):
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    import gym as old_gym
    import d4rl  # noqa: F401
    import h5py
    e = old_gym.make(variant)
    fp = e.dataset_filepath
    if not os.path.exists(fp):
        try:
            e.get_dataset()
        except Exception:
            pass
    e.close()
    with h5py.File(fp, "r") as f:
        s = f["observations"][:].astype(np.float32)
        a = f["actions"][:].astype(np.float32)
        r = f["rewards"][:].astype(np.float32)
        s2 = np.empty_like(s)
        s2[:-1] = s[1:]
        s2[-1] = s[-1]
        d = f["terminals"][:].astype(np.float32)
        if "timeouts" in f:
            d = np.maximum(d, f["timeouts"][:].astype(np.float32))
    if max_size is not None and len(s) > max_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(s), max_size, replace=False)
        s, a, r, s2, d = s[idx], a[idx], r[idx], s2[idx], d[idx]
    return s, a, r, s2, d


def _train_guidance_one_epoch(sdice, loader, device, do_update_v0=True, do_update_wt=True):
    for batch in loader:
        gd = {k: v.to(device) for k, v in batch.items()}
        if do_update_v0 and hasattr(sdice, "update_v0"):
            try:
                sdice.update_v0(gd)
            except TypeError:
                pass
        if do_update_wt and hasattr(sdice, "update_wt"):
            gd2 = {"s": gd["s"], "a": gd["a"], "weights": torch.ones_like(gd["r"])}
            try:
                sdice.update_wt(gd2)
            except TypeError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_path", type=str, default="data/adroit/metadata.json")
    ap.add_argument("--num_clients", type=int, default=8)
    ap.add_argument("--first_client_id", type=int, default=0)
    ap.add_argument("--d4rl_max_size", type=int, default=100_000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_behavior_epochs", type=int, default=40)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dg_hidden_dim", type=int, default=64)
    ap.add_argument("--dg_horizon", type=int, default=64)
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--guidance_mode", type=str, default="warmup",
                    choices=["off", "warmup", "interleave"])
    ap.add_argument("--guidance_warmup_epochs", type=int, default=80)
    ap.add_argument("--guidance_interval", type=int, default=5)
    ap.add_argument("--guidance_epochs_per_call", type=int, default=1)
    ap.add_argument("--guidance_scale_init", type=float, default=0.0)
    ap.add_argument("--guidance_scale_target", type=float, default=1.0)
    ap.add_argument("--guidance_scale_warmup_epochs", type=int, default=30)
    ap.add_argument("--q_ensemble_num", type=int, default=0)
    ap.add_argument("--value_lr", type=float, default=1e-4)
    ap.add_argument("--wt_lr", type=float, default=1e-4)
    ap.add_argument("--min_value_lr", type=float, default=1e-5)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--hidden_dim", type=int, default=256)

    ap.add_argument("--save_root", type=str, default="./model/models_prior")
    ap.add_argument("--env_name", type=str, default="Adroit")

    args = ap.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    metadata_path = str(_PROJECT_ROOT / args.metadata_path)
    print(f"[pretrain_adroit] using metadata: {metadata_path}")
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    clients = meta["clients"]

    for offset in range(args.num_clients):
        cid = args.first_client_id + offset
        if cid >= len(clients):
            break
        task = clients[cid]["task"]
        print(f"\n[Pretrain] Adroit client {cid} | task={task}")

        s, a, r, s2, d = _collect_d4rl_dataset(task, max_size=args.d4rl_max_size, seed=args.seed + cid)
        s_dim = int(s.shape[1])
        a_dim = int(a.shape[1])
        print(f"  buffer: s={s.shape}, a={a.shape}, mean_r={r.mean():.3f}")

        prior = DiffusionGuidance(
            state_dim=s_dim, action_dim=a_dim,
            hidden_dim=args.dg_hidden_dim,
            timesteps=args.num_train_timesteps,
            horizon=args.dg_horizon,
        ).to(device)
        opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sa_loader = DataLoader(_SADataset(s, a), batch_size=args.batch_size, shuffle=True, drop_last=True)

        sdice = None
        gd_loader = None
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
            sdice = SDICE_Critic(adim=a_dim, sdim=s_dim, args=c).to(device)
            sdice.guidance_scale = args.guidance_scale_init
            gd_loader = DataLoader(
                _TransitionDataset(s, a, r, s2, d),
                batch_size=args.batch_size, shuffle=True, drop_last=True,
            )

        save_dir = Path(args.save_root) / args.env_name / f"client_{cid}"
        (save_dir / "final").mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.n_behavior_epochs + 1):
            prior.train()
            total = 0.0
            n = 0
            for batch in sa_loader:
                s_b = batch[:, :s_dim].to(device)
                a_b = batch[:, s_dim:].to(device)
                opt.zero_grad()
                loss = prior.update(s_b, a_b, lr=args.lr)
                total += loss
                n += 1
            avg_loss = total / max(1, n)
            if args.guidance_mode == "interleave" and sdice is not None and epoch % args.guidance_interval == 0:
                if args.guidance_scale_warmup_epochs > 0:
                    ratio = min(1.0, epoch / args.guidance_scale_warmup_epochs)
                    sdice.guidance_scale = args.guidance_scale_init + \
                        (args.guidance_scale_target - args.guidance_scale_init) * ratio
                for _ in range(args.guidance_epochs_per_call):
                    _train_guidance_one_epoch(sdice, gd_loader, device, True, True)
            if epoch % 50 == 0 or epoch == args.n_behavior_epochs:
                msg = f"  [client {cid}] epoch {epoch}/{args.n_behavior_epochs} | prior_loss={avg_loss:.6f}"
                if sdice is not None:
                    msg += f" | guidance_scale={sdice.guidance_scale:.3f}"
                print(msg)

        if args.guidance_mode == "warmup" and sdice is not None and args.guidance_warmup_epochs > 0:
            print(f"  [client {cid}] guidance warmup: {args.guidance_warmup_epochs} epochs")
            for e in range(1, args.guidance_warmup_epochs + 1):
                ratio = min(1.0, e / max(1, args.guidance_scale_warmup_epochs))
                sdice.guidance_scale = args.guidance_scale_init + \
                    (args.guidance_scale_target - args.guidance_scale_init) * ratio
                _train_guidance_one_epoch(sdice, gd_loader, device, True, True)

        torch.save({
            "prior": prior.state_dict(),
            "state_dim": s_dim, "action_dim": a_dim,
            "hidden_dim": args.dg_hidden_dim,
            "timesteps": args.num_train_timesteps,
            "horizon": args.dg_horizon,
            "prior_type": "diffusion",
        }, save_dir / "final" / "torch_prior.pth")
        if sdice is not None:
            torch.save(sdice.state_dict(), save_dir / "final" / "guidance_sdice.pth")
            print(f"[client {cid}] saved prior + guidance to {save_dir/'final'}")
        else:
            print(f"[client {cid}] saved prior to {save_dir/'final'}")

    print("\n[Pretrain] All Adroit clients done.")


if __name__ == "__main__":
    main()
