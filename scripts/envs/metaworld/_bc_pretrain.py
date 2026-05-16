"""BC pretrain (warm-start) for MetaWorld ML10 FedGuide policies.

Each ML10 task gets a Behaviour-Cloning warm-start trained on the same data
the prior was pretrained on. The architecture matches FedguideAgent.policy
so the saved state_dict loads via ``actor_ckpt`` cleanly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.envs.metaworld._pretrain import (
    _make_metaworld_env,
    _collect_rollouts,
)


def _make_policy(state_dim: int, action_dim: int, hidden_dim: int = 256,
                 activation: str = "tanh") -> nn.Sequential:
    act = nn.Tanh if activation == "tanh" else nn.ReLU
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim), act(),
        nn.Linear(hidden_dim, hidden_dim), act(),
        nn.Linear(hidden_dim, action_dim),
    )


def _bc_one_client(s, a, *, state_dim, action_dim, hidden_dim, activation,
                   epochs, batch_size, lr, init_log_std, device):
    policy = _make_policy(state_dim, action_dim, hidden_dim, activation).to(device)
    log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std), device=device))
    opt = torch.optim.Adam(list(policy.parameters()) + [log_std], lr=lr)
    s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
    a_t = torch.as_tensor(a, dtype=torch.float32, device=device)
    ds = TensorDataset(s_t, a_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    losses = []
    for ep in range(1, epochs + 1):
        ep_loss, n = 0.0, 0
        for sb, ab in loader:
            mu = policy(sb)
            sigma = log_std.exp().clamp(min=1e-3)
            nll = 0.5 * ((ab - mu) ** 2 / (sigma ** 2)).sum(-1) \
                + log_std.sum() + 0.5 * action_dim * math.log(2 * math.pi)
            loss = nll.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + [log_std], 1.0)
            opt.step()
            ep_loss += float(loss.detach().cpu()); n += 1
        avg = ep_loss / max(1, n)
        losses.append(avg)
        if ep == 1 or ep % 20 == 0 or ep == epochs:
            print(f"    epoch {ep:>3}/{epochs} | bc_nll={avg:+.4f}")
    return policy, log_std.detach().cpu(), losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_path", type=str, default="data/metaworld/metadata.json")
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--first_client_id", type=int, default=0)
    ap.add_argument("--rollout_steps", type=int, default=5000)
    ap.add_argument("--behaviour", type=str, default="scripted",
                    choices=["scripted", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--policy_activation", type=str, default="tanh", choices=["tanh", "relu"])
    ap.add_argument("--init_log_std", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_root", type=str, default="./model/bc_policy")
    ap.add_argument("--env_name", type=str, default="MetaWorld")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metadata_path = str(_PROJECT_ROOT / args.metadata_path)
    save_root = _PROJECT_ROOT / args.save_root / args.env_name
    save_root.mkdir(parents=True, exist_ok=True)
    print(f"[bc_pretrain] device={device} save_root={save_root}")

    with open(metadata_path, "r") as f:
        meta = json.load(f)
    clients = meta["clients"]

    for offset in range(args.num_clients):
        cid = args.first_client_id + offset
        if cid >= len(clients):
            break
        cfg = clients[cid]
        task_name = cfg["task"]
        print(f"\n[BC] client {cid} | task={task_name}")
        env = _make_metaworld_env(task_name, seed=args.seed + cid)
        state_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
        s, a, _r, _s2, _d = _collect_rollouts(env, args.rollout_steps, args.behaviour, task_name, seed=args.seed + cid)
        print(f"  buffer: s={s.shape}, a={a.shape}")
        policy, log_std, losses = _bc_one_client(
            s, a,
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=args.hidden_dim, activation=args.policy_activation,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            init_log_std=args.init_log_std, device=device,
        )
        save_dir = save_root / f"client_{cid}" / "final"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
            "log_std": log_std,
            "state_dim": state_dim, "action_dim": action_dim,
            "hidden_dim": args.hidden_dim, "policy_activation": args.policy_activation,
            "bc_final_loss": float(losses[-1]), "bc_epochs": int(args.epochs),
        }, save_dir / "policy.pth")
        print(f"  → saved {save_dir/'policy.pth'} (final NLL={losses[-1]:+.4f})")
        try:
            env.close()
        except Exception:
            pass

    print("\n[bc_pretrain] done.")


if __name__ == "__main__":
    main()
