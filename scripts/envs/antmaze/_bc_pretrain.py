"""BC pretrain (warm-start) for AntMaze FedGuide policies.

AntMaze's prior is a state-conditional UNet1D diffusion model — there's no
single μ to copy into the policy bias the way bandit2d's GaussianBehaviorPrior
allows. The "D-fix" warm-start equivalent for antmaze is **behavior cloning**:
fit a policy network to the same offline data the prior was trained on so
each client's policy starts at its own behavior policy (per-cluster goal).

For each client `i`:
    1. Collect offline rollouts on its hetero env (random or PPO behaviour),
       same as `_pretrain.py`.
    2. Train an MLP policy `π_i(a|s)` by minimizing a Gaussian negative
       log-likelihood: ½‖a − μ_θ(s)‖² / σ² + log σ, with σ = exp(log_std).
       Architecture matches `FedguideAgent.policy` (Linear → activation →
       Linear → activation → Linear) so the saved state_dict loads cleanly.
    3. Save to `<save_root>/<env>/client_<i>/final/policy.pth` as
       `{"policy": <state_dict>, "log_std": <tensor>, "state_dim": s_dim,
        "action_dim": a_dim, "policy_activation": "tanh"}`.

The FedguideAgent's existing `actor_ckpt` loader picks up the "policy" key,
so plumbing into the federated runner is just "set `bc_dir` in the config
and the client wires up `actor_ckpt` per `cluster_id`".

Usage:
    python scripts/envs/antmaze/_bc_pretrain.py --num_clients 8 \\
        --rollout_steps 5000 --behaviour random \\
        --epochs 100 --hidden_dim 256 \\
        --save_root ./model/bc_policy
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

# Reuse the rollout-collection helpers from the prior pretrain.
from scripts.envs.antmaze._pretrain import (
    _collect_random_rollouts,
    _collect_ppo_rollouts,
    _collect_d4rl_dataset,
)
from fedguide.envs.antmaze_hetero import make_hetero_antmaze_env_from_metadata


def _make_policy(state_dim: int, action_dim: int, hidden_dim: int = 256,
                 activation: str = "tanh") -> nn.Sequential:
    """Mirror FedguideAgent.policy exactly so the saved state_dict loads
    via `actor_ckpt → load_state_dict(strict=False)` without renaming."""
    act = nn.Tanh if activation == "tanh" else nn.ReLU
    return nn.Sequential(
        nn.Linear(state_dim, hidden_dim), act(),
        nn.Linear(hidden_dim, hidden_dim), act(),
        nn.Linear(hidden_dim, action_dim),
    )


def _bc_one_client(s: np.ndarray, a: np.ndarray, *,
                   state_dim: int, action_dim: int,
                   hidden_dim: int, activation: str,
                   epochs: int, batch_size: int, lr: float,
                   init_log_std: float, device: torch.device) -> tuple[nn.Sequential, torch.Tensor, list[float]]:
    policy = _make_policy(state_dim, action_dim, hidden_dim, activation).to(device)
    log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std), device=device))
    opt = torch.optim.Adam(list(policy.parameters()) + [log_std], lr=lr)

    s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
    a_t = torch.as_tensor(a, dtype=torch.float32, device=device)
    ds = TensorDataset(s_t, a_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    losses: list[float] = []
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        n_batches = 0
        for sb, ab in loader:
            mu = policy(sb)
            sigma = log_std.exp().clamp(min=1e-3)
            # Gaussian NLL summed over action dims.
            nll = 0.5 * ((ab - mu) ** 2 / (sigma ** 2)).sum(-1) \
                  + log_std.sum() \
                  + 0.5 * action_dim * math.log(2 * math.pi)
            loss = nll.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + [log_std], 1.0)
            opt.step()
            ep_loss += float(loss.detach().cpu())
            n_batches += 1
        avg = ep_loss / max(1, n_batches)
        losses.append(avg)
        if ep == 1 or ep % 20 == 0 or ep == epochs:
            print(f"    epoch {ep:>3}/{epochs} | bc_nll={avg:+.4f} | log_std={log_std.detach().cpu().tolist()}")

    return policy, log_std.detach().cpu(), losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_path", type=str, default="data/antmaze/metadata.json")
    ap.add_argument("--num_clients", type=int, default=8)
    ap.add_argument("--first_client_id", type=int, default=0)
    ap.add_argument("--rollout_steps", type=int, default=5000)
    ap.add_argument("--behaviour", type=str, default="random",
                    choices=["random", "ppo", "d4rl"])
    ap.add_argument("--ppo_steps", type=int, default=20_000)
    ap.add_argument("--d4rl_max_size", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--policy_activation", type=str, default="tanh",
                    choices=["tanh", "relu"])
    ap.add_argument("--init_log_std", type=float, default=0.0)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_root", type=str, default="./model/bc_policy")
    ap.add_argument("--env_name", type=str, default="AntMaze")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metadata_path = str(_PROJECT_ROOT / args.metadata_path)
    save_root = _PROJECT_ROOT / args.save_root / args.env_name
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"[bc_pretrain] device={device} save_root={save_root}")
    print(f"[bc_pretrain] num_clients={args.num_clients} behaviour={args.behaviour}")

    for offset in range(args.num_clients):
        client_id = args.first_client_id + offset
        print(f"\n[BC] client {client_id}")

        env = None
        if args.behaviour == "d4rl":
            import json as _json
            with open(metadata_path, "r") as _f:
                _meta = _json.load(_f)
            variant = _meta["clients"][client_id]["variant"]
            print(f"  [behaviour=d4rl] loading offline dataset for variant={variant}")
            s, a, _r, _s2, _d = _collect_d4rl_dataset(
                variant, max_size=args.d4rl_max_size, seed=args.seed + client_id
            )
            state_dim = int(s.shape[1])
            action_dim = int(a.shape[1])
        else:
            env = make_hetero_antmaze_env_from_metadata(
                metadata_path, client_id, seed=args.seed + client_id
            )
            state_dim = int(env.observation_space.shape[0])
            action_dim = int(env.action_space.shape[0])
            if args.behaviour == "ppo":
                print(f"  collecting PPO rollouts ({args.rollout_steps} steps after {args.ppo_steps} train)...")
                s, a, _r, _s2, _d = _collect_ppo_rollouts(
                    env, args.rollout_steps, args.ppo_steps, seed=args.seed + client_id
                )
            else:
                print(f"  collecting random rollouts ({args.rollout_steps} steps)...")
                s, a, _r, _s2, _d = _collect_random_rollouts(
                    env, args.rollout_steps, seed=args.seed + client_id
                )
        print(f"  buffer: s={s.shape}, a={a.shape}")

        policy, log_std, losses = _bc_one_client(
            s, a,
            state_dim=state_dim, action_dim=action_dim,
            hidden_dim=args.hidden_dim, activation=args.policy_activation,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            init_log_std=args.init_log_std, device=device,
        )

        save_dir = save_root / f"client_{client_id}" / "final"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
            "log_std": log_std,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "policy_activation": args.policy_activation,
            "bc_final_loss": float(losses[-1]),
            "bc_epochs": int(args.epochs),
        }, save_dir / "policy.pth")
        print(f"  → saved {save_dir/'policy.pth'} (final NLL={losses[-1]:+.4f})")

        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    print("\n[bc_pretrain] done.")


if __name__ == "__main__":
    main()
