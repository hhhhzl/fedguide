"""Pretrain DiffusionGuidance + SDICE_Critic for federated HalfCheetah.

Reacher has no direct D4RL dataset, so we collect offline (s, a, r, s')
trajectories per client by running a behaviour policy in that client's
the federated HalfCheetah env (uses the heterogeneity from
``data/halfcheetah/metadata_mild64.json``: per-client goal region, action noise,
reward scale, angle noise). The behaviour policy can be either:

* ``--behaviour random`` — sample from action_space.uniform (fast, works
  out of the box, gives a wide-coverage prior).
* ``--behaviour ppo`` — train PPO from scratch on the client env for
  ``--ppo_steps`` env steps, then collect rollouts. Slower but produces
  a more value-aware behaviour buffer (closer to "medium" quality).

After each client's buffer is collected we pretrain ``DiffusionGuidance``
on the (s, a) pairs and ``SDICE_Critic`` on (s, a, r, s', d).

Outputs:
    ./model/models_prior/HalfCheetah/client_{i}/final/torch_prior.pth
    ./model/models_prior/HalfCheetah/client_{i}/final/guidance_sdice.pth

Usage:
    # Quick smoke test (1 client, random behaviour, 200 epochs, small buffer):
    python scripts/envs/halfcheetah/_pretrain.py \
        --num_clients 1 --behaviour random \
        --rollout_steps 5000 --n_behavior_epochs 200 \
        --device cuda --save_root ./model/models_prior

    # Full pretrain (8 clients, longer):
    python scripts/envs/halfcheetah/_pretrain.py \
        --num_clients 8 --behaviour random \
        --rollout_steps 20000 --n_behavior_epochs 1500 \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from fedguide.envs.halfcheetah_hetero import make_hetero_halfcheetah_env_from_metadata
from fedguide.guidance.diffusion_prior import DiffusionGuidance
from fedguide.guidance.model import SDICE_Critic


# ---------- offline data collection ----------------------------------------

class _SADataset(Dataset):
    """(s, a) flat tensors for prior pretrain."""

    def __init__(self, s, a):
        self.s = torch.as_tensor(np.asarray(s), dtype=torch.float32)
        self.a = torch.as_tensor(np.asarray(a), dtype=torch.float32)

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, i):
        return torch.cat([self.s[i], self.a[i]], dim=-1)


class _TransitionDataset(Dataset):
    """(s, a, r, s_next, done) for SDICE."""

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


def _collect_random_rollouts(env, total_steps: int, seed: int = 0):
    """Collect total_steps transitions from a random policy."""
    s_buf, a_buf, r_buf, s2_buf, d_buf = [], [], [], [], []
    obs, _ = env.reset(seed=seed)
    rng = np.random.RandomState(seed)
    for _ in range(total_steps):
        a = env.action_space.sample()
        out = env.step(a)
        if len(out) == 5:
            obs2, r, terminated, truncated, _ = out
            done = bool(terminated) or bool(truncated)
        else:
            obs2, r, done, _ = out
        s_buf.append(np.asarray(obs, dtype=np.float32))
        a_buf.append(np.asarray(a, dtype=np.float32))
        r_buf.append(float(r))
        s2_buf.append(np.asarray(obs2, dtype=np.float32))
        d_buf.append(1.0 if done else 0.0)
        obs = obs2
        if done:
            obs, _ = env.reset()
    return np.stack(s_buf), np.stack(a_buf), np.array(r_buf), np.stack(s2_buf), np.array(d_buf)


def _collect_ppo_rollouts(env, total_steps: int, ppo_steps: int, seed: int):
    """Train a quick PPO on this env, then collect rollouts."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        raise RuntimeError(
            "stable-baselines3 not installed; install it or use --behaviour random"
        ) from e
    venv = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", venv, verbose=0, seed=seed, n_steps=512, batch_size=64)
    model.learn(total_timesteps=int(ppo_steps))

    s_buf, a_buf, r_buf, s2_buf, d_buf = [], [], [], [], []
    obs = venv.reset()
    for _ in range(total_steps):
        a, _ = model.predict(obs, deterministic=False)
        nobs, r, done, _info = venv.step(a)
        s_buf.append(np.asarray(obs[0], dtype=np.float32))
        a_buf.append(np.asarray(a[0], dtype=np.float32))
        r_buf.append(float(r[0]))
        s2_buf.append(np.asarray(nobs[0], dtype=np.float32))
        d_buf.append(1.0 if bool(done[0]) else 0.0)
        obs = nobs
    return np.stack(s_buf), np.stack(a_buf), np.array(r_buf), np.stack(s2_buf), np.array(d_buf)


# ---------- guidance helper -------------------------------------------------

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


# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_path", type=str, default="data/halfcheetah/metadata_mild64.json")
    ap.add_argument("--num_clients", type=int, default=8)
    ap.add_argument("--first_client_id", type=int, default=0)
    ap.add_argument("--rollout_steps", type=int, default=20000)
    ap.add_argument("--behaviour", type=str, default="random",
                    choices=["random", "ppo"])
    ap.add_argument("--ppo_steps", type=int, default=50_000)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    # Diffusion prior args
    ap.add_argument("--n_behavior_epochs", type=int, default=1500)
    ap.add_argument("--save_interval", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dg_hidden_dim", type=int, default=64)
    ap.add_argument("--dg_horizon", type=int, default=64,
                    help="UNet1D sample length; needs to leave ≥4 spatial dim after 3 /2 downblocks → horizon ≥ 64 in practice")
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # Guidance args
    ap.add_argument("--guidance_mode", type=str, default="interleave",
                    choices=["off", "warmup", "interleave"])
    ap.add_argument("--guidance_warmup_epochs", type=int, default=200)
    ap.add_argument("--guidance_interval", type=int, default=5)
    ap.add_argument("--guidance_epochs_per_call", type=int, default=1)
    ap.add_argument("--guidance_scale_init", type=float, default=0.0)
    ap.add_argument("--guidance_scale_target", type=float, default=1.0)
    ap.add_argument("--guidance_scale_warmup_epochs", type=int, default=50)
    ap.add_argument("--q_ensemble_num", type=int, default=0)
    ap.add_argument("--value_lr", type=float, default=1e-4)
    ap.add_argument("--wt_lr", type=float, default=1e-4)
    ap.add_argument("--min_value_lr", type=float, default=1e-5)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--hidden_dim", type=int, default=256)

    ap.add_argument("--save_root", type=str, default="./model/models_prior")
    ap.add_argument("--env_name", type=str, default="HalfCheetah")

    args = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    metadata_path = str(_PROJECT_ROOT / args.metadata_path)
    print(f"[pretrain_halfcheetah] using metadata: {metadata_path}")
    print(f"[pretrain_halfcheetah] device: {device}")

    for offset in range(args.num_clients):
        client_id = args.first_client_id + offset
        print(f"\n[Pretrain] HalfCheetah client {client_id} | behaviour={args.behaviour}")

        env = make_hetero_halfcheetah_env_from_metadata(
            metadata_path, client_id, seed=args.seed + client_id
        )
        s_dim = int(env.observation_space.shape[0])
        a_dim = int(env.action_space.shape[0])

        # 1) Collect offline buffer.
        if args.behaviour == "ppo":
            print(f"  [behaviour=ppo] training PPO for {args.ppo_steps} steps then collecting {args.rollout_steps}...")
            s, a, r, s2, d = _collect_ppo_rollouts(env, args.rollout_steps, args.ppo_steps, seed=args.seed + client_id)
        else:
            print(f"  [behaviour=random] collecting {args.rollout_steps} steps...")
            s, a, r, s2, d = _collect_random_rollouts(env, args.rollout_steps, seed=args.seed + client_id)
        print(f"  buffer: s={s.shape}, a={a.shape}, mean_r={r.mean():.3f}")

        # 2) Pretrain DiffusionGuidance on (s, a).
        prior = DiffusionGuidance(
            state_dim=s_dim, action_dim=a_dim,
            hidden_dim=args.dg_hidden_dim,
            timesteps=args.num_train_timesteps,
            horizon=args.dg_horizon,
        ).to(device)
        opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        sa_loader = DataLoader(_SADataset(s, a), batch_size=args.batch_size, shuffle=True, drop_last=True)

        # 3) Optional SDICE_Critic guidance.
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

        save_dir = Path(args.save_root) / args.env_name / f"client_{client_id}"
        (save_dir / "final").mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.n_behavior_epochs + 1):
            prior.train()
            total = 0.0
            n = 0
            for batch in sa_loader:
                # batch is concat(s, a). Re-split.
                s_b = batch[:, :s_dim].to(device)
                a_b = batch[:, s_dim:].to(device)
                opt.zero_grad()
                # log_prob is @torch.no_grad — use update method instead.
                loss = prior.update(s_b, a_b, lr=args.lr)  # update() does its own backward
                total += loss
                n += 1
            avg_loss = total / max(1, n)

            if args.guidance_mode == "interleave" and sdice is not None:
                if epoch % args.guidance_interval == 0:
                    if args.guidance_scale_warmup_epochs > 0:
                        ratio = min(1.0, epoch / args.guidance_scale_warmup_epochs)
                        sdice.guidance_scale = args.guidance_scale_init + \
                            (args.guidance_scale_target - args.guidance_scale_init) * ratio
                    for _ in range(args.guidance_epochs_per_call):
                        _train_guidance_one_epoch(sdice, gd_loader, device, do_update_v0=True, do_update_wt=True)

            if epoch % 50 == 0 or epoch == args.n_behavior_epochs:
                msg = f"  [client {client_id}] epoch {epoch}/{args.n_behavior_epochs} | prior_loss={avg_loss:.6f}"
                if sdice is not None:
                    msg += f" | guidance_scale={sdice.guidance_scale:.3f}"
                print(msg)

        if args.guidance_mode == "warmup" and sdice is not None and args.guidance_warmup_epochs > 0:
            print(f"  [client {client_id}] guidance warmup: {args.guidance_warmup_epochs} epochs")
            for e in range(1, args.guidance_warmup_epochs + 1):
                ratio = min(1.0, e / max(1, args.guidance_scale_warmup_epochs))
                sdice.guidance_scale = args.guidance_scale_init + \
                    (args.guidance_scale_target - args.guidance_scale_init) * ratio
                _train_guidance_one_epoch(sdice, gd_loader, device, do_update_v0=True, do_update_wt=True)

        torch.save({
            "prior": prior.state_dict(),
            "state_dim": s_dim,
            "action_dim": a_dim,
            "hidden_dim": args.dg_hidden_dim,
            "timesteps": args.num_train_timesteps,
            "horizon": args.dg_horizon,
            "prior_type": "diffusion",
        }, save_dir / "final" / "torch_prior.pth")
        if sdice is not None:
            torch.save(sdice.state_dict(), save_dir / "final" / "guidance_sdice.pth")
            print(f"[client {client_id}] saved prior + guidance to {save_dir/'final'}")
        else:
            print(f"[client {client_id}] saved prior to {save_dir/'final'}")

        try:
            env.close()
        except Exception:
            pass

    print("\n[Pretrain] All clients done.")


if __name__ == "__main__":
    main()
