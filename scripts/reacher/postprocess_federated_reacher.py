#!/usr/bin/env python3
"""
Post-process federated Reacher (FedKL / FedAvg / FedRep) runs:

1) Plot loss vs round and eval/return vs round from:
   - Flower training_history.pkl (preferred when metrics_distributed_fit is populated), or
   - training log: "[Round k] Aggregated metrics:" blocks (FedKL), or
   - FedRep-style client lines: "Round k: loss = ... eval_return = ..." (mean per round).
   If the pickle has no points (common for FedRep), falls back to
   logs/reacher_<prefix>_seed0_cuda_chain.log when present.
2) Render a 2x4 montage GIF: same global policy on each client's Reacher env (FedKL only).

Montage requires final_global_policy_flat.pkl in metrics_dir (FedKL saves on last round).
FedRep does not emit this file yet; use --skip-montage or ignore the warning.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def _history_metric_pairs(history, key: str) -> Dict[int, float]:
    metrics = getattr(history, "metrics_distributed_fit", None) or getattr(
        history, "metrics_centralized_fit", {}
    )
    if not metrics or key not in metrics or not metrics[key]:
        return {}
    return {int(rnd): float(val) for rnd, val in metrics[key]}


def load_curves_from_history_pkl(path: Path) -> Tuple[List[int], List[float], List[float]]:
    with open(path, "rb") as f:
        history = pickle.load(f)
    loss_m = _history_metric_pairs(history, "loss")
    ev_m = _history_metric_pairs(history, "eval/return")
    rounds = sorted(set(loss_m.keys()) | set(ev_m.keys()))
    losses = [loss_m.get(r, float("nan")) for r in rounds]
    evals = [ev_m.get(r, float("nan")) for r in rounds]
    return rounds, losses, evals


def _parse_aggregated_metric_blocks(text: str) -> Tuple[List[int], List[float], List[float]]:
    """FedKL-style server log: `[Round k] Aggregated metrics:` + loss / eval/return lines."""
    rounds: List[int] = []
    losses: List[float] = []
    evals: List[float] = []

    i = 0
    header_re = re.compile(r"\[Round\s+(\d+)\]\s+Aggregated metrics:")
    loss_re = re.compile(r"^\s*loss:\s*([-\d.]+|N/A)\s*$", re.MULTILINE)
    eval_re = re.compile(r"^\s*eval/return:\s*([-\d.]+)\s*$", re.MULTILINE)

    while True:
        m = header_re.search(text, i)
        if not m:
            break
        rnd = int(m.group(1))
        block_start = m.end()
        next_m = header_re.search(text, block_start)
        block = text[block_start : next_m.start()] if next_m else text[block_start:]

        lm = loss_re.search(block)
        em = eval_re.search(block)
        if lm:
            loss_s = lm.group(1)
            loss_v = float("nan") if loss_s == "N/A" else float(loss_s)
        else:
            loss_v = float("nan")
        ev_v = float(em.group(1)) if em else float("nan")

        rounds.append(rnd)
        losses.append(loss_v)
        evals.append(ev_v)
        i = next_m.start() if next_m else len(text)
        if not next_m:
            break

    return rounds, losses, evals


def _parse_fedrep_style_client_lines(text: str) -> Tuple[List[int], List[float], List[float]]:
    """
    FedRep (and similar) client lines, e.g.:
    [FedRepClient ...] Round 12: loss = 30.68, train_return = ..., eval_return = -17.76, success = True
    Averages per round over all matching lines (Ray may deduplicate some lines).
    """
    line_re = re.compile(
        r"Round\s+(\d+):\s+loss\s*=\s*([-\d.eE+]+),[^\n]*eval_return\s*=\s*([-\d.eE+]+)",
    )
    loss_by_r: Dict[int, List[float]] = defaultdict(list)
    ev_by_r: Dict[int, List[float]] = defaultdict(list)
    for m in line_re.finditer(text):
        r = int(m.group(1))
        loss_by_r[r].append(float(m.group(2)))
        ev_by_r[r].append(float(m.group(3)))
    rounds = sorted(set(loss_by_r.keys()) | set(ev_by_r.keys()))
    losses = [float(np.mean(loss_by_r[r])) if loss_by_r[r] else float("nan") for r in rounds]
    evals = [float(np.mean(ev_by_r[r])) if ev_by_r[r] else float("nan") for r in rounds]
    return rounds, losses, evals


def parse_curves_from_log(path: Path) -> Tuple[List[int], List[float], List[float]]:
    text = strip_ansi(path.read_text(encoding="utf-8", errors="ignore"))
    rounds, losses, evals = _parse_aggregated_metric_blocks(text)
    if rounds:
        return rounds, losses, evals
    return _parse_fedrep_style_client_lines(text)


def plot_curves(
    rounds: List[int],
    losses: List[float],
    evals: List[float],
    out_dir: Path,
    prefix: str = "fedrun",
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, losses, "b.-", label="loss (aggregated)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.set_title("Aggregated loss vs round")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_loss_vs_round.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, evals, "g.-", label="eval/return (aggregated)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Eval return")
    ax.set_title("Aggregated eval/return vs round")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_eval_return_vs_round.png", dpi=150)
    plt.close(fig)
    print(f"Wrote plots under {out_dir}")


def _flat_to_agent_dict(agent, flat: List[np.ndarray]):
    import torch

    policy_state = agent.policy.state_dict()
    new_policy_state = {}
    idx = 0
    for k, v in policy_state.items():
        new_policy_state[k] = torch.tensor(flat[idx], dtype=v.dtype)
        idx += 1
    return {"policy": new_policy_state, "log_std": torch.tensor(flat[idx])}


def render_montage_gif(
    policy_pkl: Path,
    metadata_path: Path,
    out_gif: Path,
    num_clients: int = 8,
    device: str = "cpu",
    fps: int = 10,
) -> None:
    from PIL import Image

    from fedguide.baselines.fedKL.agent import FedKLAgent
    from fedguide.envs.reacher import make_hetero_reacher_env_from_metadata

    with open(policy_pkl, "rb") as f:
        flat = pickle.load(f)

    envs = [
        make_hetero_reacher_env_from_metadata(
            str(metadata_path), i, seed=1000 + i, render_mode="rgb_array"
        )
        for i in range(num_clients)
    ]
    obs_dims = [int(e.observation_space.shape[0]) for e in envs]
    act_dims = [int(e.action_space.shape[0]) for e in envs]
    if len(set(obs_dims)) != 1 or len(set(act_dims)) != 1:
        raise ValueError(f"Hetero env dims differ: obs={obs_dims} act={act_dims}")
    obs_dim, act_dim = obs_dims[0], act_dims[0]

    agent = FedKLAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=256,
        lr=3e-4,
        device=device,
    )
    agent.set_parameters(_flat_to_agent_dict(agent, flat))

    obs_list: List[np.ndarray] = []
    last_frames: List[np.ndarray] = []
    for env in envs:
        o, _ = env.reset(seed=None)
        obs_list.append(np.asarray(o, dtype=np.float32))
        fr = env.render()
        last_frames.append(
            np.asarray(fr if fr is not None else np.zeros((480, 480, 3), dtype=np.uint8))
        )

    active = [True] * num_clients
    montage_frames: List[Image.Image] = []

    def resize_cell(arr: np.ndarray, tw: int, th: int) -> Image.Image:
        im = Image.fromarray(arr)
        return im.resize((tw, th), Image.Resampling.LANCZOS)

    h0, w0 = last_frames[0].shape[0], last_frames[0].shape[1]
    tw, th = max(160, w0 // 2), max(120, h0 // 2)

    def frames_to_montage(frames: List[np.ndarray]) -> Image.Image:
        cells = [resize_cell(f, tw, th) for f in frames]
        row1 = np.hstack([np.asarray(c) for c in cells[:4]])
        row2 = np.hstack([np.asarray(c) for c in cells[4:]])
        return Image.fromarray(np.vstack([row1, row2]))

    montage_frames.append(frames_to_montage(last_frames))

    while any(active):
        for i in range(num_clients):
            if not active[i]:
                continue
            a, _, _ = agent.select_action(obs_list[i], deterministic=True)
            a = np.asarray(a)[0] if np.asarray(a).ndim > 1 else a
            obs_list[i], _r, term, trunc, _ = envs[i].step(a)
            done = bool(term) or bool(trunc)
            fr = envs[i].render()
            last_frames[i] = np.asarray(
                fr if fr is not None else np.zeros((h0, w0, 3), dtype=np.uint8)
            )
            if done:
                active[i] = False
        montage_frames.append(frames_to_montage(last_frames))

    for env in envs:
        env.close()

    if not montage_frames:
        raise RuntimeError("No frames captured for GIF")

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / max(1, fps))
    montage_frames[0].save(
        out_gif,
        save_all=True,
        append_images=montage_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Wrote montage GIF to {out_gif}")


def resolve_history_pkl(metrics_dir: Optional[Path]) -> Optional[Path]:
    if not metrics_dir:
        return None
    metrics_dir = metrics_dir.resolve()
    cand = metrics_dir / "training_history.pkl"
    if cand.is_file():
        return cand
    subs = sorted(metrics_dir.glob("seed_*/training_history.pkl"))
    return subs[-1] if subs else None


def resolve_default_log_for_prefix(prefix: str) -> Optional[Path]:
    """Typical unified-runner logs: logs/reacher_<prefix>_seed0_cuda_chain.log."""
    for name in (
        f"reacher_{prefix}_seed0_cuda_chain.log",
        f"reacher_{prefix}_seed0_cuda.log",
    ):
        p = _ROOT / "logs" / name
        if p.is_file():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help="e.g. metrics/reacher/fedavg/seed_0",
    )
    ap.add_argument("--history-pkl", type=str, default=None)
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="plots/reacher/fedavg_post")
    ap.add_argument("--prefix", type=str, default="fedavg")
    ap.add_argument("--policy-pkl", type=str, default=None)
    ap.add_argument("--metadata", type=str, default="data/reacher/metadata.json")
    ap.add_argument("--num-clients", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--skip-montage", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else None

    hist = Path(args.history_pkl) if args.history_pkl else None
    if hist is None and metrics_dir:
        hist = resolve_history_pkl(metrics_dir)

    if not args.skip_plots:
        rounds: List[int] = []
        losses: List[float] = []
        evals: List[float] = []
        if hist and hist.is_file():
            rounds, losses, evals = load_curves_from_history_pkl(hist)
            print(f"Loaded curves from {hist} ({len(rounds)} points)")
        if not rounds:
            log_path: Optional[Path] = Path(args.log) if args.log else None
            if log_path is None or not log_path.is_file():
                log_path = resolve_default_log_for_prefix(args.prefix)
            if log_path and log_path.is_file():
                rounds, losses, evals = parse_curves_from_log(log_path)
                print(f"Parsed curves from log {log_path} ({len(rounds)} points)")
            elif not rounds:
                print(
                    "No aggregated metrics in training_history.pkl (Flower often empty for FedRep) "
                    "and no usable log; pass --log or ensure logs/reacher_<prefix>_seed0_cuda_chain.log exists.",
                    file=sys.stderr,
                )
        if rounds:
            plot_curves(rounds, losses, evals, out_dir, prefix=args.prefix)

    if not args.skip_montage:
        pol = Path(args.policy_pkl) if args.policy_pkl else None
        if pol is None and metrics_dir:
            pol = (metrics_dir / "final_global_policy_flat.pkl").resolve()
        if pol and pol.is_file():
            meta = Path(args.metadata)
            if not meta.is_file():
                meta = _ROOT / args.metadata
            render_montage_gif(
                pol,
                meta,
                out_dir / f"{args.prefix}_clients_montage.gif",
                num_clients=args.num_clients,
                device=args.device,
                fps=args.gif_fps,
            )
        else:
            print(
                f"Policy file not found ({pol}). "
                "Finish a training run (saves on last round) or pass --policy-pkl.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
