"""Unified plotting & analysis tool for bandit2d Phase-1 metrics.

Subcommands:
    curves          reward curves (train/eval) over rounds, multi-algo overlay
    ring            per-client policy density grid (4 clients + mean) + multi-algo
                    side-by-side ring comparison
    prior           visualize each client's saved diffusion prior log-density
                    (the Gaussian behaviour prior dumped to ./model/models_prior_gauss)
    density_eval    integrate per-client policy density × per-client reward field
                    → markdown table written to metrics/bandit2d_phase1/SUMMARY.md
    all             run everything in sequence

Reads the Flower training_history.pkl + the bandit2d Bandit2DMetricsCollector
pickles produced by ``run_baselines.py`` / ``run_main.py``.

Examples:
    python scripts/envs/bandit2d/plots.py curves --algos fedavg fedkl fedguide_prior fedguide_pg
    python scripts/envs/bandit2d/plots.py ring   --algos fedavg fedkl fedguide_all fedguide_prior fedguide_pg
    python scripts/envs/bandit2d/plots.py density_eval
    python scripts/envs/bandit2d/plots.py all
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))


_DISPLAY = {
    "fedavg": "FedAvg",
    "fedkl": "FedKL",
    "fedrep": "FedRep",
    "fedmomentum": "FedMomentum",
    "fmarl": "FMARL",
    "fedrl": "FedRL-DDPG",
    "fedguide_prior": "FedGuide-prior (Thm 3)",
    "fedguide_pg": "FedGuide-pg (Thm 4)",
    "fedguide_all": "FedGuide-all (Thm 5)",
}


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

def _flower_series(history_path: Path) -> Dict[str, np.ndarray]:
    """metric_name -> np.array from Flower History."""
    with open(history_path, "rb") as f:
        h = pickle.load(f)
    out: Dict[str, np.ndarray] = {}
    md = getattr(h, "metrics_distributed_fit", None) or {}
    for k, pairs in md.items():
        if not pairs:
            continue
        try:
            out[k] = np.array([v for _, v in pairs], dtype=np.float64)
        except Exception:
            continue
    return out


def _round_density(metrics_pkl: Path, round_num: int = -1) -> Tuple[Optional[Dict[int, np.ndarray]], Tuple[float, float]]:
    """Per-client policy density grid for a chosen round."""
    with open(metrics_pkl, "rb") as f:
        coll = pickle.load(f)
    if hasattr(coll, "metrics_history"):
        mh = list(coll.metrics_history)
    elif isinstance(coll, dict) and "metrics_history" in coll:
        mh = list(coll["metrics_history"])
    else:
        return None, (-1.5, 1.5)
    if not mh:
        return None, (-1.5, 1.5)
    if round_num < 0:
        round_num = len(mh) - 1
    round_num = max(0, min(round_num, len(mh) - 1))
    md = mh[round_num]
    cm = md.get("client_metrics") or {}
    densities: Dict[int, np.ndarray] = {}
    for cid, m in cm.items():
        pd = m.get("policy_density")
        if pd is None:
            continue
        densities[int(cid)] = np.asarray(pd, dtype=np.float64)
    bounds = (-1.5, 1.5)
    if hasattr(coll, "bounds"):
        bounds = tuple(coll.bounds)
    return densities, bounds


# --------------------------------------------------------------------------
# subcommand: reward curves
# --------------------------------------------------------------------------

def _plot_curves(args):
    root = _PROJECT_ROOT / args.root
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for algo in args.algos:
        path = root / algo / f"seed_{args.seed}" / "training_history.pkl"
        if not path.exists():
            print(f"[curves] missing: {path}")
            continue
        s = _flower_series(path)
        tr = s.get("train/return")
        ev = s.get("eval/return")
        label = _DISPLAY.get(algo, algo)
        if tr is not None:
            axes[0].plot(tr, label=label, alpha=0.85)
        if ev is not None:
            axes[1].plot(ev, label=label, alpha=0.85)
    axes[0].set_title("train/return")
    axes[1].set_title("eval/return")
    for ax in axes:
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Bandit2D — reward curves", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = _PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[curves] saved {out}")


# --------------------------------------------------------------------------
# subcommand: ring (per-client + multi-algo side-by-side)
# --------------------------------------------------------------------------

def _draw_unit_circle(ax):
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="cyan", linewidth=1.0, alpha=0.6)


def _peak_xy(K: int = 4):
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    return np.cos(angles), np.sin(angles)


def _plot_ring(args):
    root = _PROJECT_ROOT / args.root
    rows = []
    for algo in args.algos:
        path = root / algo / f"seed_{args.seed}" / "bandit2d_metrics.pkl"
        if not path.exists():
            print(f"[ring] missing: {path}")
            continue
        densities, bounds = _round_density(path, args.round)
        if not densities:
            continue
        cids = sorted(densities.keys())
        rows.append((algo, cids, densities, bounds))
    if not rows:
        print("[ring] no data — did you run run_baselines / run_main first?")
        return

    n_rows = len(rows)
    n_cols = max(len(r[1]) for r in rows) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    px, py = _peak_xy(4)
    for i, (algo, cids, densities, bounds) in enumerate(rows):
        all_d = np.concatenate([densities[cid].ravel() for cid in cids])
        valid = all_d[~np.isnan(all_d)]
        vmax = max(float(np.percentile(valid, args.vmax_q)), 1e-8) if valid.size else 1.0
        for j, cid in enumerate(cids):
            ax = axes[i, j]
            ax.imshow(
                densities[cid], origin="lower",
                extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
                cmap="hot", vmin=0.0, vmax=vmax,
            )
            _draw_unit_circle(ax)
            ax.scatter(px, py, color="lime", marker="x", s=24, linewidths=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"client {cid}", fontsize=10)
            if j == 0:
                ax.set_ylabel(_DISPLAY.get(algo, algo), fontsize=9)
        mean_d = np.mean([densities[cid] for cid in cids], axis=0)
        ax = axes[i, n_cols - 1]
        ax.imshow(
            mean_d, origin="lower",
            extent=[bounds[0], bounds[1], bounds[0], bounds[1]],
            cmap="hot", vmin=0.0, vmax=vmax,
        )
        _draw_unit_circle(ax)
        ax.scatter(px, py, color="lime", marker="x", s=24, linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title("mean", fontsize=10)
        for j in range(len(cids), n_cols - 1):
            axes[i, j].set_visible(False)

    fig.suptitle("Bandit2D — per-client policy density (4 peaks on unit circle)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = _PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ring] saved {out}")


# --------------------------------------------------------------------------
# subcommand: prior viz (saved Gaussian behavior priors)
# --------------------------------------------------------------------------

def _plot_prior(args):
    """Visualize each client's saved Gaussian prior log-density on a 2-D grid."""
    import torch
    from fedguide.guidance.diffusion_prior import GaussianBehaviorPrior

    base = _PROJECT_ROOT / args.prior_dir / "Bandit2D"
    n = args.num_clients
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    if n == 1:
        axes = [axes]

    grid = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(grid, grid, indexing="xy")
    A = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

    px, py = _peak_xy(4)
    for cid in range(n):
        path = base / f"client_{cid}" / "final" / "torch_prior.pth"
        if not path.exists():
            print(f"[prior] missing: {path}")
            continue
        prior = GaussianBehaviorPrior(state_dim=2, action_dim=2)
        sd = torch.load(path, map_location="cpu")
        prior.load_state_dict(sd["prior"] if "prior" in sd else sd, strict=False)
        prior.to(A.device).eval()
        with torch.no_grad():
            lp = prior.log_prob(A, A).cpu().numpy().reshape(X.shape)
        ax = axes[cid]
        im = ax.imshow(lp, origin="lower", extent=[-1.5, 1.5, -1.5, 1.5], cmap="viridis")
        _draw_unit_circle(ax)
        ax.scatter(px, py, color="red", marker="x", s=18, linewidths=1.0)
        ax.set_title(f"client {cid} prior", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Bandit2D — pretrained Gaussian behaviour prior log-density", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = _PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[prior] saved {out}")


# --------------------------------------------------------------------------
# subcommand: density_eval markdown table
# --------------------------------------------------------------------------

def _client_reward_field(grid_xy: np.ndarray, mu: np.ndarray, sigma: float, peak_weights: np.ndarray) -> np.ndarray:
    diffs = grid_xy[..., None, :] - mu[None, None, :, :]
    d2 = (diffs ** 2).sum(axis=-1)
    r = peak_weights[None, None, :] * np.exp(-d2 / (2 * sigma ** 2))
    return r.max(axis=-1)


def _density_eval_one(metrics_pkl: Path, K: int, sigma: float, hetero: bool, round_num: int = -1) -> Dict:
    densities, bounds = _round_density(metrics_pkl, round_num)
    if not densities:
        return {"density_eval": float("nan"), "n_clients": 0, "round": round_num}
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    mu = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    per_client: Dict[int, float] = {}
    for cid, pd in densities.items():
        pd = np.asarray(pd, dtype=np.float64)
        H, W = pd.shape
        xs = np.linspace(bounds[0], bounds[1], W)
        ys = np.linspace(bounds[0], bounds[1], H)
        XY = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)
        if hetero:
            wts = np.full(K, 0.1, dtype=np.float32)
            wts[int(cid) % K] = 1.0
        else:
            wts = np.ones(K, dtype=np.float32)
        R = _client_reward_field(XY, mu, sigma, wts)
        dx = (bounds[1] - bounds[0]) / max(W - 1, 1)
        dy = (bounds[1] - bounds[0]) / max(H - 1, 1)
        z = pd * dx * dy
        s = float(z.sum())
        z = z / s if s > 1e-12 else np.zeros_like(z)
        per_client[int(cid)] = float((z * R).sum())
    if not per_client:
        return {"density_eval": float("nan"), "n_clients": 0, "round": round_num}
    return {
        "density_eval": float(np.mean(list(per_client.values()))),
        "per_client": per_client,
        "n_clients": len(per_client),
        "round": round_num,
    }


def _plot_density_eval(args):
    root = _PROJECT_ROOT / args.root
    if args.algos:
        algos = list(args.algos)
    else:
        algos = sorted([p.name for p in root.iterdir() if p.is_dir()])
    rows = []
    for algo in algos:
        p = root / algo / f"seed_{args.seed}" / "bandit2d_metrics.pkl"
        if not p.exists():
            continue
        res = _density_eval_one(p, args.K, args.sigma, args.hetero, args.round)
        rows.append((algo, res))

    out_lines = ["# Bandit2D density-eval summary", ""]
    out_lines.append(
        f"`density_eval` = ∑ π(a|s) · R(a) over the 200×200 action grid for each "
        f"client, averaged across clients. σ={args.sigma}, hetero peak-weights = "
        f"`{1.0 if args.hetero else 1.0}/{0.1 if args.hetero else 1.0}`.\n"
    )
    out_lines.append("| algo | mean | per-client (c0/c1/c2/c3) | balance (std/mean) |")
    out_lines.append("|---|---|---|---|")
    for algo, res in rows:
        if res["n_clients"] == 0:
            out_lines.append(f"| {algo} | — | — | — |")
            continue
        pc = res["per_client"]
        cells = " / ".join(f"{pc[k]:.3f}" for k in sorted(pc.keys()))
        vals = np.array(list(pc.values()))
        bal = float(vals.std() / max(vals.mean(), 1e-9))
        out_lines.append(f"| {_DISPLAY.get(algo, algo)} | {res['density_eval']:.4f} | {cells} | {bal:.2f} |")

    text = "\n".join(out_lines) + "\n"
    print(text)
    out = _PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"[density_eval] wrote {out}")


# --------------------------------------------------------------------------
# subcommand: all (run everything)
# --------------------------------------------------------------------------

def _plot_all(args):
    # default algos = baseline + main if not specified
    if not args.algos:
        args.algos = ["fedavg", "fedkl", "fedrep", "fedmomentum", "fmarl",
                      "fedguide_prior", "fedguide_pg", "fedguide_all"]
    args.out = "plots/bandit2d_phase1/curves.png"; _plot_curves(args)
    args.out = "plots/bandit2d_phase1/ring_comparison.png"; _plot_ring(args)
    args.out = "plots/bandit2d_phase1/prior_density.png"; _plot_prior(args)
    args.out = "metrics/bandit2d_phase1/SUMMARY.md"; _plot_density_eval(args)


# --------------------------------------------------------------------------
# argparse glue
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="metrics/bandit2d_phase1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--round", type=int, default=-1)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--hetero", action="store_true",
                    help="density_eval: use per-client preferred-peak weights")
    ap.add_argument("--prior_dir", type=str, default="model/models_prior_gauss")
    ap.add_argument("--num_clients", type=int, default=4)
    ap.add_argument("--vmax_q", type=float, default=99.0)

    sub = ap.add_subparsers(dest="cmd", required=True)
    p_curves = sub.add_parser("curves")
    p_curves.add_argument("--algos", type=str, nargs="+",
                          default=["fedavg", "fedkl", "fedguide_prior", "fedguide_pg", "fedguide_all"])
    p_curves.add_argument("--out", type=str, default="plots/bandit2d_phase1/curves.png")
    p_curves.set_defaults(func=_plot_curves)

    p_ring = sub.add_parser("ring")
    p_ring.add_argument("--algos", type=str, nargs="+",
                        default=["fedavg", "fedkl", "fedguide_all", "fedguide_prior", "fedguide_pg"])
    p_ring.add_argument("--out", type=str, default="plots/bandit2d_phase1/ring_comparison.png")
    p_ring.set_defaults(func=_plot_ring)

    p_prior = sub.add_parser("prior")
    p_prior.add_argument("--out", type=str, default="plots/bandit2d_phase1/prior_density.png")
    p_prior.set_defaults(func=_plot_prior)

    p_de = sub.add_parser("density_eval")
    p_de.add_argument("--algos", type=str, nargs="+", default=None)
    p_de.add_argument("--out", type=str, default="metrics/bandit2d_phase1/SUMMARY.md")
    p_de.set_defaults(func=_plot_density_eval)

    p_all = sub.add_parser("all")
    p_all.add_argument("--algos", type=str, nargs="+", default=None)
    p_all.add_argument("--out", type=str, default=None)
    p_all.set_defaults(func=_plot_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
