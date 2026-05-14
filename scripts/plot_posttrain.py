"""Unified post-train eval/return plots: per-env curves + cross-env summary.

For each (env, algo), aggregate over seeds: mean ±1 SE on
`metrics_distributed_fit['eval/return']`. Smoothed window=5.

Outputs:
    plots/posttrain/<env>_posttrain.{png,pdf}   — one panel per env
    plots/posttrain/all_envs_grid.{png,pdf}     — N×1 grid of all envs
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# Display name + color for each algorithm.
ALGOS = [
    ("fedavg",     "FedAvg",      "#ca6c0f"),
    ("fedkl",      "FedKL",       "#888888"),
    ("ppo",        "PPO",         "#444444"),
    ("fedrl",      "FedRL-DDPG",  "#1f77b4"),
    ("fedguide_p", "FedGuide-p",  "#6f4ec5"),
    ("fedguide",   "FedGuide",    "#1f6b3f"),
    ("fedguide_a", "FedGuide-a",  "#a3214c"),
]

ENVS_DEFAULT = ["halfcheetah", "walker", "hopper", "reacher", "metaworld"]

METRICS_ROOT = Path("metrics")
OUT_DIR      = Path("plots/posttrain")


def load_seed_curve(env: str, algo: str, seed: int):
    p = METRICS_ROOT / f"{env}_phase1" / algo / f"seed_{seed}" / "training_history.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        h = pickle.load(f)
    ev = h.metrics_distributed_fit.get("eval/return", [])
    if not ev:
        return None
    rounds = [int(r) for (r, _) in ev]
    vals   = [float(v) for (_, v) in ev]
    return np.asarray(rounds), np.asarray(vals)


def smooth(x: np.ndarray, w: int = 5):
    if len(x) < w:
        return x.copy(), np.arange(len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid"), np.arange(w - 1, len(x))


def aggregate_across_seeds(env: str, algo: str, seeds):
    curves = []
    for s in seeds:
        c = load_seed_curve(env, algo, s)
        if c is not None:
            curves.append(c)
    if not curves:
        return None
    # Align by intersection of rounds
    common_rounds = sorted(set(curves[0][0]).intersection(*[set(c[0]) for c in curves[1:]]))
    if not common_rounds:
        return None
    rounds = np.asarray(common_rounds)
    stack = []
    for r_arr, v_arr in curves:
        d = dict(zip(r_arr.tolist(), v_arr.tolist()))
        stack.append(np.asarray([d[r] for r in common_rounds]))
    stack = np.stack(stack, axis=0)  # (n_seeds, n_rounds)
    mean = stack.mean(axis=0)
    se = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 else np.zeros_like(mean)
    return rounds, mean, se, stack.shape[0]


def plot_one_env(env: str, seeds, ax=None):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    plotted = 0
    for algo, label, color in ALGOS:
        agg = aggregate_across_seeds(env, algo, seeds)
        if agg is None:
            continue
        rounds, mean, se, n_seeds = agg
        smean, idx = smooth(mean, 5)
        sse,   _   = smooth(se, 5)
        xs = rounds[idx]
        suffix = f" (n={n_seeds})"
        ax.plot(xs, smean, color=color, linewidth=2.2, label=label + suffix)
        if (sse > 0).any():
            ax.fill_between(xs, smean - sse, smean + sse, color=color, alpha=0.15)
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Round")
    ax.set_ylabel("post-train eval/return")
    ax.set_title(env)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    if own_fig:
        fig.tight_layout()
        for ext in ("png", "pdf"):
            out = OUT_DIR / f"{env}_posttrain.{ext}"
            fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
            print(f"  → {out}")
        plt.close(fig)


def plot_grid(envs, seeds):
    n = len(envs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)
    for i, env in enumerate(envs):
        r, c = divmod(i, cols)
        plot_one_env(env, seeds, ax=axes[r][c])
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)
    fig.suptitle("Post-train local eval/return across heterogeneous RL environments", fontsize=13, y=1.005)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"all_envs_grid.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"  → {out}")
    plt.close(fig)


def summary_table(envs, seeds):
    print(f"\n{'env':<14} {'algo':<14} {'n_seeds':>8} {'mean(last10)':>14} {'max':>10}")
    print("-" * 75)
    for env in envs:
        for algo, label, _ in ALGOS:
            agg = aggregate_across_seeds(env, algo, seeds)
            if agg is None:
                continue
            rounds, mean, se, n = agg
            last10 = mean[-10:].mean() if len(mean) >= 10 else mean.mean()
            print(f"{env:<14} {label:<14} {n:>8} {last10:>14.1f} {mean.max():>10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=str, default=",".join(ENVS_DEFAULT),
                    help="comma-separated env names")
    ap.add_argument("--seeds", type=str, default="0,1,2",
                    help="comma-separated seeds")
    args = ap.parse_args()
    envs  = [e.strip() for e in args.envs.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== per-env figures ===")
    for env in envs:
        plot_one_env(env, seeds)

    print("\n=== grid figure ===")
    plot_grid(envs, seeds)

    summary_table(envs, seeds)


if __name__ == "__main__":
    main()
