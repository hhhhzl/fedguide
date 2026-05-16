"""Unified post-train eval/return plots: per-env curves + cross-env summary.

For each (env, algo), aggregate over seeds: mean ±1 SE on
`metrics_distributed_fit['eval/return']`. Smoothed window=5.

Outputs:
    plots/posttrain/<env>_posttrain.{png,pdf}   — one panel per env
    plots/posttrain/all_envs_grid.{png,pdf}     — N×1 grid of all envs
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "fedguide-matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt


def _install_numpy_pickle_aliases():
    """Allow numpy-2 pickles to load in older numpy environments."""
    try:
        import numpy.core as np_core
        import numpy.core.multiarray as np_multiarray
        import numpy.core.numeric as np_numeric
        import numpy.core.umath as np_umath
    except Exception:
        return
    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)
    sys.modules.setdefault("numpy._core.umath", np_umath)


_install_numpy_pickle_aliases()


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
BANDIT2D_ALGOS = [
    ("fedavg", "FedAvg", "#1f77b4"),
    ("fedguide_p", "FedGuide-p", "#d62728"),
    ("fedguide_a", "FedGuide-a", "#ff7f0e"),
    ("fedguide", "FedGuide", "#2ca02c"),
]

METRICS_ROOT = Path("metrics")
OUT_DIR      = Path("plots/posttrain")
BANDIT2D_METRICS_ROOT = Path("metrics/bandit2d")


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


def _parse_seed_arg(value: str) -> list[int]:
    return [int(s.strip()) for s in value.split(",") if s.strip()]


def discover_bandit2d_seeds(algos=None) -> list[int]:
    algos = [a for a, _, _ in BANDIT2D_ALGOS] if algos is None else algos
    seeds: set[int] = set()
    for algo in algos:
        algo_dir = BANDIT2D_METRICS_ROOT / algo
        if not algo_dir.exists():
            continue
        for path in algo_dir.glob("seed_*"):
            try:
                seeds.add(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return sorted(seeds)


def load_bandit2d_curve(algo: str, seed: int, metric: str):
    p = BANDIT2D_METRICS_ROOT / algo / f"seed_{seed}" / "training_history.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            h = pickle.load(f)
    except Exception as exc:
        print(f"[bandit2d] could not load {p}: {exc}")
        return None
    ev = getattr(h, "metrics_distributed_fit", {}).get(metric, [])
    if not ev:
        return None
    rounds = np.asarray([int(r) for (r, _) in ev], dtype=int)
    vals = np.asarray([float(v) for (_, v) in ev], dtype=float)
    return rounds, vals


def _align_curves(curves):
    if not curves:
        return None
    common_rounds = sorted(set(curves[0][0]).intersection(*[set(c[0]) for c in curves[1:]]))
    if not common_rounds:
        return None
    stack = []
    for r_arr, v_arr in curves:
        d = dict(zip(r_arr.tolist(), v_arr.tolist()))
        stack.append(np.asarray([d[r] for r in common_rounds], dtype=float))
    return np.asarray(common_rounds, dtype=int), np.stack(stack, axis=0)


def _smooth_stack(stack: np.ndarray, w: int = 3):
    if stack.shape[1] < w or w <= 1:
        return stack.copy(), np.arange(stack.shape[1])
    kernel = np.ones(w, dtype=float) / float(w)
    out = np.stack([np.convolve(row, kernel, mode="valid") for row in stack], axis=0)
    return out, np.arange(w - 1, stack.shape[1])


def plot_bandit2d_returns(seeds: list[int], out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [("eval/return", "eval/return"), ("train/return", "train/return")]
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), squeeze=False)

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        plotted = 0
        for algo, label, color in BANDIT2D_ALGOS:
            curves = []
            used_seeds = []
            for seed in seeds:
                c = load_bandit2d_curve(algo, seed, metric)
                if c is not None:
                    curves.append(c)
                    used_seeds.append(seed)
            aligned = _align_curves(curves)
            if aligned is None:
                continue
            rounds, stack = aligned
            smooth_stack, idx = _smooth_stack(stack, w=15)
            xs = rounds[idx]
            mean = smooth_stack.mean(axis=0)
            lo = smooth_stack.min(axis=0)
            hi = smooth_stack.max(axis=0)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.16, linewidth=0)
            ax.plot(xs, mean, color=color, linewidth=2.0,
                    label=f"{label} (n={len(used_seeds)})")
            plotted += 1

        if plotted == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(f"Bandit2D {title}")
        ax.set_xlabel("Round")
        ax.set_ylabel(title)
        if metric == "eval/return":
            ax.axhline(1.0, color="0.35", alpha=0.8, linestyle="--", linewidth=1.0)
            ax.text(0.99, 0.96, "upper bound = 1", color="0.25",
                    ha="right", va="top", transform=ax.transAxes, fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, -0.03), fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    paths = []
    for ext in ("png", "pdf"):
        out = out_dir / f"bandit2d_returns_all_seeds.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else None, bbox_inches="tight")
        print(f"  -> {out}")
        paths.append(out)
    plt.close(fig)
    return paths


def _import_bandit2d_viz_modules():
    bandit_script_dir = Path(__file__).resolve().parent / "envs" / "bandit2d"
    if str(bandit_script_dir) not in sys.path:
        sys.path.insert(0, str(bandit_script_dir))
    import viz_priors
    import viz_distribution
    return viz_priors, viz_distribution


def run_bandit2d(seeds: list[int]):
    if not seeds:
        print("[bandit2d] no seeds found under metrics/bandit2d")
        return

    viz_priors, viz_distribution = _import_bandit2d_viz_modules()

    print("=== bandit2d OT-MoE global prior ===")
    viz_priors.plot_global_prior(
        metrics_path="metrics/bandit2d/fedguide_p/seed_0/bandit2d_metrics.pkl",
        metadata="data/bandit2d/metadata.json",
        out="plots/bandit2d_priors/aggregate_prior.png",
        grid=240,
        bound=1.5,
        source="ring",
        write_pdf=True,
    )

    print("\n=== bandit2d policy distributions ===")
    viz_distribution.plot_bandit2d_policy_distributions(
        metrics_root=str(BANDIT2D_METRICS_ROOT),
        metadata="data/bandit2d/metadata.json",
        out_dir="plots/bandit2d_policy_density",
        algos=[a for a, _, _ in BANDIT2D_ALGOS],
        seeds=seeds,
        bound=1.5,
        write_pdf=True,
    )

    print("\n=== bandit2d return curves ===")
    plot_bandit2d_returns(seeds, OUT_DIR)


def main():
    argv = sys.argv[1:]
    seeds_explicit = any(a == "--seeds" or a.startswith("--seeds=") for a in argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandit2d", action="store_true",
                    help="generate the Bandit2D prior, policy-density, diagnostics, and all-seed return plots")
    ap.add_argument("--envs", type=str, default=",".join(ENVS_DEFAULT),
                    help="comma-separated env names")
    ap.add_argument("--seeds", type=str, default="0,1,2",
                    help="comma-separated seeds")
    args = ap.parse_args()

    if args.bandit2d:
        seeds = _parse_seed_arg(args.seeds) if seeds_explicit else discover_bandit2d_seeds()
        run_bandit2d(seeds)
        return

    envs  = [e.strip() for e in args.envs.split(",") if e.strip()]
    seeds = _parse_seed_arg(args.seeds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== per-env figures ===")
    for env in envs:
        plot_one_env(env, seeds)

    print("\n=== grid figure ===")
    plot_grid(envs, seeds)

    summary_table(envs, seeds)


if __name__ == "__main__":
    main()
