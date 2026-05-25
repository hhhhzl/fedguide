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
    ("ppo",        "PPO",         "#444444"),
    ("fedavg",     "FedAvg",      "#CC79A7"),
    ("fedkl",      "FedKL",       "#9467bd"),
    ("fedrl",      "FedRL",  "#2ca02c"),
    ("fedguide_p", "FedGuide-P",  "#d62728"),
    ("fedguide",   "FedGuide",    "#1f77b4"),
    ("fedguide_a", "FedGuide-A",  "#ff7f0e"),
]

ENVS_DEFAULT = ["halfcheetah", "walker", "hopper", "reacher", "metaworld"]
BANDIT2D_ALGOS = [
    ("fedavg", "FedAvg", "#1f77b4"),
    ("fedguide_p", "FedGuide-p", "#d62728"),
    ("fedguide_a", "FedGuide-a", "#ff7f0e"),
    ("fedguide", "FedGuide", "#2ca02c"),
]

# Main-mode: the 6 algos shipped per env in metrics/<env>/<dir>/seed_*.
# Tuple is (directory name on disk, display label, color).
MAIN_ALGOS = [
    ("fedavg",     "FedAvg",     "#CC79A7"),
    ("fedkl",      "FedKL",      "#9467bd"),
    ("fedrl_ddpg", "FedRL",      "#2ca02c"),
    ("fedguide_a", "FedGuide-A", "#ff7f0e"),
    ("fedguide_p", "FedGuide-P", "#d62728"),
    ("fedguide",   "FedGuide",   "#1f77b4"),
]

MAIN_ENVS = ["reacher", "hopper", "walker", "halfcheetah", "metaworld"]

ENV_DISPLAY = {
    "reacher":     "Reacher",
    "hopper":      "Hopper",
    "walker":      "Walker2D",
    "halfcheetah": "HalfCheetah",
    "metaworld":   "MetaWorld10",
}

# (slug, ylabel, (history attribute, key inside the dict))
MAIN_METRICS = [
    ("posttrain_eval_return", "Client Average Return", ("metrics_distributed_fit", "eval/return")),
    ("global_eval_return",    "Global eval/return",    ("metrics_distributed",     "eval/return")),
]

# Per-(env, algo) smoothing window. On the 4 MuJoCo envs fedavg/fedkl have ~3-5x
# lower inherent round-to-round variance than the rest (see analysis), so w=5
# collapses them into near-flat lines; we relax them to w=1 to match the others
# visually. On metaworld all algos share similar noise (~0.21-0.25), so w=5 stays
# for everyone there.
DEFAULT_SMOOTH_W = 5
RELAXED_SMOOTH_W = 1
RELAX_BASELINES_ON = {"reacher", "hopper", "walker", "halfcheetah"}
RELAXED_BASELINES = {"fedavg", "fedkl"}


def smooth_window_for(env: str, algo: str) -> int:
    # Substring match so that hard / ablation variants (e.g. "hopper_hard",
    # "reacher/ablation/C") inherit the same baseline-relaxation rule.
    if algo in RELAXED_BASELINES and any(k in env for k in RELAX_BASELINES_ON):
        return RELAXED_SMOOTH_W
    return DEFAULT_SMOOTH_W

METRICS_ROOT = Path("metrics")
OUT_DIR      = Path("plots/posttrain")
MAIN_OUT_DIR = Path("plots/posttrain")
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
    """Trailing mean with adaptive leading-edge window.

    Output length == len(x), so every algo's curve starts at the same round
    regardless of its per-algo `w` (see smooth_window_for). Position i
    averages x[max(0, i-w+1) : i+1] — the first w-1 entries use shrunk
    windows (1, 2, ..., w-1), then it settles into the full window. With
    w=1 this is identity.

    Previously this used np.convolve(..., mode='valid'), which trimmed
    w-1 points off the left and made FedAvg/FedKL (w=1) visually have
    "extra" early-round data compared to other algos (w=5). Trimming the
    baselines erased their genuine early-training trend on Walker2D, so
    we switched to trailing-mean instead.
    """
    n = len(x)
    if n == 0 or w <= 1:
        return x.copy(), np.arange(n)
    out = np.empty(n, dtype=float)
    cumsum = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    for i in range(n):
        start = max(0, i - w + 1)
        out[i] = (cumsum[i + 1] - cumsum[start]) / (i + 1 - start)
    return out, np.arange(n)


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


def load_seed_curve_at(env_root: Path, algo_dir: str, seed: int, attr: str, key: str):
    """Load one (env_root, algo, seed) curve for the requested metric."""
    p = Path(env_root) / algo_dir / f"seed_{seed}" / "training_history.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            h = pickle.load(f)
    except Exception as exc:
        print(f"[main] could not load {p}: {exc}")
        return None
    container = getattr(h, attr, None)
    if not isinstance(container, dict):
        return None
    series = container.get(key, [])
    if not series:
        return None
    rounds = np.asarray([int(r) for (r, _) in series], dtype=int)
    vals = np.asarray([float(v) for (_, v) in series], dtype=float)
    return rounds, vals


def aggregate_at(env_root: Path, algo_dir: str, seeds, attr: str, key: str):
    curves = []
    for s in seeds:
        c = load_seed_curve_at(env_root, algo_dir, s, attr, key)
        if c is not None:
            curves.append(c)
    if not curves:
        return None
    common = sorted(set(curves[0][0]).intersection(*[set(c[0]) for c in curves[1:]]))
    if not common:
        return None
    rounds = np.asarray(common, dtype=int)
    stack = []
    for r_arr, v_arr in curves:
        d = dict(zip(r_arr.tolist(), v_arr.tolist()))
        stack.append(np.asarray([d[r] for r in common], dtype=float))
    stack = np.stack(stack, axis=0)
    mean = stack.mean(axis=0)
    se = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 else np.zeros_like(mean)
    return rounds, mean, se, stack.shape[0]


# Back-compat shims used by the existing --main flow.
def load_main_seed_curve(env, algo_dir, seed, attr, key):
    return load_seed_curve_at(METRICS_ROOT / env, algo_dir, seed, attr, key)

def aggregate_main(env, algo_dir, seeds, attr, key):
    return aggregate_at(METRICS_ROOT / env, algo_dir, seeds, attr, key)


def _draw_env_panel(ax, env_root, env_key, display_name, seeds, attr, key, ylabel, yscale="linear"):
    """Render one env panel with every MAIN_ALGOS curve. Returns {label: Line2D}."""
    handles: dict[str, object] = {}
    for algo_dir, label, color in MAIN_ALGOS:
        agg = aggregate_at(Path(env_root), algo_dir, seeds, attr, key)
        if agg is None:
            continue
        rounds, mean, se, _n_seeds = agg
        w = smooth_window_for(env_key, algo_dir)
        smean, idx = smooth(mean, w)
        sse, _ = smooth(se, w)
        xs = rounds[idx]
        line, = ax.plot(xs, smean, color=color, linewidth=2.8, label=label, solid_capstyle="round")
        if (sse > 0).any():
            ax.fill_between(xs, smean - sse, smean + sse, color=color, alpha=0.22, linewidth=0)
        handles[label] = line
    if not handles:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=10, linscale=1.0)
    elif yscale == "log":
        ax.set_yscale("log")
    ax.set_title(display_name)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.6)
    ax.tick_params(width=1.3, length=5)
    return handles


def _legend_handles_in_order(per_panel_handles):
    merged: dict[str, object] = {}
    for h in per_panel_handles:
        for k, v in h.items():
            merged.setdefault(k, v)
    ordered = [(lbl, merged[lbl]) for _, lbl, _ in MAIN_ALGOS if lbl in merged]
    return ordered


def _add_rl_legend(fig, ordered_handles, position="top"):
    """Wide horizontal legend that spans the full figure width."""
    if not ordered_handles:
        return
    if position == "top":
        # 4-tuple bbox = (x0, y0, width, height) in figure fraction; placed just
        # above the axes so mode="expand" stretches it across the whole row.
        bbox = (0.0, 1.005, 1.0, 0.09)
        loc = "lower left"
    else:
        bbox = (0.0, -0.10, 1.0, 0.09)
        loc = "upper left"
    leg = fig.legend(
        [h for _, h in ordered_handles],
        [l for l, _ in ordered_handles],
        loc=loc,
        ncol=len(ordered_handles),
        bbox_to_anchor=bbox,
        mode="expand",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="0.3",
        handlelength=4.0,
        handleheight=1.3,
        handletextpad=0.9,
        columnspacing=3.0,
        borderpad=0.9,
        borderaxespad=0.0,
        prop={"size": 26},
    )
    leg.get_frame().set_linewidth(1.3)
    for line in leg.get_lines():
        line.set_linewidth(4.2)


def plot_main_per_env(env, seeds, metric_slug, ylabel, attr, key, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    handles = _draw_env_panel(
        ax, METRICS_ROOT / env, env, ENV_DISPLAY.get(env, env),
        seeds, attr, key, ylabel,
    )
    if handles:
        ordered = [(lbl, handles[lbl]) for _, lbl, _ in MAIN_ALGOS if lbl in handles]
        leg = ax.legend(
            [h for _, h in ordered],
            [l for l, _ in ordered],
            loc="best",
            frameon=True,
            fancybox=True,
            framealpha=0.92,
            edgecolor="0.3",
            handlelength=2.4,
            handletextpad=0.6,
            prop={"size": 16},
        )
        leg.get_frame().set_linewidth(1.0)
        for line in leg.get_lines():
            line.set_linewidth(3.2)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_dir / f"{env}.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else None, bbox_inches="tight")
        print(f"  -> {out}")
    plt.close(fig)


def plot_main_summary(envs, seeds, metric_slug, ylabel, attr, key, out_dir):
    """1xN grid of envs with all algos overlaid; one bold shared legend below."""
    n = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    per_panel = []
    for i, env in enumerate(envs):
        ax = axes[0][i]
        per_panel.append(_draw_env_panel(
            ax, METRICS_ROOT / env, env, ENV_DISPLAY.get(env, env),
            seeds, attr, key, ylabel,
        ))
        if i > 0:
            ax.set_ylabel("")
    ordered = _legend_handles_in_order(per_panel)
    fig.tight_layout()
    _add_rl_legend(fig, ordered, position="bottom")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_dir / f"summary_all_algos.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else None, bbox_inches="tight")
        print(f"  -> {out}")
    plt.close(fig)


def _apply_rl_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 22,
        "font.weight": "normal",
        "axes.titlesize": 28,
        "axes.titleweight": "normal",
        "axes.labelsize": 24,
        "axes.labelweight": "normal",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 26,
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.9,
        "lines.linewidth": 2.8,
        "savefig.bbox": "tight",
    })


def run_main(envs, seeds, base_out: Path = MAIN_OUT_DIR):
    _apply_rl_style()
    base_out.mkdir(parents=True, exist_ok=True)
    for metric_slug, ylabel, (attr, key) in MAIN_METRICS:
        print(f"\n=== metric: {ylabel} ({metric_slug}) ===")
        metric_dir = base_out / metric_slug
        metric_dir.mkdir(parents=True, exist_ok=True)
        for env in envs:
            plot_main_per_env(env, seeds, metric_slug, ylabel, attr, key, metric_dir)
        print(f"-- summary grid for {metric_slug}")
        plot_main_summary(envs, seeds, metric_slug, ylabel, attr, key, metric_dir)


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
    viz_priors.plot_ground_truth_distributions(
        metadata="data/bandit2d/metadata.json",
        out="plots/bandit2d_priors/ground_truth_distributions.png",
        grid=240,
        bound=1.5,
        write_pdf=True,
    )
    viz_priors.plot_ground_truth_peaks(
        metadata="data/bandit2d/metadata.json",
        out="plots/bandit2d_priors/ground_truth_peaks.png",
        grid=240,
        bound=1.5,
        write_pdf=True,
    )
    viz_priors.plot_ground_truth_ring(
        metadata="data/bandit2d/metadata.json",
        out="plots/bandit2d_priors/ground_truth_ring.png",
        grid=240,
        bound=1.5,
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
    ap.add_argument("--main", action="store_true",
                    help="plot the 3 main metrics (post-train eval/return, global eval/return, loss) for the 6 algos x 5 envs; one panel per (algo,env) plus one unified-legend summary per metric")
    ap.add_argument("--envs", type=str, default=",".join(MAIN_ENVS),
                    help="comma-separated env names")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4",
                    help="comma-separated seeds")
    args = ap.parse_args()

    if args.bandit2d:
        seeds = _parse_seed_arg(args.seeds) if seeds_explicit else discover_bandit2d_seeds()
        run_bandit2d(seeds)
        return

    envs  = [e.strip() for e in args.envs.split(",") if e.strip()]
    seeds = _parse_seed_arg(args.seeds)

    if args.main:
        run_main(envs, seeds)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== per-env figures ===")
    for env in envs:
        plot_one_env(env, seeds)

    print("\n=== grid figure ===")
    plot_grid(envs, seeds)

    summary_table(envs, seeds)


if __name__ == "__main__":
    main()
