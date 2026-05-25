"""Hard-mode ablation plots: 4 envs (reacher + 3 hard MuJoCo) x 2 metrics.

Mirrors `plot_posttrain.py --main` style but for the "hard" ablation suite.
Reacher is loaded from `metrics/reacher/ablation/C` (the variant where all
three FedGuide variants dominate on global eval); the other three are the
`*_hard` siblings of the main suite. Reacher uses symlog because the FedAvg
and FedKL curves crash to ~-3000 while FedGuide variants sit near +100, and
linear scale flattens everything below the baselines.

Outputs (per metric_slug in {posttrain_eval_return, global_eval_return}):
    plots/ablation_hard/<metric_slug>/<env>.{png,pdf}        — one panel per env
    plots/ablation_hard/<metric_slug>/summary_2x2.{png,pdf}  — 2x2 grid w/ shared legend
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "fedguide-matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from scripts.viz.plot_posttrain import (
    MAIN_ALGOS,
    MAIN_METRICS,
    _apply_rl_style,
    _add_rl_legend,
    _draw_env_panel,
    _legend_handles_in_order,
    _parse_seed_arg,
)


# (env_key, display_name, env_root, yscale)
# Grid layout (row-major, 2x2):
#     Reacher    | Hopper
#     Walker2D   | HalfCheetah
ABLATION_ENVS = [
    ("reacher_hard",     "Reacher-Hard",     "metrics/reacher/ablation/C", "symlog"),
    ("hopper_hard",      "Hopper-Hard",      "metrics/hopper_hard",        "linear"),
    ("walker_hard",      "Walker2D-Hard",    "metrics/walker_hard",        "linear"),
    ("halfcheetah_hard", "HalfCheetah-Hard", "metrics/halfcheetah_hard",   "linear"),
]

OUT_ROOT = Path("plots/posttrain/ablation_hard")


def plot_ablation_per_env(env_key, display_name, env_root, yscale,
                          seeds, attr, key, ylabel, out_dir: Path):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    handles = _draw_env_panel(
        ax, env_root, env_key, display_name,
        seeds, attr, key, ylabel, yscale=yscale,
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
        out = out_dir / f"{env_key}.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else None, bbox_inches="tight")
        print(f"  -> {out}")
    plt.close(fig)


def _add_2x2_legend(fig, ordered_handles):
    """Bottom legend spanning the full figure width as a 2 row x 3 col grid.

    Target visual layout:
        Row 1:  FedAvg       FedKL        FedRL
        Row 2:  FedGuide-A   FedGuide-P   FedGuide

    Matplotlib fills legends in column-major order with ncol=3, so to land
    on the target row-major appearance we permute [A, B, C, D, E, F] into
    [A, D, B, E, C, F]. mode='expand' stretches the legend across the bbox
    width so it fills the column rather than being a centered narrow box.
    """
    if not ordered_handles:
        return
    n = len(ordered_handles)
    ncol = 3
    if n == 6:
        # transpose 2x3 row-major -> column-major
        permuted = [ordered_handles[i] for i in (0, 3, 1, 4, 2, 5)]
    else:
        permuted = ordered_handles
    leg = fig.legend(
        [h for _, h in permuted],
        [l for l, _ in permuted],
        loc="upper left",
        ncol=ncol,
        bbox_to_anchor=(0.0, -0.06, 1.0, 0.09),
        mode="expand",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="0.3",
        handlelength=3.0,
        handleheight=1.2,
        handletextpad=0.7,
        columnspacing=2.5,
        borderpad=0.7,
        borderaxespad=0.0,
        prop={"size": 26},  # match posttrain summary legend (plot_posttrain._add_rl_legend)
    )
    leg.get_frame().set_linewidth(1.2)
    for line in leg.get_lines():
        line.set_linewidth(3.6)


def plot_ablation_summary_2x2(seeds, attr, key, ylabel, out_dir: Path):
    """2x2 grid of all four envs, one shared legend below."""
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), squeeze=False)
    per_panel = []
    for idx, (env_key, display_name, env_root, yscale) in enumerate(ABLATION_ENVS):
        r, c = divmod(idx, 2)
        ax = axes[r][c]
        per_panel.append(_draw_env_panel(
            ax, env_root, env_key, display_name,
            seeds, attr, key, ylabel, yscale=yscale,
        ))
        # Only label outer axes — matches the clean look of the main summary.
        if c > 0:
            ax.set_ylabel("")
        if r < 1:
            ax.set_xlabel("")
    ordered = _legend_handles_in_order(per_panel)
    fig.tight_layout()
    _add_2x2_legend(fig, ordered)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_dir / f"summary_2x2.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else None, bbox_inches="tight")
        print(f"  -> {out}")
    plt.close(fig)


def run(seeds, base_out: Path = OUT_ROOT):
    _apply_rl_style()
    base_out.mkdir(parents=True, exist_ok=True)
    for metric_slug, ylabel, (attr, key) in MAIN_METRICS:
        print(f"\n=== metric: {ylabel} ({metric_slug}) ===")
        metric_dir = base_out / metric_slug
        metric_dir.mkdir(parents=True, exist_ok=True)
        for env_key, display_name, env_root, yscale in ABLATION_ENVS:
            plot_ablation_per_env(env_key, display_name, env_root, yscale,
                                  seeds, attr, key, ylabel, metric_dir)
        print(f"-- summary 2x2 grid for {metric_slug}")
        plot_ablation_summary_2x2(seeds, attr, key, ylabel, metric_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4",
                    help="comma-separated seeds")
    args = ap.parse_args()
    run(_parse_seed_arg(args.seeds))


if __name__ == "__main__":
    main()
