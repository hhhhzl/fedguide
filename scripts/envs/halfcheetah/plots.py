"""HalfCheetah Phase-1 plotting utilities (mirrors bandit2d/plots.py).

Subcommands:
    curves         — train/return + eval/return curves for selected algos.
    summary        — per-algo final / best / AUC table written to SUMMARY.md.
    list_videos    — glob the per-client mp4s saved at the final render round
                     and print one Markdown link per file (handy for embedding
                     in the README).

Usage:
    python scripts/envs/halfcheetah/plots.py curves
    python scripts/envs/halfcheetah/plots.py summary
    python scripts/envs/halfcheetah/plots.py list_videos
    python scripts/envs/halfcheetah/plots.py all          # curves + summary + list_videos
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


_DISPLAY = {
    "fedavg":         "FedAvg",
    "fedkl":          "FedKL",
    "fedrep":         "FedRep",
    "fedmomentum":    "FedMomentum",
    "fmarl":          "FMARL",
    "fedrl":          "FedRL-DDPG",
    "mfpo":           "MFPO",
    "fedguide_prior": "FedGuide-prior (Thm 3)",
    "fedguide_pg":    "FedGuide-pg (Thm 4)",
    "fedguide_all":   "FedGuide-all (Thm 5)",
}

_DEFAULT_ALGOS = [
    "fedavg", "fedkl",
    "fedguide_prior", "fedguide_pg", "fedguide_all",
]


def _flower_series(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        h = pickle.load(f)
    out: Dict[str, np.ndarray] = {}
    md = getattr(h, "metrics_distributed_fit", None) or {}
    for k, pairs in md.items():
        if not pairs:
            continue
        try:
            arr = np.array([v for _, v in pairs], dtype=np.float64)
        except Exception:
            continue
        out[k] = arr
    return out


# --------------------------------------------------------------------------
# subcommand: curves
# --------------------------------------------------------------------------

def _plot_curves(args):
    root = _PROJECT_ROOT / args.root
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    titles = {"train/return": "Train Return (sum-over-episode)",
              "eval/return":  "Eval Return (deterministic)"}

    for ax, key in zip(axes, ["train/return", "eval/return"]):
        for algo in args.algos:
            path = root / algo / f"seed_{args.seed}" / "training_history.pkl"
            if not path.exists():
                print(f"[curves] missing: {path}")
                continue
            ser = _flower_series(path)
            arr = ser.get(key)
            if arr is None or arr.size == 0:
                continue
            label = _DISPLAY.get(algo, algo)
            ax.plot(np.arange(1, arr.size + 1), arr, label=label, lw=1.6)
        ax.set_title(titles[key])
        ax.set_xlabel("round")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"HalfCheetah Phase-1 (8 hetero clients) — seed={args.seed}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = _PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[curves] saved {out}")


# --------------------------------------------------------------------------
# subcommand: summary
# --------------------------------------------------------------------------

def _agg(vals: List[float]) -> Tuple[float, float]:
    arr = np.array([v for v in vals if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def _summary(args):
    root = _PROJECT_ROOT / args.root
    out_lines = ["# HalfCheetah Phase-1 summary", ""]
    out_lines.append(
        f"Reading from `{args.root}` over seeds={args.seeds}, expected rounds={args.rounds}.\n"
    )
    out_lines.append("Mean ± std over seeds; N = number of seeds present.")
    out_lines.append("")
    out_lines.append("| algo | N | train final | train best | train AUC | eval final | eval best | eval AUC |")
    out_lines.append("|---|---|---|---|---|---|---|---|")

    def _f(p):
        m, sd = p
        return "—" if math.isnan(m) else f"{m:.2f} ± {sd:.2f}"

    for algo in args.algos:
        adir = root / algo
        if not adir.exists():
            continue
        train_finals, train_bests, train_aucs = [], [], []
        eval_finals, eval_bests, eval_aucs = [], [], []
        n_seeds = 0
        for s in args.seeds:
            p = adir / f"seed_{s}" / "training_history.pkl"
            if not p.exists():
                continue
            ser = _flower_series(p)
            tr = ser.get("train/return")
            ev = ser.get("eval/return")
            if tr is None or tr.size == 0:
                continue
            n_seeds += 1
            train_finals.append(float(tr[-1]))
            train_bests.append(float(tr.max()))
            train_aucs.append(float(tr.mean()))
            if ev is not None and ev.size:
                eval_finals.append(float(ev[-1]))
                eval_bests.append(float(ev.max()))
                eval_aucs.append(float(ev.mean()))
            else:
                eval_finals.append(float("nan"))
                eval_bests.append(float("nan"))
                eval_aucs.append(float("nan"))
        if not n_seeds:
            continue
        out_lines.append(
            f"| {_DISPLAY.get(algo, algo)} | {n_seeds} | "
            f"{_f(_agg(train_finals))} | {_f(_agg(train_bests))} | {_f(_agg(train_aucs))} | "
            f"{_f(_agg(eval_finals))} | {_f(_agg(eval_bests))} | {_f(_agg(eval_aucs))} |"
        )

    text = "\n".join(out_lines) + "\n"
    print(text)
    out_path = _PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"[summary] wrote {out_path}")


# --------------------------------------------------------------------------
# subcommand: list_videos
# --------------------------------------------------------------------------

def _list_videos(args):
    root = _PROJECT_ROOT / args.video_root
    if not root.exists():
        print(f"[list_videos] no videos under {root}")
        return
    lines = ["# HalfCheetah Phase-1 — final-round per-client videos", ""]
    for algo in sorted(p.name for p in root.iterdir() if p.is_dir()):
        mp4s = sorted((root / algo).rglob("*.mp4"))
        if not mp4s:
            continue
        lines.append(f"## {_DISPLAY.get(algo, algo)}")
        for mp4 in mp4s:
            rel = mp4.relative_to(_PROJECT_ROOT)
            lines.append(f"- [{rel.name}]({rel})")
        lines.append("")
    text = "\n".join(lines) + "\n"
    out_path = _PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"[list_videos] wrote {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="metrics/halfcheetah_phase1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--algos", type=str, nargs="+", default=_DEFAULT_ALGOS)
    ap.add_argument("--video_root", type=str, default="plots/halfcheetah_phase1")

    sub = ap.add_subparsers(dest="cmd", required=True)
    cu = sub.add_parser("curves")
    cu.add_argument("--out", type=str, default="plots/halfcheetah_phase1/reward_curves.png")
    cu.set_defaults(fn=_plot_curves)

    su = sub.add_parser("summary")
    su.add_argument("--out", type=str, default="metrics/halfcheetah_phase1/SUMMARY.md")
    su.set_defaults(fn=_summary)

    lv = sub.add_parser("list_videos")
    lv.add_argument("--out", type=str, default="plots/halfcheetah_phase1/VIDEOS.md")
    lv.set_defaults(fn=_list_videos)

    al = sub.add_parser("all")
    al.set_defaults(fn=None)

    args = ap.parse_args()

    if args.cmd == "all":
        args.out = "plots/halfcheetah_phase1/reward_curves.png"
        _plot_curves(args)
        args.out = "metrics/halfcheetah_phase1/SUMMARY.md"
        _summary(args)
        args.out = "plots/halfcheetah_phase1/VIDEOS.md"
        _list_videos(args)
    else:
        args.fn(args)


if __name__ == "__main__":
    main()
