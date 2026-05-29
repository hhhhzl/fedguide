"""Compute and print the FedGuide main results table to stdout.

Two pure-performance metrics (apples-to-apples across all 7 algos):
    1. Final Return  (mean +- seed std, last-10-round mean) — endpoint performance
       with seed-robustness embedded.
    2. Worst-Seed    (min over 5 seeds of last-10-round mean) — hard per-seed
       lower bound; couples performance with worst-case robustness.

Just prints a plain ASCII table; does not emit LaTeX or write any file.
"""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np


def _install_stubs():
    flwr = types.ModuleType("flwr")
    server = types.ModuleType("flwr.server")
    history = types.ModuleType("flwr.server.history")

    class History:
        pass

    history.History = History
    sys.modules.setdefault("flwr", flwr)
    sys.modules.setdefault("flwr.server", server)
    sys.modules.setdefault("flwr.server.history", history)
    try:
        import numpy.core as nc
        import numpy.core.multiarray as nm
        import numpy.core.numeric as nn
        import numpy.core.umath as nu
    except Exception:
        return
    sys.modules.setdefault("numpy._core", nc)
    sys.modules.setdefault("numpy._core.multiarray", nm)
    sys.modules.setdefault("numpy._core.numeric", nn)
    sys.modules.setdefault("numpy._core.umath", nu)


_install_stubs()


ENVS = [
    ("reacher",          "Reacher",        "metrics/reacher"),
    ("hopper",           "Hopper",         "metrics/hopper"),
    ("walker",           "Walker2D",       "metrics/walker"),
    ("halfcheetah",      "HalfCheetah",    "metrics/halfcheetah"),
    ("metaworld",        "MetaWorld10",    "metrics/metaworld"),
    ("reacher_hard",     "Reacher-H",      "metrics/reacher/ablation/C"),
    ("hopper_hard",      "Hopper-H",       "metrics/hopper_hard"),
    ("walker_hard",      "Walker2D-H",     "metrics/walker_hard"),
    ("halfcheetah_hard", "HalfCheetah-H",  "metrics/halfcheetah_hard"),
]
MAIN_BLOCK_SIZE = 5

ALGOS = [
    ("fedavg",      "FedAvg"),
    ("fedkl",       "FedKL"),
    ("fedrl_ddpg",  "FedRL"),
    ("fedmomentum", "FedSVRPG-M"),
    ("fedguide_a",  "FedGuide-A"),
    ("fedguide_p",  "FedGuide-P"),
    ("fedguide",    "FedGuide"),
]

SEEDS = [0, 1, 2, 3, 4]
TAIL = 10
METRIC_ATTR = "metrics_distributed_fit"
METRIC_KEY  = "eval/return"
def load_stack(env_root: str, algo: str):
    curves = []
    for s in SEEDS:
        p = Path(env_root) / algo / f"seed_{s}" / "training_history.pkl"
        if not p.exists():
            continue
        try:
            with open(p, "rb") as f:
                h = pickle.load(f)
        except Exception as exc:
            print(f"!! could not load {p}: {exc}", file=sys.stderr)
            continue
        ev = getattr(h, METRIC_ATTR, {}).get(METRIC_KEY, [])
        if not ev:
            continue
        rounds = np.asarray([int(r) for (r, _) in ev], dtype=int)
        vals = np.asarray([float(v) for (_, v) in ev], dtype=float)
        curves.append((rounds, vals))
    if not curves:
        return None
    common = sorted(set(curves[0][0]).intersection(*[set(c[0]) for c in curves[1:]]))
    if not common:
        return None
    return np.stack([
        np.asarray([dict(zip(r.tolist(), v.tolist()))[k] for k in common], dtype=float)
        for r, v in curves
    ], axis=0)


def compute(stack: np.ndarray) -> tuple[float, float, float]:
    """(final_mean, final_std, worst_seed_final)."""
    n_seeds, n_rounds = stack.shape
    tail = min(TAIL, n_rounds)
    last_means = stack[:, -tail:].mean(axis=1)
    final_mean = float(last_means.mean())
    final_std  = float(last_means.std(ddof=1)) if n_seeds > 1 else 0.0
    worst_seed = float(last_means.min())
    return final_mean, final_std, worst_seed


def fmt_value(v: float) -> str:
    av = abs(v)
    if av >= 1000: return f"{v:.0f}"
    if av >= 100:  return f"{v:.0f}"
    if av >= 10:   return f"{v:.1f}"
    return f"{v:.2f}"


def fmt_final(mean: float, std: float) -> str:
    return f"{fmt_value(mean)}+-{fmt_value(std)}"


CELL_W = 14   # column width for each algo cell


def _row(env_label: str, metric_label: str, vals, best_idx, second_idx, formatter):
    """Return one printable line. Mark best with '*', second-best with '~'."""
    cells = []
    for i, v in enumerate(vals):
        if v is None:
            s = "-"
        else:
            s = formatter(v, i)
            if i == best_idx:
                s = f"*{s}"
            elif i == second_idx:
                s = f"~{s}"
        cells.append(s.rjust(CELL_W))
    return f"{env_label:<14}{metric_label:<8}" + "".join(cells)


def print_table(per_env):
    """Print a plain-text table to stdout. * = best per row, ~ = second-best."""
    algo_displays = [a[1] for a in ALGOS]
    header = f"{'Env':<14}{'Metric':<8}" + "".join(a.rjust(CELL_W) for a in algo_displays)
    rule = "-" * len(header)
    print()
    print("Main results — Final (mean+-seed std, last 10 rounds) | Worst (min seed of last-10 mean)")
    print("All metrics: higher is better. * = best in row, ~ = second-best.")
    print(rule)
    print(header)
    print(rule)
    for env_idx, (env_key, env_display, _root) in enumerate(ENVS):
        em = per_env.get(env_key)
        if not em:
            continue
        for m_idx, m_key in enumerate(("final", "worst")):
            vals = []
            for d in algo_displays:
                t = em.get(d)
                vals.append(None if t is None else (t[0] if m_key == "final" else t[2]))
            present = [(i, v) for i, v in enumerate(vals) if v is not None]
            ordered = sorted(present, key=lambda iv: -iv[1])
            best = ordered[0][0] if ordered else -1
            second = ordered[1][0] if len(ordered) > 1 else -1
            if m_key == "final":
                fn = lambda v, i: fmt_final(em[algo_displays[i]][0], em[algo_displays[i]][1])
                metric_label = "Final"
            else:
                fn = lambda v, i: fmt_value(v)
                metric_label = "Worst"
            env_label = env_display if m_idx == 0 else ""
            print(_row(env_label, metric_label, vals, best, second, fn))
        if env_idx == MAIN_BLOCK_SIZE - 1:
            print(rule)  # separator between main suite and hard ablation
    print(rule)


def main():
    per_env = {}
    for env_key, env_display, env_root in ENVS:
        d = {}
        for algo_dir, display in ALGOS:
            st = load_stack(env_root, algo_dir)
            if st is None:
                continue
            d[display] = compute(st)
        per_env[env_key] = d

    print_table(per_env)


if __name__ == "__main__":
    main()
