"""Within-family ablation: FedGuide vs FedGuide-P vs FedGuide-A on
CV_sigmaT = (within-seed temporal std on the last 20 rounds, averaged across
seeds) / |Final|. Scale-invariant temporal-noise metric; lower is better.

Just prints a plain ASCII table to stdout; does not emit LaTeX or write any file.
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

FAMILY = [
    ("fedguide_a", "FG-A"),
    ("fedguide_p", "FG-P"),
    ("fedguide",   "FG"),
]
SEEDS = [0, 1, 2, 3, 4]
TAIL_FINAL = 10
TAIL_NOISE = 20
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


def cv_sigma_t(stack: np.ndarray) -> float:
    n_seeds, n_rounds = stack.shape
    tf = min(TAIL_FINAL, n_rounds)
    final_mean = float(stack[:, -tf:].mean(axis=1).mean())
    tn = min(TAIL_NOISE, n_rounds)
    block = stack[:, -tn:]
    sigma_T = float(block.std(axis=1, ddof=1).mean()) if tn > 1 else 0.0
    return sigma_T / (abs(final_mean) + 1e-9)


CELL_W = 10


def print_table(per_env):
    """Print a plain-text table to stdout. * = best (lowest CV) per row,
    ~ = second-best."""
    family_displays = [f[1] for f in FAMILY]
    header = f"{'Env':<14}" + "".join(d.rjust(CELL_W) for d in family_displays)
    rule = "-" * len(header)
    print()
    print("FedGuide family ablation -- CV_sigma_T = sigma_T(last 20) / |Final|")
    print("Lower is better. * = best in row, ~ = second-best.")
    print(rule)
    print(header)
    print(rule)
    for env_idx, (env_key, env_display, _root) in enumerate(ENVS):
        em = per_env.get(env_key)
        if not em:
            continue
        cvs = [em.get(d) for d in family_displays]
        present = [(i, v) for i, v in enumerate(cvs) if v is not None]
        ordered = sorted(present, key=lambda iv: iv[1])  # lower-is-better
        best = ordered[0][0] if ordered else -1
        second = ordered[1][0] if len(ordered) > 1 else -1
        cells = []
        for i, v in enumerate(cvs):
            if v is None:
                s = "-"
            else:
                s = f"{v:.3f}"
                if i == best:
                    s = f"*{s}"
                elif i == second:
                    s = f"~{s}"
            cells.append(s.rjust(CELL_W))
        print(f"{env_display:<14}" + "".join(cells))
        if env_idx == MAIN_BLOCK_SIZE - 1:
            print(rule)  # separator between main suite and hard ablation
    print(rule)


def main():
    per_env = {}
    for env_key, env_display, env_root in ENVS:
        d = {}
        for algo_dir, display in FAMILY:
            st = load_stack(env_root, algo_dir)
            if st is None:
                continue
            d[display] = cv_sigma_t(st)
        per_env[env_key] = d

    print_table(per_env)


if __name__ == "__main__":
    main()
