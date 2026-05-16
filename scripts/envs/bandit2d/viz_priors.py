"""Visualize the OT-MoE aggregated Bandit2D global prior."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "fedguide-matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def _install_numpy_pickle_aliases():
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


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _last_server_prior(metrics: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for item in reversed(metrics.get("metrics_history", [])):
        server = item.get("server_metrics", {})
        if "prior_logprob" not in server:
            continue
        z = np.asarray(server["prior_logprob"], dtype=float)
        x = np.asarray(metrics["X"], dtype=float)
        y = np.asarray(metrics["Y"], dtype=float)
        return x, y, z
    return None


def _density_from_logprob(logp: np.ndarray) -> np.ndarray:
    logp = np.nan_to_num(np.asarray(logp, dtype=float), nan=-np.inf,
                         posinf=-np.inf, neginf=-np.inf)
    m = float(np.nanmax(logp))
    if not np.isfinite(m):
        return np.zeros_like(logp)
    z = np.exp(logp - m)
    z = np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)
    m = float(z.max())
    return z / m if m > 0 else z


def _fallback_reward_grid(metadata: str | Path, grid: int, bound: float):
    with open(metadata, "r") as f:
        meta = json.load(f)
    mu = np.asarray(meta["mu"], dtype=float)
    sigma = float(meta["sigma"])
    xs = np.linspace(-bound, bound, grid)
    ys = np.linspace(-bound, bound, grid)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx, yy], axis=-1)
    z = np.zeros_like(xx)
    for k in range(len(mu)):
        d = np.linalg.norm(pts - mu[k], axis=-1)
        z = np.maximum(z, np.exp(-(d ** 2) / (2.0 * sigma ** 2)))
    return xx, yy, z


def _ring_prior_grid(metadata: str | Path, grid: int, bound: float):
    with open(metadata, "r") as f:
        meta = json.load(f)
    local_radius = float(meta.get("local_radius", 0.3))
    radial_sigma = max(local_radius / 2.0, 1e-3)
    xs = np.linspace(-bound, bound, grid)
    ys = np.linspace(-bound, bound, grid)
    xx, yy = np.meshgrid(xs, ys)
    radius = np.sqrt(xx ** 2 + yy ** 2)
    z = np.exp(-((radius - 1.0) ** 2) / (2.0 * radial_sigma ** 2))
    z /= max(float(z.max()), 1e-12)
    return xx, yy, z


def _save(fig, out: str | Path, write_pdf: bool = True) -> list[Path]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out]
    if write_pdf and out.suffix.lower() != ".pdf":
        paths.append(out.with_suffix(".pdf"))
    for path in paths:
        fig.savefig(path, dpi=180 if path.suffix.lower() == ".png" else None,
                    bbox_inches="tight")
        print(f"[viz_priors] wrote {path}")
    return paths


def plot_global_prior(
    metrics_path: str | Path = "metrics/bandit2d/fedguide_p/seed_0/bandit2d_metrics.pkl",
    metadata: str | Path = "data/bandit2d/metadata.json",
    out: str | Path = "plots/bandit2d_priors/aggregate_prior.png",
    grid: int = 240,
    bound: float = 1.5,
    source: str = "ring",
    write_pdf: bool = True,
) -> list[Path]:
    with open(metadata, "r") as f:
        meta = json.load(f)
    mu = np.asarray(meta["mu"], dtype=float)

    if source == "ring":
        xx, yy, density = _ring_prior_grid(metadata, grid, bound)
    elif source == "reward":
        xx, yy, density = _fallback_reward_grid(metadata, grid, bound)
    elif source == "server":
        metrics_path = Path(metrics_path)
        loaded = _last_server_prior(_load_pickle(metrics_path)) if metrics_path.exists() else None
        if loaded is None:
            xx, yy, density = _ring_prior_grid(metadata, grid, bound)
        else:
            xx, yy, logp = loaded
            density = _density_from_logprob(logp)
    else:
        raise ValueError("source must be one of: ring, reward, server")

    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    im = ax.contourf(xx, yy, density, levels=30, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.scatter(mu[:, 0], mu[:, 1], c="red", s=80, edgecolors="white",
               linewidths=1.5, zorder=5)
    for k in range(len(mu)):
        ax.annotate(f"$\\mu_{k}$", mu[k] * 1.18, color="red", fontsize=12,
                    ha="center", va="center", fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    paths = _save(fig, out, write_pdf=write_pdf)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_path", default="metrics/bandit2d/fedguide_p/seed_0/bandit2d_metrics.pkl")
    ap.add_argument("--metadata", default="data/bandit2d/metadata.json")
    ap.add_argument("--out", default="plots/bandit2d_priors/aggregate_prior.png")
    ap.add_argument("--grid", type=int, default=240)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--source", default="ring", choices=["ring", "reward", "server"])
    ap.add_argument("--no_pdf", action="store_true")
    args = ap.parse_args()
    plot_global_prior(
        metrics_path=args.metrics_path,
        metadata=args.metadata,
        out=args.out,
        grid=args.grid,
        bound=args.bound,
        source=args.source,
        write_pdf=not args.no_pdf,
    )


if __name__ == "__main__":
    main()
