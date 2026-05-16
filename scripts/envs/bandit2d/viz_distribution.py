"""Bandit2D final policy-density grids."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "fedguide-matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


BANDIT_ALGOS = ["fedavg", "fedguide_p", "fedguide_a", "fedguide"]


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


def _parse_csv(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def discover_seeds(metrics_root: str | Path, algos: Iterable[str]) -> list[int]:
    root = Path(metrics_root)
    seeds: set[int] = set()
    for algo in algos:
        for path in (root / algo).glob("seed_*"):
            try:
                seeds.add(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return sorted(seeds)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_bandit_metrics(metrics_root: str | Path, algo: str, seed: int):
    path = Path(metrics_root) / algo / f"seed_{seed}" / "bandit2d_metrics.pkl"
    if not path.exists():
        return None
    return _load_pickle(path)


def _last_round_with_clients(metrics: dict):
    for item in reversed(metrics.get("metrics_history", [])):
        if item.get("client_metrics"):
            return item
    return None


def _prob_mass(value) -> np.ndarray:
    z = np.asarray(value, dtype=float)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.maximum(z, 0.0)
    total = float(z.sum())
    return z / total if total > 0 else z


def _display_density(value) -> np.ndarray:
    z = _prob_mass(value)
    m = float(z.max())
    return z / m if m > 0 else z


def _client_policy_masses(round_metrics: dict) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for cid, metrics in round_metrics.get("client_metrics", {}).items():
        if "policy_density" in metrics:
            out[int(cid)] = _prob_mass(metrics["policy_density"])
    return out


def _global_mixture(client_masses: dict[int, np.ndarray], client_ids: Iterable[int]) -> np.ndarray | None:
    arrs = [client_masses[cid] for cid in client_ids if cid in client_masses]
    if not arrs:
        return None
    return np.mean(np.stack(arrs, axis=0), axis=0)


def _extent(metrics: dict, bound: float):
    if metrics is None:
        return (-bound, bound, -bound, bound)
    x = np.asarray(metrics.get("X"), dtype=float)
    y = np.asarray(metrics.get("Y"), dtype=float)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _grid_xy(metrics: dict):
    return np.asarray(metrics["X"], dtype=float), np.asarray(metrics["Y"], dtype=float)


def _add_red_moon(ax, xy, radius: float = 0.055):
    x, y = float(xy[0]), float(xy[1])
    ax.add_patch(Circle((x, y), radius, facecolor="red", edgecolor="white",
                        linewidth=0.8, zorder=6))
    ax.add_patch(Circle((x + radius * 0.42, y + radius * 0.05), radius * 0.9,
                        facecolor=ax.get_facecolor(), edgecolor="none", zorder=7))


def _draw_bandit2d_reference_ring(ax):
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(np.cos(theta), np.sin(theta), color="red",
            linewidth=1.8, alpha=0.98, zorder=5)


def _draw_panel(
    ax,
    density: np.ndarray | None,
    extent,
    mu: np.ndarray,
    bound: float,
    xx: np.ndarray,
    yy: np.ndarray,
    global_mass: np.ndarray | None,
):
    ax.set_facecolor("black")
    if density is not None:
        ax.imshow(_display_density(density), extent=extent, origin="lower",
                  cmap="hot", interpolation="bilinear", vmin=0.0, vmax=1.0)
    _draw_bandit2d_reference_ring(ax)
    for peak in mu:
        _add_red_moon(ax, peak)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig, out: str | Path, write_pdf: bool = True) -> list[Path]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out]
    if write_pdf and out.suffix.lower() != ".pdf":
        paths.append(out.with_suffix(".pdf"))
    for path in paths:
        fig.savefig(path, dpi=180 if path.suffix.lower() == ".png" else None,
                    bbox_inches="tight", pad_inches=0.01)
        print(f"[viz_distribution] wrote {path}")
    return paths


def plot_policy_grid_for_seed(
    metrics_root: str | Path,
    metadata: str | Path,
    out: str | Path,
    algos: Iterable[str] = BANDIT_ALGOS,
    seed: int = 0,
    bound: float = 1.5,
    write_pdf: bool = True,
) -> list[Path]:
    with open(metadata, "r") as f:
        meta = json.load(f)
    mu = np.asarray(meta["mu"], dtype=float)
    client_ids = list(range(int(meta.get("n_clients", len(mu)))))
    algos = list(algos)

    loaded = {algo: load_bandit_metrics(metrics_root, algo, seed) for algo in algos}
    first = next((m for m in loaded.values() if m is not None), None)
    if first is None:
        raise FileNotFoundError(f"No Bandit2D metrics found under {metrics_root} for seed {seed}")
    extent = _extent(first, bound)
    xx, yy = _grid_xy(first)

    fig, axes = plt.subplots(
        len(algos), len(client_ids) + 1,
        figsize=(2.05 * (len(client_ids) + 1), 2.05 * len(algos)),
        squeeze=False,
    )
    for r, algo in enumerate(algos):
        metrics = loaded[algo]
        last = _last_round_with_clients(metrics) if metrics is not None else None
        masses = _client_policy_masses(last) if last is not None else {}
        global_mass = _global_mixture(masses, client_ids)
        for c, cid in enumerate(client_ids):
            _draw_panel(axes[r, c], masses.get(cid), extent, mu, bound, xx, yy, global_mass)
        _draw_panel(axes[r, -1], global_mass, extent, mu, bound, xx, yy, global_mass)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.03, hspace=0.03)
    paths = _save(fig, out, write_pdf=write_pdf)
    plt.close(fig)
    return paths


def plot_policy_grid_for_algo_seed(
    metrics_root: str | Path,
    metadata: str | Path,
    out: str | Path,
    algo: str,
    seed: int = 0,
    bound: float = 1.5,
    write_pdf: bool = True,
) -> list[Path]:
    with open(metadata, "r") as f:
        meta = json.load(f)
    mu = np.asarray(meta["mu"], dtype=float)
    client_ids = list(range(int(meta.get("n_clients", len(mu)))))

    metrics = load_bandit_metrics(metrics_root, algo, seed)
    if metrics is None:
        raise FileNotFoundError(f"No Bandit2D metrics found for {algo} seed {seed}")
    extent = _extent(metrics, bound)
    xx, yy = _grid_xy(metrics)
    last = _last_round_with_clients(metrics)
    masses = _client_policy_masses(last) if last is not None else {}
    global_mass = _global_mixture(masses, client_ids)

    fig, axes = plt.subplots(
        1, len(client_ids) + 1,
        figsize=(2.15 * (len(client_ids) + 1), 2.15),
        squeeze=False,
    )
    for c, cid in enumerate(client_ids):
        _draw_panel(axes[0, c], masses.get(cid), extent, mu, bound, xx, yy, global_mass)
    _draw_panel(axes[0, -1], global_mass, extent, mu, bound, xx, yy, global_mass)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.03, hspace=0.0)
    paths = _save(fig, out, write_pdf=write_pdf)
    plt.close(fig)
    return paths


def _reward_surfaces(x: np.ndarray, y: np.ndarray, mu: np.ndarray, sigma: float):
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    d2 = ((points[:, None, :] - mu[None, :, :]) ** 2).sum(axis=-1)
    base = np.exp(-d2 / (2.0 * sigma ** 2))
    client_rewards = []
    for cid in range(len(mu)):
        weights = np.ones(len(mu), dtype=float) * 0.1
        weights[cid % len(mu)] = 1.0
        client_rewards.append(np.max(base * weights[None, :], axis=1).reshape(x.shape))
    global_reward = np.max(base, axis=1).reshape(x.shape)
    return client_rewards, global_reward, points


def _nearest_modes(points: np.ndarray, mu: np.ndarray):
    return np.argmin(((points[:, None, :] - mu[None, :, :]) ** 2).sum(axis=-1), axis=1)


def _mode_entropy(mass: np.ndarray, nearest: np.ndarray, n_modes: int) -> float:
    flat = _prob_mass(mass).ravel()
    probs = np.array([flat[nearest == k].sum() for k in range(n_modes)], dtype=float)
    probs /= max(float(probs.sum()), 1e-12)
    nz = probs[probs > 0]
    if len(nz) == 0:
        return 0.0
    return float(-(nz * np.log(nz)).sum() / np.log(n_modes))


def _target_mode_mass(mass: np.ndarray, nearest: np.ndarray, cid: int, n_modes: int) -> float:
    flat = _prob_mass(mass).ravel()
    return float(flat[nearest == (cid % n_modes)].sum())


def _diagnostics(metrics_root: str | Path, metadata: str | Path, algos, seeds):
    with open(metadata, "r") as f:
        meta = json.load(f)
    mu = np.asarray(meta["mu"], dtype=float)
    sigma = float(meta.get("sigma", 0.2))
    client_ids = list(range(int(meta.get("n_clients", len(mu)))))
    first = None
    for algo in algos:
        for seed in seeds:
            first = load_bandit_metrics(metrics_root, algo, seed)
            if first is not None:
                break
        if first is not None:
            break
    if first is None:
        raise FileNotFoundError(f"No Bandit2D metrics found under {metrics_root}")

    xx, yy = _grid_xy(first)
    client_rewards, global_reward, points = _reward_surfaces(xx, yy, mu, sigma)
    nearest = _nearest_modes(points, mu)
    out = {algo: {"per_client": [], "global": [], "entropy": [], "target_mass": []}
           for algo in algos}
    for algo in algos:
        for seed in seeds:
            metrics = load_bandit_metrics(metrics_root, algo, seed)
            if metrics is None:
                continue
            last = _last_round_with_clients(metrics)
            if last is None:
                continue
            masses = _client_policy_masses(last)
            if not masses:
                continue
            scores = []
            target_masses = []
            for cid in client_ids:
                if cid not in masses:
                    scores.append(np.nan)
                    target_masses.append(np.nan)
                    continue
                scores.append(float((masses[cid] * client_rewards[cid]).sum()))
                target_masses.append(_target_mode_mass(masses[cid], nearest, cid, len(mu)))
            mix = _global_mixture(masses, client_ids)
            if mix is None:
                continue
            out[algo]["per_client"].append(scores)
            out[algo]["global"].append(float((mix * global_reward).sum()))
            out[algo]["entropy"].append(_mode_entropy(mix, nearest, len(mu)))
            out[algo]["target_mass"].append(float(np.nanmean(target_masses)))
    return out, client_ids


def plot_federal_diagnostics(
    metrics_root: str | Path = "metrics/bandit2d",
    metadata: str | Path = "data/bandit2d/metadata.json",
    out: str | Path = "plots/bandit2d_policy_density/federal_diagnostics.png",
    algos: Iterable[str] = BANDIT_ALGOS,
    seeds: Iterable[int] | None = None,
    write_pdf: bool = True,
) -> list[Path]:
    algos = list(algos)
    seeds = list(seeds) if seeds is not None else discover_seeds(metrics_root, algos)
    diag, client_ids = _diagnostics(metrics_root, metadata, algos, seeds)
    labels = ["FedAvg", "FedGuide-p", "FedGuide-a", "FedGuide"]

    heat = np.full((len(algos), len(client_ids)), np.nan)
    mean_return = np.full(len(algos), np.nan)
    min_return = np.full(len(algos), np.nan)
    entropy = np.full(len(algos), np.nan)
    target_mass = np.full(len(algos), np.nan)
    for i, algo in enumerate(algos):
        pc = np.asarray(diag[algo]["per_client"], dtype=float)
        if pc.size:
            heat[i] = np.nanmean(pc, axis=0)
            mean_return[i] = np.nanmean(np.nanmean(pc, axis=1))
            min_return[i] = np.nanmean(np.nanmin(pc, axis=1))
        if diag[algo]["entropy"]:
            entropy[i] = np.nanmean(diag[algo]["entropy"])
        if diag[algo]["target_mass"]:
            target_mass[i] = np.nanmean(diag[algo]["target_mass"])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2),
                             gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    im = axes[0].imshow(heat, cmap="viridis", vmin=0.0, vmax=max(0.65, np.nanmax(heat)))
    axes[0].set_xticks(range(len(client_ids)), [f"c{cid + 1}" for cid in client_ids])
    axes[0].set_yticks(range(len(algos)), labels)
    axes[0].set_title("Per-client density return")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                axes[0].text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center",
                             color="white", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    x = np.arange(len(algos))
    width = 0.36
    axes[1].bar(x - width / 2, mean_return, width, label="mean", color="#4c78a8")
    axes[1].bar(x + width / 2, min_return, width, label="min", color="#f58518")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("density return")
    axes[1].set_title("Mean vs min-client")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x - width / 2, entropy, width, label="global mode entropy", color="#54a24b")
    axes[2].bar(x + width / 2, target_mass, width, label="target-mode mass", color="#e45756")
    axes[2].set_xticks(x, labels, rotation=25, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Mode preservation")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, out, write_pdf=write_pdf)


def plot_bandit2d_policy_distributions(
    metrics_root: str | Path = "metrics/bandit2d",
    metadata: str | Path = "data/bandit2d/metadata.json",
    out_dir: str | Path = "plots/bandit2d_policy_density",
    algos: Iterable[str] = BANDIT_ALGOS,
    seeds: Iterable[int] | None = None,
    bound: float = 1.5,
    write_pdf: bool = True,
) -> list[Path]:
    algos = list(algos)
    seeds = list(seeds) if seeds is not None else discover_seeds(metrics_root, algos)
    paths: list[Path] = []
    for seed in seeds:
        for algo in algos:
            paths.extend(
                plot_policy_grid_for_algo_seed(
                    metrics_root=metrics_root,
                    metadata=metadata,
                    out=Path(out_dir) / f"policy_density_{algo}_seed{seed}.png",
                    algo=algo,
                    seed=int(seed),
                    bound=bound,
                    write_pdf=write_pdf,
                )
            )
    paths.extend(
        plot_federal_diagnostics(
            metrics_root=metrics_root,
            metadata=metadata,
            out=Path(out_dir) / "federal_diagnostics.png",
            algos=algos,
            seeds=seeds,
            write_pdf=write_pdf,
        )
    )
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_root", default="metrics/bandit2d")
    ap.add_argument("--metadata", default="data/bandit2d/metadata.json")
    ap.add_argument("--out_dir", default="plots/bandit2d_policy_density")
    ap.add_argument("--algos", default=",".join(BANDIT_ALGOS))
    ap.add_argument("--seeds", default="auto")
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--no_pdf", action="store_true")
    args = ap.parse_args()

    algos = _parse_csv(args.algos)
    seeds = None if args.seeds == "auto" else [int(s) for s in _parse_csv(args.seeds)]
    plot_bandit2d_policy_distributions(
        metrics_root=args.metrics_root,
        metadata=args.metadata,
        out_dir=args.out_dir,
        algos=algos,
        seeds=seeds,
        bound=args.bound,
        write_pdf=not args.no_pdf,
    )


if __name__ == "__main__":
    main()
