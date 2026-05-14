"""Visualize the pretrained Gaussian behavior prior for each bandit2d client.

For each client i in 0..N-1, plot the prior log_prob heatmap over the action grid
and overlay (a) all 4 ring peaks, and (b) the peak that *should* match this
client (peak_i). If the prior was fitted correctly, the dense (red) region of
the heatmap should sit on top of peak_i.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fedguide.guidance.diffusion_prior import GaussianBehaviorPrior


def load_prior(ckpt_path: Path, action_dim: int = 2) -> GaussianBehaviorPrior:
    prior = GaussianBehaviorPrior(state_dim=2, action_dim=action_dim)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Saved format: {'prior': state_dict, 'state_dim':..., 'action_dim':..., 'prior_type':...}
    if isinstance(state, dict) and "prior" in state and not isinstance(state["prior"], torch.Tensor):
        state = state["prior"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = prior.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_prior] {ckpt_path} missing keys: {missing}")
    prior.eval()
    return prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="model/models_prior_gauss/Bandit2D")
    ap.add_argument("--metadata", default="data/bandit2d/metadata.json")
    ap.add_argument("--out", default="metrics/bandit2d_phase1/prior_diagnostic.png")
    ap.add_argument("--grid", type=int, default=200)
    ap.add_argument("--bound", type=float, default=1.5)
    args = ap.parse_args()

    meta = json.load(open(args.metadata))
    K = meta["K"]
    mu = np.array(meta["mu"])  # (K, 2) ground-truth peaks

    n_clients = sorted(int(d.split("_")[1]) for d in os.listdir(args.prior_dir) if d.startswith("client_"))
    n = len(n_clients)

    xs = np.linspace(-args.bound, args.bound, args.grid)
    ys = np.linspace(-args.bound, args.bound, args.grid)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    grid_t = torch.tensor(grid)

    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0), squeeze=False)
    for idx, cid in enumerate(n_clients):
        ax = axes[0, idx]
        ckpt = Path(args.prior_dir) / f"client_{cid}" / "final" / "torch_prior.pth"
        if not ckpt.exists():
            ax.set_title(f"client {cid}: ckpt missing")
            ax.axis("off")
            continue
        prior = load_prior(ckpt)
        with torch.no_grad():
            logp = prior.log_prob(grid_t).detach().cpu().numpy().reshape(args.grid, args.grid)
        prob = np.exp(logp - logp.max())  # normalize for visualization

        im = ax.contourf(XX, YY, prob, levels=20, cmap="viridis")
        # all peaks (small dots)
        ax.scatter(mu[:, 0], mu[:, 1], c="white", s=30, edgecolors="black", linewidths=1, zorder=3)
        # this client's expected peak (large red ×)
        target = mu[cid % K]
        ax.scatter([target[0]], [target[1]], c="red", marker="x", s=200, linewidths=3, zorder=4,
                   label=f"target peak {cid % K}")
        # learned mean of the prior
        learned_mu = prior.head_mu.detach().cpu().numpy()
        ax.scatter([learned_mu[0]], [learned_mu[1]], c="lime", marker="+", s=200, linewidths=3,
                   zorder=4, label=f"prior μ")
        sigma = prior.head_log_sigma.detach().exp().cpu().numpy()
        ax.set_title(f"client {cid}\nμ=({learned_mu[0]:+.2f},{learned_mu[1]:+.2f}) σ=({sigma[0]:.3f},{sigma[1]:.3f})",
                     fontsize=10)
        ax.set_xlim(-args.bound, args.bound)
        ax.set_ylim(-args.bound, args.bound)
        ax.set_aspect("equal")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Bandit2D pretrained Gaussian priors — does each client's prior peak land on its target peak?",
                 fontsize=11)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[viz_priors] wrote {out}")

    # Also dump a table of (client, target_peak, learned_mu, dist_to_target_peak, dist_to_each_peak)
    print()
    print(f"{'cid':>4} {'target_peak':>11} {'learned_mu':>22} "
          f"{'dist→target':>12} {'closest_peak':>13}")
    for cid in n_clients:
        ckpt = Path(args.prior_dir) / f"client_{cid}" / "final" / "torch_prior.pth"
        if not ckpt.exists():
            continue
        prior = load_prior(ckpt)
        m = prior.head_mu.detach().cpu().numpy()
        target = mu[cid % K]
        dists = np.linalg.norm(mu - m[None, :], axis=1)
        closest = int(np.argmin(dists))
        print(f"{cid:>4} {cid % K:>11} ({m[0]:+.3f},{m[1]:+.3f})       "
              f"{np.linalg.norm(target - m):>12.3f} {closest:>13}")


if __name__ == "__main__":
    main()
