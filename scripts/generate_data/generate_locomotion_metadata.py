"""Generate hetero metadata.json for Walker2D / Ant / Hopper.

8-client federated heterogeneity, mirroring the halfcheetah design:
  * dynamics_preset (categorical) — picks a {mass, damping, friction} regime
  * preference_preset (categorical) — picks a {forward, ctrl, contact, unstable}
    reward-weight regime
  * jittered numeric values around each preset center

Usage:
    python scripts/envs/_gen_locomotion_metadata.py --env walker
    python scripts/envs/_gen_locomotion_metadata.py --env ant
    python scripts/envs/_gen_locomotion_metadata.py --env hopper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


ENV_GYM_ID = {
    "walker": "Walker2d-v4",
    "hopper": "Hopper-v4",
}


# Per-env reward kwargs scale (defaults from gym source).
ENV_REWARD_DEFAULTS = {
    "walker":  dict(forward_reward_weight=1.0, ctrl_cost_weight=1e-3, reset_noise_scale=5e-3),
    "hopper":  dict(forward_reward_weight=1.0, ctrl_cost_weight=1e-3, reset_noise_scale=5e-3),
}


DYNAMICS_PRESETS = {
    "nominal":             dict(mass=1.00, damping=1.00, friction=1.00, action_gain=1.00),
    "heavy":               dict(mass=1.40, damping=1.10, friction=1.05, action_gain=0.95),
    "weak_actuator":       dict(mass=1.00, damping=0.85, friction=0.95, action_gain=0.80),
    "high_damping_friction": dict(mass=1.05, damping=1.25, friction=1.30, action_gain=0.90),
}


PREFERENCE_PRESETS = {
    # multipliers on default reward weights.
    "speed":        dict(forward=1.10, ctrl=1.0, contact=1.0, unstable=0.0),
    "efficiency":   dict(forward=1.00, ctrl=1.5, contact=1.0, unstable=0.0),
    "stable":       dict(forward=1.00, ctrl=1.2, contact=1.5, unstable=0.020),
}


def gen_clients(env: str, n: int = 8, seed: int = 2026):
    rng = np.random.RandomState(seed)
    dyn_keys = list(DYNAMICS_PRESETS.keys())
    pref_keys = list(PREFERENCE_PRESETS.keys())
    defaults = ENV_REWARD_DEFAULTS[env]
    clients = []
    for cid in range(n):
        d_key = dyn_keys[cid % len(dyn_keys)]
        p_key = pref_keys[cid % len(pref_keys)]
        d = DYNAMICS_PRESETS[d_key]
        p = PREFERENCE_PRESETS[p_key]
        # Jitter ±10% on dynamics, ±20% on preferences.
        client = {
            "client_id": cid,
            "env_name": ENV_GYM_ID[env],
            "dynamics_preset": d_key,
            "preference_preset": p_key,
            "mass_scale":     float(d["mass"] * rng.uniform(0.95, 1.05)),
            "damping_scale":  float(d["damping"] * rng.uniform(0.90, 1.10)),
            "ground_friction": float(d["friction"] * rng.uniform(0.90, 1.10)),
            "action_gain":    float(d["action_gain"] * rng.uniform(0.90, 1.10)),
            "forward_reward_weight": float(defaults["forward_reward_weight"] * p["forward"] * rng.uniform(0.90, 1.10)),
            "ctrl_cost_weight":      float(defaults["ctrl_cost_weight"] * p["ctrl"] * rng.uniform(0.80, 1.20)),
            "unstable_cost_weight":  float(p["unstable"] * rng.uniform(0.50, 1.50)),
            "reset_noise_scale":     float(defaults["reset_noise_scale"] * rng.uniform(0.90, 1.10)),
        }
        if "contact_cost_weight" in defaults:
            client["contact_cost_weight"] = float(
                defaults["contact_cost_weight"] * p["contact"] * rng.uniform(0.80, 1.20)
            )
        clients.append(client)
    return clients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(ENV_GYM_ID.keys()))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out_dir = _PROJECT_ROOT / "data" / args.env
    out_dir.mkdir(parents=True, exist_ok=True)
    clients = gen_clients(args.env, n=args.n, seed=args.seed)
    meta = {
        "env": args.env,
        "env_name": ENV_GYM_ID[args.env],
        "n_clients": args.n,
        "hetero_type": "both",
        "seed": args.seed,
        "clients": clients,
    }
    out = out_dir / "metadata.json"
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.env}] wrote {out}")
    for c in clients:
        print(f"  cid={c['client_id']} dyn={c['dynamics_preset']} pref={c['preference_preset']}")


if __name__ == "__main__":
    main()
