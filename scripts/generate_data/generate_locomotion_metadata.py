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


# Standard ("medium") profile — current default used by metadata.json.
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


# Hard profile — much more extreme dynamics + reward-shape heterogeneity.
# Centers are farther from nominal, ranges roughly 2× the standard profile.
DYNAMICS_PRESETS_HARD = {
    "nominal":             dict(mass=1.00, damping=1.00, friction=1.00, action_gain=1.00),
    "very_heavy":          dict(mass=1.65, damping=1.20, friction=1.10, action_gain=0.85),
    "very_weak_actuator":  dict(mass=1.00, damping=0.70, friction=0.85, action_gain=0.55),
    "extreme_damping_friction": dict(mass=1.10, damping=1.55, friction=1.65, action_gain=0.80),
    "low_friction":        dict(mass=1.00, damping=0.85, friction=0.45, action_gain=0.90),
    "heavy_low_damping":   dict(mass=1.50, damping=0.65, friction=0.95, action_gain=0.90),
    "weak_low_friction":   dict(mass=1.00, damping=0.90, friction=0.55, action_gain=0.65),
    "inverted_dynamics":   dict(mass=0.75, damping=1.40, friction=1.45, action_gain=0.75),
}

PREFERENCE_PRESETS_HARD = {
    # multipliers on default reward weights — pulled further apart so each
    # client has a meaningfully different optimal policy shape.
    "sprint":      dict(forward=1.30, ctrl=0.7, contact=0.8, unstable=0.000),
    "miser":       dict(forward=0.85, ctrl=3.0, contact=1.2, unstable=0.000),
    "tightrope":   dict(forward=0.95, ctrl=1.5, contact=2.0, unstable=0.050),
    "explorer":    dict(forward=1.10, ctrl=1.0, contact=1.0, unstable=0.010),
}


def gen_clients(
    env: str,
    n: int = 8,
    seed: int = 2026,
    profile: str = "standard",
):
    rng = np.random.RandomState(seed)
    if profile == "hard":
        dyn_presets = DYNAMICS_PRESETS_HARD
        pref_presets = PREFERENCE_PRESETS_HARD
        dyn_jitter = (0.85, 1.15)          # ±15% on dynamics
        pref_fwd_jitter = (0.80, 1.20)     # ±20% on forward reward
        pref_ctrl_jitter = (0.65, 1.35)    # ±35% on ctrl cost
        pref_unstable_jitter = (0.40, 1.60)
        reset_noise_jitter = (0.80, 1.20)
    else:
        dyn_presets = DYNAMICS_PRESETS
        pref_presets = PREFERENCE_PRESETS
        dyn_jitter = (0.95, 1.05)
        pref_fwd_jitter = (0.90, 1.10)
        pref_ctrl_jitter = (0.80, 1.20)
        pref_unstable_jitter = (0.50, 1.50)
        reset_noise_jitter = (0.90, 1.10)

    dyn_keys = list(dyn_presets.keys())
    pref_keys = list(pref_presets.keys())
    defaults = ENV_REWARD_DEFAULTS[env]
    clients = []
    for cid in range(n):
        d_key = dyn_keys[cid % len(dyn_keys)]
        p_key = pref_keys[cid % len(pref_keys)]
        d = dyn_presets[d_key]
        p = pref_presets[p_key]
        client = {
            "client_id": cid,
            "env_name": ENV_GYM_ID[env],
            "dynamics_preset": d_key,
            "preference_preset": p_key,
            "mass_scale":     float(d["mass"] * rng.uniform(*dyn_jitter)),
            "damping_scale":  float(d["damping"] * rng.uniform(*dyn_jitter)),
            "ground_friction": float(d["friction"] * rng.uniform(*dyn_jitter)),
            "action_gain":    float(d["action_gain"] * rng.uniform(*dyn_jitter)),
            "forward_reward_weight": float(defaults["forward_reward_weight"] * p["forward"] * rng.uniform(*pref_fwd_jitter)),
            "ctrl_cost_weight":      float(defaults["ctrl_cost_weight"] * p["ctrl"] * rng.uniform(*pref_ctrl_jitter)),
            "unstable_cost_weight":  float(p["unstable"] * rng.uniform(*pref_unstable_jitter)),
            "reset_noise_scale":     float(defaults["reset_noise_scale"] * rng.uniform(*reset_noise_jitter)),
        }
        if "contact_cost_weight" in defaults:
            client["contact_cost_weight"] = float(
                defaults["contact_cost_weight"] * p["contact"] * rng.uniform(*pref_ctrl_jitter)
            )
        clients.append(client)
    return clients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(ENV_GYM_ID.keys()))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--profile", type=str, default="standard", choices=["standard", "hard"])
    ap.add_argument("--out", type=str, default=None,
                    help="output path; defaults to data/<env>/metadata[_<profile>].json")
    args = ap.parse_args()

    out_dir = _PROJECT_ROOT / "data" / args.env
    out_dir.mkdir(parents=True, exist_ok=True)
    clients = gen_clients(args.env, n=args.n, seed=args.seed, profile=args.profile)
    meta = {
        "env": args.env,
        "env_name": ENV_GYM_ID[args.env],
        "n_clients": args.n,
        "hetero_type": "both",
        "seed": args.seed,
        "profile": args.profile,
        "clients": clients,
    }
    if args.out is not None:
        out = Path(args.out)
    else:
        suffix = "" if args.profile == "standard" else f"_{args.profile}"
        out = out_dir / f"metadata{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.env}] wrote {out} (profile={args.profile})")
    for c in clients:
        print(f"  cid={c['client_id']} dyn={c['dynamics_preset']} pref={c['preference_preset']}")


if __name__ == "__main__":
    main()
