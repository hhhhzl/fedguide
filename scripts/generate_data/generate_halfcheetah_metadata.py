#!/usr/bin/env python3
"""Generate HalfCheetah metadata with selectable heterogeneity profiles."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clipped_gauss(rng: random.Random, mean: float, std: float, lo: float, hi: float) -> float:
    return clip(rng.gauss(mean, std), lo, hi)


def _make_client_from_profile(rng: random.Random, cid: int, dyn: str, pref: str, profile: str) -> dict:
    if profile == "nominal":
        mass_scale = 1.0
        damping_scale = 1.0
        ground_friction = 1.0
        action_gain = 1.0
        w_vel = 1.0
        w_ctrl = 0.1
        w_unstable = 0.0
        reset_noise_scale = 0.1
    else:
        if profile == "mild":
            if dyn == "nominal":
                mass_scale = clipped_gauss(rng, 1.00, 0.02, 0.96, 1.05)
                damping_scale = clipped_gauss(rng, 1.00, 0.025, 0.95, 1.08)
                ground_friction = clipped_gauss(rng, 1.00, 0.03, 0.94, 1.10)
                action_gain = clipped_gauss(rng, 1.00, 0.02, 0.95, 1.05)
            elif dyn == "heavy_body":
                mass_scale = clipped_gauss(rng, 1.08, 0.03, 1.02, 1.15)
                damping_scale = clipped_gauss(rng, 1.02, 0.03, 0.96, 1.10)
                ground_friction = clipped_gauss(rng, 1.02, 0.03, 0.95, 1.12)
                action_gain = clipped_gauss(rng, 0.98, 0.02, 0.93, 1.03)
            elif dyn == "weak_actuator":
                mass_scale = clipped_gauss(rng, 1.00, 0.02, 0.95, 1.06)
                damping_scale = clipped_gauss(rng, 1.01, 0.03, 0.95, 1.10)
                ground_friction = clipped_gauss(rng, 1.00, 0.03, 0.94, 1.10)
                action_gain = clipped_gauss(rng, 0.92, 0.03, 0.85, 0.98)
            else:
                mass_scale = clipped_gauss(rng, 1.02, 0.02, 0.96, 1.08)
                damping_scale = clipped_gauss(rng, 1.10, 0.04, 1.02, 1.22)
                ground_friction = clipped_gauss(rng, 1.12, 0.04, 1.03, 1.25)
                action_gain = clipped_gauss(rng, 0.98, 0.02, 0.93, 1.03)

            if pref == "speed":
                w_vel = clipped_gauss(rng, 1.05, 0.02, 1.0, 1.10)
                w_ctrl = clipped_gauss(rng, 0.09, 0.006, 0.075, 0.105)
                w_unstable = 0.0
            elif pref == "efficiency":
                w_vel = clipped_gauss(rng, 0.98, 0.02, 0.94, 1.04)
                w_ctrl = clipped_gauss(rng, 0.11, 0.007, 0.095, 0.125)
                w_unstable = 0.0
            else:
                w_vel = clipped_gauss(rng, 0.97, 0.02, 0.93, 1.03)
                w_ctrl = clipped_gauss(rng, 0.105, 0.007, 0.09, 0.12)
                w_unstable = 0.0

            reset_noise_scale = clipped_gauss(rng, 0.10, 0.01, 0.08, 0.12)
        else:  # standard
            if dyn == "nominal":
                mass_scale = clipped_gauss(rng, 1.00, 0.04, 0.90, 1.12)
                damping_scale = clipped_gauss(rng, 1.00, 0.05, 0.85, 1.15)
                ground_friction = clipped_gauss(rng, 1.00, 0.06, 0.85, 1.20)
                action_gain = clipped_gauss(rng, 1.00, 0.05, 0.85, 1.15)
            elif dyn == "heavy_body":
                mass_scale = clipped_gauss(rng, 1.20, 0.06, 1.05, 1.35)
                damping_scale = clipped_gauss(rng, 1.05, 0.06, 0.90, 1.25)
                ground_friction = clipped_gauss(rng, 1.02, 0.06, 0.85, 1.25)
                action_gain = clipped_gauss(rng, 0.98, 0.05, 0.82, 1.12)
            elif dyn == "weak_actuator":
                mass_scale = clipped_gauss(rng, 1.00, 0.05, 0.88, 1.18)
                damping_scale = clipped_gauss(rng, 1.02, 0.06, 0.88, 1.20)
                ground_friction = clipped_gauss(rng, 1.00, 0.06, 0.84, 1.22)
                action_gain = clipped_gauss(rng, 0.80, 0.06, 0.65, 0.95)
            else:
                mass_scale = clipped_gauss(rng, 1.03, 0.05, 0.90, 1.20)
                damping_scale = clipped_gauss(rng, 1.30, 0.08, 1.10, 1.55)
                ground_friction = clipped_gauss(rng, 1.25, 0.08, 1.05, 1.50)
                action_gain = clipped_gauss(rng, 0.97, 0.05, 0.82, 1.12)

            if pref == "speed":
                w_vel = clipped_gauss(rng, 1.12, 0.05, 1.00, 1.25)
                w_ctrl = clipped_gauss(rng, 0.085, 0.012, 0.06, 0.11)
                w_unstable = clipped_gauss(rng, 0.012, 0.004, 0.004, 0.025)
            elif pref == "efficiency":
                w_vel = clipped_gauss(rng, 0.98, 0.05, 0.88, 1.12)
                w_ctrl = clipped_gauss(rng, 0.145, 0.015, 0.11, 0.19)
                w_unstable = clipped_gauss(rng, 0.014, 0.004, 0.006, 0.028)
            else:
                w_vel = clipped_gauss(rng, 0.96, 0.05, 0.86, 1.10)
                w_ctrl = clipped_gauss(rng, 0.115, 0.014, 0.085, 0.16)
                w_unstable = clipped_gauss(rng, 0.035, 0.007, 0.02, 0.06)

            reset_noise_scale = clipped_gauss(rng, 0.11, 0.025, 0.06, 0.18)

    return {
        "client_id": cid,
        "env_name": "HalfCheetah-v4",
        "dynamics_preset": dyn,
        "preference_preset": pref,
        "mass_scale": round(mass_scale, 4),
        "damping_scale": round(damping_scale, 4),
        "ground_friction": round(ground_friction, 4),
        "action_gain": round(action_gain, 4),
        "forward_reward_weight": round(w_vel, 4),
        "ctrl_cost_weight": round(w_ctrl, 4),
        "unstable_cost_weight": round(w_unstable, 4),
        "reset_noise_scale": round(reset_noise_scale, 4),
    }


def generate_halfcheetah_metadata(
    n_clients: int = 64,
    seed: int = 2026,
    out_path: str | Path = "data/halfcheetah/metadata.json",
    profile: str = "standard",
) -> None:
    rng = random.Random(seed)

    if profile not in ("standard", "mild", "nominal"):
        raise ValueError("profile must be one of: standard, mild, nominal")

    base = n_clients // 4
    dyn_pool = (
        ["nominal"] * base
        + ["heavy_body"] * base
        + ["weak_actuator"] * base
        + ["high_damping_friction"] * (n_clients - 3 * base)
    )
    rng.shuffle(dyn_pool)

    n_pref = n_clients // 3
    pref_pool = (
        ["speed"] * n_pref
        + ["efficiency"] * n_pref
        + ["stability"] * (n_clients - 2 * n_pref)
    )
    rng.shuffle(pref_pool)

    clients = [
        _make_client_from_profile(rng, cid, dyn_pool[cid], pref_pool[cid], profile)
        for cid in range(n_clients)
    ]

    meta = {
        "env": "halfcheetah",
        "env_name": "HalfCheetah-v4",
        "n_clients": n_clients,
        "hetero_type": "both" if profile != "nominal" else "iid",
        "seed": seed,
        "profile": profile,
        "clients": clients,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {n_clients} clients ({profile}) to {out.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-clients", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--profile", type=str, default="standard", choices=["standard", "mild", "nominal"])
    p.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "halfcheetah" / "metadata.json"),
    )
    args = p.parse_args()
    generate_halfcheetah_metadata(
        n_clients=args.n_clients,
        seed=args.seed,
        out_path=args.out,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
