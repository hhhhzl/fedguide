"""Generate reacher hetero metadata.json files at three difficulty levels.

  A — strong numerical heterogeneity (no preference, no structural)
  B — A + categorical reward preference (speed / stability / efficiency / default)
  C — B + structural heterogeneity (per-client mass / damping / actuator gain)

Reacher physical reach is ~0.21, so the goal is constrained to within 0.25;
heterogeneity in goal position comes from spreading clients across 8 octants
of an annular region rather than a wider radius.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = _PROJECT_ROOT / "data" / "reacher"
OUT_DIR.mkdir(parents=True, exist_ok=True)


GOAL_RADIUS = 0.18
GOAL_BOX_RADIAL = 0.06
GOAL_BOX_ANGULAR_DEG = 30.0


def _polar_goal_box(center_angle_deg: float):
    theta = np.deg2rad(center_angle_deg)
    dtheta = np.deg2rad(GOAL_BOX_ANGULAR_DEG / 2.0)
    rmin = GOAL_RADIUS - GOAL_BOX_RADIAL / 2.0
    rmax = GOAL_RADIUS + GOAL_BOX_RADIAL / 2.0
    pts = []
    for r in (rmin, rmax):
        for a in (theta - dtheta, theta + dtheta):
            pts.append((r * np.cos(a), r * np.sin(a)))
    xs, ys = zip(*pts)
    return [[float(min(xs)), float(max(xs))], [float(min(ys)), float(max(ys))]]


PREFERENCES_4 = ("default", "speed", "stability", "efficiency")


def gen_clients(level: str, n_clients: int = 8, seed: int = 42):
    rng = np.random.RandomState(seed)
    clients = []
    angles_deg = np.linspace(0, 360, n_clients, endpoint=False)
    for cid in range(n_clients):
        center = float(angles_deg[cid])
        client = {
            "client_id": cid,
            "variant": "medium-v2",
            "goal_center_angle_deg": center,
            "qpos_high_low": _polar_goal_box(center),
            "action_noise": np.clip(rng.normal(0.0, 0.8, 2), -1.0, 1.0).tolist(),
            "reward_scale": float(rng.uniform(0.6, 1.6)),
            "angle_noise": float(rng.uniform(-0.2, 0.2)),
        }
        if level in ("B", "C"):
            client["preference"] = PREFERENCES_4[cid % len(PREFERENCES_4)]
        if level == "C":
            client["mass_scale"] = float(rng.uniform(0.5, 2.0))
            client["damping_scale"] = float(rng.uniform(0.5, 2.0))
            client["gear_scale"] = float(rng.uniform(0.5, 2.0))
        clients.append(client)
    return clients


def main():
    n_clients = 8
    seed = 42
    for level in ("A", "B", "C"):
        clients = gen_clients(level, n_clients=n_clients, seed=seed)
        meta = {
            "n_clients": n_clients,
            "hetero_type": "both",
            "seed": seed,
            "level": level,
            "variants": ["medium-v2"],
            "clients": clients,
        }
        out = OUT_DIR / f"metadata_{level}.json"
        with open(out, "w") as f:
            json.dump(meta, f, indent=2)
        # Diagnostic
        an = [np.linalg.norm(c["action_noise"]) for c in clients]
        rs = [c["reward_scale"] for c in clients]
        prefs = sorted({c.get("preference", "default") for c in clients})
        ms = [c.get("mass_scale", 1.0) for c in clients]
        ds = [c.get("damping_scale", 1.0) for c in clients]
        gs = [c.get("gear_scale", 1.0) for c in clients]
        print(f"[{level}] {out.name}")
        print(
            f"    action_noise norm: [{min(an):.2f}, {max(an):.2f}]"
            f"  reward_scale: [{min(rs):.2f}, {max(rs):.2f}]"
            f"  preferences: {prefs}"
            f"  mass: [{min(ms):.2f}, {max(ms):.2f}]"
            f"  damping: [{min(ds):.2f}, {max(ds):.2f}]"
            f"  gear: [{min(gs):.2f}, {max(gs):.2f}]"
        )


if __name__ == "__main__":
    main()
