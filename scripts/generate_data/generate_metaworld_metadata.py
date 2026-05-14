"""Generate MetaWorld ML10 federated metadata.json.

Each client owns one ML10 task — this fixed task assignment is the simplest
form of manipulation-task heterogeneity. 10 train tasks → 10 clients.

Usage:
    python scripts/generate_data/generate_metaworld_metadata.py
    python scripts/generate_data/generate_metaworld_metadata.py --seed 2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


METAWORLD_ML10_TRAIN_TASKS = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "sweep-v3",
    "basketball-v3",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str,
                    default=str(_PROJECT_ROOT / "data" / "metaworld" / "metadata.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "env": "metaworld_ml10",
        "n_clients": len(METAWORLD_ML10_TRAIN_TASKS),
        "seed": int(args.seed),
        "clients": [
            {"client_id": i, "task": task}
            for i, task in enumerate(METAWORLD_ML10_TRAIN_TASKS)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {out_path}  ({len(METAWORLD_ML10_TRAIN_TASKS)} clients)")


if __name__ == "__main__":
    main()
