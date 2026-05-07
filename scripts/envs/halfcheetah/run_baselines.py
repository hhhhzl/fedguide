"""Run all halfcheetah baseline configs (federated baselines only).

Each (algo, seed) is launched as its own subprocess via the unified
``scripts/run_from_config.py`` so every Flower simulation gets a clean
Ray/Python state. Per-seed metrics land in
``metrics/halfcheetah_phase1/<algo>/seed_<i>/``.

Configs are loaded from ``configs/halfcheetah/baseline/``:
    fedavg, fedkl, fedrep, fedmomentum, fmarl, fedrl_ddpg

Usage:
    python scripts/envs/halfcheetah/run_baselines.py --seeds 0 --rounds 50
    python scripts/envs/halfcheetah/run_baselines.py --only fedavg,fedkl
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


JOBS = [
    # (key, algorithm, config_path)
    ("fedavg",      "fedkl",        "configs/halfcheetah/baseline/fedavg.yaml"),       # FedAvg = FedKL with λ=0
    ("fedkl",       "fedkl",        "configs/halfcheetah/baseline/fedkl.yaml"),
    ("fedrep",      "fedrep",       "configs/halfcheetah/baseline/fedrep.yaml"),
    ("fedmomentum", "fedmomentum",  "configs/halfcheetah/baseline/fedmomentum.yaml"),
    ("fmarl",       "fmarl",        "configs/halfcheetah/baseline/fmarl.yaml"),
    ("fedrl",       "fedrl",        "configs/halfcheetah/baseline/fedrl_ddpg.yaml"),
    ("mfpo",        "mfpo",         "configs/halfcheetah/baseline/mfpo.yaml"),
]


def run_one(key: str, algorithm: str, config_path: str, seed: int, rounds: int,
            log_dir: Path, metrics_root: Path) -> tuple[int, float]:
    out_metrics = metrics_root / key / f"seed_{seed}"
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_models = _PROJECT_ROOT / "model" / "policy" / "halfcheetah_phase1" / key / f"seed_{seed}"
    log_path = log_dir / f"{key}_s{seed}.log"

    env = os.environ.copy()
    env["FG_PHASE1_SWEEP"] = "1"
    env["FG_PHASE1_METRICS_DIR"] = str(out_metrics)
    env["FG_PHASE1_OUTPUT_DIR"] = str(out_models)
    env["FG_PHASE1_SEED"] = str(seed)
    env["FG_PHASE1_ROUNDS"] = str(rounds)
    env.setdefault("FG_PHASE1_DEVICE", os.environ.get("FG_PHASE1_DEVICE", "cuda"))
    env.setdefault("FG_PHASE1_GPUS_PER_CLIENT", os.environ.get("FG_PHASE1_GPUS_PER_CLIENT", "0.125"))
    env.setdefault("RAY_DEDUP_LOGS", "1")
    # Per-client mp4 render at the final round (every client, not just cid 0).
    env.setdefault("FEDGUIDE_FEDERATED_RENDER_ALL_CLIENTS", "1")

    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "run_from_config.py"),
        str(_PROJECT_ROOT / config_path),
        "--algorithm", algorithm,
        "--seeds", str(seed),
        "--rounds", str(rounds),
    ]

    print(f"[baseline] >>> {key} seed={seed} rounds={rounds}")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(_PROJECT_ROOT))
    dur = time.time() - t0
    print(f"[baseline] <<< {key} seed={seed} rc={proc.returncode} time={dur:.1f}s")
    return proc.returncode, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of {fedavg, fedkl, fedrep, fedmomentum, fmarl, fedrl, mfpo}")
    ap.add_argument("--metrics_root", type=str, default="metrics/halfcheetah_phase1")
    ap.add_argument("--log_dir", type=str, default="logs/halfcheetah_phase1")
    args = ap.parse_args()

    log_dir = _PROJECT_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_root = _PROJECT_ROOT / args.metrics_root

    only = set(args.only.split(",")) if args.only else None
    jobs = [j for j in JOBS if (only is None or j[0] in only)]

    summary = []
    for key, algo, cfg in jobs:
        for s in args.seeds:
            rc, dur = run_one(key, algo, cfg, s, args.rounds, log_dir, metrics_root)
            summary.append((key, s, rc, dur))

    print("\n========= BASELINE SUMMARY =========")
    for key, s, rc, dur in summary:
        status = "OK" if rc == 0 else "FAIL"
        print(f"{key:>16s}  seed={s}  status={status}  time={dur:6.1f}s")


if __name__ == "__main__":
    main()
