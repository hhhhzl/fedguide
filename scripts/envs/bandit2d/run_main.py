"""Run the 3 main FedGuide variants on bandit2d.

Variants in ``configs/bandit2d/main/``:
    fedguide_prior  (Theorem 3, prior aggregated, no log-std anneal, λ_guide=0.5)
    fedguide_pg     (Theorem 4, prior + value-guidance aggregated, guide_coef=0.1)
    fedguide_all    (Theorem 5, policy + prior + guidance aggregated)

This script also handles the "ensure pretrain" step: if the Gaussian
behaviour priors at ``./model/models_prior_gauss/Bandit2D/client_{0..N}/``
are missing, it runs the closed-form Gaussian pretrain (≈ 10 s) before
launching any sweep job. Pass ``--skip_pretrain`` to bypass.

Usage:
    python scripts/envs/bandit2d/run_main.py --seeds 0 --rounds 60
    python scripts/envs/bandit2d/run_main.py --only fedguide_prior,fedguide_pg
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
    ("fedguide_prior", "fedguide", "configs/bandit2d/main/fedguide_prior.yaml"),
    ("fedguide_pg",    "fedguide", "configs/bandit2d/main/fedguide_pg.yaml"),
    ("fedguide_all",   "fedguide", "configs/bandit2d/main/fedguide_all.yaml"),
]


def _gaussian_priors_present(num_clients: int = 4,
                             root: str = "./model/models_prior_gauss") -> bool:
    base = _PROJECT_ROOT / root / "Bandit2D"
    for cid in range(num_clients):
        if not (base / f"client_{cid}" / "final" / "torch_prior.pth").exists():
            return False
    return True


def ensure_pretrain(num_clients: int = 4,
                    rollout_seed: int = 42,
                    save_root: str = "./model/models_prior_gauss",
                    n_behavior_epochs: int = 500,
                    guidance_warmup_epochs: int = 200,
                    device: str = "cuda") -> None:
    """Run the closed-form Gaussian pretrain if the ckpts aren't on disk."""
    if _gaussian_priors_present(num_clients=num_clients, root=save_root):
        print(f"[pretrain] already present at {save_root}/Bandit2D/")
        return
    print(f"[pretrain] running Gaussian pretrain → {save_root}/Bandit2D/ ...")
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "envs" / "bandit2d" / "_pretrain.py"),
        "--num_clients", str(num_clients),
        "--K", "4",
        "--sigma", "0.2",
        "--seed", str(rollout_seed),
        "--save_root", save_root,
        "--prior_type", "gaussian",
        "--n_behavior_epochs", str(n_behavior_epochs),
        "--guidance_warmup_epochs", str(guidance_warmup_epochs),
        "--device", device,
    ]
    res = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"pretrain failed (rc={res.returncode})")
    print("[pretrain] done.")


def run_one(key: str, algorithm: str, config_path: str, seed: int, rounds: int,
            log_dir: Path, metrics_root: Path) -> tuple[int, float]:
    out_metrics = metrics_root / key / f"seed_{seed}"
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_models = _PROJECT_ROOT / "model" / "policy" / "bandit2d_phase1" / key / f"seed_{seed}"
    log_path = log_dir / f"{key}_s{seed}.log"

    env = os.environ.copy()
    env["FG_PHASE1_SWEEP"] = "1"
    env["FG_PHASE1_METRICS_DIR"] = str(out_metrics)
    env["FG_PHASE1_OUTPUT_DIR"] = str(out_models)
    env["FG_PHASE1_SEED"] = str(seed)
    env["FG_PHASE1_ROUNDS"] = str(rounds)
    env.setdefault("FG_PHASE1_DEVICE", os.environ.get("FG_PHASE1_DEVICE", "cuda"))
    env.setdefault("FG_PHASE1_GPUS_PER_CLIENT", os.environ.get("FG_PHASE1_GPUS_PER_CLIENT", "0.25"))
    env.setdefault("RAY_DEDUP_LOGS", "1")

    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "run_from_config.py"),
        str(_PROJECT_ROOT / config_path),
        "--algorithm", algorithm,
        "--seeds", str(seed),
        "--rounds", str(rounds),
    ]

    print(f"[main] >>> {key} seed={seed} rounds={rounds}")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(_PROJECT_ROOT))
    dur = time.time() - t0
    print(f"[main] <<< {key} seed={seed} rc={proc.returncode} time={dur:.1f}s")
    return proc.returncode, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of {fedguide_prior, fedguide_pg, fedguide_all}")
    ap.add_argument("--num_clients", type=int, default=4)
    ap.add_argument("--metrics_root", type=str, default="metrics/bandit2d_phase1")
    ap.add_argument("--log_dir", type=str, default="logs/bandit2d_phase1")
    ap.add_argument("--skip_pretrain", action="store_true",
                    help="don't run the Gaussian pretrain even if ckpts are missing")
    args = ap.parse_args()

    if not args.skip_pretrain:
        ensure_pretrain(num_clients=args.num_clients)

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

    print("\n========= MAIN SUMMARY =========")
    for key, s, rc, dur in summary:
        status = "OK" if rc == 0 else "FAIL"
        print(f"{key:>16s}  seed={s}  status={status}  time={dur:6.1f}s")


if __name__ == "__main__":
    main()
