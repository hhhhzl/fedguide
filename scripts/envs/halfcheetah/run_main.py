"""Run the 3 main FedGuide variants on halfcheetah.

Variants in ``configs/halfcheetah/main/``:
    fedguide_prior  (Theorem 3, prior aggregated only)
    fedguide_pg     (Theorem 4, prior + value-guidance aggregated)
    fedguide_all    (Theorem 5, policy + prior + guidance aggregated)

This script also handles the "ensure pretrain" step: if the per-client
DiffusionGuidance priors at
``./model/models_prior/HalfCheetah/client_{0..7}/final/{torch_prior.pth,guidance_sdice.pth}``
are missing, it runs the diffusion pretrain before launching any sweep
job. Pass ``--skip_pretrain`` to bypass.

Usage:
    python scripts/envs/halfcheetah/run_main.py --seeds 0 --rounds 50
    python scripts/envs/halfcheetah/run_main.py --only fedguide_prior,fedguide_pg
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
    ("fedguide_prior", "fedguide", "configs/halfcheetah/main/fedguide_prior.yaml"),
    ("fedguide_pg",    "fedguide", "configs/halfcheetah/main/fedguide_pg.yaml"),
    ("fedguide_all",   "fedguide", "configs/halfcheetah/main/fedguide_all.yaml"),
]


def _halfcheetah_priors_present(num_clients: int = 8,
                            root: str = "./model/models_prior") -> bool:
    base = _PROJECT_ROOT / root / "HalfCheetah"
    for cid in range(num_clients):
        d = base / f"client_{cid}" / "final"
        if not (d / "torch_prior.pth").exists():
            return False
    return True


def _bc_policies_present(num_clients: int = 8,
                         root: str = "./model/bc_policy") -> bool:
    base = _PROJECT_ROOT / root / "HalfCheetah"
    for cid in range(num_clients):
        if not (base / f"client_{cid}" / "final" / "policy.pth").exists():
            return False
    return True


def ensure_pretrain(num_clients: int = 8,
                    save_root: str = "./model/models_prior",
                    device: str = "cuda",
                    n_behavior_epochs: int = 300,
                    rollout_steps: int = 5000,
                    guidance_warmup_epochs: int = 100) -> None:
    """Run the halfcheetah diffusion pretrain if the ckpts aren't on disk.

    Defaults are scaled down (300 epochs / 5000 transitions) so first-time
    sweep fits in ~30min on one GPU. Bump back up to 1500/20000 for final
    paper runs where prior quality matters more.
    """
    if _halfcheetah_priors_present(num_clients=num_clients, root=save_root):
        print(f"[pretrain] already present at {save_root}/HalfCheetah/")
        return
    print(f"[pretrain] running diffusion pretrain → {save_root}/HalfCheetah/ "
          f"(n_behavior_epochs={n_behavior_epochs}, rollout_steps={rollout_steps})")
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "envs" / "halfcheetah" / "_pretrain.py"),
        "--num_clients", str(num_clients),
        "--save_root", save_root,
        "--device", device,
        "--n_behavior_epochs", str(n_behavior_epochs),
        "--rollout_steps", str(rollout_steps),
        "--guidance_warmup_epochs", str(guidance_warmup_epochs),
    ]
    res = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"pretrain failed (rc={res.returncode})")
    print("[pretrain] done.")


def ensure_bc_pretrain(num_clients: int = 8,
                       save_root: str = "./model/bc_policy",
                       device: str = "cuda",
                       behaviour: str = "random",
                       rollout_steps: int = 5000,
                       epochs: int = 100) -> None:
    """Run BC pretrain (warm-start) if the per-client policy ckpts are missing."""
    if _bc_policies_present(num_clients=num_clients, root=save_root):
        print(f"[bc_pretrain] already present at {save_root}/HalfCheetah/")
        return
    print(f"[bc_pretrain] running BC pretrain → {save_root}/HalfCheetah/ ...")
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "envs" / "halfcheetah" / "_bc_pretrain.py"),
        "--num_clients", str(num_clients),
        "--save_root", save_root,
        "--device", device,
        "--behaviour", behaviour,
        "--rollout_steps", str(rollout_steps),
        "--epochs", str(epochs),
    ]
    res = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"bc_pretrain failed (rc={res.returncode})")
    print("[bc_pretrain] done.")


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
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of {fedguide_prior, fedguide_pg, fedguide_all}")
    ap.add_argument("--num_clients", type=int, default=8)
    ap.add_argument("--metrics_root", type=str, default="metrics/halfcheetah_phase1")
    ap.add_argument("--log_dir", type=str, default="logs/halfcheetah_phase1")
    ap.add_argument("--skip_pretrain", action="store_true",
                    help="don't run the diffusion pretrain even if ckpts are missing")
    ap.add_argument("--skip_bc", action="store_true",
                    help="don't run the BC warm-start pretrain even if ckpts are missing")
    args = ap.parse_args()

    if not args.skip_pretrain:
        ensure_pretrain(num_clients=args.num_clients)
    if not args.skip_bc:
        ensure_bc_pretrain(num_clients=args.num_clients)

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
