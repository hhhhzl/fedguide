#!/usr/bin/env bash
# Run the full bandit2d Phase-1 pipeline:
#   1. ensure Gaussian behaviour priors are pretrained (fast, ~10s)
#   2. baselines sweep (fedavg / fedkl / fedrep / fedmomentum / fmarl / fedrl)
#   3. main FedGuide sweep (prior / pg / all)
#   4. all plots + density-eval markdown table
#
# Usage:
#   bash scripts/envs/bandit2d/run_all.sh                     # 1 seed × 60 rounds
#   bash scripts/envs/bandit2d/run_all.sh --seeds "0 1" --rounds 60
set -euo pipefail
cd "$(dirname "$0")/../../.."

SEEDS="0"
ROUNDS=60
ONLY_MAIN=""
ONLY_BASE=""
SKIP_BASE=0
SKIP_MAIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)        SEEDS="$2"; shift 2 ;;
    --rounds)       ROUNDS="$2"; shift 2 ;;
    --only-main)    ONLY_MAIN="--only $2"; shift 2 ;;
    --only-base)    ONLY_BASE="--only $2"; shift 2 ;;
    --skip-base)    SKIP_BASE=1; shift ;;
    --skip-main)    SKIP_MAIN=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

read -r -a SEED_ARR <<< "$SEEDS"

if [[ "$SKIP_BASE" -eq 0 ]]; then
  echo "===== [run_all] baselines ====="
  python scripts/envs/bandit2d/run_baselines.py \
    --seeds "${SEED_ARR[@]}" --rounds "$ROUNDS" $ONLY_BASE
fi

if [[ "$SKIP_MAIN" -eq 0 ]]; then
  echo "===== [run_all] main FedGuide ====="
  python scripts/envs/bandit2d/run_main.py \
    --seeds "${SEED_ARR[@]}" --rounds "$ROUNDS" $ONLY_MAIN
fi

echo "===== [run_all] plots + density_eval ====="
python scripts/envs/bandit2d/plots.py all --hetero

echo "===== [run_all] DONE ====="
