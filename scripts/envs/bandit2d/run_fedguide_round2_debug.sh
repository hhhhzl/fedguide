#!/usr/bin/env bash
# FedGuide Bandit2D quick diagnostic (single config).
# (Former debug_shared / debug_ablation / debug_routed YAMLs were removed.)

set -e
cd "$(dirname "$0")/../../.."
PROJECT_ROOT="$(pwd)"

if [ -f "$PROJECT_ROOT/.venv_bandit/bin/python" ]; then
  PYTHON="$PROJECT_ROOT/.venv_bandit/bin/python"
else
  PYTHON=${PYTHON:-python3}
fi

ROUNDS=${ROUNDS:-20}
SEED=${SEED:-0}

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Using: $($PYTHON --version 2>&1)"
echo "Rounds: $ROUNDS, Seed: $SEED"

echo ""
echo "=== [fedguide] ==="
$PYTHON scripts/envs/bandit2d/run_fedguide_bandit2d.py --config configs/bandit2d/fedguide.yaml --rounds "$ROUNDS" --seed "$SEED"
$PYTHON scripts/envs/bandit2d/check_bandit2d_hetero_metrics.py --metrics_path metrics/bandit2d/fedguide/bandit2d_metrics.pkl --also_round1 --round_num -1
ray stop --force 2>/dev/null || true

echo ""
echo "Done."
