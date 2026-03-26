#!/usr/bin/env bash
# Round-2 FedGuide diagnostics with strict controls:
# 1) shared prior/guidance
# 2) ablation (lambda_guide=0)
# 3) routed experts (client-specific)

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

# Stabilize Ray+Torch on macOS OpenMP
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Using: $($PYTHON --version 2>&1)"
echo "Rounds: $ROUNDS, Seed: $SEED"

run_case() {
  local name="$1"
  local config="$2"
  local metrics="$3"
  echo ""
  echo "=== [$name] ==="
  $PYTHON scripts/envs/bandit2d/run_fedguide_bandit2d.py --config "$config" --rounds "$ROUNDS" --seed "$SEED"
  $PYTHON scripts/envs/bandit2d/check_bandit2d_hetero_metrics.py --metrics_path "$metrics" --also_round1 --round_num -1
  ray stop --force 2>/dev/null || true
  sleep 2
}

run_case \
  "debug_shared(prior+guidance)" \
  "configs/bandit2d/fedguide_debug_shared.yaml" \
  "metrics/bandit2d/fedguide_debug_shared/bandit2d_metrics.pkl"

run_case \
  "debug_ablation(lambda_guide=0)" \
  "configs/bandit2d/fedguide_debug_ablation.yaml" \
  "metrics/bandit2d/fedguide_debug_ablation/bandit2d_metrics.pkl"

run_case \
  "debug_routed(client-specific experts)" \
  "configs/bandit2d/fedguide_debug_routed.yaml" \
  "metrics/bandit2d/fedguide_debug_routed/bandit2d_metrics.pkl"

echo ""
echo "Done. Compare round1 vs last-round heterogeneity across three strict controls."
