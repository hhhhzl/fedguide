#!/usr/bin/env bash
# FedGuide Bandit2D: baseline run + heterogeneity check.
# (Former ablation/routed configs were removed; use fedguide.yaml toggles if needed.)

set -e
cd "$(dirname "$0")/../../.."
PROJECT_ROOT="$(pwd)"

if [ -f "$PROJECT_ROOT/.venv_bandit/bin/python" ]; then
  PYTHON="$PROJECT_ROOT/.venv_bandit/bin/python"
else
  PYTHON=${PYTHON:-python3}
fi

ROUNDS=${ROUNDS:-60}
SEED=${SEED:-0}

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
  "fedguide(baseline)" \
  "configs/bandit2d/fedguide.yaml" \
  "metrics/bandit2d/fedguide/bandit2d_metrics.pkl"

echo ""
echo "Done."
