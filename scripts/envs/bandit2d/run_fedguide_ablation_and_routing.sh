#!/usr/bin/env bash
# FedGuide Bandit2D diagnostics:
# 1) baseline (shared prior/guidance)
# 2) ablation (disable shared pull)
# 3) routed experts (client-specific expert routing)

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
  "baseline(shared prior/guidance)" \
  "configs/bandit2d/fedguide.yaml" \
  "metrics/bandit2d/fedguide/bandit2d_metrics.pkl"

run_case \
  "ablation(no shared prior/guidance pull)" \
  "configs/bandit2d/fedguide_ablation_noshared.yaml" \
  "metrics/bandit2d/fedguide_ablation_noshared/bandit2d_metrics.pkl"

run_case \
  "routed experts(bandit2d experimental switch)" \
  "configs/bandit2d/fedguide_bandit2d_routed.yaml" \
  "metrics/bandit2d/fedguide_routed/bandit2d_metrics.pkl"

echo ""
echo "Done. Compare round1 vs last-round heterogeneity across the three cases."
