#!/usr/bin/env bash
# Re-analyze (plots + density_eval table) using already-completed sweep
# metrics under metrics/bandit2d_phase1/. Does NOT trigger any training.
#
# Useful when you've changed plotting code or want to regenerate figures
# without re-running the (slow) Flower simulations.
#
# Usage:
#   bash scripts/envs/bandit2d/analyze.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

python scripts/envs/bandit2d/plots.py all --hetero
echo "===== [analyze] DONE ====="
