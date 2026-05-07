#!/usr/bin/env bash
# Re-analyze (curves + summary + videos manifest) using already-completed
# sweep metrics under metrics/reacher_phase1/. Does NOT trigger any training.
#
# Usage:
#   bash scripts/envs/reacher/analyze.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

python scripts/envs/reacher/plots.py all
echo "===== [analyze] DONE ====="
