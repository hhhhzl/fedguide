#!/usr/bin/env bash
# Re-analyze (curves + summary + videos manifest) using already-completed
# sweep metrics under metrics/halfcheetah_phase1/. Does NOT trigger any training.
#
# Usage:
#   bash scripts/envs/halfcheetah/analyze.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

python scripts/envs/halfcheetah/plots.py all
echo "===== [analyze] DONE ====="
