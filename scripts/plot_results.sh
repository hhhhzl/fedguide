#!/usr/bin/env bash
# Unified figure generation: aggregate post-train eval/return per (env, algo)
# across seeds, plot per-env curves with ±std band.
#
# Inputs:  ./metrics/<env>_phase1/<algo>/seed_<s>/training_history.pkl
# Outputs: ./plots/posttrain/<env>_posttrain.{png,pdf}
#          ./plots/posttrain/all_envs_summary.{png,pdf}
#
# Usage:  bash scripts/plot_results.sh
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p plots/posttrain
python -u scripts/plot_posttrain.py "$@"
