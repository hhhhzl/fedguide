#!/usr/bin/env bash
# Chain: re-run reacher fedguide_pg/_all (now that online_guidance is plumbed),
# then reacher's remaining baselines, then halfcheetah pretrain + full sweep.
# All sequential on a single GPU (no parallel jobs).
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/chain
LOG=logs/chain/run_$(date +%Y%m%d_%H%M%S).log
echo "[chain] log file: $LOG"

run() {
    echo "" >> "$LOG"
    echo "===== [chain] $* =====" | tee -a "$LOG"
    "$@" >> "$LOG" 2>&1
    rc=$?
    echo "[chain] rc=$rc :: $*" | tee -a "$LOG"
    return 0  # never abort the chain on a single failure
}

echo "===== [chain] STAGE 1: reacher fedguide_pg + fedguide_all (re-run with online_guidance) ====="
run python -u scripts/envs/reacher/run_main.py \
    --seeds 0 --rounds 50 --only fedguide_pg,fedguide_all --skip_pretrain --skip_bc

echo "===== [chain] STAGE 2: reacher remaining federated baselines ====="
run python -u scripts/envs/reacher/run_baselines.py \
    --seeds 0 --rounds 50 --only fedrep,fedmomentum,fmarl,fedrl,mfpo

echo "===== [chain] STAGE 3: halfcheetah baselines (auto-pretrain skipped — uses random init) ====="
run python -u scripts/envs/halfcheetah/run_baselines.py \
    --seeds 0 --rounds 50

echo "===== [chain] STAGE 4: halfcheetah main (auto-trigger diffusion + BC pretrain, then 3 FedGuide variants) ====="
run python -u scripts/envs/halfcheetah/run_main.py \
    --seeds 0 --rounds 50

echo "===== [chain] STAGE 5: regenerate plots/summaries ====="
run bash scripts/envs/reacher/analyze.sh
run bash scripts/envs/halfcheetah/analyze.sh

echo "===== [chain] DONE =====" | tee -a "$LOG"
