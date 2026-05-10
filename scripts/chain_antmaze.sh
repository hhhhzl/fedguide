#!/usr/bin/env bash
# Antmaze chain: pretrain (diffusion prior + BC) → fedavg → fedguide_all.
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/chain
LOG=logs/chain/antmaze_$(date +%Y%m%d_%H%M%S).log
echo "[chain] log: $LOG"

run() {
    echo "" >> "$LOG"
    echo "===== [chain] $* =====" | tee -a "$LOG"
    "$@" >> "$LOG" 2>&1
    rc=$?
    echo "[chain] rc=$rc :: $*" | tee -a "$LOG"
    return 0
}

echo "===== STAGE 1: antmaze diffusion prior pretrain ====="
run python -u scripts/envs/antmaze/_pretrain.py \
    --num_clients 8 --save_root ./model/models_prior \
    --device cuda --n_behavior_epochs 200 --rollout_steps 3000 \
    --guidance_warmup_epochs 50

echo "===== STAGE 2: antmaze BC pretrain ====="
run python -u scripts/envs/antmaze/_bc_pretrain.py \
    --num_clients 8 --save_root ./model/bc_policy \
    --device cuda --rollout_steps 3000 --epochs 50

echo "===== STAGE 3: antmaze fedavg (single seed × 50 rounds) ====="
mkdir -p logs/antmaze_phase1
run env FG_PHASE1_DEVICE=cuda FG_PHASE1_GPUS_PER_CLIENT=0.125 \
    python -u scripts/run_from_config.py configs/antmaze/baseline/fedavg.yaml \
    --algorithm fedkl --seeds 0 --rounds 50

echo "===== STAGE 4: antmaze fedguide_all ====="
run env FG_PHASE1_DEVICE=cuda FG_PHASE1_GPUS_PER_CLIENT=0.125 \
    python -u scripts/run_from_config.py configs/antmaze/main/fedguide_all.yaml \
    --algorithm fedguide --seeds 0 --rounds 50

echo "===== [chain] DONE =====" | tee -a "$LOG"
