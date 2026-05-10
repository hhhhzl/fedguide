#!/usr/bin/env bash
# Antmaze D4RL chain:
#   1. Pretrain diffusion prior + SDICE on per-client D4RL variant
#   2. BC pretrain policy from D4RL data (per-client variant)
#   3. fedavg (50 rounds × 8 clients)
#   4. fedguide_all (50 rounds × 8 clients)
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/chain
LOG=logs/chain/antmaze_d4rl_$(date +%Y%m%d_%H%M%S).log
echo "[chain] log: $LOG"

run() {
    echo "" >> "$LOG"
    echo "===== [chain] $* =====" | tee -a "$LOG"
    "$@" >> "$LOG" 2>&1
    rc=$?
    echo "[chain] rc=$rc :: $*" | tee -a "$LOG"
    return 0
}

echo "===== STAGE 1: antmaze diffusion prior pretrain (D4RL data) ====="
run python -u scripts/envs/antmaze/_pretrain.py \
    --num_clients 8 --save_root ./model/models_prior \
    --behaviour d4rl --d4rl_max_size 30000 \
    --device cuda --n_behavior_epochs 100 --guidance_warmup_epochs 50

echo "===== STAGE 2: antmaze BC pretrain (D4RL data) ====="
run python -u scripts/envs/antmaze/_bc_pretrain.py \
    --num_clients 8 --save_root ./model/bc_policy \
    --behaviour d4rl --d4rl_max_size 30000 \
    --device cuda --epochs 50

echo "===== STAGE 3: antmaze fedavg ====="
run env FG_PHASE1_DEVICE=cuda FG_PHASE1_GPUS_PER_CLIENT=0.125 \
    python -u scripts/run_from_config.py configs/antmaze/baseline/fedavg.yaml \
    --algorithm fedkl --seeds 0 --rounds 50

echo "===== STAGE 4: antmaze fedguide_all ====="
run env FG_PHASE1_DEVICE=cuda FG_PHASE1_GPUS_PER_CLIENT=0.125 \
    python -u scripts/run_from_config.py configs/antmaze/main/fedguide_all.yaml \
    --algorithm fedguide --seeds 0 --rounds 50

echo "===== [chain] DONE =====" | tee -a "$LOG"
