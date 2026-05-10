#!/usr/bin/env bash
# Resume after the (still-running) antmaze fedavg finishes:
#   1. wait for fedavg PID 270842 to exit
#   2. diffusion + BC pretrain
#   3. fedguide_all
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/chain
LOG=logs/chain/antmaze_resume_$(date +%Y%m%d_%H%M%S).log
echo "[chain] log: $LOG"

run() {
    echo "" >> "$LOG"
    echo "===== [chain] $* =====" | tee -a "$LOG"
    "$@" >> "$LOG" 2>&1
    rc=$?
    echo "[chain] rc=$rc :: $*" | tee -a "$LOG"
    return 0
}

echo "===== STAGE 0: wait for orphaned fedavg (PID 270842) to exit =====" | tee -a "$LOG"
while kill -0 270842 2>/dev/null; do
    sleep 60
done
echo "[chain] fedavg exited at $(date)" | tee -a "$LOG"

echo "===== STAGE 1: antmaze diffusion prior pretrain ====="
run python -u scripts/envs/antmaze/_pretrain.py \
    --num_clients 8 --save_root ./model/models_prior \
    --device cuda --n_behavior_epochs 200 --rollout_steps 3000 \
    --guidance_warmup_epochs 50

echo "===== STAGE 2: antmaze BC pretrain ====="
run python -u scripts/envs/antmaze/_bc_pretrain.py \
    --num_clients 8 --save_root ./model/bc_policy \
    --device cuda --rollout_steps 3000 --epochs 50

echo "===== STAGE 3: antmaze fedguide_all ====="
run env FG_PHASE1_DEVICE=cuda FG_PHASE1_GPUS_PER_CLIENT=0.125 \
    python -u scripts/run_from_config.py configs/antmaze/main/fedguide_all.yaml \
    --algorithm fedguide --seeds 0 --rounds 50

echo "===== [chain] DONE =====" | tee -a "$LOG"
