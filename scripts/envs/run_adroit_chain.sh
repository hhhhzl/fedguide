#!/usr/bin/env bash
# Run pretrain + 4 fed variants for D4RL Adroit.
set -uo pipefail
cd "$(dirname "$0")/../.."

mkdir -p logs/adroit_phase1

LOG=logs/chain/adroit_$(date +%Y%m%d_%H%M%S).log
echo "[chain] log: $LOG"

run_step() {
  echo "===== $* ====="
  "$@"
  rc=$?
  echo "[chain] rc=$rc :: $*"
  return $rc
}

# Stage 1: prior + SDICE pretrain (D4RL data)
run_step python -u scripts/envs/adroit/_pretrain.py \
    --num_clients 8 \
    --d4rl_max_size 100000 --n_behavior_epochs 40 --batch_size 512 \
    --guidance_mode warmup --guidance_warmup_epochs 80 \
    --guidance_scale_warmup_epochs 30 \
    --device cuda --save_root ./model/models_prior \
    > logs/adroit_phase1/pretrain.log 2>&1

# Stage 2: BC pretrain
run_step python -u scripts/envs/adroit/_bc_pretrain.py \
    --num_clients 8 \
    --d4rl_max_size 30000 --epochs 40 --batch_size 512 \
    --device cuda --save_root ./model/bc_policy \
    > logs/adroit_phase1/bc.log 2>&1

# Stage 3: federated runs
for variant in fedavg fedguide_prior_strict fedguide_pg_p4 fedguide_all_p4; do
  algo="fedguide"
  if [ "$variant" = "fedavg" ]; then algo="fedkl"; fi
  cfg=configs/adroit/main/${variant}.yaml
  [ "$variant" = "fedavg" ] && cfg=configs/adroit/baseline/fedavg.yaml
  run_step python -u scripts/run_from_config.py "$cfg" \
      --algorithm $algo --seeds 0 --rounds 50 \
      > logs/adroit_phase1/${variant}_s0.log 2>&1
done

echo "===== adroit chain DONE ====="
