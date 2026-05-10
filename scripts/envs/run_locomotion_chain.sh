#!/usr/bin/env bash
# Run pretrain + 4 fed variants for one locomotion env (walker / ant / hopper).
# Usage:  bash scripts/envs/run_locomotion_chain.sh walker
set -uo pipefail
cd "$(dirname "$0")/../.."

if [ "$#" -lt 1 ]; then
  echo "usage: $0 {walker|ant|hopper}"; exit 2
fi
ENV="$1"
case "$ENV" in
  walker) D4RL_VAR="walker2d-medium-v2"; ENV_TITLE="Walker2D";;
  ant)    D4RL_VAR="ant-medium-v2"; ENV_TITLE="Ant";;
  hopper) D4RL_VAR="hopper-medium-v2"; ENV_TITLE="Hopper";;
  *) echo "unknown env: $ENV"; exit 2;;
esac

mkdir -p logs/${ENV}_phase1

LOG=logs/chain/${ENV}_$(date +%Y%m%d_%H%M%S).log
echo "[chain] log: $LOG"

run_step() {
  echo "===== $* ====="
  "$@"
  rc=$?
  echo "[chain] rc=$rc :: $*"
  return $rc
}

# Stage 1: prior + SDICE pretrain (D4RL medium)
run_step python -u scripts/envs/${ENV}/_pretrain.py \
    --num_clients 8 --behaviour d4rl --d4rl_variant "${D4RL_VAR}" \
    --d4rl_max_size 100000 --n_behavior_epochs 40 --batch_size 512 \
    --guidance_mode warmup --guidance_warmup_epochs 80 \
    --guidance_scale_warmup_epochs 30 \
    --device cuda --save_root ./model/models_prior \
    > logs/${ENV}_phase1/pretrain.log 2>&1

# Stage 2: BC pretrain
run_step python -u scripts/envs/${ENV}/_bc_pretrain.py \
    --num_clients 8 --behaviour d4rl --d4rl_variant "${D4RL_VAR}" \
    --d4rl_max_size 30000 --epochs 40 --batch_size 512 \
    --device cuda --save_root ./model/bc_policy \
    > logs/${ENV}_phase1/bc.log 2>&1

# Stage 3: federated runs
for variant in fedavg fedguide_prior_strict fedguide_pg_p4 fedguide_all_p4; do
  algo="fedguide"
  if [ "$variant" = "fedavg" ]; then algo="fedkl"; fi  # fedavg = fedkl with lambda_global=0
  cfg=configs/${ENV}/main/${variant}.yaml
  [ "$variant" = "fedavg" ] && cfg=configs/${ENV}/baseline/fedavg.yaml
  run_step python -u scripts/run_from_config.py "$cfg" \
      --algorithm $algo --seeds 0 --rounds 50 \
      > logs/${ENV}_phase1/${variant}_s0.log 2>&1
done

echo "===== ${ENV} chain DONE ====="
