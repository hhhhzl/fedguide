#!/usr/bin/env bash
# Pretrain DiffusionGuidance(UNet) prior + SDICE_Critic guidance for ALL kept envs.
#
# Outputs:  ./model/models_prior/<EnvName>/client_<i>/final/torch_prior.pth
#           ./model/models_prior/<EnvName>/client_<i>/final/guidance_sdice.pth
# Envs:     halfcheetah, walker, hopper, reacher, metaworld, bandit2d
#
# Usage:
#   bash scripts/pretrain_prior.sh                  # all envs
#   bash scripts/pretrain_prior.sh halfcheetah      # one env
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/pretrain_prior

if [ "$#" -eq 0 ]; then
  ENVS=(halfcheetah walker hopper reacher metaworld bandit2d)
else
  ENVS=("$@")
fi

run_one() {
  env="$1"
  script="scripts/envs/${env}/_pretrain.py"
  if [ ! -f "$script" ]; then
    echo "[prior] SKIP $env: $script not found"
    return 0
  fi
  echo "===== Prior pretrain: $env =====" | tee -a logs/pretrain_prior/${env}.log
  python -u "$script" \
      --num_clients 8 --n_behavior_epochs 40 --batch_size 512 \
      --guidance_mode warmup --guidance_warmup_epochs 80 \
      --guidance_scale_warmup_epochs 30 --device cuda \
      --save_root ./model/models_prior >> logs/pretrain_prior/${env}.log 2>&1
  echo "[prior] rc=$? :: $env"
}

for env in "${ENVS[@]}"; do
  run_one "$env"
done

echo "===== Prior pretrain done for: ${ENVS[*]} ====="
