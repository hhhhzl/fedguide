#!/usr/bin/env bash
# Pretrain BC policies (one per client) for ALL kept environments.
#
# Outputs:  ./model/bc_policy/<EnvName>/client_<i>/final/policy.pth
# Envs:     halfcheetah, walker, hopper, reacher, metaworld, bandit2d (skip if no data)
#
# Usage:
#   bash scripts/pretrain_bc.sh                 # all envs
#   bash scripts/pretrain_bc.sh halfcheetah     # one env
#   bash scripts/pretrain_bc.sh halfcheetah walker hopper
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/pretrain_bc

if [ "$#" -eq 0 ]; then
  ENVS=(halfcheetah walker hopper reacher metaworld)
else
  ENVS=("$@")
fi

run_one() {
  env="$1"
  script="scripts/envs/${env}/_bc_pretrain.py"
  if [ ! -f "$script" ]; then
    echo "[bc] SKIP $env: $script not found"
    return 0
  fi
  echo "===== BC pretrain: $env =====" | tee -a logs/pretrain_bc/${env}.log
  python -u "$script" \
      --num_clients 8 --epochs 40 --batch_size 512 --device cuda \
      --save_root ./model/bc_policy >> logs/pretrain_bc/${env}.log 2>&1
  echo "[bc] rc=$? :: $env"
}

for env in "${ENVS[@]}"; do
  run_one "$env"
done

echo "===== BC pretrain done for: ${ENVS[*]} ====="
