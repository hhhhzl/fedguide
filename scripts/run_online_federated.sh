#!/usr/bin/env bash
# Run online federated learning experiments across all (env, algorithm, seed).
#
# Algorithms (all start from BC warm-start by default):
#   fedavg, fedkl, ppo, fedrl, fedguide, fedguide_a, fedguide_p
#
# Optional algorithms (kept supported, run only if requested):
#   fedmomentum, fedrep, fmarl, mfpo, sac
#
# Outputs:  ./metrics/<env>_phase1/<algo>/seed_<s>/training_history.pkl
#
# Usage:
#   bash scripts/run_online_federated.sh                                    # default 5-algo grid × 3 seeds × 5 envs
#   bash scripts/run_online_federated.sh --envs halfcheetah                 # one env
#   bash scripts/run_online_federated.sh --algos fedguide                   # one algo
#   bash scripts/run_online_federated.sh --seeds 0,1,2                      # custom seeds
#   bash scripts/run_online_federated.sh --rounds 50                        # override rounds
#   bash scripts/run_online_federated.sh --envs halfcheetah --algos fedguide --seeds 0 --rounds 50
set -uo pipefail
cd "$(dirname "$0")/.."

ENVS="halfcheetah walker hopper reacher metaworld"
ALGOS="fedavg fedkl ppo fedrl fedguide fedguide_a fedguide_p"
SEEDS="0 1 2"
ROUNDS=""  # empty = use config's rounds

while [ $# -gt 0 ]; do
  case "$1" in
    --envs)   ENVS="${2//,/ }"; shift 2;;
    --algos)  ALGOS="${2//,/ }"; shift 2;;
    --seeds)  SEEDS="${2//,/ }"; shift 2;;
    --rounds) ROUNDS="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

mkdir -p logs/online_federated

config_path() {
  env="$1"; algo="$2"
  # fedguide variants live in main/, baselines in baseline/
  case "$algo" in
    fedguide|fedguide_a|fedguide_p)
      echo "configs/${env}/main/${algo}.yaml" ;;
    *)
      echo "configs/${env}/baseline/${algo}.yaml" ;;
  esac
}

# algorithm → flwr factory key (the --algorithm CLI flag)
algo_key() {
  case "$1" in
    fedguide|fedguide_a|fedguide_p) echo "fedguide" ;;
    fedavg|fedkl)                   echo "fedkl"    ;;
    fedrl)                          echo "fedrl"    ;;
    *)                              echo "$1"        ;;
  esac
}

for env in $ENVS; do
  for algo in $ALGOS; do
    cfg=$(config_path "$env" "$algo")
    if [ ! -f "$cfg" ]; then
      echo "[skip] $env/$algo: config $cfg missing"
      continue
    fi
    for seed in $SEEDS; do
      log="logs/online_federated/${env}_${algo}_s${seed}.log"
      echo "===== $env / $algo / seed=$seed ====="
      cmd=(python -u scripts/run_from_config.py "$cfg"
           --algorithm "$(algo_key $algo)" --seeds "$seed")
      [ -n "$ROUNDS" ] && cmd+=(--rounds "$ROUNDS")
      "${cmd[@]}" > "$log" 2>&1
      echo "[run] rc=$? :: $env / $algo / seed=$seed"
    done
  done
done

echo "===== online federated DONE ====="
echo "Envs:  $ENVS"
echo "Algos: $ALGOS"
echo "Seeds: $SEEDS"
