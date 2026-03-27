#!/usr/bin/env bash
# After FedKL (configs/reacher/fedkl.yaml) exits, run Reacher baselines sequentially on CUDA, seed 0.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

FEDKL_PATTERN="run_from_config.py configs/reacher/fedkl.yaml"

wait_for_fedkl() {
  while pgrep -f "${FEDKL_PATTERN}" >/dev/null 2>&1; do
    echo "[$(date -Iseconds)] Waiting for FedKL to finish..."
    sleep 60
  done
  echo "[$(date -Iseconds)] FedKL done (no matching python process)."
}

run_one() {
  local cfg="$1"
  local tag="$2"
  local log="$LOGDIR/reacher_${tag}_seed0_cuda_chain.log"
  echo "[$(date -Iseconds)] === ${tag}: ${cfg} ===" | tee -a "$log"
  ray stop --force 2>/dev/null || true
  sleep 2
  # PIPESTATUS[0] is python's exit code; plain $? after a pipeline is often tee's (0), and $? in else after `if pipeline` is wrong too.
  python -u scripts/run_from_config.py "${cfg}" --seeds 0 --device cuda 2>&1 | tee -a "$log"
  local ec=${PIPESTATUS[0]}
  if [ "$ec" -eq 0 ]; then
    echo "[$(date -Iseconds)] OK ${tag}" | tee -a "$log"
  else
    echo "[$(date -Iseconds)] FAILED ${tag} (python exit ${ec}; 137 often OOM/SIGKILL)" | tee -a "$log"
  fi
}

wait_for_fedkl

run_one configs/reacher/ppo.yaml ppo

echo "[$(date -Iseconds)] Chain finished." | tee -a "$LOGDIR/reacher_chain_master.log"
