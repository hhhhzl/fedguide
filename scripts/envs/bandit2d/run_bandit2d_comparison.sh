#!/usr/bin/env bash
# Run Bandit2D comparison: FedGuide, FedKL, FedAvg
# Usage: conda activate fedguide && bash scripts/envs/bandit2d/run_bandit2d_comparison.sh
# Multi-seed: SEEDS="0,1,2,3,4" bash scripts/envs/bandit2d/run_bandit2d_comparison.sh

set -e
cd "$(dirname "$0")/../../.."
PROJECT_ROOT="$(pwd)"

echo "=== Bandit2D Comparison: FedGuide vs FedKL vs FedAvg ==="
echo "Project root: $PROJECT_ROOT"

# Seeds: single seed or comma-separated for multi-seed (default: 0 for low memory)
SEEDS=${SEEDS:-0}
ROUNDS=60
N_STEPS=200

# Use venv or conda python (prefer .venv_bandit for reproducibility)
if [ -f "$PROJECT_ROOT/.venv_bandit/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv_bandit/bin/python"
else
    PYTHON=${PYTHON:-python}
fi
if ! command -v $PYTHON &>/dev/null 2>&1; then
    PYTHON=python3
fi
echo "Using: $($PYTHON --version 2>&1)"
echo "Seeds: $SEEDS"

# 1. FedGuide (requires pretrain with prior + guidance)
if [ ! -f "$PROJECT_ROOT/model/models_prior/Bandit2D/client_0/final/guidance_sdice.pth" ]; then
    echo "Running pretrain for FedGuide (prior + guidance)..."
    $PYTHON scripts/envs/bandit2d/pretrain_bandit2d.py \
        --num_clients 4 \
        --samples_per_client 10000 \
        --n_behavior_epochs 200 \
        --batch_size 512 \
        --guidance_mode interleave \
        --device cuda 2>/dev/null || $PYTHON scripts/envs/bandit2d/pretrain_bandit2d.py \
        --num_clients 4 \
        --samples_per_client 10000 \
        --n_behavior_epochs 200 \
        --batch_size 512 \
        --guidance_mode interleave \
        --device cpu
fi

# Use run_from_config for multi-seed if SEEDS contains comma
if [[ "$SEEDS" == *","* ]]; then
    echo ""
    echo ">>> Running FedGuide (multi-seed: $SEEDS)..."
    $PYTHON scripts/run_from_config.py configs/bandit2d/fedguide.yaml --algorithm fedguide --seeds "$SEEDS" 2>&1 | tee /tmp/fedguide_bandit2d.log

    echo ""
    echo ">>> Running FedKL (multi-seed: $SEEDS)..."
    $PYTHON scripts/run_from_config.py configs/bandit2d/fedkl.yaml --algorithm fedkl --seeds "$SEEDS" 2>&1 | tee /tmp/fedkl_bandit2d.log

    echo ""
    echo ">>> Running FedAvg (multi-seed: $SEEDS, uses FedKL with lambda=0)..."
    $PYTHON scripts/run_from_config.py configs/bandit2d/fedavg.yaml --algorithm fedkl --seeds "$SEEDS" 2>&1 | tee /tmp/fedavg_bandit2d.log
else
    echo ""
    echo ">>> Running FedGuide (seed $SEEDS)..."
    $PYTHON scripts/envs/bandit2d/run_fedguide_bandit2d.py \
        --num_clients 4 \
        --rounds $ROUNDS \
        --seed $SEEDS 2>&1 | tee /tmp/fedguide_bandit2d.log
    ray stop --force 2>/dev/null || true
    sleep 3

    echo ""
    echo ">>> Running FedKL (seed $SEEDS)..."
    $PYTHON scripts/envs/bandit2d/run_fedkl_bandit2d.py \
        --num_clients 4 \
        --rounds $ROUNDS \
        --seed $SEEDS 2>&1 | tee /tmp/fedkl_bandit2d.log
    ray stop --force 2>/dev/null || true
    sleep 3

    echo ""
    echo ">>> Running FedAvg (seed $SEEDS)..."
    $PYTHON scripts/envs/bandit2d/run_fedavg_bandit2d.py \
        --num_clients 4 \
        --rounds $ROUNDS \
        --seed $SEEDS 2>&1 | tee /tmp/fedavg_bandit2d.log
    ray stop --force 2>/dev/null || true
fi

echo ""
echo "=== Done. Summary (multi-seed if SEEDS had multiple): ==="
$PYTHON scripts/envs/bandit2d/analyze_returns.py --metrics_dir ./metrics/bandit2d/fedguide --label FedGuide
$PYTHON scripts/envs/bandit2d/analyze_returns.py --metrics_dir ./metrics/bandit2d/fedkl --label FedKL
$PYTHON scripts/envs/bandit2d/analyze_returns.py --metrics_dir ./metrics/bandit2d/fedavg --label FedAvg
echo ""
echo "Single-seed: use --history_path ./metrics/bandit2d/fedguide/seed_0/training_history.pkl"
