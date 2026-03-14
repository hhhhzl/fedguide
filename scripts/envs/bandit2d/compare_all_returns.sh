#!/usr/bin/env bash
# Compare train/return and eval/return across FedGuide, FedKL, FedAvg
# Run after experiments complete: conda activate fedguide && bash scripts/envs/bandit2d/compare_all_returns.sh

set -e
cd "$(dirname "$0")/../../.."
PYTHON=${PYTHON:-python}
if ! command -v $PYTHON &>/dev/null; then
    PYTHON=python3
fi

echo "=== Bandit2D Return Comparison: FedGuide vs FedKL vs FedAvg ==="
echo ""

# FedGuide: saved by run_fedguide_bandit2d.py to metrics/bandit2d/fedguide/
# FedKL: existing run_from_config uses seed_0; run_fedkl saves to metrics/bandit2d/fedkl/
# FedAvg: run_fedavg saves to metrics/bandit2d/fedavg/

for label in FedGuide FedKL FedAvg; do
    case $label in
        FedGuide) path="./metrics/bandit2d/fedguide/training_history.pkl" ;;
        FedKL)    path="./metrics/bandit2d/fedkl/training_history.pkl"
                  [ ! -f "$path" ] && path="./metrics/bandit2d/fedkl/seed_0/training_history.pkl" ;;
        FedAvg)   path="./metrics/bandit2d/fedavg/training_history.pkl" ;;
    esac
    if [ -f "$path" ]; then
        $PYTHON scripts/envs/bandit2d/analyze_returns.py --history_path "$path" --label "$label" 2>&1
    else
        echo "[SKIP] $label: $path not found"
    fi
    echo ""
done

echo "=== Summary Table (copy to paper) ==="
echo "Method    | train/return (final) | eval/return (final) | train/return (best) | eval/return (best)"
echo "----------|----------------------|---------------------|---------------------|-------------------"
