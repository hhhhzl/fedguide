#!/usr/bin/env bash
# Generate all per-env metadata + offline datasets + env preview images.
#
# This is the entry point for all data/asset generation. Run once before
# pretrain/run/plot.
#
# Outputs:
#   data/bandit2d/metadata.json + offline buffers
#   data/halfcheetah/metadata.json
#   data/walker/metadata.json
#   data/hopper/metadata.json
#   data/reacher/{metadata.json (= ablation A, main), metadata_B.json, metadata_C.json, offline buffers}
#   data/metaworld/metadata.json
#   assets/envs/<env>/{single.png, heterogeneity.png}
#
# Usage:
#   bash scripts/generate_all.sh                       # generate everything
#   bash scripts/generate_all.sh --skip-images         # data only
#   bash scripts/generate_all.sh --only-images         # images only
#   bash scripts/generate_all.sh --envs halfcheetah    # one env's metadata
set -uo pipefail
cd "$(dirname "$0")/.."

SKIP_IMAGES=0
ONLY_IMAGES=0
ENVS="bandit2d halfcheetah walker hopper reacher metaworld"

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-images) SKIP_IMAGES=1; shift;;
    --only-images) ONLY_IMAGES=1; shift;;
    --envs)        ENVS="${2//,/ }"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

mkdir -p logs/generate_all

if [ "$ONLY_IMAGES" -eq 0 ]; then
  for env in $ENVS; do
    case "$env" in
      bandit2d)
        echo "===== bandit2d data + metadata ====="
        python -u scripts/generate_data/generate_bandit2d_data.py \
            > logs/generate_all/bandit2d.log 2>&1 \
            && echo "  ok" || echo "  FAIL (see logs/generate_all/bandit2d.log)"
        ;;
      halfcheetah)
        echo "===== halfcheetah metadata (mild) ====="
        python -u scripts/generate_data/generate_halfcheetah_metadata.py \
            --profile mild --n 8 \
            > logs/generate_all/halfcheetah.log 2>&1 \
            && echo "  ok" || echo "  FAIL"
        ;;
      walker|hopper)
        echo "===== $env metadata (locomotion hetero) ====="
        python -u scripts/generate_data/generate_locomotion_metadata.py \
            --env $env --n 8 \
            > logs/generate_all/${env}.log 2>&1 \
            && echo "  ok" || echo "  FAIL"
        ;;
      reacher)
        echo "===== reacher data + hetero metadata ====="
        python -u scripts/generate_data/generate_reacher_data.py \
            > logs/generate_all/reacher.log 2>&1 \
            && echo "  reacher base ok" || echo "  reacher base FAIL"
        python -u scripts/generate_data/generate_reacher_hetero_metadata.py \
            >> logs/generate_all/reacher.log 2>&1 \
            && echo "  reacher hetero A/B/C ok" || echo "  reacher hetero FAIL"
        ;;
      metaworld)
        echo "===== metaworld ML10 metadata ====="
        python -u scripts/generate_data/generate_metaworld_metadata.py \
            > logs/generate_all/metaworld.log 2>&1 \
            && echo "  ok" || echo "  FAIL"
        ;;
      *) echo "[warn] unknown env: $env";;
    esac
  done
fi

if [ "$SKIP_IMAGES" -eq 0 ]; then
  echo "===== render env preview images → assets/envs/<env>/ ====="
  python -u scripts/generate_data/generate_envs.py \
      --envs $ENVS \
      > logs/generate_all/render_envs.log 2>&1 \
      && echo "  ok" || echo "  partial (see logs/generate_all/render_envs.log)"
fi

echo "===== generate_all DONE ====="
