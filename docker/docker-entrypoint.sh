#!/usr/bin/env bash
set -e

# Run conditional CUDA libs setup (safe & idempotent)
if [[ -x /opt/setup_cuda.sh ]]; then
  /opt/setup_cuda.sh || echo "[entrypoint] CUDA setup script exited non-zero; continuing."
fi

#if [[ -x /workspace/fedguide/scripts/setup.sh ]]; then
#  /workspace/fedguide/scripts/setup.sh || echo "[entrypoint] Project environments building."
#fi

exec "$@"