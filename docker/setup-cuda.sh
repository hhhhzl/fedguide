#!/usr/bin/env bash
set -euo pipefail

# Only run if:
#  1) AUTO_INSTALL_CUDA_ON_GPU=1
#  2) Some sign of NVIDIA GPU is present (nvidia-smi or /proc/driver/nvidia)
if [[ "${AUTO_INSTALL_CUDA_ON_GPU:-1}" != "1" ]]; then
  echo "[CUDA setup] Skipping (AUTO_INSTALL_CUDA_ON_GPU != 1)"
  exit 0
fi

if ! command -v sudo >/dev/null 2>&1; then
  apt-get update && apt-get install -y --no-install-recommends sudo && rm -rf /var/lib/apt/lists/*
fi

has_gpu="0"
if command -v nvidia-smi >/dev/null 2>&1; then
  has_gpu="1"
elif [[ -e /proc/driver/nvidia/version ]]; then
  has_gpu="1"
elif [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES}" != "void" ]]; then
  has_gpu="1"
fi

if [[ "${has_gpu}" != "1" ]]; then
  echo "[CUDA setup] No GPU detected; skipping CUDA installation."
  exit 0
fi

echo "[CUDA setup] GPU detected. Installing CUDA user-space libraries (CUDA ${CUDA_VERSION:-11-8})..."

# Add NVIDIA CUDA apt repo (Ubuntu 22.04)
tmpdeb="/tmp/cuda-keyring.deb"
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o "${tmpdeb}"
sudo dpkg -i "${tmpdeb}" && rm -f "${tmpdeb}"
sudo apt-get update

# Install CUDA runtime (or toolkit/devel if you want compilers)
# For a leaner image use cuda-runtime; for development use cuda-toolkit.
sudo apt-get install -y --no-install-recommends \
  cuda-runtime-${CUDA_VERSION}

# (Optional) cuDNN from NVIDIA apt (may require agreement; if it fails, comment/remove)
# sudo apt-get install -y --no-install-recommends libcudnn8 libcudnn8-dev

sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

echo "[CUDA setup] Done."