#!/bin/bash

MUJOCO_DIR="$HOME/.mujoco"
TAR_PATH="./assets/mujoco/linux/mujoco210-linux-x86_64.tar.gz"
EXTRACTED_DIR="$MUJOCO_DIR/mujoco210-linux-x86_64"
TARGET_DIR="$MUJOCO_DIR/mujoco210"

apt update
apt install -y xvfb libglew-dev
# Create MuJoCo directory
mkdir -p "$MUJOCO_DIR"
tar -xf "$TAR_PATH" -C "$MUJOCO_DIR"
if [ -d "$EXTRACTED_DIR" ]; then
    mv "$EXTRACTED_DIR" "$TARGET_DIR"
fi


if [ -n "$ZSH_VERSION" ] || [ "$(basename "$SHELL")" = "zsh" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi
touch "$SHELL_RC"
echo -e 'export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PATH="$LD_LIBRARY_PATH:$PATH"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> "$SHELL_RC"
source "$SHELL_RC"

echo "MuJoCo Install Successfully! Run Simulation To Test"

# Run simulation
# shellcheck disable=SC2164
#cd ~/.mujoco/mujoco210/bin
#./simulate ../model/humanoid.xml