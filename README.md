# FedGuide


## Installation

**Requirements**
- Python ≥ 3.10

**Tested configurations**
- Ubuntu 22.04
- CUDA 11.8.0 / 12.1.1
- PyTorch 2.1.0 / 2.2.0

**Install Mujoco && Mujoco-py (only support for Linux):**
```
./scripts/setup/install_mujoco_linux.sh
./scripts/setup/install_mujoco_py_linux.sh
```
Note: make sure your "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" is activate by
```
source ~/.bashrc
```
**Environment Setup**
```
./scripts/setup/setup.sh
```


## Post-Train Plot