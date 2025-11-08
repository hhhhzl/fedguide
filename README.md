# FedGuide


## Installation

#### Env Requirements: 
  - Python = 3.10
  - ubuntu = 22.04
  - cuda = 11.8.0
  - pytorch = 2.1.0
#### Install Mujoco && Mujoco-py (only support for Linux):
```
./scripts/install_mujoco_linux.sh
./scripts/install_mujoco_py_linux.sh
```
Note: make sure your "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" is activate by
```
source ~/.bashrc
```
#### Environment Setup:
```
./scripts/setup.sh
```