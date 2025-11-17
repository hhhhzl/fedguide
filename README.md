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


torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2
flwr[simulation]>=1.9.0
gym==0.23.1
gymnasium==0.29.1
pygame==2.5.2
numpy==1.26.4
scipy==1.13.1
pot==0.9.3            # Python Optimal Transport
scikit-learn==1.5.2
einops==0.8.0
opt-einsum==3.3.0
tqdm==4.66.4
matplotlib==3.9.1
pandas==2.2.2
rich==13.7.1
diffusers==0.30.0
transformers==4.44.2
accelerate==0.33.0
psutil==5.9.8
PyYAML==6.0.2
stable_baselines3==2.4.1
wandb
mujoco==3.3.6
cython==0.29.22