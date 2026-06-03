# FedGuide: Diffusion Prior Alignment and Value Baseline Guidance for Heterogeneous Federated Reinforcement Learning

![FedGuide policy rollouts across environments](assets/gifs/all_envs.gif)

> Round-100 evaluation rollouts. Columns are methods (FedGuide / FedGuide-A / FedGuide-P), rows are clients, grouped by environment (Reacher, Hopper, Walker2D, HalfCheetah, MetaWorld10).

## Pipeline

```
generate_all.sh  →  pretrain_bc.sh + pretrain_prior.sh  →  run_online_federated.sh  →  plot_posttrain.py
```

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

## Generate Heterogeneous Data

Per-env metadata, heterogeneous datasets, and env preview images (`assets/envs/`):

```
./scripts/generate_all.sh                 # everything (data + images)
./scripts/generate_all.sh --envs reacher  # one env
```

## Pretrain Diffusion Prior

BC warm-start policies and the DiffusionGuidance(UNet) prior + SDICE critic, one per client:

```
./scripts/pretrain_bc.sh        # → model/bc_policy/<Env>/client_<i>/
./scripts/pretrain_prior.sh     # → model/models_prior/<Env>/client_<i>/
```

Pass an env name to either script to run just one (e.g. `./scripts/pretrain_prior.sh halfcheetah`).

## FRL Training

Online federated training over the (env, algorithm, seed) grid. Algorithms: `fedavg`, `fedkl`, `fedrl`, `fedmomentum`, `fedguide`, `fedguide_a`, `fedguide_p`.

```
./scripts/run_online_federated.sh                                  # full grid (5 envs × 7 algos × 3 seeds)
./scripts/run_online_federated.sh --envs halfcheetah --algos fedguide --seeds 0
```

Outputs:
- Metrics: `metrics/<env>_phase1/<algo>/seed_<s>/training_history.pkl`
- Evaluation rollout videos: `plots/<env>/<algo>/seed_<s>/client_<i>/round_<NNNN>.mp4`

## Post-Train Plot

```
python scripts/viz/plot_posttrain.py --main   # main-env curves
```

GIF gallery of evaluation rollouts (regenerates `assets/gifs/all_envs.gif` + `index.html`):

```
python scripts/make_combined_gif.py
python scripts/make_gif_gallery.py
```