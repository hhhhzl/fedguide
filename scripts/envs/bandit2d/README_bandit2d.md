# 2D Bandit Toy Experiment Guide

This guide explains how to run federated learning experiments on the 2D Bandit environment.

## Environment Description

- **Action = State**: `a = (x, y) ∈ [-1.5, 1.5]²`
- **Global reward**: K peaks placed on the unit circle (default K=4)
- **Reward formula**: `R(a) = max_{i=1..K} exp(-||a - μ_i||² / (2σ²))`
- **Client heterogeneity**: Each client i only sees data near μ_i

## Quick Start (use client = 4 as example)

### 1. Generate Dataset (Optional, pretrain will auto-generate)

```bash
python scripts/envs/bandit2d/generate_bandit2d_data.py \
    --K 4 \
    --n_clients 4 \
    --samples_per_client 1000 \
    --sigma 0.2 \
    --local_radius 0.3 \
    --seed 42
```

### 2. Pretrain (Train Prior and Guidance)

```bash
python scripts/envs/bandit2d/pretrain_bandit2d.py \
    --num_clients 4 \
    --samples_per_client 10000 \
    --n_behavior_epochs 200 \
    --batch_size 512 \
    --lr 1e-4 \
    --device cuda
```

**Output**: Models are saved to `./model/models_prior/Bandit2D/client_{0..n-1}/final/`

### 3. Run FedGuide

```bash
python3 scripts/envs/bandit2d/run_fedguide_bandit2d.py \
    --num_clients 4 \
    --rounds 60 \
    --n_steps 200 \
    --lambda_local 0.25 \
    --lambda_guide 0.2 \
    --cpus_per_client 2
```

### 4. Run FedKL Baseline

```bash
python3 scripts/envs/bandit2d/run_fedkl_bandit2d.py \
    --num_clients 4 \
    --rounds 60 \
    --n_steps 200 \
    --lambda_global 0.1 \
    --lambda_local 0.05 \
    --cpus_per_client 2
```

### 5. Run SAC Baseline
```bash
python scripts/run_from_config.py configs/bandit2d/sac.yaml --algorithm sac
```
Or simply (algorithm is auto-detected from config path):
```bash
python scripts/run_from_config.py configs/bandit2d/sac.yaml
```
Visualize:
```bash
python scripts/envs/bandit2d/visualize_rl_training.py \
    --config configs/bandit2d/sac.yaml \
    --output_dir ./plots/bandit2d/sac
```

### 6. Run PPO Baseline
```bash
python scripts/run_from_config.py configs/bandit2d/ppo.yaml --algorithm ppo
```
Or simply (algorithm is auto-detected from config path):
```bash
python scripts/run_from_config.py configs/bandit2d/ppo.yaml
```
Visualize:
```bash
python scripts/envs/bandit2d/visualize_rl_training.py \
    --config configs/bandit2d/ppo.yaml \
    --output_dir ./plots/bandit2d/ppo
```
Note: The same visualization script works for both SAC and PPO baselines. It automatically detects the algorithm type and displays appropriate metrics (e.g., Q Value for SAC, Value and Entropy for PPO).

### 7. Visualize Prior Models

After pretraining, you can visualize the prior models:

**Visualize aggregated prior (recommended):**
```bash
python3 scripts/envs/bandit2d/visualize_prior_aggregated.py \
    --base_path ./model/models_prior/Bandit2D \
    --client_ids 0 1 2 3 \
    --ckpt_dir final \
    --output_path ./prior_aggregated.png
```

**Visualize aggregated + individual client priors:**
```bash
python3 scripts/envs/bandit2d/visualize_prior_aggregated.py \
    --base_path ./model/models_prior/Bandit2D \
    --client_ids 0 1 2 3 \
    --ckpt_dir final \
    --output_path ./prior_comparison.png
```

### 8. Visualize Training Results
```bash
python3 scripts/envs/bandit2d/visualize_bandit2d.py \
    --metrics_path ./metrics/bandit2d_fedguide/bandit2d_metrics.pkl \
    --metrics_fedkl_path ./metrics/bandit2d_fedkl/bandit2d_metrics.pkl \
    --fedguide_history ./metrics/bandit2d_fedguide/training_history.pkl \
    --fedkl_history ./metrics/bandit2d_fedkl/training_history.pkl \
    --output_path ./data/figures/bandit2d_comparison.png \
    --comparison
```

```bash
python3 scripts/envs/bandit2d/calc_summary.py \
    --history_path ./metrics/bandit2d_fedkl/training_history.pkl \
    --label FedKL
```

## Parameter Description

### Dataset Generation Parameters
- `--K`: Number of peaks (default: 4)
- `--n_clients`: Number of clients (default: 4)
- `--samples_per_client`: Number of data samples per client (default: 1000)
- `--sigma`: Standard deviation for reward function (default: 0.2)
- `--local_radius`: Data sampling radius for each client (default: 0.3)

### Pretrain Parameters
- `--n_behavior_epochs`: Number of prior training epochs (default: 1500)
- `--batch_size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 1e-4)
- `--guidance_mode`: Guidance mode: "off", "warmup", "interleave" (default: "off")

### FedGuide Parameters
- `--rounds`: Number of federated learning rounds (default: 60)
- `--n_steps`: Number of steps collected per round (default: 200)
- `--lambda_local`: Local loss weight (default: 0.25)
- `--lambda_guide`: Guidance weight (default: 0.2)

### FedKL Parameters
- `--lambda_global`: Global KL penalty coefficient (default: 0.1)
- `--lambda_local`: Local KL penalty coefficient (default: 0.05)
- `--update_epochs`: Number of update epochs per round (default: 10)

## Important Notes

1. **Pretrain is required**: FedGuide needs to run pretrain first to train the prior model
2. **Environment registration**: Bandit2D environment is automatically registered, can be used directly with `env_id="Bandit2D"`
3. **Data format**: Dataset uses `TrajectoryDataset` format, returns concatenated `[obs, action]`
4. **Model path**: Pretrain models are automatically saved, FedGuide will attempt to load them (loading logic needs to be implemented)
5. **Prior visualization**: After pretraining, use `visualize_prior_aggregated.py` to visualize the aggregated prior model, which represents the global prior distribution learned from all clients

## Verify Environment

Test if the environment works correctly:

```bash
python -c "
from fedguide.envs.bandit2d import Bandit2D
env = Bandit2D(K=4, sigma=0.2)
obs, _ = env.reset()
print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')
print(f'Peak locations: {env.get_peak_locations()}')
action = env.action_space.sample()
obs, reward, done, _, _ = env.step(action)
print(f'Action: {action}, Reward: {reward:.4f}')
"
```

## Troubleshooting

1. **Import errors**: Make sure to run scripts from the project root directory
2. **CUDA errors**: If no GPU is available, pretrain will automatically use CPU
3. **Model loading failures**: Check if the pretrain output path is correct
