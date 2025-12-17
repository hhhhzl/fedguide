# Centralized SAC Baseline

This module implements a centralized Soft Actor-Critic (SAC) baseline that learns from multiple clients' data without federated aggregation.

## Architecture

- **No Federated Aggregation**: All client data is merged into a single replay buffer
- **Centralized Training**: One central SAC agent trains on the mixed data
- **Off-Policy Learning**: Uses replay buffer for sample-efficient learning

## Components

### `SACAgent`
- Actor network (policy)
- Two Q-networks (critics) for double Q-learning
- Target networks with soft updates
- Supports both training and evaluation modes

### `CentralSACTrainer`
- Merges multiple client datasets into replay buffer
- Performs off-policy training with SAC algorithm
- Supports evaluation on environment
- Collects training metrics

## Usage

### Running on Bandit2D

```bash
# Basic usage
python scripts/envs/bandit2d/run_sac_centralized_bandit2d.py \
    --num_clients 4 \
    --rounds 100 \
    --update_steps 1000 \
    --batch_size 256

# With custom parameters
python scripts/envs/bandit2d/run_sac_centralized_bandit2d.py \
    --num_clients 4 \
    --rounds 200 \
    --update_steps 2000 \
    --batch_size 512 \
    --lr 1e-3 \
    --hidden_dim 512 \
    --output_dir ./results/sac_bandit2d
```

### Programmatic Usage

```python
from fedguide.baselines.sac import SACAgent, CentralSACTrainer
from fedguide.envs.bandit2d import Bandit2D
from fedguide.datasets.base import TransitionDataset

# Load datasets (list of TransitionDataset from multiple clients)
datasets = [...]  # Your datasets

# Create environment
env = Bandit2D(K=4, sigma=0.2)

# Create agent
agent = SACAgent(
    state_dim=2,
    action_dim=2,
    hidden_dim=256,
    lr=3e-4,
)

# Create trainer
trainer = CentralSACTrainer(
    agent=agent,
    datasets=datasets,
    env=env,
    batch_size=256,
    update_steps=1000,
)

# Train
for round_num in range(100):
    metrics = trainer.train_one_round()
    print(f"Round {round_num}: loss={metrics['loss']:.4f}")
```

## Metrics

The trainer returns the following metrics:

- `loss`: Total loss (actor + critic)
- `train/loss/actor`: Actor loss
- `train/loss/critic`: Critic loss
- `train/q_value`: Average Q-value
- `train/q_value_min`: Minimum Q-value
- `train/buffer_size`: Replay buffer size
- `eval/return`: Evaluation return (if evaluation enabled)
- `data/num_clients`: Number of clients
- `data/total_transitions`: Total number of transitions

## Differences from FedKL/FedAvg

| Feature | FedKL/FedAvg | Centralized SAC |
|---------|--------------|-----------------|
| Architecture | Federated (server-client) | Centralized (no server) |
| Data | Local per client | Merged into single buffer |
| Training | Local training → Aggregation | Direct training on mixed data |
| Communication | Client-server communication | No communication needed |

