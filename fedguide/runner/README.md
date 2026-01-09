# Runner Module Architecture

This module provides a unified interface for running RL training experiments across different environments and algorithms.

The architecture uses a **registry system** for automatic discovery and registration of runners, making it easy to add new environments and algorithms without modifying core code.

## Directory Structure

```
fedguide/runner/
├── __init__.py              # Module exports
├── registry.py              # Registry system for auto-discovery
├── auto_discover.py         # Auto-discovery mechanism
├── run_from_config.py       # Unified config runner (supports multi-seed)
├── bandit2d/
│   ├── __init__.py          # Auto-registers runners
│   ├── ppo.py              # PPO runner for Bandit2D
│   ├── sac.py              # SAC runner for Bandit2D
│   ├── fedguide.py         # FedGuide federated runner
│   ├── fedkl.py            # FedKL federated runner
│   └── _common.py          # Shared utilities for federated runners
├── d4rl/
│   ├── __init__.py          # Auto-registers runners
│   ├── ppo.py              # PPO runner for D4RL environments
│   └── sac.py              # SAC runner for D4RL environments
├── minari/
│   ├── __init__.py          # Auto-registers runners
│   ├── ppo.py              # PPO runner for Minari environments
│   └── sac.py              # SAC runner for Minari environments
└── reacher/
    ├── __init__.py          # Auto-registers runners
    ├── ppo.py              # PPO runner for Reacher (heterogeneous)
    └── sac.py              # SAC runner for Reacher (heterogeneous)
```

## Usage

### Using the Unified Config Runner

The recommended way to run training is through the unified config runner:

```bash
# PPO training
python -m fedguide.runner.run_from_config configs/bandit2d/ppo.yaml --algorithm ppo

# SAC training
python -m fedguide.runner.run_from_config configs/bandit2d/sac.yaml --algorithm sac

# Or use the unified script interface (recommended)
python scripts/run_from_config.py configs/bandit2d/ppo.yaml
python scripts/run_from_config.py configs/bandit2d/sac.yaml --algorithm sac
```

### Direct Runner Usage

You can also call individual runners directly:

```bash
# Bandit2D PPO
python -m fedguide.runner.bandit2d.ppo --config configs/bandit2d/ppo.yaml

# D4RL SAC
python -m fedguide.runner.d4rl.sac --env_name maze2d-umaze-v1 --rounds 100
```

## Environment Types

- **bandit2d**: 2D multi-armed bandit environment
- **d4rl**: D4RL benchmark environments (maze2d, antmaze, flow, reacher, etc.)
- **minari**: Minari dataset environments
- **reacher_hetero**: Reacher environment with client heterogeneity

## Algorithms

- **ppo**: Proximal Policy Optimization (on-policy)
- **sac**: Soft Actor-Critic (off-policy)
- **fedguide**: FedGuide federated learning (federated)
- **fedkl**: FedKL federated learning (federated)

## Registry System

The registry system enables automatic discovery of runners. When you import an environment module, it automatically registers its available runners:

```python
# In fedguide/runner/bandit2d/__init__.py
from fedguide.runner.registry import register_env, register_runner

register_env('bandit2d', 'bandit2d')
register_runner('bandit2d', 'ppo')
register_runner('bandit2d', 'sac')
register_runner('bandit2d', 'fedguide')
register_runner('bandit2d', 'fedkl')
```

## Adding New Environments

To add a new environment (e.g., `cartpole`):

1. Create directory: `fedguide/runner/cartpole/`
2. Create runner files: `ppo.py`, `sac.py` (with `main()` function)
3. Create `__init__.py`:
```python
from fedguide.runner.registry import register_env, register_runner

register_env('cartpole', 'cartpole')
register_runner('cartpole', 'ppo')
register_runner('cartpole', 'sac')
```

That's it! The runner will be automatically discovered and available.

## Adding New Algorithms

To add a new algorithm (e.g., `td3`) to an existing environment:

1. Create runner file: `fedguide/runner/bandit2d/td3.py` (with `main()` function)
2. Update `__init__.py`:
```python
register_runner('bandit2d', 'td3')
```

The algorithm will be automatically available for that environment.

## Scripts Directory

The `scripts/` directory now contains only a single unified interface:
- `scripts/run_from_config.py`: Unified entry point for all training (supports all algorithms and environments)

All runner implementations have been moved to `fedguide/runner/` and organized by environment type.

