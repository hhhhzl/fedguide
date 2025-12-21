# Configuration Files for SAC Training

This directory contains YAML configuration files for running SAC training on different environments.

## Usage

Run training from a configuration file:

```bash
# From project root
python scripts/run_sac_from_config.py configs/bandit2d/sac.yaml

# Or with relative path
python scripts/run_sac_from_config.py bandit2d/sac.yaml
```

## Multi-Seed Support

The configuration files support multiple random seeds. If `seed` is a list, the script will run training for each seed sequentially.

### Example: Single Seed
```yaml
seed: 42  # Run with seed 42 only
```

### Example: Multiple Seeds
```yaml
seed: [0, 1, 2, 3, 4]  # Run 5 times with different seeds
```

### Override Seeds from Command Line
```bash
# Override seeds from config file
python scripts/run_sac_from_config.py configs/bandit2d/sac.yaml --seeds "10,20,30"
```

## Configuration File Structure

### Bandit2D Environment
- `env_type: "bandit2d"` - Environment type
- `data_dir` - Directory containing bandit2d data
- `num_clients` - Number of clients
- `K` - Number of peaks in bandit
- `sigma` - Standard deviation for reward function
- `collect_logprob` - Whether to collect policy log probability distribution on grid (default: true)
- `logprob_grid_size` - Grid size for logprob evaluation (default: 200)
- `logprob_bounds` - Bounds for logprob grid evaluation [min, max] (default: [-1.5, 1.5])

### D4RL Environment
- `env_type: "d4rl"` - Environment type
- `env_name` - D4RL environment name (e.g., "reacher-medium-v2")
- `num_clients` - Number of clients (for data splitting)

### Minari Environment
- `env_type: "minari"` - Environment type
- `dataset_id` - Minari dataset ID (e.g., "D4RL/pointmaze/medium-v2", "D4RL/maze2d/umaze-v1", "D4RL/antmaze/umaze-v0")
- `env_name` - Optional: Environment name (if dataset doesn't provide it, e.g., "PointMaze_UMaze-v3")
- `num_clients` - Number of clients (for data splitting)
- `download` - Whether to download dataset if not found locally (default: true)

### Common Parameters
- `rounds` - Number of training rounds
- `update_steps` - Number of update steps per round
- `batch_size` - Batch size for training
- `hidden_dim` - Hidden dimension for networks
- `lr` - Learning rate
- `gamma` - Discount factor
- `tau` - Soft update coefficient
- `alpha` - Temperature parameter (entropy regularization)
- `eval_episodes` - Number of episodes for evaluation
- `output_dir` - Directory to save models
- `metrics_dir` - Directory to save metrics
- `save_every` - Save results every N rounds
- `device` - Device to use ("auto", "cpu", or "cuda")
- `seed` - Random seed (int or list of ints)

## Available Configurations

- `bandit2d/sac.yaml` - Bandit2D environment
- `reacher/sac.yaml` - Reacher D4RL environment
- `maze2d/sac.yaml` - Maze2D D4RL environment
- `antmaze/sac.yaml` - AntMaze D4RL environment
- `flow/sac.yaml` - Flow D4RL environment
- `minari/sac.yaml` - Minari environment (pointmaze, maze2d, antmaze, etc.)

## Output

For each seed, the training results are saved to separate subdirectories:
- Models: `{output_dir}/seed_{seed}/checkpoint_round_{round_num}.pkl`
- Metrics: `{metrics_dir}/seed_{seed}/training_history.pkl`

### Example Output Structure

If you run with `seed: [0, 1, 2]` and `output_dir: "./model/policy/bandit2d/sac"`, the structure will be:

```
./model/policy/bandit2d/sac/
  ├── seed_0/
  │   ├── checkpoint_round_10.pkl
  │   ├── checkpoint_round_20.pkl
  │   └── ...
  ├── seed_1/
  │   ├── checkpoint_round_10.pkl
  │   └── ...
  └── seed_2/
      ├── checkpoint_round_10.pkl
      └── ...

./metrics/bandit2d/sac/
  ├── seed_0/
  │   └── training_history.pkl
  ├── seed_1/
  │   └── training_history.pkl
  └── seed_2/
      └── training_history.pkl
```

Each seed's results are completely isolated in its own subdirectory, making it easy to compare results across different seeds.

## Visualization

After training, you can visualize the results using the visualization script:

### Using Config File (Recommended for Multiple Seeds)

```bash
# Visualize all seeds from config
python scripts/envs/bandit2d/visualize_sac_training.py --config configs/bandit2d/sac.yaml

# Visualize specific seeds
python scripts/envs/bandit2d/visualize_sac_training.py --config configs/bandit2d/sac.yaml --seeds "0,1,2"

# Save plots to directory
python scripts/envs/bandit2d/visualize_sac_training.py \
    --config configs/bandit2d/sac.yaml \
    --output_dir ./plots/bandit2d/sac
```

### Using Direct History Path (Single Seed)

```bash
# Visualize a single seed
python scripts/envs/bandit2d/visualize_sac_training.py \
    --history_path ./metrics/bandit2d/sac/seed_0/training_history.pkl \
    --output_dir ./plots/bandit2d/sac/seed_0
```

### Visualization Options

- `--combined`: Also create a combined plot with dual y-axis (loss and return)
- `--plot_logprob`: Plot policy log probability distribution (default: True)
- `--logprob_rounds`: Specific round numbers to plot logprob (if None, auto-selects first, middle, last)

The visualization script generates:
- `training_curves.png`: Training curves (loss, return, Q value)
- `training_combined.png`: Combined plot (if `--combined` is used)
- `policy_logprob_distribution.png`: Policy log probability distribution heatmap (if `--plot_logprob` is used)

