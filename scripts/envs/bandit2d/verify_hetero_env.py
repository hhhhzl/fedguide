#!/usr/bin/env python3
"""
Standalone verification: train 4 independent policies, each with preferred_peak 0,1,2,3.
No federation - proves env heterogeneity works. Output: 4 distinct peaks.
"""
import sys
import os
import pickle
import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _project_root)

from fedguide.envs.bandit2d import Bandit2D
from fedguide.baselines.fedKL.agent import FedKLAgent
from fedguide.baselines.fedKL.trainer import FedKLTrainer
from fedguide.utils.bandit2d_metrics import Bandit2DMetricsCollector

def main():
    num_clients = 4
    n_steps = 1000
    rounds = 60
    seed = 0
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create 4 envs with different preferred_peak (0,1,2,3 -> right, top, left, bottom)
    # sigma=0.4 for wider reward region so policy gets gradient signal
    envs = [Bandit2D(K=4, sigma=0.4, seed=seed + i, preferred_peak=i) for i in range(num_clients)]
    
    # Create 4 agents (same init)
    agents = []
    for i in range(num_clients):
        agent = FedKLAgent(state_dim=2, action_dim=2, hidden_dim=256, lr=3e-4, device="cpu",
                          init_log_std=-1.0)
        # Set global = local so KL penalty is 0 (vanilla PPO)
        agent.set_parameters(agent.get_parameters())
        agents.append(agent)
    
    # Train each agent on its own env (no federation)
    for cid in range(num_clients):
        env = envs[cid]
        agent = agents[cid]
        trainer = FedKLTrainer(agent=agent, env=env, n_steps=n_steps, gamma=0.99, gae_lambda=0.95,
                              clip_eps=0.2, entropy_coef=0.01, value_coef=0.5, update_epochs=10,
                              minibatch_size=64, lambda_global=0.0, lambda_local=0.0, device="cpu")
        for r in range(rounds):
            result = trainer.train_one_round()
            if (r + 1) % 10 == 0:
                ret = result.get('train/return', 0) or 0
                print(f"Client {cid} (peak {cid}) round {r+1}: return ~{ret:.3f}")
    
    # Collect policy densities and save
    collector = Bandit2DMetricsCollector(save_dir="/tmp", grid_size=200, bounds=(-1.5, 1.5))
    round_metrics = {'round': rounds, 'client_metrics': {}, 'server_metrics': {}}
    for cid in range(num_clients):
        round_metrics['client_metrics'][cid] = collector.evaluate_on_grid(agents[cid], client_id=cid, round_num=rounds)
    
    out_dir = os.path.join(_project_root, "metrics/bandit2d/verify_hetero")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bandit2d_metrics.pkl")
    data = {
        'client_actions': {},
        'metrics_history': [round_metrics],
        'grid_points': collector.grid_points,
        'X': collector.X,
        'Y': collector.Y,
        'grid_size': collector.grid_size,
        'bounds': collector.bounds,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved to {out_path}")
    return out_path

if __name__ == "__main__":
    main()
