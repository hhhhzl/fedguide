# run_simulation.py
import flwr as fl
from flwr.common import Context
from flwr.server import ServerConfig
from fedguide.fed.server import FedGuideServer
from fedguide.fed.client import FedGuideClient
from fedguide.agents import BaseAgent, PPOAgent, SACAgent
from fedguide.trainers.local_trainer import LocalTrainer
from fedguide.envs.make_env import make_env
from fedguide.envs.pointmaze_narrow import PointMazeNarrow
from fedguide.priors.diffusion_prior import DiffusionGuidance, SimpleDiffusionPrior
import random, torch, numpy as np


def client_fn(context: Context):
    try:
        cid = str(context.node_id)
    except Exception:
        cid = str(getattr(context, "client_id", "0"))

    base = 42 + (abs(hash(cid)) % 10000)
    torch.manual_seed(base)
    np.random.seed(base)
    random.seed(base)

    env = PointMazeNarrow()
    agent = PPOAgent(2, 2)
    prior = SimpleDiffusionPrior(state_dim=2, action_dim=2)
    trainer = LocalTrainer(agent, env, prior=prior, lambda_local=0.2, lambda_guide=0.1)

    c = FedGuideClient(agent, env, trainer).to_client()
    setattr(c, "cid", cid)
    return c


def main():
    config = ServerConfig(num_rounds=50)
    strategy = FedGuideServer(num_experts=2)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        strategy=strategy,
        config=config,
        client_resources={"num_cpus": 4}
    )


if __name__ == "__main__":
    main()

