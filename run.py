# run_simulation.py
import flwr as fl
from flwr.common import Context
from flwr.server import ServerConfig
from fedguide.fed.fedguide.server import FedGuideServer
from fedguide.fed.client import FedGuideClient
from fedguide.agents import PPOAgent
from fedguide.trainers.local_trainer import LocalTrainer
from fedguide.envs.pointmaze_narrow import PointMazeNarrow
from fedguide.priors.diffusion_prior import DiffusionGuidance
import random, torch, numpy as np
from viz_clients import main as visual


def client_fn(context: Context):
    cid = str(getattr(context, "client_id", None) or getattr(context, "node_id", None) or "0")

    base = 42 + (abs(hash(cid)) % 10000)
    torch.manual_seed(base)
    np.random.seed(base)
    random.seed(base)

    env = PointMazeNarrow()
    agent = PPOAgent(2, 2)
    _hp = dict(
        clip_eps=0.20,
        lr=5e-4,
        entropy_coef=0.02,
        value_coef=0.5,
        gae_lambda=0.95,
        max_grad_norm=0.5,
    )
    for k, v in _hp.items():
        if hasattr(agent, k):
            setattr(agent, k, v)

    prior = DiffusionGuidance(state_dim=2, action_dim=2)
    if hasattr(prior, "temperature"):
        prior.temperature = 0.9

    trainer = LocalTrainer(
        agent,
        env,
        n_steps=512,
        prior=prior,
        lambda_local=0.25,
        lambda_guide=0.2,
    )

    client = FedGuideClient(agent, env, trainer, aggregate_mode='policy')
    setattr(client, "cid", cid)
    return client.to_client()


def main():
    config = ServerConfig(
        num_rounds=80
    )
    strategy = FedGuideServer(
        fraction_fit=1.0,
        min_fit_clients=5,
        min_available_clients=5,
        on_fit_config_fn=lambda rnd: {"server_round": rnd}
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        strategy=strategy,
        config=config,
        client_resources={"num_cpus": 4}
    )


if __name__ == "__main__":
    main()
    visual()

