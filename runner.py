import argparse, random, numpy as np, torch
import gymnasium as gym
import flwr as fl
from flwr.common.context import Context
from fedguide.fed.fedguide.server import FedGuideServer
from fedguide.fed.client import FedGuideClient
from fedguide.trainers.local_trainer import LocalTrainer
from fedguide.agents import PPOAgent, A2CAgent


def is_box1d(space) -> bool:
    try:
        import gymnasium.spaces as spaces
        return isinstance(space, spaces.Box) and len(space.shape) == 1
    except Exception:
        return hasattr(space, "low") and hasattr(space, "high") and len(space.shape) == 1


def make_env(env_id: str, seed: int):
    if env_id.lower() in ["pointmazenarrow", "pointmaze_narrow", "pointnarrow", "pointmaze-narrow"]:
        from fedguide.envs.pointmaze_narrow import PointMazeNarrow
        env = PointMazeNarrow()
        env.reset(seed=seed)
        return env

    if env_id.lower() in ["bandit2d", "bandit_2d", "2dbandit"]:
        from fedguide.envs.bandit2d import Bandit2D
        env = Bandit2D(K=4, sigma=0.2, seed=seed)
        env.reset(seed=seed)
        return env

    env = gym.make(env_id)
    env.reset(seed=seed)

    try:
        from gymnasium.wrappers import TransformObservation, ClipAction
        if is_box1d(env.observation_space):
            env = TransformObservation(env, lambda obs: np.asarray(obs, dtype=np.float32))
        if is_box1d(env.action_space):
            env = ClipAction(env)
    except Exception:
        pass
    return env


def client_fn_builder(env_id: str, algo: str):
    def client_fn(context: Context):
        cid = str(getattr(context, "client_id", None) or getattr(context, "node_id", None) or "0")
        base = 42 + (abs(hash(cid)) % 10000)
        random.seed(base)
        np.random.seed(base)
        torch.manual_seed(base)

        env = make_env(env_id, seed=base)
        obs_space, act_space = env.observation_space, env.action_space
        assert is_box1d(obs_space) and is_box1d(act_space), "Only Support 1D Box"

        state_dim = int(obs_space.shape[0])
        action_dim = int(act_space.shape[0])

        if algo.lower() == "a2c":
            agent = A2CAgent(state_dim, action_dim, lr=3e-4, entropy_coef=0.0, value_coef=0.5)
        else:
            agent = PPOAgent(state_dim, action_dim)
            _ppo_hp = dict(
                clip_eps=0.20,
                lr=3e-4,
                entropy_coef=0.02,
                value_coef=0.5,
                gae_lambda=0.95,
                max_grad_norm=0.5,
            )
            for k, v in _ppo_hp.items():
                if hasattr(agent, k):
                    setattr(agent, k, v)

        prior = None
        try:
            from fedguide.guidance.diffusion_prior import DiffusionGuidance
            prior = DiffusionGuidance(state_dim=state_dim, action_dim=action_dim)
        except Exception:
            pass

        trainer = LocalTrainer(
            env=env,
            agent=agent,
            prior=prior,
            n_steps=200,
            gamma=0.99,
        )

        client = FedGuideClient(agent, env, trainer, aggregate_mode='policy')
        setattr(client, "cid", cid)
        return client.to_client()

    return client_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id",
        type=str,
        default="Reacher-v4",
        help="PointMazeNarrow / Reacher-v4 / other Gymnasium envs"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="a2c",
        choices=["a2c", "ppo"]
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=5
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=60
    )
    parser.add_argument(
        "--cpus_per_client",
        type=int,
        default=4
    )
    args = parser.parse_args()

    strategy = FedGuideServer(
        fraction_fit=1.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
    )

    from flwr.server import ServerConfig
    config = ServerConfig(num_rounds=args.rounds)

    fl.simulation.start_simulation(
        client_fn=client_fn_builder(args.env_id, args.algo),
        num_clients=args.num_clients,
        strategy=strategy,
        config=config,
        client_resources={"num_cpus": args.cpus_per_client},
    )


if __name__ == "__main__":
    main()
    # vis()
