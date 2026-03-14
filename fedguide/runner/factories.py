"""
Factory functions for creating agents, trainers, and environments.

Register these factories in the unified runner registry to enable
dynamic creation of components based on configuration.
"""

from typing import Dict, Any, List, Optional, Callable
import torch
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)


# Registry for factory functions
class RunnerRegistry:
    """Registry for environment and algorithm factory functions."""
    
    def __init__(self):
        self._env_factories: Dict[str, Callable] = {}
        self._agent_factories: Dict[str, Callable] = {}
        self._trainer_factories: Dict[str, Callable] = {}
        self._federated_client_factories: Dict[str, Callable] = {}
        self._federated_server_factories: Dict[str, Callable] = {}
        self._hooks: Dict[tuple, List] = {}  # (env_type, algorithm) -> list of hooks
    
    def register_env_factory(self, env_type: str, factory: Callable):
        """Register an environment factory function."""
        self._env_factories[env_type] = factory
    
    def register_agent_factory(self, algorithm: str, factory: Callable):
        """Register an agent factory function."""
        self._agent_factories[algorithm] = factory
    
    def register_trainer_factory(self, algorithm: str, factory: Callable):
        """Register a trainer factory function."""
        self._trainer_factories[algorithm] = factory
    
    def register_federated_client_factory(self, algorithm: str, factory: Callable):
        """Register a federated client factory function."""
        self._federated_client_factories[algorithm] = factory
    
    def register_federated_server_factory(self, algorithm: str, factory: Callable):
        """Register a federated server factory function."""
        self._federated_server_factories[algorithm] = factory
    
    def register_hook(self, env_type: str, algorithm: str, hook: Any):
        """Register a hook for environment-specific logic."""
        key = (env_type, algorithm)
        if key not in self._hooks:
            self._hooks[key] = []
        self._hooks[key].append(hook)
    
    def create_env(self, env_type: str, config: Dict[str, Any], **kwargs):
        """Create an environment using registered factory."""
        factory = self._env_factories.get(env_type)
        if factory is None:
            raise ValueError(f"No factory registered for environment type: {env_type}")
        return factory(config, **kwargs)
    
    def create_agent(self, algorithm: str, env, config: Dict[str, Any], **kwargs):
        """Create an agent using registered factory."""
        factory = self._agent_factories.get(algorithm.lower())
        if factory is None:
            raise ValueError(f"No factory registered for algorithm: {algorithm}")
        return factory(env, config, **kwargs)
    
    def create_trainer(self, algorithm: str, agent, env, datasets, config: Dict[str, Any], **kwargs):
        """Create a trainer using registered factory."""
        factory = self._trainer_factories.get(algorithm.lower())
        if factory is None:
            raise ValueError(f"No factory registered for algorithm: {algorithm}")
        return factory(agent, env, datasets, config, **kwargs)
    
    def create_federated_client_fn(self, algorithm: str, config: Dict[str, Any], **kwargs):
        """Create a federated client function using registered factory."""
        factory = self._federated_client_factories.get(algorithm.lower())
        if factory is None:
            raise ValueError(f"No factory registered for federated algorithm: {algorithm}")
        return factory(config, **kwargs)
    
    def create_federated_server(self, algorithm: str, config: Dict[str, Any], **kwargs):
        """Create a federated server using registered factory."""
        factory = self._federated_server_factories.get(algorithm.lower())
        if factory is None:
            raise ValueError(f"No factory registered for federated algorithm: {algorithm}")
        return factory(config, **kwargs)
    
    def get_hooks(self, env_type: str, algorithm: str) -> List:
        """Get hooks for a specific environment and algorithm."""
        key = (env_type, algorithm.lower())
        return self._hooks.get(key, [])


# Global registry instance
_registry = RunnerRegistry()


def get_registry() -> RunnerRegistry:
    """Get the global registry instance."""
    return _registry


# ============= Environment Factories =============

def _create_bandit2d_env(config: Dict[str, Any], **kwargs):
    """Create Bandit2D environment."""
    from fedguide.envs.bandit2d import Bandit2D
    seed = kwargs.get('seed', config.get('seed', 42))
    return Bandit2D(
        K=config.get('K', 4),
        sigma=config.get('sigma', 0.2),
        seed=seed
    )


def _create_reacher_env(config: Dict[str, Any], **kwargs):
    """Create Reacher environment."""
    from fedguide.envs.reacher import CustomizedReacherEnv
    from gymnasium.wrappers import TimeLimit
    from fedguide.utils.seeds import set_all_seeds
    
    seed = kwargs.get('seed', config.get('seed', 42))
    
    # Load metadata if provided
    metadata_path = config.get('metadata_path')
    if metadata_path and os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        client_configs = metadata.get('clients', [])
        
        if client_configs:
            # Use first client config or select based on num_clients
            num_clients = config.get('num_clients')
            if num_clients and num_clients <= len(client_configs):
                client_config = client_configs[0]  # Use first client for training
            else:
                client_config = client_configs[0]
            
            variant = client_config.get('variant', 'medium-v2')
            env = TimeLimit(
                CustomizedReacherEnv(
                    qpos_high_low=client_config["qpos_high_low"],
                    action_noise=client_config["action_noise"],
                    reward_scale=client_config["reward_scale"],
                    angle_noise=client_config["angle_noise"],
                    variant=variant
                ),
                max_episode_steps=50
            )
            set_all_seeds(seed, env)
            return env
    
    # Default configuration
    env = TimeLimit(
        CustomizedReacherEnv(
            qpos_high_low=[[-0.2, 0.2], [-0.2, 0.2]],
            action_noise=[0, 0],
            reward_scale=1.0,
            angle_noise=0.0,
            variant='medium-v2'
        ),
        max_episode_steps=50
    )
    set_all_seeds(seed, env)
    return env


def _create_d4rl_env(config: Dict[str, Any], **kwargs):
    """Create D4RL environment."""
    import gymnasium as gym
    import d4rl
    
    env_name = config.get('env_name', 'halfcheetah-medium-v2')
    env = gym.make(env_name)
    
    seed = kwargs.get('seed', config.get('seed', 42))
    env.reset(seed=seed)
    return env


def _create_minari_env(config: Dict[str, Any], **kwargs):
    """Create Minari environment."""
    import gymnasium as gym
    import minari
    from fedguide.utils.seeds import set_all_seeds
    
    dataset_id = config.get('dataset_id')
    env_name = config.get('env_name')
    seed = kwargs.get('seed', config.get('seed', 42))
    
    if dataset_id:
        dataset = minari.load_dataset(dataset_id)
        env = dataset.recover_environment()
    elif env_name:
        env = gym.make(env_name)
    else:
        raise ValueError("Either dataset_id or env_name must be provided for Minari environment")
    
    env.reset(seed=seed)
    set_all_seeds(seed, env)
    return env


# Register environment factories
_registry.register_env_factory('bandit2d', _create_bandit2d_env)
_registry.register_env_factory('reacher_hetero', _create_reacher_env)
_registry.register_env_factory('reacher', _create_reacher_env)
_registry.register_env_factory('d4rl', _create_d4rl_env)
_registry.register_env_factory('minari', _create_minari_env)


# ============= Agent Factories =============

def _create_ppo_agent(env, config: Dict[str, Any], **kwargs):
    """Create PPO agent."""
    from fedguide.baselines.ppo.agent import PPOAgent
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = kwargs.get('action_low')
    action_high = kwargs.get('action_high')
    device = kwargs.get('device', 'cpu')
    
    return PPOAgent(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config.get('hidden_dim', 256),
        lr=config.get('lr', 3e-4),
        gamma=config.get('gamma', 0.99),
        clip_eps=config.get('clip_eps', 0.2),
        gae_lambda=config.get('gae_lambda', 0.95),
        entropy_coef=config.get('entropy_coef', 0.01),
        value_coef=config.get('value_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        action_std=config.get('action_std', 0.1),
        learnable_std=config.get('learnable_std', True),
        device=device,
        action_low=action_low,
        action_high=action_high,
    )


def _create_sac_agent(env, config: Dict[str, Any], **kwargs):
    """Create SAC agent."""
    from fedguide.baselines.sac.agent import SACAgent
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = kwargs.get('action_low')
    action_high = kwargs.get('action_high')
    device = kwargs.get('device', 'cpu')
    
    return SACAgent(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=config.get('hidden_dim', 256),
        lr=config.get('lr', 3e-4),
        gamma=config.get('gamma', 0.99),
        tau=config.get('tau', 0.005),
        alpha=config.get('alpha', 0.2),
        device=device,
        action_low=action_low,
        action_high=action_high,
        action_std=config.get('action_std', 0.1),
    )


# Register agent factories
_registry.register_agent_factory('ppo', _create_ppo_agent)
_registry.register_agent_factory('sac', _create_sac_agent)


# ============= Trainer Factories =============

def _create_ppo_trainer(agent, env, datasets, config: Dict[str, Any], **kwargs):
    """Create PPO trainer."""
    from fedguide.baselines.ppo.trainer import CentralPPOTrainer
    
    device = kwargs.get('device', 'cpu')
    
    return CentralPPOTrainer(
        agent=agent,
        datasets=datasets,
        env=env,
        steps_per_round=config.get('steps_per_round', 2000),
        update_epochs=config.get('update_epochs', 4),
        minibatch_size=config.get('minibatch_size'),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        eval_episodes=config.get('eval_episodes', 10),
        eval_stochastic_samples=config.get('eval_stochastic_samples', 64),
        device=device,
        render_eval=config.get('render_eval', False),
        render_mode=config.get('render_mode', 'video'),
        render_save_dir=config.get('render_save_dir'),
        render_every_n_rounds=config.get('render_every_n_rounds', 10),
        render_episodes=config.get('render_episodes', 1),
    )


def _create_sac_trainer(agent, env, datasets, config: Dict[str, Any], **kwargs):
    """Create SAC trainer."""
    from fedguide.baselines.sac.trainer import CentralSACTrainer
    
    device = kwargs.get('device', 'cpu')
    
    # For SAC, use update_steps instead of steps_per_round for offline training
    # If steps_per_round is provided, use it; otherwise use update_steps
    update_steps = config.get('steps_per_round', config.get('update_steps', 1000))
    
    return CentralSACTrainer(
        agent=agent,
        datasets=datasets,
        env=env,
        batch_size=config.get('batch_size', 256),
        update_steps=update_steps,
        gamma=config.get('gamma', 0.99),
        eval_episodes=config.get('eval_episodes', 10),
        eval_stochastic_samples=config.get('eval_stochastic_samples', 64),
        device=device,
        render_eval=config.get('render_eval', False),
        render_mode=config.get('render_mode', 'video'),
        render_save_dir=config.get('render_save_dir'),
        render_every_n_rounds=config.get('render_every_n_rounds', 10),
        render_episodes=config.get('render_episodes', 1),
    )


# Register trainer factories
_registry.register_trainer_factory('ppo', _create_ppo_trainer)
_registry.register_trainer_factory('sac', _create_sac_trainer)


# ============= Federated Client Factories =============

def _create_fedguide_client_fn(config: Dict[str, Any], **kwargs):
    """Create FedGuide client function."""
    from fedguide.fed.fedguide.client import client_fn_builder
    
    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    
    return client_fn_builder(
        env_id=env_type,
        algo=config.get('algo', 'ppo'),
        aggregate_mode=config.get('aggregate_mode', 'policy'),
        n_steps=config.get('n_steps', 200),
        lambda_local=config.get('lambda_local', 0.05),
        lambda_guide=config.get('lambda_guide', 0.05),
        lambda_guide_anneal=config.get('lambda_guide_anneal', False),
        lambda_guide_decay_rounds=config.get('lambda_guide_decay_rounds', 40),
        init_log_std=config.get('init_log_std', 0.0),
        update_epochs=config.get('update_epochs', 10),
        minibatch_size=config.get('minibatch_size', 64),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project'),
        run_name=config.get('run_name') or f"{env_type.lower()}-fedguide",
        metrics_collector=metrics_collector,
        num_clients=config.get('num_clients', 4),
    )


def _create_fedkl_client_fn(config: Dict[str, Any], **kwargs):
    """Create FedKL client function."""
    from fedguide.baselines.fedKL.client import client_fn_builder

    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')

    return client_fn_builder(
        env_id=env_type,
        algo=config.get('algo', 'ppo'),
        n_steps=config.get('n_steps', 200),
        lambda_global=config.get('lambda_global', 15.0),
        lambda_local=config.get('lambda_local', 0.05),
        update_epochs=config.get('update_epochs', 10),
        minibatch_size=config.get('minibatch_size', 64),
        clip_eps=config.get('clip_eps', 0.2),
        entropy_coef=config.get('entropy_coef', 0.01),
        init_log_std=config.get('init_log_std', 0.0),
        log_std_anneal=config.get('log_std_anneal', False),
        log_std_anneal_rounds=config.get('log_std_anneal_rounds', 40),
        log_std_anneal_target=config.get('log_std_anneal_target', -2.0),
        metrics_collector=metrics_collector,
        num_clients=config.get('num_clients', 4),
        cid_mapping_file=config.get('cid_mapping_file'),
        sigma=config.get('sigma', 0.2),
    )


def _create_fmarl_client_fn(config: Dict[str, Any], **kwargs):
    """Create FMARL client function."""
    from fedguide.baselines.fmarl.client import client_fn_builder
    
    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    
    return client_fn_builder(
        env_id=env_type,
        algo=config.get('algo', 'ppo'),
        n_steps=config.get('n_steps', 200),
        lambda_global=config.get('lambda_global', 15.0),
        lambda_local=config.get('lambda_local', 0.05),
        update_epochs=config.get('update_epochs', 10),
        minibatch_size=config.get('minibatch_size', 64),
        clip_eps=config.get('clip_eps', 0.2),
        entropy_coef=config.get('entropy_coef', 0.01),
        metrics_collector=metrics_collector,
        num_clients=config.get('num_clients', 4),
    )


def _create_fedrl_client_fn(config: Dict[str, Any], **kwargs):
    """Create FedRL client function (DQN or DDPG)."""
    from fedguide.baselines.fedrl.client import client_fn_builder
    
    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    algo = config.get('algo', 'ddpg')
    
    common = dict(
        env_id=env_type,
        algo=algo,
        gamma=float(config.get('gamma', 0.99)),
        lr=float(config.get('lr', 1e-4)),
        hidden_dim=int(config.get('hidden_dim', 256)),
        metrics_collector=metrics_collector,
        num_clients=config.get('num_clients', 4),
        merge_interval=int(config.get('merge_interval', 500)),
        batch_size=int(config.get('batch_size', 64)),
        replay_size=int(config.get('replay_size', 50000)),
        replay_initial=config.get('replay_initial'),
        eval_episodes=int(config.get('eval_episodes', 1)),
        device=str(config.get('device', 'cpu')),
    )
    if algo.lower() == 'dqn':
        return client_fn_builder(
            **common,
            epsilon=float(config.get('epsilon', 1.0)),
            epsilon_decay=float(config.get('epsilon_decay', 0.99)),
            epsilon_min=float(config.get('epsilon_min', 0.01)),
            sync_interval=int(config.get('sync_interval', 10)),
        )
    else:  # ddpg
        return client_fn_builder(
            **common,
            tau=float(config.get('tau', 0.001)),
            threshold=float(config.get('threshold', 2.0)),
            aggregate_critic=bool(config.get('aggregate_critic', False)),
            add_noise=bool(config.get('add_noise', True)),
        )


def _create_fedrep_client_fn(config: Dict[str, Any], **kwargs):
    """Create FedRep client function."""
    from fedguide.baselines.fedrep.client import client_fn_builder
    
    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    
    return client_fn_builder(
        env_id=env_type,
        algo='fedrep',
        n_steps=int(config.get('n_steps', 200)),
        gamma=float(config.get('gamma', 0.99)),
        gae_lambda=float(config.get('gae_lambda', 0.95)),
        clip_eps=float(config.get('clip_eps', 0.2)),
        entropy_coef=float(config.get('entropy_coef', 0.01)),
        value_coef=float(config.get('value_coef', 0.5)),
        update_epochs=int(config.get('update_epochs', 10)),
        minibatch_size=int(config.get('minibatch_size', 64)),
        max_grad_norm=float(config.get('max_grad_norm', 0.5)),
        hidden_dim=int(config.get('hidden_dim', 256)),
        lr=float(config.get('lr', 3e-4)),
        metrics_collector=metrics_collector,
        num_clients=config.get('num_clients', 4),
    )


def _create_fedmomentum_client_fn(config: Dict[str, Any], **kwargs):
    """Create FedMomentum client function."""
    from fedguide.baselines.fedmomentum.client import client_fn_builder

    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')

    return client_fn_builder(
        env_id=env_type,
        n_steps=int(config.get('n_steps', 200)),
        gamma=float(config.get('gamma', 0.99)),
        gae_lambda=float(config.get('gae_lambda', 0.95)),
        clip_eps=float(config.get('clip_eps', 0.2)),
        entropy_coef=float(config.get('entropy_coef', 0.01)),
        value_coef=float(config.get('value_coef', 0.5)),
        update_epochs=int(config.get('update_epochs', 4)),
        minibatch_size=int(config.get('minibatch_size', 64)),
        max_grad_norm=float(config.get('max_grad_norm', 0.5)),
        hidden_dim=int(config.get('hidden_dim', 256)),
        lr=float(config.get('lr', 3e-4)),
        algorithm=config.get('algorithm_type', 'svrpg'),
        reference_update_freq=config.get('reference_update_freq', 5),
        use_svrpg=config.get('use_svrpg', True),
        hessian_alpha=config.get('hessian_alpha', 0.1),
        use_diagonal_approx=config.get('use_diagonal_approx', True),
        fisher_update_freq=config.get('fisher_update_freq', 1),
        use_fisher_info=config.get('use_fisher_info', True),
        metrics_collector=metrics_collector,
        num_clients=config.get('num_clients', 4),
    )


# Register federated client factories
_registry.register_federated_client_factory('fedguide', _create_fedguide_client_fn)
_registry.register_federated_client_factory('fedkl', _create_fedkl_client_fn)
_registry.register_federated_client_factory('fmarl', _create_fmarl_client_fn)
_registry.register_federated_client_factory('fedrl', _create_fedrl_client_fn)
_registry.register_federated_client_factory('fedrep', _create_fedrep_client_fn)
_registry.register_federated_client_factory('fedmomentum', _create_fedmomentum_client_fn)


# ============= Federated Server Factories =============

def _create_fedguide_server(config: Dict[str, Any], **kwargs):
    """Create FedGuide server strategy."""
    from fedguide.fed.fedguide.server import FedGuideStrategy as FedGuideServer
    
    num_clients = config.get('num_clients', 4)
    evaluate_fn = kwargs.get('evaluate_fn')
    
    return FedGuideServer(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        evaluate_fn=evaluate_fn,
        moe_enable=config.get('moe_enable', True),
        num_experts_prior=config.get('num_experts_prior', 1),
        num_experts_guidance=config.get('num_experts_guidance', 1),
    )


def _create_fedkl_server(config: Dict[str, Any], **kwargs):
    """Create FedKL server strategy."""
    from fedguide.baselines.fedKL.server import FedKLStrategy
    
    num_clients = config.get('num_clients', 4)
    evaluate_fn = kwargs.get('evaluate_fn')
    
    return FedKLStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        evaluate_fn=evaluate_fn,
    )


def _create_fmarl_server(config: Dict[str, Any], **kwargs):
    """Create FMARL server strategy."""
    from fedguide.baselines.fmarl.server import FMARLStrategy as FMARLServer
    
    num_clients = config.get('num_clients', 4)
    evaluate_fn = kwargs.get('evaluate_fn')
    
    return FMARLServer(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        evaluate_fn=evaluate_fn,
    )


def _create_fedrl_server(config: Dict[str, Any], **kwargs):
    """Create FedRL server strategy."""
    from fedguide.baselines.fedrl.server import FedRLStrategy
    
    num_clients = config.get('num_clients', 4)
    evaluate_fn = kwargs.get('evaluate_fn')
    
    return FedRLStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        evaluate_fn=evaluate_fn,
    )


def _create_fedrep_server(config: Dict[str, Any], **kwargs):
    """Create FedRep server strategy."""
    from fedguide.baselines.fedrep.server import FedRepStrategy as FedRepServer
    
    num_clients = config.get('num_clients', 4)
    evaluate_fn = kwargs.get('evaluate_fn')
    
    return FedRepServer(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        evaluate_fn=evaluate_fn,
    )


def _create_fedmomentum_server(config: Dict[str, Any], **kwargs):
    """Create FedMomentum server strategy."""
    from fedguide.baselines.fedmomentum.server import FedMomentumStrategy
    
    num_clients = config.get('num_clients', 4)
    evaluate_fn = kwargs.get('evaluate_fn')
    
    return FedMomentumStrategy(
        momentum_beta=config.get('momentum_beta', 0.9),
        server_lr=config.get('server_lr', 0.001),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        evaluate_fn=evaluate_fn,
    )


# Register federated server factories
_registry.register_federated_server_factory('fedguide', _create_fedguide_server)
_registry.register_federated_server_factory('fedkl', _create_fedkl_server)
_registry.register_federated_server_factory('fmarl', _create_fmarl_server)
_registry.register_federated_server_factory('fedrl', _create_fedrl_server)
_registry.register_federated_server_factory('fedrep', _create_fedrep_server)
_registry.register_federated_server_factory('fedmomentum', _create_fedmomentum_server)


# Register hooks for environment-specific logic
# This import must be at the end to avoid circular imports
try:
    from fedguide.runner.hooks import register_default_hooks
    register_default_hooks(_registry)
except ImportError:
    # If hooks module isn't available, continue without hooks
    pass

