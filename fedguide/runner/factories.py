"""
Factory functions for creating agents, trainers, and environments.

Register these factories in the unified runner registry to enable
dynamic creation of components based on configuration.
"""

from typing import Dict, Any, List, Optional, Callable
import json
import numpy as np
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


def _reacher_render_mode_for_config(config: Dict[str, Any]) -> Optional[str]:
    """Gymnasium needs render_mode at env creation for env.render() to return pixels."""
    if not config.get("render_eval"):
        return None
    rm = str(config.get("render_mode", "rgb_array")).lower()
    if rm in ("video", "rgb_array"):
        return "rgb_array"
    if rm == "human":
        return "human"
    return None


def _create_reacher_env(config: Dict[str, Any], **kwargs):
    """Create Reacher environment."""
    seed = kwargs.get('seed', config.get('seed', 42))
    render_mode = _reacher_render_mode_for_config(config)
    # Headless (no DISPLAY): MuJoCo must use EGL before GLFW/X11 loads.
    if render_mode == "rgb_array":
        from fedguide.utils.mujoco_headless import ensure_mujoco_headless_gl_if_needed

        ensure_mujoco_headless_gl_if_needed()

    from fedguide.envs.reacher import CustomizedReacherEnv
    from gymnasium.wrappers import TimeLimit
    from fedguide.utils.seeds import set_all_seeds
    
    reacher_kw: Dict[str, Any] = {}
    if render_mode is not None:
        reacher_kw["render_mode"] = render_mode
    
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
                    variant=variant,
                    **reacher_kw,
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
            variant='medium-v2',
            **reacher_kw,
        ),
        max_episode_steps=50
    )
    set_all_seeds(seed, env)
    return env


def _set_gym_mujoco_render_mode_rgb(env) -> None:
    """
    Gym 0.26+ MuJoCo envs require render_mode='rgb_array' for pixel output.
    D4RL-built envs leave it unset; calling render() then errors or returns None,
    and the PPO trainer swallows exceptions — so no video is saved.
    """
    try:
        from gym.envs.mujoco import mujoco_env as _gym_mujoco
    except ImportError:
        return
    cur = env
    for _ in range(32):
        if isinstance(cur, _gym_mujoco.MujocoEnv):
            cur.render_mode = "rgb_array"
            return
        nxt = getattr(cur, "env", None) or getattr(cur, "_wrapped_env", None)
        if nxt is None:
            break
        cur = nxt


def _create_d4rl_env(config: Dict[str, Any], **kwargs):
    """Create D4RL environment (or Gymnasium HalfCheetah when metadata env=halfcheetah)."""
    from fedguide.utils.mujoco_headless import ensure_mujoco_headless_gl_if_needed

    ensure_mujoco_headless_gl_if_needed()

    metadata_path = config.get("metadata_path")
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        clients = meta.get("clients") or []
        if str(meta.get("env", "")).lower() == "halfcheetah":
            from fedguide.envs.halfcheetah_hetero import make_hetero_halfcheetah_env_from_metadata

            seed = kwargs.get("seed", config.get("seed", 42))
            return make_hetero_halfcheetah_env_from_metadata(
                metadata_path,
                0,
                seed=seed,
                render_mode=None,
                render_eval=bool(config.get("render_eval")),
            )
        if meta.get("env") == "antmaze" or (
            clients and str(clients[0].get("variant", "")).startswith("antmaze-")
        ):
            from fedguide.envs.antmaze_hetero import make_hetero_antmaze_env_from_metadata

            seed = kwargs.get("seed", config.get("seed", 42))
            return make_hetero_antmaze_env_from_metadata(
                metadata_path,
                0,
                seed=seed,
                reward_type=config.get("reward_type"),
                render_eval=bool(config.get("render_eval")),
            )

    # D4RL registers envs with the `gym` package, not Gymnasium's registry.
    import gym as gym_legacy
    import d4rl  # noqa: F401 — register envs
    from fedguide.envs.antmaze_hetero import build_d4rl_make_kwargs

    class _D4RLObservationSpaceFix(gym_legacy.Wrapper):
        """Some D4RL envs (e.g. antmaze) report a wrong Box shape vs actual reset/step obs."""

        def __init__(self, env, obs_dim: int):
            super().__init__(env)
            self.observation_space = gym_legacy.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

    env_name = config.get('env_name', 'halfcheetah-medium-v2')
    mkw = build_d4rl_make_kwargs(env_name, config)
    env = gym_legacy.make(env_name, **mkw)

    seed = kwargs.get('seed', config.get('seed', 42))
    try:
        out = env.reset(seed=seed)
    except TypeError:
        env.reset()
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        out = env.reset()
    o0 = out[0] if isinstance(out, tuple) else out
    actual_dim = int(np.asarray(o0, dtype=np.float32).ravel().shape[0])
    decl_dim = int(np.asarray(env.observation_space.shape).prod())
    if actual_dim != decl_dim:
        env = _D4RLObservationSpaceFix(env, actual_dim)

    if config.get("render_eval"):
        _set_gym_mujoco_render_mode_rgb(env)

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
    # Registry calls factory(env, config) only — device/bounds live on config, not kwargs.
    action_low = config.get('action_low')
    action_high = config.get('action_high')
    device = config.get('device', 'cpu')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # YAML may load scientific notation (e.g. 3e-4) as str; optimizers require float.
    return PPOAgent(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(config.get('hidden_dim', 256)),
        lr=float(config.get('lr', 3e-4)),
        gamma=float(config.get('gamma', 0.99)),
        clip_eps=float(config.get('clip_eps', 0.2)),
        gae_lambda=float(config.get('gae_lambda', 0.95)),
        entropy_coef=float(config.get('entropy_coef', 0.01)),
        value_coef=float(config.get('value_coef', 0.5)),
        max_grad_norm=float(config.get('max_grad_norm', 0.5)),
        action_std=float(config.get('action_std', 0.1)),
        learnable_std=bool(config.get('learnable_std', True)),
        device=device,
        action_low=action_low,
        action_high=action_high,
    )


def _create_sac_agent(env, config: Dict[str, Any], **kwargs):
    """Create SAC agent."""
    from fedguide.baselines.sac.agent import SACAgent
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = config.get('action_low')
    action_high = config.get('action_high')
    device = config.get('device', 'cpu')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return SACAgent(
        state_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(config.get('hidden_dim', 256)),
        lr=float(config.get('lr', 3e-4)),
        gamma=float(config.get('gamma', 0.99)),
        tau=float(config.get('tau', 0.005)),
        alpha=float(config.get('alpha', 0.2)),
        device=device,
        action_low=action_low,
        action_high=action_high,
        action_std=float(config.get('action_std', 0.1)),
    )


# Register agent factories
_registry.register_agent_factory('ppo', _create_ppo_agent)
_registry.register_agent_factory('sac', _create_sac_agent)


# ============= Trainer Factories =============

def _create_ppo_trainer(agent, env, datasets, config: Dict[str, Any], **kwargs):
    """Create PPO trainer."""
    from fedguide.baselines.ppo.trainer import CentralPPOTrainer
    
    device = config.get('device', 'cpu')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mb = config.get('minibatch_size')
    return CentralPPOTrainer(
        agent=agent,
        datasets=datasets,
        env=env,
        steps_per_round=int(config.get('steps_per_round', 2000)),
        update_epochs=int(config.get('update_epochs', 4)),
        minibatch_size=int(mb) if mb is not None else None,
        gamma=float(config.get('gamma', 0.99)),
        gae_lambda=float(config.get('gae_lambda', 0.95)),
        eval_episodes=int(config.get('eval_episodes', 10)),
        eval_stochastic_samples=int(config.get('eval_stochastic_samples', 64)),
        device=device,
        render_eval=bool(config.get('render_eval', False)),
        render_mode=str(config.get('render_mode', 'video')),
        render_save_dir=config.get('render_save_dir'),
        render_every_n_rounds=int(config.get('render_every_n_rounds', 10)),
        render_episodes=int(config.get('render_episodes', 1)),
    )


def _create_sac_trainer(agent, env, datasets, config: Dict[str, Any], **kwargs):
    """Create SAC trainer."""
    from fedguide.baselines.sac.trainer import CentralSACTrainer
    
    device = config.get('device', 'cpu')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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

def _federated_client_env_id(config: Dict[str, Any], env_type: str) -> str:
    """For D4RL, Flower clients need the concrete env id (e.g. antmaze-umaze-v0), not 'd4rl'."""
    if env_type == "d4rl":
        mp = config.get("metadata_path")
        if mp and os.path.isfile(mp):
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if str(meta.get("env", "")).lower() == "halfcheetah":
                    return str(meta.get("env_name") or "HalfCheetah-v4")
            except (json.JSONDecodeError, OSError):
                pass
        return config.get("env_name") or "halfcheetah-medium-v2"
    return env_type


def _create_fedguide_client_fn(config: Dict[str, Any], **kwargs):
    """Create FedGuide client function."""
    from fedguide.fed.fedguide.client import client_fn_builder
    from fedguide.utils.federated_render import reacher_env_render_mode_from_config

    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    env_id = _federated_client_env_id(config, env_type)
    return client_fn_builder(
        env_id=env_id,
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
        cid_mapping_file=config.get('cid_mapping_file'),
        sigma=config.get('sigma', 0.2),
        prior_adapt_fallback_all=config.get('prior_adapt_fallback_all', False),
        use_pretrained_models=config.get('use_pretrained_models', True),
        metadata_path=config.get('metadata_path'),
        reward_type=config.get('reward_type'),
        render_eval=bool(config.get('render_eval', False)),
        render_mode=str(config.get('render_mode', 'video')),
        render_save_dir=config.get('render_save_dir'),
        render_every_n_rounds=int(config.get('render_every_n_rounds', 10)),
        render_episodes=int(config.get('render_episodes', 5)),
        reacher_render_mode=reacher_env_render_mode_from_config(
            bool(config.get('render_eval', False)),
            str(config.get('render_mode', 'video')),
        ),
        policy_activation=str(config.get('policy_activation', 'tanh')),
        action_clamp_low=config.get('action_clamp_low'),
        action_clamp_high=config.get('action_clamp_high'),
        log_std_anneal=bool(config.get('log_std_anneal', False)),
        log_std_anneal_target=float(config.get('log_std_anneal_target', -2.0)),
        log_std_anneal_rounds=int(config.get('log_std_anneal_rounds', 40)),
        prior_dir=str(config.get('prior_dir', './model/models_prior')),
        bc_dir=config.get('bc_dir'),
        bc_env_name=config.get('bc_env_name'),
        online_guidance=bool(config.get('online_guidance', False)),
        online_prior=bool(config.get('online_prior', False)),
        guide_coef=float(config.get('guide_coef', 1.0)),
        guidance_eta=float(config.get('guidance_eta', 0.1)),
        prior_reshape=bool(config.get('prior_reshape', False)),
        reshape_beta=float(config.get('reshape_beta', 0.1)),
        dice_reward_eta=float(config.get('dice_reward_eta', 0.0)),
        dice_v_blend_alpha=float(config.get('dice_v_blend_alpha', 1.0)),
        dice_adv_beta=float(config.get('dice_adv_beta', 0.0)),
    )


def _create_fedkl_client_fn(config: Dict[str, Any], **kwargs):
    """Create FedKL client function."""
    from fedguide.baselines.fedKL.client import client_fn_builder
    from fedguide.utils.federated_render import reacher_env_render_mode_from_config

    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    dev = kwargs.get('device')
    if dev is None:
        dev = config.get('device', 'auto')
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    env_id = _federated_client_env_id(config, env_type)
    return client_fn_builder(
        env_id=env_id,
        algo=config.get('algo', 'ppo'),
        n_steps=int(config.get('n_steps', 200)),
        gamma=float(config.get('gamma', 0.99)),
        gae_lambda=float(config.get('gae_lambda', 0.95)),
        value_coef=float(config.get('value_coef', 0.5)),
        lambda_global=float(config.get('lambda_global', 15.0)),
        lambda_local=float(config.get('lambda_local', 0.05)),
        update_epochs=int(config.get('update_epochs', 10)),
        minibatch_size=int(config.get('minibatch_size', 64)),
        clip_eps=float(config.get('clip_eps', 0.2)),
        entropy_coef=float(config.get('entropy_coef', 0.01)),
        hidden_dim=int(config.get('hidden_dim', 256)),
        lr=float(config.get('lr', 3e-4)),
        max_grad_norm=float(config.get('max_grad_norm', 0.5)),
        eval_episodes=int(config.get('eval_episodes', 1)),
        init_log_std=float(config.get('init_log_std', 0.0)),
        log_std_anneal=bool(config.get('log_std_anneal', False)),
        log_std_anneal_rounds=int(config.get('log_std_anneal_rounds', 40)),
        log_std_anneal_target=float(config.get('log_std_anneal_target', -2.0)),
        metrics_collector=metrics_collector,
        num_clients=int(config.get('num_clients', 4)),
        cid_mapping_file=config.get('cid_mapping_file'),
        sigma=float(config.get('sigma', 0.2)),
        metadata_path=config.get('metadata_path'),
        reward_type=config.get('reward_type'),
        device=str(dev),
        render_eval=bool(config.get('render_eval', False)),
        render_mode=str(config.get('render_mode', 'video')),
        render_save_dir=config.get('render_save_dir'),
        render_every_n_rounds=int(config.get('render_every_n_rounds', 10)),
        render_episodes=int(config.get('render_episodes', 5)),
        reacher_render_mode=reacher_env_render_mode_from_config(
            bool(config.get('render_eval', False)),
            str(config.get('render_mode', 'video')),
        ),
        prior_dir=config.get('prior_dir'),
        prior_env_name=config.get('prior_env_name'),
    )


def _create_fmarl_client_fn(config: Dict[str, Any], **kwargs):
    """Create FMARL client function."""
    from fedguide.baselines.fmarl.client import client_fn_builder
    from fedguide.utils.federated_render import reacher_env_render_mode_from_config

    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    dev = kwargs.get('device')
    if dev is None:
        dev = config.get('device', 'auto')
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    env_id = _federated_client_env_id(config, env_type)
    return client_fn_builder(
        env_id=env_id,
        algo=config.get('algo', 'fmarl'),
        n_steps=int(config.get('n_steps', 200)),
        gamma=float(config.get('gamma', 0.99)),
        gae_lambda=float(config.get('gae_lambda', 0.95)),
        clip_eps=float(config.get('clip_eps', 0.2)),
        entropy_coef=float(config.get('entropy_coef', 0.01)),
        value_coef=float(config.get('value_coef', 0.5)),
        update_epochs=int(config.get('update_epochs', 10)),
        minibatch_size=int(config.get('minibatch_size', 64)),
        lambda_global=float(config.get('lambda_global', 15.0)),
        lambda_local=float(config.get('lambda_local', 0.05)),
        max_grad_norm=float(config.get('max_grad_norm', 0.5)),
        hidden_dim=int(config.get('hidden_dim', 256)),
        lr=float(config.get('lr', 3e-4)),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project'),
        run_name=config.get('run_name'),
        metrics_collector=metrics_collector,
        num_clients=int(config.get('num_clients', 4)),
        cid_mapping_file=config.get('cid_mapping_file'),
        sigma=float(config.get('sigma', 0.2)),
        metadata_path=config.get('metadata_path'),
        render_eval=bool(config.get('render_eval', False)),
        render_mode=str(config.get('render_mode', 'video')),
        render_save_dir=config.get('render_save_dir'),
        render_every_n_rounds=int(config.get('render_every_n_rounds', 10)),
        render_episodes=int(config.get('render_episodes', 5)),
        reacher_render_mode=reacher_env_render_mode_from_config(
            bool(config.get('render_eval', False)),
            str(config.get('render_mode', 'video')),
        ),
        device=str(dev),
    )


def _create_fedrl_client_fn(config: Dict[str, Any], **kwargs):
    """Create FedRL client function."""
    from fedguide.baselines.fedrl.client import client_fn_builder
    from fedguide.utils.federated_render import reacher_env_render_mode_from_config

    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    dev = config.get('device', 'cpu')
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    replay_initial = config.get('replay_initial')
    if replay_initial is None and str(config.get('algo', 'dqn')).lower() == 'ddpg':
        replay_initial = 1000
    elif replay_initial is not None:
        replay_initial = int(replay_initial)

    env_id = _federated_client_env_id(config, env_type)
    return client_fn_builder(
        env_id=env_id,
        algo=config.get('algo', 'dqn'),
        gamma=float(config.get('gamma', 0.99)),
        lr=float(config.get('lr', 1e-4)),
        hidden_dim=int(config.get('hidden_dim', 256)),
        epsilon=float(config.get('epsilon', 1.0)),
        epsilon_decay=float(config.get('epsilon_decay', 0.99)),
        epsilon_min=float(config.get('epsilon_min', 0.01)),
        tau=float(config.get('tau', 0.001)),
        threshold=float(config.get('threshold', 2.0)),
        aggregate_critic=bool(config.get('aggregate_critic', False)),
        batch_size=int(config.get('batch_size', 64)),
        replay_size=int(config.get('replay_size', 100000)),
        replay_initial=replay_initial,
        sync_interval=int(config.get('sync_interval', 10)),
        merge_interval=int(config.get('merge_interval', config.get('n_steps', 200))),
        eval_episodes=int(config.get('eval_episodes', 1)),
        add_noise=bool(config.get('add_noise', True)),
        replay_persist_across_rounds=bool(config.get('replay_persist_across_rounds', False)),
        ou_enabled=bool(config.get('ou_enabled', True)),
        ou_mu=float(config.get('ou_mu', 0.0)),
        ou_theta=float(config.get('ou_theta', 0.15)),
        ou_sigma=float(config.get('ou_sigma', 0.2)),
        ou_epsilon=float(config.get('ou_epsilon', 1.0)),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project'),
        run_name=config.get('run_name'),
        metrics_collector=metrics_collector,
        num_clients=int(config.get('num_clients', 4)),
        device=str(dev),
        cid_mapping_file=config.get('cid_mapping_file'),
        metadata_path=config.get('metadata_path'),
        render_eval=bool(config.get('render_eval', False)),
        render_mode=str(config.get('render_mode', 'video')),
        render_save_dir=config.get('render_save_dir'),
        render_every_n_rounds=int(config.get('render_every_n_rounds', 10)),
        render_episodes=int(config.get('render_episodes', 5)),
        reacher_render_mode=reacher_env_render_mode_from_config(
            bool(config.get('render_eval', False)),
            str(config.get('render_mode', 'video')),
        ),
    )


def _create_fedrep_client_fn(config: Dict[str, Any], **kwargs):
    """Create FedRep client function."""
    from fedguide.baselines.fedrep.client import client_fn_builder
    from fedguide.utils.federated_render import reacher_env_render_mode_from_config

    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    dev = config.get("device", "cpu")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    env_id = _federated_client_env_id(config, env_type)
    return client_fn_builder(
        env_id=env_id,
        algo=config.get('fedrep_algo', 'fedrep'),
        n_steps=int(config.get('n_steps', 200)),
        gamma=float(config.get('gamma', 0.99)),
        gae_lambda=float(config.get('gae_lambda', 0.95)),
        update_epochs=int(config.get('update_epochs', 10)),
        minibatch_size=int(config.get('minibatch_size', 64)),
        clip_eps=float(config.get('clip_eps', 0.2)),
        entropy_coef=float(config.get('entropy_coef', 0.01)),
        value_coef=float(config.get('value_coef', 0.5)),
        max_grad_norm=float(config.get('max_grad_norm', 0.5)),
        hidden_dim=int(config.get('hidden_dim', 256)),
        lr=float(config.get('lr', 3e-4)),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project'),
        run_name=config.get('run_name'),
        metrics_collector=metrics_collector,
        num_clients=int(config.get('num_clients', 4)),
        cid_mapping_file=config.get('cid_mapping_file'),
        sigma=float(config.get('sigma', 0.2)),
        metadata_path=config.get('metadata_path'),
        device=str(dev),
        render_eval=bool(config.get('render_eval', False)),
        render_mode=str(config.get('render_mode', 'video')),
        render_save_dir=config.get('render_save_dir'),
        render_every_n_rounds=int(config.get('render_every_n_rounds', 10)),
        render_episodes=int(config.get('render_episodes', 5)),
        eval_episodes=int(config.get('eval_episodes', 1)),
        reacher_render_mode=reacher_env_render_mode_from_config(
            bool(config.get('render_eval', False)),
            str(config.get('render_mode', 'video')),
        ),
    )


def _create_fedmomentum_client_fn(config: Dict[str, Any], **kwargs):
    """Create FedMomentum client function."""
    from fedguide.baselines.fedmomentum.client import client_fn_builder
    from fedguide.utils.federated_render import reacher_env_render_mode_from_config

    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    dev = config.get('device', 'cpu')
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    env_id = _federated_client_env_id(config, env_type)
    return client_fn_builder(
        env_id=env_id,
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
        reference_update_freq=int(config.get('reference_update_freq', 5)),
        use_svrpg=bool(config.get('use_svrpg', True)),
        hessian_alpha=float(config.get('hessian_alpha', 0.1)),
        use_diagonal_approx=bool(config.get('use_diagonal_approx', True)),
        fisher_update_freq=int(config.get('fisher_update_freq', 1)),
        use_fisher_info=bool(config.get('use_fisher_info', True)),
        eval_episodes=int(config.get('eval_episodes', 1)),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project'),
        run_name=config.get('run_name'),
        metrics_collector=metrics_collector,
        num_clients=int(config.get('num_clients', 4)),
        device=str(dev),
        cid_mapping_file=config.get('cid_mapping_file'),
        sigma=float(config.get('sigma', 0.2)),
        metadata_path=config.get('metadata_path'),
        render_eval=bool(config.get('render_eval', False)),
        render_mode=str(config.get('render_mode', 'video')),
        render_save_dir=config.get('render_save_dir'),
        render_every_n_rounds=int(config.get('render_every_n_rounds', 10)),
        render_episodes=int(config.get('render_episodes', 5)),
        reacher_render_mode=reacher_env_render_mode_from_config(
            bool(config.get('render_eval', False)),
            str(config.get('render_mode', 'video')),
        ),
        use_fedsvrpgm_strict=bool(config.get('use_fedsvrpgm_strict', False)),
        fedsvrpgm_eta=float(config.get('fedsvrpgm_eta', 0.01)),
        fedsvrpgm_beta=float(config.get('fedsvrpgm_beta', 0.2)),
        local_steps_k=int(config.get('local_steps_k', 5)),
        fedsvrpgm_max_horizon=int(config.get('fedsvrpgm_max_horizon', config.get('max_horizon', 500))),
    )


def _create_mfpo_client_fn(config: Dict[str, Any], **kwargs):
    """Create MFPO (INFOCOM 2024) federated client — 1:1 with MFPO-INFOCOM24."""
    from fedguide.baselines.mfpo.client import client_fn_builder
    from fedguide.utils.federated_render import reacher_env_render_mode_from_config

    env_type = kwargs.get('env_type', 'Bandit2D')
    metrics_collector = kwargs.get('metrics_collector')
    dev = config.get('device', 'cpu')
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    env_id = _federated_client_env_id(config, env_type)
    return client_fn_builder(
        env_id=env_id,
        batch_size=int(config.get('batch_size', 20)),
        local_update=int(config.get('local_update', 10)),
        device=str(dev),
        num_clients=int(config.get('num_clients', 4)),
        cid_mapping_file=config.get('cid_mapping_file'),
        metadata_path=config.get('metadata_path'),
        render_eval=bool(config.get('render_eval', False)),
        render_mode=str(config.get('render_mode', 'video')),
        render_save_dir=config.get('render_save_dir'),
        render_every_n_rounds=int(config.get('render_every_n_rounds', 10)),
        render_episodes=int(config.get('render_episodes', 5)),
        reacher_render_mode=reacher_env_render_mode_from_config(
            bool(config.get('render_eval', False)),
            str(config.get('render_mode', 'video')),
        ),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project'),
        run_name=config.get('run_name'),
        metrics_collector=metrics_collector,
        learning_rate_a=float(config.get('learning_rate_a', config.get('lr_a', 1e-4))),
        learning_rate_c=float(config.get('learning_rate_c', config.get('lr_c', 1e-4))),
        gamma=float(config.get('gamma', 0.99)),
        eps=float(config.get('eps', 1e-5)),
        average_type=str(config.get('average_type', 'target')),
        c=float(config.get('c', 3.0)),
        decay_rate=float(config.get('decay_rate', 0.99)),
        decay_start_iter_id=int(config.get('decay_start_iter_id', 500)),
        fault_type=config.get('fault_type'),
        mfpo_test_episodes=int(config.get('mfpo_test_episodes', 10)),
    )


# Register federated client factories
_registry.register_federated_client_factory('fedguide', _create_fedguide_client_fn)
_registry.register_federated_client_factory('fedkl', _create_fedkl_client_fn)
_registry.register_federated_client_factory('fmarl', _create_fmarl_client_fn)
_registry.register_federated_client_factory('fedrl', _create_fedrl_client_fn)
_registry.register_federated_client_factory('fedrep', _create_fedrep_client_fn)
_registry.register_federated_client_factory('fedmomentum', _create_fedmomentum_client_fn)
_registry.register_federated_client_factory('mfpo', _create_mfpo_client_fn)


# ============= Federated Server Factories =============

def _create_fedguide_server(config: Dict[str, Any], **kwargs):
    """Create FedGuide server strategy."""
    from fedguide.fed.fedguide.server import FedGuideStrategy as FedGuideServer
    
    num_clients = config.get('num_clients', 4)
    evaluate_fn = kwargs.get('evaluate_fn')
    
    min_fit_clients = int(config.get('min_fit_clients', num_clients))
    min_eval_clients = int(config.get('min_evaluate_clients', num_clients))
    min_available_clients = int(config.get('min_available_clients', num_clients))

    return FedGuideServer(
        fraction_fit=config.get('fraction_fit', 1.0),
        fraction_evaluate=config.get('fraction_evaluate', 1.0),
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        evaluate_fn=evaluate_fn,
        moe_enable=config.get('moe_enable', True),
        num_experts_prior=config.get('num_experts_prior', 1),
        num_experts_guidance=config.get('num_experts_guidance', 1),
        ot_mode=str(config.get('ot_mode', 'sinkhorn')),
        ot_reg=float(config.get('ot_reg', 0.05)),
        personalized_routing=bool(config.get('personalized_routing', True)),
        client_specific_expert_routing=config.get('client_specific_expert_routing', False),
        cid_mapping_file=config.get('cid_mapping_file'),
        num_clients=num_clients,
        routing_debug=config.get('routing_debug', False),
        policy_agg_every_k=int(config.get('policy_agg_every_k', 1)),
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
        policy_save_dir=config.get("metrics_dir"),
        total_rounds=int(config.get("rounds", 60)),
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


def _create_mfpo_server(config: Dict[str, Any], **kwargs):
    """MFPO uses the same FedAvg aggregation as MFPO-INFOCOM24 server.average_weights."""
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
        momentum_beta=float(config.get('momentum_beta', 0.9)),
        server_lr=float(config.get('server_lr', 0.001)),
        use_server_momentum=bool(config.get('use_server_momentum', False)),
        use_fedsvrpgm_strict=bool(config.get('use_fedsvrpgm_strict', False)),
        eta=float(config.get('fedsvrpgm_eta', 0.01)),
        local_steps_k=int(config.get('local_steps_k', 5)),
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
_registry.register_federated_server_factory('mfpo', _create_mfpo_server)
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

