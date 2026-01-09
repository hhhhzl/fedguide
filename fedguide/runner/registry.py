"""
Runner registry for automatic discovery and registration of runners.

This module provides a plugin-style architecture where new environments
and algorithms can be registered without modifying core code.
"""

from typing import Dict, Callable, Optional, List, Tuple
from abc import ABC, abstractmethod


class RunnerRegistry:
    """Registry for runner modules."""
    
    def __init__(self):
        self._env_module_map: Dict[str, str] = {}
        self._runners: Dict[Tuple[str, str], str] = {}  # (env_type, algorithm) -> module_path
        self._env_arg_builders: Dict[str, Callable] = {}
        self._algo_arg_builders: Dict[str, Callable] = {}
        self._supported_algorithms: List[str] = []
    
    def register_env(self, env_type: str, module_name: str):
        """Register an environment type.
        
        Args:
            env_type: Environment type identifier (e.g., 'bandit2d', 'd4rl')
            module_name: Module name in fedguide.runner (e.g., 'bandit2d', 'd4rl')
        """
        self._env_module_map[env_type] = module_name
    
    def register_runner(self, env_type: str, algorithm: str, module_path: Optional[str] = None):
        """Register a runner module.
        
        Args:
            env_type: Environment type identifier
            algorithm: Algorithm name (e.g., 'ppo', 'sac', 'fedguide', 'fedkl')
            module_path: Full module path. If None, auto-generates from convention.
        """
        if module_path is None:
            env_module = self._env_module_map.get(env_type, env_type)
            module_path = f"fedguide.runner.{env_module}.{algorithm}"
        
        self._runners[(env_type, algorithm)] = module_path
        if algorithm not in self._supported_algorithms:
            self._supported_algorithms.append(algorithm)
    
    def register_env_arg_builder(self, env_type: str, builder: Callable[[Dict, List[str]], None]):
        """Register an argument builder for an environment type.
        
        Args:
            env_type: Environment type identifier
            builder: Function that takes (config, cmd) and modifies cmd list
        """
        self._env_arg_builders[env_type] = builder
    
    def register_algo_arg_builder(self, algorithm: str, builder: Callable[[Dict, List[str]], None]):
        """Register an argument builder for an algorithm.
        
        Args:
            algorithm: Algorithm name
            builder: Function that takes (config, cmd) and modifies cmd list
        """
        self._algo_arg_builders[algorithm] = builder
    
    def get_runner_module(self, env_type: str, algorithm: str) -> str:
        """Get runner module path for given env_type and algorithm.
        
        Args:
            env_type: Environment type identifier
            algorithm: Algorithm name
        
        Returns:
            Full module path (e.g., 'fedguide.runner.bandit2d.ppo')
        
        Raises:
            ValueError: If runner not found and cannot auto-generate
        """
        module_path = self._runners.get((env_type, algorithm))
        if module_path is None:
            # Try to auto-generate from convention
            env_module = self._env_module_map.get(env_type, env_type)
            module_path = f"fedguide.runner.{env_module}.{algorithm}"
        return module_path
    
    def build_env_args(self, env_type: str, config: Dict, cmd: List[str]):
        """Build environment-specific arguments.
        
        Args:
            env_type: Environment type identifier
            config: Configuration dictionary
            cmd: Command list to append arguments to
        """
        builder = self._env_arg_builders.get(env_type)
        if builder:
            builder(config, cmd)
        else:
            # Default: pass common environment config keys
            default_keys = {
                'bandit2d': ['num_clients', 'data_dir', 'K', 'sigma'],
                'd4rl': ['env_name', 'num_clients'],
                'minari': ['dataset_id', 'env_name', 'download'],
                'reacher_hetero': ['metadata_path', 'num_clients', 'random_select_clients'],
            }
            
            keys = default_keys.get(env_type, [])
            for key in keys:
                value = config.get(key)
                if value is not None:
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{key}")
                    else:
                        cmd.extend([f"--{key}", str(value)])
    
    def build_algo_args(self, algorithm: str, config: Dict, cmd: List[str]):
        """Build algorithm-specific arguments.
        
        Args:
            algorithm: Algorithm name
            config: Configuration dictionary
            cmd: Command list to append arguments to
        """
        builder = self._algo_arg_builders.get(algorithm)
        if builder:
            builder(config, cmd)
        else:
            # Default: pass all config keys that are not common
            common_keys = {'env_type', 'algorithm', 'seed', 'device', 'output_dir', 
                          'metrics_dir', 'save_every', 'eval_episodes', 'render_eval',
                          'render_mode', 'render_save_dir', 'render_every_n_rounds',
                          'render_episodes', 'collect_logprob', 'logprob_grid_size',
                          'logprob_bounds', 'data_dir', 'K', 'sigma'}  # Environment-specific common keys
            
            for key, value in config.items():
                if key not in common_keys and value is not None:
                    # Skip None/null values
                    if value is None or (isinstance(value, str) and value.lower() == 'null'):
                        continue
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{key}")
                    elif isinstance(value, list):
                        if key == 'logprob_bounds':
                            cmd.extend([f"--{key}"] + [str(v) for v in value])
                    else:
                        cmd.extend([f"--{key}", str(value)])
    
    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported algorithms."""
        return self._supported_algorithms.copy()
    
    def get_supported_envs(self) -> List[str]:
        """Get list of supported environment types."""
        return list(self._env_module_map.keys())
    
    def is_runner_registered(self, env_type: str, algorithm: str) -> bool:
        """Check if a runner is registered."""
        return (env_type, algorithm) in self._runners


# Global registry instance
_registry = RunnerRegistry()


def register_env(env_type: str, module_name: str):
    """Register an environment type.
    
    Args:
        env_type: Environment type identifier
        module_name: Module name in fedguide.runner
    """
    _registry.register_env(env_type, module_name)


def register_runner(env_type: str, algorithm: str, module_path: Optional[str] = None):
    """Register a runner.
    
    Args:
        env_type: Environment type identifier
        algorithm: Algorithm name
        module_path: Optional full module path. If None, auto-generates.
    """
    _registry.register_runner(env_type, algorithm, module_path)


def register_env_arg_builder(env_type: str, builder: Callable[[Dict, List[str]], None]):
    """Register an argument builder for an environment.
    
    Args:
        env_type: Environment type identifier
        builder: Function that takes (config, cmd) and modifies cmd list
    """
    _registry.register_env_arg_builder(env_type, builder)


def register_algo_arg_builder(algorithm: str, builder: Callable[[Dict, List[str]], None]):
    """Register an argument builder for an algorithm.
    
    Args:
        algorithm: Algorithm name
        builder: Function that takes (config, cmd) and modifies cmd list
    """
    _registry.register_algo_arg_builder(algorithm, builder)


def get_registry() -> RunnerRegistry:
    """Get the global registry instance."""
    return _registry

