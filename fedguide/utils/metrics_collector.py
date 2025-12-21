"""
Unified Metrics Collector

This module provides a unified interface for collecting metrics across different
baselines (SAC, FedKL, FedGuide) and environments (Bandit2D, Maze2D, AntMaze, Flow, Reacher).

The collector is designed to be optional and non-intrusive - existing code will
continue to work without modification.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch


class UniversalMetricsCollector:
    """
    Universal metrics collector that works across all baselines and environments.
    
    This collector:
    1. Extracts universal metrics from trainer results
    2. Adds environment-specific metrics based on env_type
    3. Supports both federated and centralized training
    4. Is backward compatible - existing code doesn't need to change
    """
    
    # Universal metric keys that all baselines should have
    UNIVERSAL_KEYS = [
        'loss',
        'train/return',
        'eval/return',
        'train/episode_length',
    ]
    
    # Baseline-specific metric prefixes
    SAC_KEYS = [
        'train/loss/actor',
        'train/loss/critic',
        'train/q_value',
        'train/q_value_min',
        'train/buffer_size',
    ]
    
    FEDKL_KEYS = [
        'train/loss/policy',
        'train/loss/value',
        'train/kl/global',
        'train/kl/local',
    ]
    
    FEDGUIDE_KEYS = [
        'train/loss/prior',
        'train/loss/guidance',
        'train/kl/prior',
    ]
    
    # Environment-specific metric keys
    ENV_SPECIFIC_KEYS = {
        'bandit2d': [
            'policy_density',
            'value_function',
            'q_value_grid',
            'data/coverage',
            'data/entropy',
        ],
        'maze2d': [
            'success',
            'success_rate',
            'goal_distance',
            'reached_goal',
            'passed_gate',
            'path_length',
            'exploration_rate',
        ],
        'antmaze': [
            'success',
            'success_rate',
            'goal_distance',
            'reached_goal',
            'path_length',
            'exploration_rate',
        ],
        'pointmaze': [
            'success',
            'success_rate',
            'goal_distance',
            'reached_goal',
            'passed_gate',
            'path_length',
        ],
        'flow': [
            'throughput',
            'delay',
            'speed',
            'queue_length',
            'vehicle_count',
            'collision_rate',
        ],
        'reacher': [
            'success',
            'target_distance',
            'joint_angle_error',
            'joint_velocity',
            'control_effort',
        ],
    }
    
    def __init__(
        self,
        env_type: str = 'generic',
        baseline_type: str = 'generic',
        is_federated: bool = False,
    ):
        """
        Initialize metrics collector.
        
        Args:
            env_type: Environment type ('bandit2d', 'maze2d', 'antmaze', 'flow', 'reacher', 'generic')
            baseline_type: Baseline type ('sac', 'fedkl', 'fedguide', 'generic')
            is_federated: Whether this is a federated baseline
        """
        self.env_type = env_type.lower()
        self.baseline_type = baseline_type.lower()
        self.is_federated = is_federated
        
        # Determine expected keys based on baseline and environment
        self.expected_keys = set(self.UNIVERSAL_KEYS)
        
        if self.baseline_type == 'sac':
            self.expected_keys.update(self.SAC_KEYS)
        elif self.baseline_type == 'fedkl':
            self.expected_keys.update(self.FEDKL_KEYS)
        elif self.baseline_type == 'fedguide':
            self.expected_keys.update(self.FEDGUIDE_KEYS)
        
        if self.env_type in self.ENV_SPECIFIC_KEYS:
            self.expected_keys.update(self.ENV_SPECIFIC_KEYS[self.env_type])
    
    def collect(
        self,
        trainer_result: Union[Dict[str, Any], float, None],
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Collect and normalize metrics from trainer result.
        
        Args:
            trainer_result: Result from trainer.train_one_round()
                           Can be dict, float (loss), or None
            additional_metrics: Additional metrics to include (e.g., from environment)
        
        Returns:
            Dictionary of normalized metrics
        """
        metrics = {}
        
        # Handle different trainer result formats
        if trainer_result is None:
            pass  # No metrics to collect
        elif isinstance(trainer_result, dict):
            # Extract all available metrics
            for key, value in trainer_result.items():
                if value is not None:
                    try:
                        # Convert to float if possible
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            if value.numel() == 1 or value.size == 1:
                                metrics[key] = float(value.item() if hasattr(value, 'item') else float(value))
                            else:
                                # For arrays, store as-is or compute statistics
                                if isinstance(value, torch.Tensor):
                                    value = value.detach().cpu().numpy()
                                metrics[key] = value  # Keep as array for now
                        elif isinstance(value, (int, float, np.number)):
                            metrics[key] = float(value)
                        elif isinstance(value, (list, tuple)):
                            # Convert list to numpy array if numeric
                            try:
                                arr = np.array(value)
                                if arr.size == 1:
                                    metrics[key] = float(arr.item())
                                else:
                                    metrics[key] = arr  # Keep as array
                            except (ValueError, TypeError):
                                metrics[key] = value  # Keep original if not numeric
                        else:
                            metrics[key] = value  # Keep other types as-is
                    except (ValueError, TypeError, AttributeError):
                        # Skip if conversion fails
                        pass
        elif isinstance(trainer_result, (int, float)):
            # Simple loss value
            metrics['loss'] = float(trainer_result)
        else:
            # Unknown type, try to convert
            try:
                metrics['loss'] = float(trainer_result)
            except (TypeError, ValueError):
                pass
        
        # Add additional metrics
        if additional_metrics:
            for key, value in additional_metrics.items():
                if value is not None:
                    try:
                        if isinstance(value, (int, float, np.number)):
                            metrics[key] = float(value)
                        elif isinstance(value, (torch.Tensor, np.ndarray)):
                            if hasattr(value, 'numel') and value.numel() == 1:
                                metrics[key] = float(value.item())
                            elif hasattr(value, 'size') and value.size == 1:
                                metrics[key] = float(value.item() if hasattr(value, 'item') else float(value))
                            else:
                                metrics[key] = value  # Keep arrays
                        else:
                            metrics[key] = value
                    except (TypeError, ValueError, AttributeError):
                        pass
        
        # Ensure universal metrics exist (set to 0.0 if missing)
        for key in self.UNIVERSAL_KEYS:
            if key not in metrics:
                metrics[key] = 0.0
        
        return metrics
    
    def collect_federated_metrics(
        self,
        client_results: List[Dict[str, Any]],
        server_round: int,
    ) -> Dict[str, float]:
        """
        Collect and aggregate metrics from multiple clients (federated training).
        
        Args:
            client_results: List of client metric dictionaries
            server_round: Current server round number
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not client_results:
            return {'server_round': server_round}
        
        aggregated = {
            'server_round': server_round,
            'num_clients': len(client_results),
        }
        
        # Aggregate universal metrics
        for key in self.UNIVERSAL_KEYS:
            values = []
            weights = []
            
            for result in client_results:
                if key in result:
                    value = result[key]
                    weight = result.get('num_examples', 1)
                    
                    # Handle different value types
                    if isinstance(value, (int, float, np.number)):
                        values.append(float(value))
                        weights.append(weight)
                    elif isinstance(value, (torch.Tensor, np.ndarray)):
                        if hasattr(value, 'numel') and value.numel() == 1:
                            values.append(float(value.item()))
                            weights.append(weight)
            
            if values:
                # Weighted average
                total_weight = sum(weights)
                if total_weight > 0:
                    aggregated[f'server/{key}'] = sum(v * w for v, w in zip(values, weights)) / total_weight
                    # Also compute statistics
                    aggregated[f'server/{key}/mean'] = np.mean(values)
                    aggregated[f'server/{key}/std'] = np.std(values) if len(values) > 1 else 0.0
                    aggregated[f'server/{key}/min'] = np.min(values)
                    aggregated[f'server/{key}/max'] = np.max(values)
        
        # Aggregate baseline-specific metrics
        baseline_keys = []
        if self.baseline_type == 'sac':
            baseline_keys = self.SAC_KEYS
        elif self.baseline_type == 'fedkl':
            baseline_keys = self.FEDKL_KEYS
        elif self.baseline_type == 'fedguide':
            baseline_keys = self.FEDGUIDE_KEYS
        
        for key in baseline_keys:
            values = []
            weights = []
            
            for result in client_results:
                if key in result:
                    value = result[key]
                    weight = result.get('num_examples', 1)
                    
                    if isinstance(value, (int, float, np.number)):
                        values.append(float(value))
                        weights.append(weight)
            
            if values:
                total_weight = sum(weights)
                if total_weight > 0:
                    aggregated[f'server/{key}'] = sum(v * w for v, w in zip(values, weights)) / total_weight
        
        # Aggregate environment-specific metrics
        if self.env_type in self.ENV_SPECIFIC_KEYS:
            for key in self.ENV_SPECIFIC_KEYS[self.env_type]:
                values = []
                weights = []
                
                for result in client_results:
                    if key in result:
                        value = result[key]
                        weight = result.get('num_examples', 1)
                        
                        if isinstance(value, (int, float, np.number)):
                            values.append(float(value))
                            weights.append(weight)
                
                if values:
                    total_weight = sum(weights)
                    if total_weight > 0:
                        aggregated[f'server/{key}'] = sum(v * w for v, w in zip(values, weights)) / total_weight
        
        return aggregated
    
    def add_environment_metrics(
        self,
        metrics: Dict[str, Any],
        env: Any,
        agent: Any,
        episode_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add environment-specific metrics based on environment type.
        
        Args:
            metrics: Existing metrics dictionary
            env: Environment object
            agent: Agent object
            episode_info: Optional episode information (e.g., from env.step())
        
        Returns:
            Updated metrics dictionary
        """
        if self.env_type == 'bandit2d':
            # Bandit2D specific metrics
            if hasattr(env, 'compute_reward') and hasattr(agent, 'act'):
                # Can compute coverage if we have access to peak locations
                if hasattr(env, 'get_peak_locations'):
                    peaks = env.get_peak_locations()
                    metrics['data/num_peaks'] = len(peaks)
        
        elif self.env_type in ['maze2d', 'antmaze', 'pointmaze']:
            # Maze-specific metrics
            if episode_info:
                if 'success' in episode_info:
                    metrics['success'] = float(episode_info['success'])
                if 'reached_goal' in episode_info:
                    metrics['reached_goal'] = float(episode_info['reached_goal'])
                if 'passed_gate' in episode_info:
                    metrics['passed_gate'] = float(episode_info['passed_gate'])
                if 'goal_distance' in episode_info:
                    metrics['goal_distance'] = float(episode_info['goal_distance'])
                if 'path_length' in episode_info:
                    metrics['path_length'] = float(episode_info['path_length'])
        
        elif self.env_type == 'flow':
            # Flow-specific metrics
            if episode_info:
                if 'throughput' in episode_info:
                    metrics['throughput'] = float(episode_info['throughput'])
                if 'delay' in episode_info:
                    metrics['delay'] = float(episode_info['delay'])
                if 'speed' in episode_info:
                    metrics['speed'] = float(episode_info['speed'])
        
        elif self.env_type == 'reacher':
            # Reacher-specific metrics
            if episode_info:
                if 'target_distance' in episode_info:
                    metrics['target_distance'] = float(episode_info['target_distance'])
                if 'success' in episode_info:
                    metrics['success'] = float(episode_info['success'])
        
        return metrics


# Convenience functions for easy usage
def collect_metrics(
    trainer_result: Union[Dict[str, Any], float, None],
    env_type: str = 'generic',
    baseline_type: str = 'generic',
    additional_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Convenience function to collect metrics from trainer result.
    
    Args:
        trainer_result: Result from trainer.train_one_round()
        env_type: Environment type
        baseline_type: Baseline type
        additional_metrics: Additional metrics to include
    
    Returns:
        Dictionary of collected metrics
    """
    collector = UniversalMetricsCollector(
        env_type=env_type,
        baseline_type=baseline_type,
        is_federated=False,
    )
    return collector.collect(trainer_result, additional_metrics)


def collect_federated_metrics(
    client_results: List[Dict[str, Any]],
    server_round: int,
    env_type: str = 'generic',
    baseline_type: str = 'generic',
) -> Dict[str, float]:
    """
    Convenience function to collect and aggregate federated metrics.
    
    Args:
        client_results: List of client metric dictionaries
        server_round: Current server round
        env_type: Environment type
        baseline_type: Baseline type
    
    Returns:
        Dictionary of aggregated metrics
    """
    collector = UniversalMetricsCollector(
        env_type=env_type,
        baseline_type=baseline_type,
        is_federated=True,
    )
    return collector.collect_federated_metrics(client_results, server_round)




