"""
Hooks for environment-specific and algorithm-specific logic.

Hooks allow extending the unified runner with custom logic for specific
environments (e.g., bandit2d metrics collection) without modifying the core runner.
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import os


class RunnerHook(ABC):
    """Base class for runner hooks."""
    pass


class Bandit2DMetricsHook(RunnerHook):
    """Hook for Bandit2D environment-specific metrics collection."""
    
    def __init__(self):
        self.metrics_collector = None
    
    def on_federated_start(self, config: Dict[str, Any]) -> Optional[Any]:
        """Called when federated training starts. Returns metrics collector if applicable."""
        from fedguide.runner.bandit2d._common import create_metrics_collector
        
        metrics_dir = config.get('metrics_dir', './metrics/bandit2d_fedguide')
        collect_every = config.get('collect_metrics_every', 1)
        grid_size = config.get('logprob_grid_size', 200)
        bounds = config.get('logprob_bounds', [-1.5, 1.5])
        
        if collect_every > 0:
            self.metrics_collector = create_metrics_collector(
                metrics_dir=metrics_dir,
                collect_every=collect_every,
                grid_size=grid_size,
                bounds=tuple(bounds)
            )
            return self.metrics_collector
        return None
    
    def create_evaluate_fn(self, config: Dict[str, Any], metrics_collector: Any, algorithm: str) -> Optional[Callable]:
        """Create evaluate function for metrics collection."""
        from fedguide.runner.bandit2d._common import make_evaluate_fn
        
        collect_every = config.get('collect_metrics_every', 1)
        if collect_every > 0 and metrics_collector is not None:
            return make_evaluate_fn(
                collect_every=collect_every,
                collector=metrics_collector,
                algorithm=algorithm
            )
        return None
    
    def on_federated_end(self, history: Any, metrics_collector: Any, config: Dict[str, Any], algorithm: str):
        """Called when federated training ends. Save results."""
        from fedguide.runner.bandit2d._common import save_training_results
        
        metrics_dir = config.get('metrics_dir', './metrics/bandit2d_fedguide')
        try:
            save_training_results(history, metrics_collector, metrics_dir, algorithm)
        except Exception as e:
            print(f"Warning: Failed to save training results in hook: {e}")
            import traceback
            traceback.print_exc()


def register_default_hooks(registry):
    """Register default hooks for environments and algorithms."""
    from fedguide.runner.factories import get_registry
    
    reg = registry if registry is not None else get_registry()
    
    # Register Bandit2D metrics hook for federated algorithms
    federated_algorithms = ['fedguide', 'fedkl', 'fmarl', 'fedrl', 'fedrep', 'fedmomentum', 'mfpo']
    for algorithm in federated_algorithms:
        hook = Bandit2DMetricsHook()
        reg.register_hook('bandit2d', algorithm, hook)


# Hooks will be registered when factories module is imported
# This prevents circular imports - hooks are registered in factories.py at the end

