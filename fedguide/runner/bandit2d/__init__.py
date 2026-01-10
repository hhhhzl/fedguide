"""
Bandit2D environment runners.

This module is kept for backward compatibility. All runners are now handled
by the unified runner system in fedguide.runner.unified_runner.
"""

# Export common utilities for hooks (e.g., metrics collection)
from ._common import (
    create_metrics_collector,
    make_evaluate_fn,
    save_training_results
)

__all__ = ['create_metrics_collector', 'make_evaluate_fn', 'save_training_results']
