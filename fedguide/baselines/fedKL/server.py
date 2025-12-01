"""
FedKL Server Implementation

This module implements the FedKL server strategy and provides
convenience functions for running federated training.
"""

import flwr as fl
from typing import Dict, List, Tuple, Optional, Callable, Union
from flwr.common import Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy


class FedKLStrategy(fl.server.strategy.FedAvg):
    """
    FedKL Server Strategy.
    
    Uses FedAvg for aggregation but can be customized for FedKL-specific
    aggregation logic if needed.
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average."""
        
        if not results:
            return None, {}
        
        # Call parent's aggregate_fit for standard FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Compute additional metrics
        total_samples = sum(fit_res.num_examples for _, fit_res in results)
        
        # Aggregate custom metrics (including loss)
        # Also collect client actions and grid metrics from metrics for server-side metrics collection
        metrics_aggregated = {}
        collected_actions: Dict[int, Any] = {}  # {mapped_client_id: actions}
        collected_client_metrics: Dict[int, Dict[str, Any]] = {}  # {mapped_client_id: {metric_name: value}}
        
        # Aggregate loss (weighted by num_examples)
        total_loss = 0.0
        loss_samples = 0
        for _, fit_res in results:
            if "loss" in fit_res.metrics:
                try:
                    loss_val = float(fit_res.metrics["loss"])
                    # Check for nan/inf
                    if loss_val == loss_val and loss_val != float('inf') and loss_val != float('-inf'):
                        total_loss += loss_val * fit_res.num_examples
                        loss_samples += fit_res.num_examples
                except (TypeError, ValueError):
                    pass
            
            # Collect client actions from metrics (passed from client fit method)
            if "client_actions" in fit_res.metrics and "client_id_mapped" in fit_res.metrics:
                try:
                    import json
                    import numpy as np
                    client_id_mapped = int(fit_res.metrics["client_id_mapped"])
                    actions_json = fit_res.metrics["client_actions"]
                    if isinstance(actions_json, str):
                        actions = json.loads(actions_json)
                        # Convert back to numpy array for consistency
                        actions = np.array(actions)
                        collected_actions[client_id_mapped] = actions
                except Exception as e:
                    # Silently fail if deserialization fails
                    pass
            
            # Collect client grid metrics (policy, value evaluations on grid)
            if "client_id_mapped" in fit_res.metrics:
                try:
                    import json
                    import numpy as np
                    client_id_mapped = int(fit_res.metrics["client_id_mapped"])
                    client_grid_metrics = {}
                    # Look for metrics with prefix "client_grid_"
                    for key, value in fit_res.metrics.items():
                        if key.startswith("client_grid_"):
                            metric_name = key[len("client_grid_"):]
                            if isinstance(value, str):
                                try:
                                    # Deserialize JSON string back to numpy array
                                    data = json.loads(value)
                                    if isinstance(data, list):
                                        # Try to reshape if it's a 2D grid
                                        arr = np.array(data)
                                        # Assume grid_size x grid_size if it's a square number
                                        if arr.size > 0:
                                            grid_size = int(np.sqrt(arr.size))
                                            if grid_size * grid_size == arr.size:
                                                arr = arr.reshape(grid_size, grid_size)
                                        client_grid_metrics[metric_name] = arr
                                    else:
                                        client_grid_metrics[metric_name] = np.array(data)
                                except Exception:
                                    pass
                    if client_grid_metrics:
                        collected_client_metrics[client_id_mapped] = client_grid_metrics
                except Exception as e:
                    # Silently fail if deserialization fails
                    pass
        
        if loss_samples > 0:
            metrics_aggregated["loss"] = total_loss / loss_samples
        else:
            metrics_aggregated["loss"] = 0.0
        
        # Aggregate other metrics
        for key in ["success"]:
            weighted_values = [
                fit_res.metrics.get(key, 0) * fit_res.num_examples
                for _, fit_res in results
                if key in fit_res.metrics
            ]
            if weighted_values:
                metrics_aggregated[key] = sum(weighted_values) / total_samples
        
        metrics_aggregated["total_samples"] = total_samples
        metrics_aggregated["num_clients"] = len(results)
        metrics_aggregated["server_round"] = server_round
        
        # Store collected actions and client metrics in strategy instance for evaluate_fn to access
        # This allows evaluate_fn to access client data even though collector is not shared
        if not hasattr(self, '_collected_actions'):
            self._collected_actions = {}
        self._collected_actions[server_round] = collected_actions
        
        if not hasattr(self, '_collected_client_metrics'):
            self._collected_client_metrics = {}
        self._collected_client_metrics[server_round] = collected_client_metrics
        
        return aggregated_parameters, metrics_aggregated
    
    def evaluate(
            self,
            server_round: int,
            parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current global model on the server side.
        
        This calls the evaluate_fn if provided, which can be used for metrics collection.
        Overrides parent method to pass collected actions through config.
        """
        if self.evaluate_fn is not None:
            # Call parent's evaluate to get config
            # But we need to modify config to include collected actions
            config = {}
            if self.on_evaluate_config_fn is not None:
                config = self.on_evaluate_config_fn(server_round)
            
            # Pass collected actions and client metrics to evaluate_fn through config
            # This allows evaluate_fn to access client data collected in aggregate_fit
            if hasattr(self, '_collected_actions') and server_round in self._collected_actions:
                config['_collected_actions'] = self._collected_actions[server_round]
            if hasattr(self, '_collected_client_metrics') and server_round in self._collected_client_metrics:
                config['_collected_client_metrics'] = self._collected_client_metrics[server_round]
            
            try:
                result = self.evaluate_fn(server_round, parameters, config)
                # evaluate_fn can return (loss, metrics) or None
                if result is not None and isinstance(result, tuple) and len(result) == 2:
                    return result
            except Exception as e:
                print(f"[FedKLStrategy.evaluate] Error calling evaluate_fn: {e}")
                import traceback
                traceback.print_exc()
        
        # Return None if no evaluate_fn or if it returns None
        return None


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration for each round."""
    config = {
        "server_round": server_round,
    }
    return config


def evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """Return evaluation configuration for each round."""
    config = {
        "server_round": server_round,
    }
    return config


def get_evaluate_fn(model=None):
    """
    Return an evaluation function for server-side evaluation.
    
    This is optional and can be used to evaluate the global model
    on a separate test environment.
    """
    
    def evaluate(
        server_round: int,
        parameters: Parameters,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on server side."""
        # If you want server-side evaluation, implement it here
        return None
    
    return evaluate


def run_fedkl_server(
    client_fn: Callable,
    num_rounds: int = 100,
    num_clients: int = 10,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 2,
    server_address: str = "0.0.0.0:8080",
    use_simulation: bool = True,
    evaluate_fn: Optional[Callable] = None,  # Add evaluate_fn parameter
):
    """
    Run FedKL server.
    
    Args:
        client_fn: Function to create client instances
        num_rounds: Number of federated rounds
        num_clients: Total number of clients
        fraction_fit: Fraction of clients to sample each round
        min_fit_clients: Minimum number of clients for training
        server_address: Server address for non-simulation mode
        use_simulation: Whether to use Flower simulation
    """
    
    # Create strategy - enable evaluation if evaluate_fn is provided
    strategy = FedKLStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0 if evaluate_fn is not None else 0.0,  # Enable if evaluate_fn provided
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients if evaluate_fn is not None else 0,  # Set if evaluate_fn provided
        min_available_clients=min_fit_clients,
        evaluate_fn=evaluate_fn if evaluate_fn is not None else get_evaluate_fn(),  # Use provided or default
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        accept_failures=True,
    )
    
    if use_simulation:
        # Use Flower simulation
        from flwr.simulation import start_simulation
        
        history = start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
        
        return history
    else:
        # Use standard Flower server
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )


# Example usage
if __name__ == "__main__":
    from fedguide.baselines.fedKL.client import client_fn_builder
    
    # Build client function
    client_fn = client_fn_builder(
        env_id="HalfCheetah-v4",
        n_steps=2048,
        lambda_global=0.1,
        lambda_local=0.05,
        update_epochs=10,
        minibatch_size=64,
    )
    
    # Run server
    run_fedkl_server(
        client_fn=client_fn,
        num_rounds=100,
        num_clients=5,
        fraction_fit=1.0,
        min_fit_clients=2,
        use_simulation=True,
    )
