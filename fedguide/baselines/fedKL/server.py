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
        
        # Aggregate custom metrics
        metrics_aggregated = {}
        for key in ["loss", "success"]:
            weighted_values = [
                fit_res.metrics.get(key, 0) * fit_res.num_examples
                for _, fit_res in results
                if key in fit_res.metrics
            ]
            if weighted_values:
                metrics_aggregated[key] = sum(weighted_values) / total_samples
        
        metrics_aggregated["total_samples"] = total_samples
        metrics_aggregated["num_clients"] = len(results)
        
        return aggregated_parameters, metrics_aggregated


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
    
    # Create strategy
    strategy = FedKLStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=0,
        min_available_clients=min_fit_clients,
        evaluate_fn=get_evaluate_fn(),
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
