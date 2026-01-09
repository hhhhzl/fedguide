"""
FedRep Server Implementation

Uses Flower's FedAvg strategy for encoder aggregation.
FedRep is essentially FedAvg where clients only upload encoder parameters.
"""

import flwr as fl
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from flwr.common import (
    Parameters, 
    Scalar, 
    FitRes, 
    EvaluateRes,
    FitIns,
    EvaluateIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
import json
import numpy as np

# Try to import FedRep strategy from Flower, fallback to FedAvg
try:
    from flwr.baselines.fedrep import FedRepStrategy
    StrategyClass = FedRepStrategy
    print("✓ Using: flwr.baselines.fedrep.FedRepStrategy")
except ImportError:
    try:
        from flwr.server.strategy import FedRep
        StrategyClass = FedRep
        print("✓ Using: flwr.server.strategy.FedRep")
    except ImportError:
        # FedRep is essentially FedAvg, just clients only upload encoder params
        from flwr.server.strategy import FedAvg
        StrategyClass = FedAvg
        print("✓ Using: flwr.server.strategy.FedAvg (FedRep = FedAvg with partial params)")


class FedRepStrategy(StrategyClass):
    """
    FedRep Server Strategy - uses FedAvg for encoder aggregation.
    
    FedRep is essentially FedAvg where:
    - Clients only upload encoder parameters (not head/value)
    - Server aggregates encoder parameters using weighted average
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
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        init_parameters: Optional[Parameters] = None,
    ):
        # If using FedAvg, initialize it with our parameters
        if StrategyClass.__name__ == "FedAvg":
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
                initial_parameters=initial_parameters or init_parameters,
            )
        else:
            # If using actual FedRep strategy, use its initialization
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
                initial_parameters=initial_parameters or init_parameters,
            )
    
    def __repr__(self) -> str:
        return "FedRepStrategy(fedavg aggregation for encoder)"


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration for each round."""
    return {"server_round": server_round}


def evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """Return evaluation configuration for each round."""
    return {"server_round": server_round}


def run_fedrep_server(
    client_fn: Callable,
    num_rounds: int = 100,
    num_clients: int = 10,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 2,
    server_address: str = "0.0.0.0:8080",
    use_simulation: bool = True,
    evaluate_fn: Optional[Callable] = None,
):
    """
    Run FedRep server using Flower's FedAvg strategy.
    
    Args:
        client_fn: Function to create client instances
        num_rounds: Number of federated rounds
        num_clients: Total number of clients
        fraction_fit: Fraction of clients to sample each round
        min_fit_clients: Minimum number of clients for training
        server_address: Server address for non-simulation mode
        use_simulation: Whether to use Flower simulation
        evaluate_fn: Optional evaluation function for metrics collection
    """
    
    # Create strategy (FedAvg for encoder aggregation)
    strategy = FedRepStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0 if evaluate_fn is not None else 0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients if evaluate_fn is not None else 0,
        min_available_clients=min_fit_clients,
        evaluate_fn=evaluate_fn,
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


# Alias for compatibility
FedRepServer = FedRepStrategy

