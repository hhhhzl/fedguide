"""
FedMomentum Server Implementation

This module implements the FedMomentum server strategy with momentum-based aggregation.
Based on the paper: "Momentum for the Win: Collaborative Federated Reinforcement Learning across Heterogeneous Environments"

The server aggregates policy gradients from clients and applies momentum updates.
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
import copy


class FedMomentumStrategy(Strategy):
    """
    FedMomentum Server Strategy.
    
    Uses momentum-based aggregation for policy gradients.
    Key differences from FedAvg:
    - Aggregates gradients instead of parameters
    - Maintains momentum buffer on server
    - Updates global parameters using momentum-averaged gradients
    """
    
    def __init__(
        self,
        *,
        # Momentum parameters
        momentum_beta: float = 0.9,  # Momentum coefficient (paper default: 0.9)
        server_lr: float = 0.001,  # Server learning rate for parameter updates
        # Standard Flower Strategy parameters
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
        init_parameters: Optional[Parameters] = None,  # Alias for backward compatibility
        use_gradient_aggregation: bool = True,  # If False, fallback to parameter aggregation (FedAvg)
    ):
        # Momentum parameters
        self.momentum_beta = momentum_beta
        self.server_lr = server_lr
        self.use_gradient_aggregation = use_gradient_aggregation
        
        # Server momentum buffer (initialized on first aggregation)
        self.server_momentum = None
        
        # Standard Flower Strategy parameters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        
        # Handle both initial_parameters and init_parameters for compatibility
        self.init_parameters = initial_parameters or init_parameters
    
    def __repr__(self) -> str:
        return f"FedMomentumStrategy(momentum_beta={self.momentum_beta}, server_lr={self.server_lr})"
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.init_parameters
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        num_available = len(client_manager.all())
        
        if num_available < self.min_available_clients:
            return []
        
        num_clients = int(num_available * self.fraction_fit)
        num_clients = max(num_clients, self.min_fit_clients)
        num_clients = min(num_clients, num_available)
        
        sampled_clients = client_manager.sample(
            num_clients=num_clients,
            min_num_clients=self.min_fit_clients
        )
        
        # Create client instructions with FitIns objects
        client_instructions = []
        for client in sampled_clients:
            if self.on_fit_config_fn is not None:
                fit_config = self.on_fit_config_fn(server_round)
            else:
                fit_config = {"server_round": server_round}
            
            # Create FitIns with parameters and config
            fit_ins = FitIns(
                parameters=parameters,
                config=fit_config,
            )
            client_instructions.append((client, fit_ins))
        
        return client_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate policy gradients using momentum and update global parameters."""
        
        if not self.accept_failures and failures:
            return None, {}
        
        # Initialize aggregated metrics
        aggregated_metrics: Dict[str, Scalar] = {"server_round": server_round}
        collected_actions: Dict[int, Any] = {}
        collected_client_metrics: Dict[int, Dict[str, Any]] = {}
        
        # Aggregate loss and other metrics (weighted by num_examples)
        total_loss = 0.0
        total_examples = 0
        total_success = 0.0
        total_train_return = 0.0
        total_eval_return = 0.0
        count_train_return = 0
        count_eval_return = 0
        
        # Try gradient aggregation first (if enabled and gradients are available)
        use_gradients = self.use_gradient_aggregation
        client_gradients = []
        gradient_weights = []
        
        for _, fit_res in results:
            # Handle both FitRes object and dict (for compatibility)
            if isinstance(fit_res, dict):
                metrics = fit_res.get("metrics", {})
                num_examples = fit_res.get("num_examples", 0)
            else:
                metrics = fit_res.metrics
                num_examples = fit_res.num_examples
            
            total_examples += num_examples
            
            # Try to extract policy gradients from metrics
            if use_gradients and "policy_gradient" in metrics:
                try:
                    grad_str = metrics["policy_gradient"]
                    if isinstance(grad_str, str):
                        # Deserialize gradient dictionary from JSON
                        grad_dict = json.loads(grad_str)
                        # Convert back to numpy arrays
                        for key, value in grad_dict.items():
                            if isinstance(value, list):
                                grad_dict[key] = np.array(value)
                        client_gradients.append(grad_dict)
                        gradient_weights.append(num_examples)
                except Exception as e:
                    print(f"[FedMomentum] Warning: Failed to parse gradient from client: {e}")
                    use_gradients = False  # Fallback to parameter aggregation
                    client_gradients = []
                    gradient_weights = []
            
            # Aggregate loss
            if "loss" in metrics:
                try:
                    loss_val = float(metrics["loss"])
                    if loss_val == loss_val and loss_val != float('inf') and loss_val != float('-inf'):
                        total_loss += loss_val * num_examples
                except (TypeError, ValueError):
                    pass
            
            # Aggregate success
            if "success" in metrics:
                try:
                    success_val = float(metrics["success"])
                    total_success += success_val * num_examples
                except (TypeError, ValueError):
                    pass
            
            # Aggregate returns
            if "train/return" in metrics:
                try:
                    train_return = float(metrics["train/return"])
                    if train_return == train_return:
                        total_train_return += train_return
                        count_train_return += 1
                except (TypeError, ValueError):
                    pass
            
            if "eval/return" in metrics:
                try:
                    eval_return = float(metrics["eval/return"])
                    if eval_return == eval_return:
                        total_eval_return += eval_return
                        count_eval_return += 1
                except (TypeError, ValueError):
                    pass
            
            # Collect client actions from metrics
            if "client_actions" in metrics and "client_id_mapped" in metrics:
                try:
                    client_id_mapped = int(metrics["client_id_mapped"])
                    actions_json = metrics["client_actions"]
                    if isinstance(actions_json, str):
                        actions = json.loads(actions_json)
                        actions = np.array(actions)
                        collected_actions[client_id_mapped] = actions
                except Exception:
                    pass
            
            # Collect client grid metrics
            if "client_id_mapped" in metrics:
                try:
                    client_id_mapped = int(metrics["client_id_mapped"])
                    client_grid_metrics = {}
                    
                    for key, value in metrics.items():
                        if key.startswith("client_grid_"):
                            metric_name = key[len("client_grid_"):]
                            if isinstance(value, str):
                                try:
                                    data = json.loads(value)
                                    if isinstance(data, list):
                                        arr = np.array(data)
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
                except Exception:
                    pass
        
        # Compute aggregated metrics
        if total_examples > 0:
            aggregated_metrics["loss"] = total_loss / total_examples
            aggregated_metrics["success"] = total_success / total_examples
        else:
            aggregated_metrics["loss"] = 0.0
            aggregated_metrics["success"] = 0.0
        
        if count_train_return > 0:
            aggregated_metrics["train/return"] = total_train_return / count_train_return
        
        if count_eval_return > 0:
            aggregated_metrics["eval/return"] = total_eval_return / count_eval_return
        
        aggregated_metrics["total_samples"] = total_examples
        aggregated_metrics["num_clients"] = len(results)
        
        # Store collected data for evaluate_fn
        if not hasattr(self, '_collected_actions'):
            self._collected_actions = {}
        self._collected_actions[server_round] = collected_actions
        
        if not hasattr(self, '_collected_client_metrics'):
            self._collected_client_metrics = {}
        self._collected_client_metrics[server_round] = collected_client_metrics
        
        # Aggregate using gradients with momentum (if available)
        if use_gradients and client_gradients and len(client_gradients) > 0:
            aggregated_params = self._aggregate_gradients_with_momentum(
                server_round,
                results,
                client_gradients,
                gradient_weights
            )
            
            if aggregated_params is not None:
                print(f"[Round {server_round}] Using momentum-based gradient aggregation")
                print(f"  loss: {aggregated_metrics.get('loss', 'N/A'):.6f}")
                print(f"  success: {aggregated_metrics.get('success', 'N/A'):.3f}")
                if "train/return" in aggregated_metrics:
                    print(f"  train/return: {aggregated_metrics['train/return']:.3f}")
                if "eval/return" in aggregated_metrics:
                    print(f"  eval/return: {aggregated_metrics['eval/return']:.3f}")
                
                return aggregated_params, aggregated_metrics
        
        # Fallback to parameter aggregation (FedAvg-style)
        print(f"[Round {server_round}] Falling back to parameter aggregation (FedAvg)")
        weighted_params = []
        for _, fit_res in results:
            if isinstance(fit_res, dict):
                parameters = fit_res.get("parameters", None)
                num_examples = fit_res.get("num_examples", 0)
            else:
                parameters = fit_res.parameters
                num_examples = fit_res.num_examples
            
            if parameters is not None:
                param_arrays = parameters_to_ndarrays(parameters)
                weighted_params.append((param_arrays, num_examples))
        
        if weighted_params:
            aggregated_arrays = self._fedavg_arrays(weighted_params)
            aggregated_parameters = ndarrays_to_parameters(aggregated_arrays)
            
            print(f"[Round {server_round}] Aggregated metrics:")
            print(f"  loss: {aggregated_metrics.get('loss', 'N/A'):.6f}")
            print(f"  success: {aggregated_metrics.get('success', 'N/A'):.3f}")
            if "train/return" in aggregated_metrics:
                print(f"  train/return: {aggregated_metrics['train/return']:.3f}")
            if "eval/return" in aggregated_metrics:
                print(f"  eval/return: {aggregated_metrics['eval/return']:.3f}")
            
            return aggregated_parameters, aggregated_metrics
        else:
            return None, aggregated_metrics
    
    def _aggregate_gradients_with_momentum(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        client_gradients: List[Dict[str, np.ndarray]],
        gradient_weights: List[int],
    ) -> Optional[Parameters]:
        """
        Aggregate policy gradients from clients and apply momentum update.
        
        Algorithm:
        1. Weighted average of client gradients (FedAvg-style)
        2. Update momentum buffer: m_t = β * m_{t-1} + (1-β) * g_t
        3. Update parameters: θ_t = θ_{t-1} + α * m_t
        
        Args:
            server_round: Current server round
            results: Client fit results
            client_gradients: List of gradient dictionaries from clients
            gradient_weights: Weights for each client (typically num_examples)
        
        Returns:
            Updated global parameters
        """
        if not client_gradients or not gradient_weights:
            return None
        
        # Get current global parameters
        if len(results) == 0:
            return None
        
        _, first_fit_res = results[0]
        if isinstance(first_fit_res, dict):
            current_params = first_fit_res.get("parameters", None)
        else:
            current_params = first_fit_res.parameters
        
        if current_params is None:
            return None
        
        current_param_arrays = parameters_to_ndarrays(current_params)
        
        # 1. Weighted average of client gradients
        total_weight = sum(gradient_weights)
        if total_weight == 0:
            return None
        
        # Get all gradient keys (should be consistent across clients)
        if not client_gradients:
            return None
        
        gradient_keys = list(client_gradients[0].keys())
        aggregated_grad = {}
        
        for key in gradient_keys:
            # Weighted sum of gradients
            weighted_grad = None
            for grad_dict, weight in zip(client_gradients, gradient_weights):
                if key not in grad_dict:
                    continue
                
                grad = grad_dict[key]
                if not isinstance(grad, np.ndarray):
                    grad = np.array(grad)
                
                if weighted_grad is None:
                    weighted_grad = grad * (weight / total_weight)
                else:
                    weighted_grad += grad * (weight / total_weight)
            
            if weighted_grad is not None:
                aggregated_grad[key] = weighted_grad
        
        # 2. Initialize momentum buffer if needed
        if self.server_momentum is None:
            self.server_momentum = {}
            # Initialize momentum with current parameter structure
            # We need to map gradient keys to parameter array indices
            # For now, assume gradient keys match parameter structure
            # This will be properly initialized in first update
        
        # 3. Update momentum buffer
        # Map gradients to parameter arrays by matching shapes
        # For simplicity, assume gradients are in same order as parameters
        # More robust: match by shape and key name
        param_idx = 0
        momentum_arrays = []
        
        for param_array in current_param_arrays:
            # Try to find matching gradient by shape
            matching_grad = None
            for key, grad_array in aggregated_grad.items():
                if grad_array.shape == param_array.shape:
                    matching_grad = grad_array
                    break
            
            if matching_grad is None:
                # No matching gradient, skip this parameter
                momentum_arrays.append(np.zeros_like(param_array))
                continue
            
            # Initialize momentum for this parameter if needed
            momentum_key = f"param_{param_idx}"
            if momentum_key not in self.server_momentum:
                self.server_momentum[momentum_key] = np.zeros_like(matching_grad)
            
            # Update momentum: m_t = β * m_{t-1} + (1-β) * g_t
            self.server_momentum[momentum_key] = (
                self.momentum_beta * self.server_momentum[momentum_key] +
                (1 - self.momentum_beta) * matching_grad
            )
            
            momentum_arrays.append(self.server_momentum[momentum_key].copy())
            param_idx += 1
        
        # 4. Update parameters: θ_t = θ_{t-1} + α * m_t
        updated_param_arrays = []
        for i, (param_array, momentum_array) in enumerate(zip(current_param_arrays, momentum_arrays)):
            if momentum_array.shape == param_array.shape:
                updated_param = param_array + self.server_lr * momentum_array
            else:
                # Shape mismatch, use original parameter
                updated_param = param_array.copy()
            updated_param_arrays.append(updated_param)
        
        # Convert back to Parameters object
        updated_parameters = ndarrays_to_parameters(updated_param_arrays)
        
        return updated_parameters
    
    def _fedavg_arrays(
        self, 
        weighted_arrays: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """Perform FedAvg aggregation on parameter arrays."""
        if not weighted_arrays:
            return []
        
        num_layers = len(weighted_arrays[0][0])
        total_examples = sum(n for _, n in weighted_arrays)
        
        if total_examples == 0:
            return []
        
        aggregated = []
        for layer_idx in range(num_layers):
            # Weighted sum of parameters for this layer
            layer_sum = sum(
                arrays[layer_idx] * (num_examples / total_examples)
                for arrays, num_examples in weighted_arrays
            )
            aggregated.append(layer_sum)
        
        return aggregated
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        
        num_available = len(client_manager.all())
        if num_available < self.min_available_clients:
            return []
        
        num_clients = int(num_available * self.fraction_evaluate)
        num_clients = max(num_clients, self.min_evaluate_clients)
        num_clients = min(num_clients, num_available)
        
        sampled_clients = client_manager.sample(
            num_clients=num_clients,
            min_num_clients=self.min_evaluate_clients
        )
        
        # Create client instructions with EvaluateIns objects
        client_instructions = []
        for client in sampled_clients:
            if self.on_evaluate_config_fn is not None:
                eval_config = self.on_evaluate_config_fn(server_round)
            else:
                eval_config = {"server_round": server_round}
            
            eval_ins = EvaluateIns(
                parameters=parameters,
                config=eval_config,
            )
            client_instructions.append((client, eval_ins))
        
        return client_instructions
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not self.accept_failures and failures:
            return None, {}
        
        if not results:
            return None, {}
        
        # Aggregate losses and metrics
        total_loss = 0.0
        total_examples = 0
        metrics: Dict[str, List] = {}
        
        for _, eval_res in results:
            if eval_res.loss is not None:
                total_loss += eval_res.loss * eval_res.num_examples
            total_examples += eval_res.num_examples
            
            # Collect metrics from all clients
            for key, value in eval_res.metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Average metrics based on type
        aggregated_metrics: Dict[str, Scalar] = {}
        for key, values in metrics.items():
            if isinstance(values[0], (int, float)):
                aggregated_metrics[key] = sum(values) / len(values)
            else:
                aggregated_metrics[key] = values
        
        aggregated_loss = total_loss / total_examples if total_examples > 0 else None
        aggregated_metrics["server_round"] = server_round
        
        return aggregated_loss, aggregated_metrics
    
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current global model on the server side."""
        if self.evaluate_fn is not None:
            config = {}
            if self.on_evaluate_config_fn is not None:
                config = self.on_evaluate_config_fn(server_round)
            
            # Pass collected actions and client metrics to evaluate_fn
            if hasattr(self, '_collected_actions') and server_round in self._collected_actions:
                config['_collected_actions'] = self._collected_actions[server_round]
            if hasattr(self, '_collected_client_metrics') and server_round in self._collected_client_metrics:
                config['_collected_client_metrics'] = self._collected_client_metrics[server_round]
            
            try:
                result = self.evaluate_fn(server_round, parameters, config)
                if result is not None and isinstance(result, tuple) and len(result) == 2:
                    return result
            except Exception as e:
                print(f"[FedMomentumStrategy.evaluate] Error calling evaluate_fn: {e}")
                import traceback
                traceback.print_exc()
        
        return None


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration for each round."""
    return {"server_round": server_round}


def evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """Return evaluation configuration for each round."""
    return {"server_round": server_round}


def run_fedmomentum_server(
    client_fn: Callable,
    num_rounds: int = 100,
    num_clients: int = 10,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 2,
    server_address: str = "0.0.0.0:8080",
    use_simulation: bool = True,
    evaluate_fn: Optional[Callable] = None,
    momentum_beta: float = 0.9,
    server_lr: float = 0.001,
):
    """
    Run FedMomentum server.
    
    Args:
        client_fn: Function to create client instances
        num_rounds: Number of federated rounds
        num_clients: Total number of clients
        fraction_fit: Fraction of clients to sample each round
        min_fit_clients: Minimum number of clients for training
        server_address: Server address for non-simulation mode
        use_simulation: Whether to use Flower simulation
        evaluate_fn: Optional evaluation function for metrics collection
        momentum_beta: Momentum coefficient (default: 0.9)
        server_lr: Server learning rate for parameter updates (default: 0.001)
    """
    
    # Create strategy with evaluation enabled if evaluate_fn provided
    strategy = FedMomentumStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0 if evaluate_fn is not None else 0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients if evaluate_fn is not None else 0,
        min_available_clients=min_fit_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        accept_failures=True,
        momentum_beta=momentum_beta,
        server_lr=server_lr,
        use_gradient_aggregation=True,
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
FedMomentumServer = FedMomentumStrategy

