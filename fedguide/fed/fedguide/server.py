from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Callable
import json
import numpy as np

from flwr.server.strategy import Strategy
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import ClientManager
from fedguide.fed.fedguide.aggregator import ot_moe_aggregate


def _modules_from_metrics(metrics: Dict[str, Scalar]) -> Optional[Dict[str, List[np.ndarray]]]:
    """
    expect metrics["modules"] is a dict：
    { module_name: [array_like, array_like, ...], ... }
    return {str: List[np.ndarray]}；or None。
    """
    if "modules" not in metrics:
        return None
    raw = metrics["modules"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    modules: Dict[str, List[np.ndarray]] = {}
    for k, v in raw.items():
        modules[k] = [np.asarray(x) for x in v]
    return modules


def _l2_cost_fn(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    """OT cost：sum of L2 distance"""
    return float(sum(np.sum((aa - bb) ** 2) for aa, bb in zip(a, b)))


# -------------------------------
# FedGuide Server Strategy
# -------------------------------
class FedGuideStrategy(Strategy):
    """
    FedGuide Server Strategy：
    - prior_adapt, guidance:  OT-MoE or FedAvg；
    - others (policy, log_std, value, ...): FedAvg
    """

    def __init__(
            self,
            *,
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

            # OT-MoE
            moe_enable: bool = True, # if false, all will do fedavg
            num_experts_prior: int = 1,
            num_experts_guidance: int = 1,
            cost_fn_prior: Callable[[List[np.ndarray], List[np.ndarray]], float] = _l2_cost_fn,
            cost_fn_guidance: Callable[[List[np.ndarray], List[np.ndarray]], float] = _l2_cost_fn,
            moe_keys: Tuple[str, ...] = ("prior_adapt", "guidance"),
    ):
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

        # OT-MoE parameters
        self.moe_enable = moe_enable
        self.moe_keys = tuple(moe_keys)
        self.num_experts_prior = int(num_experts_prior)
        self.num_experts_guidance = int(num_experts_guidance)
        self.cost_fn_prior = cost_fn_prior
        self.cost_fn_guidance = cost_fn_guidance

        # experts_map[module_key] = List[List[np.ndarray]]  # M x L
        self.experts_map: Dict[str, List[List[np.ndarray]]] = {}

    # ---- Strategy ----
    def __repr__(self) -> str:
        return "FedGuideStrategy(ot-moe for prior/guidance, fedavg otherwise)"

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.init_parameters

    def _fedavg_arrays(self, weighted_arrays: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        if not weighted_arrays:
            return []
        L = len(weighted_arrays[0][0])
        total = float(sum(n for _, n in weighted_arrays)) + 1e-12
        out: List[np.ndarray] = []
        for l in range(L):
            agg = sum(arrs[l] * (n / total) for arrs, n in weighted_arrays)
            out.append(agg)
        return out

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager,
    ):
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
        
        # Create configs
        configs = []
        for client in sampled_clients:
            if self.on_fit_config_fn is not None:
                fit_config = self.on_fit_config_fn(server_round)
            else:
                fit_config = {"server_round": server_round}
            configs.append((client, fit_config))
        return configs

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[Any, FitRes]],
            failures: List[Any],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not self.accept_failures and failures:
            return None, {}

        modules_list: List[Tuple[Dict[str, List[np.ndarray]], int]] = []
        for _, fitres in results:
            mods = _modules_from_metrics(fitres.metrics)
            if mods is None:
                modules_list = []
                break
            modules_list.append((mods, fitres.num_examples))

        if modules_list:
            new_global = self._aggregate_by_modules(modules_list)
            flat, layout = self._flatten_module_dict(new_global)
            params = ndarrays_to_parameters(flat)
            metrics: Dict[str, Scalar] = {
                "layout": json.dumps(layout),
                "server_round": server_round,
            }
            return params, metrics

        # 2) FedAvg
        weighted = [
            (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
            for _, fitres in results
        ]
        fedavg = self._fedavg_arrays(weighted)
        return ndarrays_to_parameters(fedavg), {"server_round": server_round}

    def configure_evaluate(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager,
    ):
        """Configure the next round of evaluation."""
        # Sample clients for evaluation
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
        
        # Create configs
        configs = []
        for client in sampled_clients:
            if self.on_evaluate_config_fn is not None:
                eval_config = self.on_evaluate_config_fn(server_round)
            else:
                eval_config = {"server_round": server_round}
            configs.append((client, eval_config))
        return configs

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[Any, EvaluateRes]],
            failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not self.accept_failures and failures:
            return None, {}
        
        if not results:
            return None, {}
        
        # Aggregate losses and metrics
        total_loss = 0.0
        total_examples = 0
        metrics: Dict[str, Scalar] = {}
        
        for _, eval_res in results:
            if eval_res.loss is not None:
                total_loss += eval_res.loss * eval_res.num_examples
            total_examples += eval_res.num_examples
            
            # Aggregate metrics
            for key, value in eval_res.metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Average metrics
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
    ):
        return None

    # ---- aggregate implementation ----
    def _aggregate_by_modules(
            self,
            modules_list: List[Tuple[Dict[str, List[np.ndarray]], int]],
    ) -> Dict[str, List[np.ndarray]]:
        """
        input：
          modules_list: [({module: [arr_l]}, num_examples), ...]
        out：
          new_global_modules: {module: [arr_l]}
        """
        buckets: Dict[str, List[Tuple[List[np.ndarray], int]]] = {}
        for mods, n in modules_list:
            for k, v in mods.items():
                buckets.setdefault(k, []).append((v, n))

        new_global: Dict[str, List[np.ndarray]] = {}

        # --- OT-MoE or Avg：prior_adapt / guidance ---
        if self.moe_enable:
            for moe_key in self.moe_keys:
                if moe_key not in buckets:
                    continue
                client_params = [arrs for arrs, _ in buckets[moe_key]]
                experts = self.experts_map.get(moe_key)
                if not experts:
                    M = self.num_experts_prior if moe_key == "prior_adapt" else self.num_experts_guidance
                    M = max(1, int(M))
                    idx = np.linspace(0, len(client_params) - 1, num=M, dtype=int).tolist()
                    experts = [[w.copy() for w in client_params[i]] for i in idx]
                    self.experts_map[moe_key] = experts

                cost_fn = self.cost_fn_prior if moe_key == "prior_adapt" else self.cost_fn_guidance
                new_experts = ot_moe_aggregate(
                    client_params=client_params,  # N x L
                    expert_params=experts,  # M x L
                    cost_fn=cost_fn,
                )
                self.experts_map[moe_key] = new_experts
                merged = self._fedavg_arrays([(e, 1) for e in new_experts])
                new_global[moe_key] = merged

        # --- 2) others FedAvg ---
        for k, wl in buckets.items():
            if self.moe_enable:
                if k in self.moe_keys:
                    continue
            new_global[k] = self._fedavg_arrays(wl)

        return new_global

    def _flatten_module_dict(
            self, modules: Dict[str, List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        order: List[Tuple[str, int]] = []
        flat: List[np.ndarray] = []
        for k in sorted(modules.keys()):
            arrs = modules[k]
            order.append((k, len(arrs)))
            flat.extend(arrs)
        layout = {"order": order}
        return flat, layout


# Alias for compatibility
FedGuideServer = FedGuideStrategy
