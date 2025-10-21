# fed/server.py
import flwr as fl
import numpy as np
from typing import List, Tuple
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from fedguide.fed.aggregator import ot_moe_aggregate


class FedGuideServer(fl.server.strategy.FedAvg):
    """Custom server strategy with OT-MoE and metric aggregation."""

    def __init__(self, num_experts=2):
        super().__init__()
        self.num_experts = num_experts

    def aggregate_fit(self, rnd, results, failures):
        # FedAvg
        aggregated_params, _ = super().aggregate_fit(rnd, results, failures)
        total_loss, total_examples = 0.0, 0

        for _, fit_res in results:
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics
            if metrics is not None and "loss" in metrics:
                total_loss += metrics["loss"] * num_examples
                total_examples += num_examples

        avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
        print(f"[Server] Round {rnd} aggregated loss: {avg_loss:.4f}")
        return aggregated_params, {"loss": avg_loss}

    # def aggregate_fit(self, rnd, results, failures):
    #     if len(results) == 0:
    #         if self.current_experts is not None:
    #             return ndarrays_to_parameters(self.current_experts[0]), {}
    #         return super().aggregate_fit(rnd, results, failures)
    #
    #     client_params_nds: List[List[np.ndarray]] = []
    #     for _, fitres in results:
    #         nds = parameters_to_ndarrays(fitres.parameters)
    #         client_params_nds.append(nds)
    #
    #     if self.current_experts is None:
    #         self.current_experts = self._avg_as_experts(client_params_nds)
    #
    #     def cost_fn(client_nds: List[np.ndarray], expert_nds: List[np.ndarray]) -> float:
    #         s = 0.0
    #         for c, e in zip(client_nds, expert_nds):
    #             diff = (c - e).ravel()
    #             s += float(np.dot(diff, diff) / diff.size)
    #         return s
    #
    #     # OT-MoE
    #     new_experts = ot_moe_aggregate(client_params_nds, self.current_experts, cost_fn)
    #     self.current_experts = new_experts
    #     return ndarrays_to_parameters(self.current_experts[0]), {}
