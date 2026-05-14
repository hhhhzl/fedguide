from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Callable
import json
import numpy as np

from flwr.server.strategy import Strategy
from flwr.common import (
    FitRes,
    EvaluateRes,
    FitIns,
    EvaluateIns,
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
            ot_mode: str = "sinkhorn",  # "sinkhorn" (default) | "hungarian" (legacy)
            ot_reg: float = 0.05,
            # Personalized routing: each client receives its OT-row-weighted expert
            # mixture for moe_keys (default ON). Without this the server falls back
            # to broadcasting the FedAvg of all experts to every client, which
            # collapses the multi-modal structure across heterogeneous clients.
            personalized_routing: bool = True,
            # Experimental: client-specific expert routing for Bandit2D (legacy
            # mapped-id-based routing; superseded by personalized_routing for
            # general use).
            client_specific_expert_routing: bool = False,
            cid_mapping_file: Optional[str] = None,
            num_clients: int = 4,
            routing_debug: bool = False,
            # Policy aggregation cadence: aggregate policy/log_std only every
            # K rounds. K=1 (default) preserves prior behavior. K>1 lets each
            # client run local PPO without policy averaging on intermediate
            # rounds — useful when client dynamics differ enough that
            # FedAvg(policy) destroys per-client adaptation.
            policy_agg_every_k: int = 1,
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
        self.ot_mode = str(ot_mode).lower()
        self.ot_reg = float(ot_reg)
        self.personalized_routing = bool(personalized_routing)
        self.client_specific_expert_routing = bool(client_specific_expert_routing)
        self.cid_mapping_file = cid_mapping_file
        self.num_clients = int(num_clients)
        self.routing_debug = bool(routing_debug)
        self.policy_agg_every_k = max(1, int(policy_agg_every_k))

        # experts_map[module_key] = List[List[np.ndarray]]  # M x L
        self.experts_map: Dict[str, List[List[np.ndarray]]] = {}
        self._latest_global_modules: Dict[str, List[np.ndarray]] = {}
        self._latest_layout_json: Optional[str] = None
        # cid -> {moe_key: List[np.ndarray]} computed from the previous round's
        # OT plan; configure_fit looks this up to send personalized priors.
        self._latest_routed_per_cid: Dict[str, Dict[str, List[np.ndarray]]] = {}

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
        
        # Create client instructions with FitIns objects
        # Flower expects: List[Tuple[ClientProxy, FitIns]]
        client_instructions = []
        for client in sampled_clients:
            if self.on_fit_config_fn is not None:
                fit_config = self.on_fit_config_fn(server_round)
            else:
                fit_config = {"server_round": server_round}

            fit_parameters = parameters
            cid_str = str(getattr(client, "cid", ""))

            # New default path (Bug 5 fix): personalized broadcast for moe_keys.
            # Each client receives an OT-row-weighted mixture of the M experts
            # so heterogeneous client priors are not collapsed into a single
            # FedAvg of experts before broadcast.
            personalized_modules = (
                self._latest_routed_per_cid.get(cid_str)
                if self.personalized_routing else None
            )
            if personalized_modules and self._latest_global_modules:
                routed_modules: Dict[str, List[np.ndarray]] = {}
                for k, arrs in self._latest_global_modules.items():
                    if k not in self.moe_keys:
                        routed_modules[k] = arrs
                for moe_key in self.moe_keys:
                    if moe_key in personalized_modules:
                        routed_modules[moe_key] = personalized_modules[moe_key]
                    elif moe_key in self._latest_global_modules:
                        routed_modules[moe_key] = self._latest_global_modules[moe_key]
                flat, layout = self._flatten_module_dict(routed_modules)
                fit_parameters = ndarrays_to_parameters(flat)
                fit_config["layout"] = json.dumps(layout)
                if self.routing_debug:
                    print(
                        f"[FedGuidePersonalized] round={server_round} cid={cid_str} "
                        f"modules={list(routed_modules.keys())}"
                    )
            # Experimental mode: route prior/guidance experts by mapped client id.
            # This keeps federated learning but uses client-specific shared experts.
            elif self.client_specific_expert_routing:
                mapped_id = self._resolve_mapped_client_id(cid_str or "0")
                fit_config["client_id_mapped"] = int(mapped_id)
                routed_modules = self._build_routed_modules(mapped_id)
                if routed_modules:
                    expert_id = int(mapped_id)
                    for moe_key in self.moe_keys:
                        experts = self.experts_map.get(moe_key)
                        if experts:
                            expert_id = int(mapped_id) % len(experts)
                            break
                    fit_config["expert_id"] = int(expert_id)
                    flat, layout = self._flatten_module_dict(routed_modules)
                    fit_parameters = ndarrays_to_parameters(flat)
                    fit_config["layout"] = json.dumps(layout)
                    if self.routing_debug:
                        print(
                            f"[FedGuideRouting] round={server_round} cid={cid_str} "
                            f"mapped={mapped_id} expert={expert_id} modules={list(routed_modules.keys())}"
                        )
                elif self._latest_layout_json is not None:
                    fit_config["layout"] = self._latest_layout_json
            elif self._latest_layout_json is not None:
                # Non-routing mode can still use layout to reconstruct module dict on clients.
                fit_config["layout"] = self._latest_layout_json
            if self.routing_debug:
                fit_config["routing_debug"] = True

            # Create FitIns with parameters and config
            fit_ins = FitIns(
                parameters=fit_parameters,
                config=fit_config,
            )
            client_instructions.append((client, fit_ins))
        
        return client_instructions

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[Any, FitRes]],
            failures: List[Any],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not self.accept_failures and failures:
            return None, {}

        modules_list: List[Tuple[Dict[str, List[np.ndarray]], int]] = []
        client_cids: List[str] = []
        for client_proxy, fitres in results:
            # Handle both FitRes object and dict (in case of serialization issues)
            if isinstance(fitres, dict):
                metrics = fitres.get("metrics", {})
                num_examples = fitres.get("num_examples", 0)
                parameters = fitres.get("parameters", None)
            else:
                metrics = fitres.metrics
                num_examples = fitres.num_examples
                parameters = fitres.parameters

            mods = _modules_from_metrics(metrics)
            if mods is None:
                modules_list = []
                client_cids = []
                break
            modules_list.append((mods, num_examples))
            client_cids.append(str(getattr(client_proxy, "cid", "")))

        # Aggregate loss and other metrics from all clients
        # Also collect client actions and grid metrics from metrics for server-side metrics collection
        total_loss = 0.0
        total_examples = 0

        total_train_return = 0.0
        count_train_return = 0
        total_eval_return = 0.0
        count_eval_return = 0

        aggregated_metrics: Dict[str, Scalar] = {"server_round": server_round}
        collected_actions: Dict[int, Any] = {}  # {mapped_client_id: actions}
        collected_client_metrics: Dict[int, Dict[str, Any]] = {}  # {mapped_client_id: {metric_name: value}}
        
        for _, fitres in results:
            # Handle both FitRes object and dict (in case of serialization issues)
            if isinstance(fitres, dict):
                metrics = fitres.get("metrics", {})
                num_examples = fitres.get("num_examples", 0)
            else:
                metrics = fitres.metrics
                num_examples = fitres.num_examples
            
            # Aggregate loss (weighted by num_examples)
            if "loss" in metrics:
                try:
                    loss_val = float(metrics["loss"])
                    # Check for nan/inf
                    if loss_val == loss_val and loss_val != float('inf') and loss_val != float('-inf'):
                        total_loss += loss_val * num_examples
                        total_examples += num_examples
                except (TypeError, ValueError):
                    pass

            # agg returns (simple average over clients that reported them)
            if "train/return" in metrics:
                try:
                    train_return = float(metrics["train/return"])
                    if train_return == train_return:  # not NaN
                        total_train_return += train_return
                        count_train_return += 1
                except (TypeError, ValueError):
                    pass

            if "eval/return" in metrics:
                try:
                    eval_return = float(metrics["eval/return"])
                    if eval_return == eval_return:  # not NaN
                        total_eval_return += eval_return
                        count_eval_return += 1
                except (TypeError, ValueError):
                    pass
            
            # Collect client actions from metrics (passed from client fit method)
            if "client_actions" in metrics and "client_id_mapped" in metrics:
                try:
                    import json
                    import numpy as np
                    client_id_mapped = int(metrics["client_id_mapped"])
                    actions_json = metrics["client_actions"]
                    if isinstance(actions_json, str):
                        actions = json.loads(actions_json)
                        # Convert back to numpy array for consistency
                        actions = np.array(actions)
                        collected_actions[client_id_mapped] = actions
                except Exception as e:
                    # Silently fail if deserialization fails
                    pass
            
            # Collect client grid metrics (policy, value, prior evaluations on grid)
            if "client_id_mapped" in metrics:
                try:
                    import json
                    import numpy as np
                    client_id_mapped = int(metrics["client_id_mapped"])
                    client_grid_metrics = {}
                    # Look for metrics with prefix "client_grid_"
                    for key, value in metrics.items():
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
        
        # Add aggregated loss to metrics
        if total_examples > 0:
            aggregated_loss = total_loss / total_examples
            aggregated_metrics["loss"] = aggregated_loss
        else:
            aggregated_metrics["loss"] = 0.0

        # agg returns to metrics (same as FedKL)
        if count_train_return > 0:
            aggregated_metrics["train/return"] = total_train_return / count_train_return

        if count_eval_return > 0:
            aggregated_metrics["eval/return"] = total_eval_return / count_eval_return

        # add sample count / client count
        aggregated_metrics["total_samples"] = total_examples
        aggregated_metrics["num_clients"] = len(results)

        # Store collected actions and client metrics in strategy instance
        if not hasattr(self, '_collected_actions'):
            self._collected_actions = {}
        self._collected_actions[server_round] = collected_actions

        if not hasattr(self, '_collected_client_metrics'):
            self._collected_client_metrics = {}
        self._collected_client_metrics[server_round] = collected_client_metrics
        
        if modules_list:
            new_global = self._aggregate_by_modules(modules_list, client_cids=client_cids, server_round=server_round)
            # Policy-aggregation cadence: on non-K rounds, drop policy/log_std
            # from the broadcast so each client keeps its own local PPO state.
            # First round (server_round == 1) always broadcasts so clients
            # share the BC warm-start initialization.
            if (
                self.policy_agg_every_k > 1
                and server_round > 1
                and (server_round % self.policy_agg_every_k) != 0
            ):
                for _k in ("policy", "log_std"):
                    new_global.pop(_k, None)
                if self.routing_debug:
                    print(
                        f"[policy-skip] round={server_round} K={self.policy_agg_every_k} "
                        f"— broadcast keys={list(new_global.keys())}"
                    )
            flat, layout = self._flatten_module_dict(new_global)
            params = ndarrays_to_parameters(flat)
            self._latest_global_modules = new_global
            self._latest_layout_json = json.dumps(layout)
            aggregated_metrics["layout"] = json.dumps(layout)
            return params, aggregated_metrics

        # 2) FedAvg
        weighted = []
        for _, fitres in results:
            # Handle both FitRes object and dict (in case of serialization issues)
            if isinstance(fitres, dict):
                parameters = fitres.get("parameters", None)
                num_examples = fitres.get("num_examples", 0)
            else:
                parameters = fitres.parameters
                num_examples = fitres.num_examples
            
            if parameters is not None:
                weighted.append((parameters_to_ndarrays(parameters), num_examples))
        fedavg = self._fedavg_arrays(weighted)
        return ndarrays_to_parameters(fedavg), aggregated_metrics

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
        
        # Create client instructions with EvaluateIns objects
        # Flower expects: List[Tuple[ClientProxy, EvaluateIns]]
        client_instructions = []
        for client in sampled_clients:
            if self.on_evaluate_config_fn is not None:
                eval_config = self.on_evaluate_config_fn(server_round)
            else:
                eval_config = {"server_round": server_round}
            
            # Create EvaluateIns with parameters and config
            eval_ins = EvaluateIns(
                parameters=parameters,  # Current global parameters
                config=eval_config,
            )
            client_instructions.append((client, eval_ins))
        
        return client_instructions

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
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current global model on the server side.
        
        This calls the evaluate_fn if provided, which can be used for metrics collection.
        """
        if self.evaluate_fn is not None:
            # Call evaluate_fn with current parameters
            # evaluate_fn signature: (server_round: int, parameters: Parameters, config: Dict[str, Scalar])
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
                print(f"[FedGuideStrategy.evaluate] Error calling evaluate_fn: {e}")
                import traceback
                traceback.print_exc()
        
        # Return None if no evaluate_fn or if it returns None
        return None

    # ---- aggregate implementation ----
    def _aggregate_by_modules(
            self,
            modules_list: List[Tuple[Dict[str, List[np.ndarray]], int]],
            client_cids: Optional[List[str]] = None,
            server_round: int = -1,
    ) -> Dict[str, List[np.ndarray]]:
        """
        input：
          modules_list: [({module: [arr_l]}, num_examples), ...]
          client_cids: optional list of cid strings, one per modules_list entry,
            used to populate ``self._latest_routed_per_cid`` for personalized
            broadcast of OT-MoE keys.
        out：
          new_global_modules: {module: [arr_l]}  (FedAvg for non-MoE keys; for
          MoE keys this is the FedAvg-of-experts fallback used when
          personalized routing has no entry for a client.)
        """
        buckets: Dict[str, List[Tuple[List[np.ndarray], int]]] = {}
        for mods, n in modules_list:
            for k, v in mods.items():
                buckets.setdefault(k, []).append((v, n))

        if client_cids is None:
            client_cids = [str(i) for i in range(len(modules_list))]

        new_global: Dict[str, List[np.ndarray]] = {}

        # Reset per-client routing for this round; we'll repopulate per moe_key.
        routed_per_cid: Dict[str, Dict[str, List[np.ndarray]]] = {
            cid: {} for cid in client_cids
        }

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

                # Recompute the OT plan once so we can both update experts and
                # route each client to its row-weighted expert mixture.
                from fedguide.fed.fedguide.aggregator import compute_ot_matrix

                N = len(client_params)
                M = len(experts)
                cost_matrix = np.zeros((N, M), dtype=float)
                for i in range(N):
                    for m in range(M):
                        cost_matrix[i, m] = float(cost_fn(client_params[i], experts[m]))
                T = compute_ot_matrix(cost_matrix, mode=self.ot_mode, reg=self.ot_reg)

                # OT routing diagnostic: log cost / transport / per-client argmax
                # along with the client cid per row, since Flower returns clients
                # in completion order (not cid order). We compare argmax to the
                # "index of the expert originally seeded from this cid" rather
                # than to the row index.
                if server_round in (1, 5, 10, 20, 30, 40, 50, 60) and N == M:
                    np.set_printoptions(precision=3, suppress=True, linewidth=200)
                    row_norm = T / (T.sum(axis=1, keepdims=True) + 1e-12)
                    argmax_per_client = row_norm.argmax(axis=1).tolist()
                    cids_per_row = list(client_cids) if client_cids else [str(i) for i in range(N)]
                    print(
                        f"[OT-diag] round={server_round} key={moe_key} "
                        f"mode={self.ot_mode} reg={self.ot_reg} N={N} M={M}"
                    )
                    print(f"[OT-diag]  cids_per_row={cids_per_row}")
                    print(f"[OT-diag]  cost_matrix=\n{cost_matrix}")
                    print(f"[OT-diag]  row_norm=\n{row_norm}")
                    print(f"[OT-diag]  cid→expert: " + ", ".join(
                        f"{cids_per_row[i]}→e{argmax_per_client[i]}" for i in range(N)
                    ))

                # Update experts as before (column-normalized convex combo).
                new_experts: List[List[np.ndarray]] = []
                for m in range(M):
                    col = T[:, m]
                    csum = float(col.sum())
                    if csum <= 1e-12:
                        new_experts.append([w.copy() for w in experts[m]])
                        continue
                    w = col / csum
                    L = len(experts[m])
                    layers = [
                        sum(w[i] * client_params[i][l] for i in range(N))
                        for l in range(L)
                    ]
                    new_experts.append(layers)
                self.experts_map[moe_key] = new_experts

                # Per-client personalized prior: row-normalized OT plan tells us
                # how much of each expert to mix for client i. With M=1 this
                # reduces to expert_0 for everyone (= FedAvg). With M=N and
                # near-permutation T, each client receives the expert that best
                # matched its own update (preserves multi-modal structure).
                if self.personalized_routing and client_cids:
                    L = len(new_experts[0])
                    for i, cid in enumerate(client_cids):
                        row = T[i, :]
                        rsum = float(row.sum())
                        if rsum <= 1e-12:
                            chosen = new_experts[i % len(new_experts)]
                            routed_per_cid[cid][moe_key] = [w.copy() for w in chosen]
                            continue
                        rw = row / rsum
                        layers = [
                            sum(rw[m] * new_experts[m][l] for m in range(M))
                            for l in range(L)
                        ]
                        routed_per_cid[cid][moe_key] = layers

                # Fallback "global" merged expert for clients we cannot route
                # (e.g. configure_fit sees a cid that wasn't in the last
                # aggregate_fit call — first round, or client churn).
                merged = self._fedavg_arrays([(e, 1) for e in new_experts])
                new_global[moe_key] = merged

        # --- 2) others FedAvg ---
        for k, wl in buckets.items():
            if self.moe_enable:
                if k in self.moe_keys:
                    continue
            new_global[k] = self._fedavg_arrays(wl)

        # Publish per-client routing for the next configure_fit call.
        if self.personalized_routing:
            self._latest_routed_per_cid = routed_per_cid

        return new_global

    def _resolve_mapped_client_id(self, cid: str) -> int:
        cid_str = str(cid)
        if self.cid_mapping_file:
            from fedguide.utils.client_id_mapping import get_mapped_client_id

            return int(get_mapped_client_id(cid_str, self.num_clients, self.cid_mapping_file))
        try:
            if cid_str.isdigit() and int(cid_str) < 10000:
                return int(cid_str) % max(self.num_clients, 1)
            import hashlib

            h = int(hashlib.sha256(cid_str.encode()).hexdigest()[:8], 16)
            return h % max(self.num_clients, 1)
        except Exception:
            return abs(hash(cid_str)) % max(self.num_clients, 1)

    def _build_routed_modules(self, mapped_id: int) -> Dict[str, List[np.ndarray]]:
        if not self._latest_global_modules and not self.experts_map:
            return {}

        routed: Dict[str, List[np.ndarray]] = {}
        # Always include non-MoE modules from latest global state.
        for key, arrs in self._latest_global_modules.items():
            if key not in self.moe_keys:
                routed[key] = arrs

        # Replace MoE keys with client-routed experts.
        for moe_key in self.moe_keys:
            experts = self.experts_map.get(moe_key)
            if experts:
                idx = int(mapped_id) % len(experts)
                routed[moe_key] = experts[idx]
            elif moe_key in self._latest_global_modules:
                routed[moe_key] = self._latest_global_modules[moe_key]
        return routed

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
