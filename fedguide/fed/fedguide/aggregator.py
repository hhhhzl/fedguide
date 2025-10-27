import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Callable


def compute_ot_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    T = np.zeros_like(cost_matrix, dtype=float)
    mass = 1.0 / len(row_ind) if len(row_ind) > 0 else 0.0
    T[row_ind, col_ind] = mass
    return T


def ot_moe_aggregate(
        client_params: List[List[np.ndarray]],  # N x L (layers)
        expert_params: List[List[np.ndarray]],  # M x L
        cost_fn: Callable[[List[np.ndarray], List[np.ndarray]], float],
) -> List[List[np.ndarray]]:
    N, M = len(client_params), len(expert_params)
    cost_matrix = np.zeros((N, M), dtype=float)
    for i in range(N):
        for m in range(M):
            cost_matrix[i, m] = cost_fn(client_params[i], expert_params[m])

    T = compute_ot_matrix(cost_matrix)  # [N, M]
    new_experts: List[List[np.ndarray]] = []
    for m in range(M):
        weights = T[:, m]  # [N]
        if weights.sum() <= 0:
            new_experts.append([w.copy() for w in expert_params[m]])
            continue
        layers = []
        L = len(expert_params[m])
        for l in range(L):
            agg = sum(weights[i] * client_params[i][l] for i in range(N))
            layers.append(agg)
        new_experts.append(layers)
    return new_experts
