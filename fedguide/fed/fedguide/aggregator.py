"""OT-MoE aggregation utilities.

Two transport modes are supported:
- "sinkhorn" (default): entropic-regularized OT producing soft client->expert
  assignments. With M=1 expert this gracefully reduces to a uniform
  FedAvg over clients. With M=N it gives a soft permutation that is
  near-uniform when client priors are similar.
- "hungarian" (legacy): exact bipartite assignment via the Hungarian
  algorithm. Kept for backward compatibility / ablation only.

In both cases each expert is updated as a convex combination of client
parameters using the column-normalized transport plan, so the result is
always a valid weighted average (no parameter scaling artefacts).
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
from scipy.optimize import linear_sum_assignment

try:  # POT is optional; fall back to a tiny numpy Sinkhorn if missing.
    import ot as _pot  # type: ignore
    _HAS_POT = True
except Exception:  # pragma: no cover - tested via fallback path below.
    _HAS_POT = False


def _normalize_cost(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    cmax = float(np.max(C)) if C.size > 0 else 0.0
    if cmax > 1e-12:
        C = C / cmax
    return C


def _sinkhorn_numpy(a: np.ndarray, b: np.ndarray, C: np.ndarray, reg: float, n_iter: int = 200) -> np.ndarray:
    """Tiny vanilla Sinkhorn used when POT is unavailable."""
    K = np.exp(-C / max(reg, 1e-8))
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        v = b / (K.T @ u + 1e-30)
        u = a / (K @ v + 1e-30)
    T = (u[:, None] * K) * v[None, :]
    return T


def compute_ot_matrix(
    cost_matrix: np.ndarray,
    *,
    mode: str = "sinkhorn",
    reg: float = 0.05,
) -> np.ndarray:
    """Return an N x M transport plan T.

    Convention: T[i, m] is the mass of client i assigned to expert m.
    Marginals are uniform: row sums = 1/N, col sums = 1/M.
    The aggregator normalizes columns before mixing parameters.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    N, M = cost_matrix.shape
    if N == 0 or M == 0:
        return np.zeros_like(cost_matrix, dtype=float)

    mode = (mode or "sinkhorn").lower()

    if mode == "hungarian":
        # Legacy path — bipartite matching, kept for ablation.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        T = np.zeros_like(cost_matrix, dtype=float)
        if len(row_ind) > 0:
            mass = 1.0 / len(row_ind)
            T[row_ind, col_ind] = mass
        return T

    # Default: Sinkhorn with uniform marginals on a normalized cost.
    a = np.full((N,), 1.0 / N, dtype=float)
    b = np.full((M,), 1.0 / M, dtype=float)
    C = _normalize_cost(cost_matrix)
    if _HAS_POT:
        T = _pot.sinkhorn(a, b, C, reg)
    else:
        T = _sinkhorn_numpy(a, b, C, reg)
    return np.asarray(T, dtype=float)


def ot_moe_aggregate(
    client_params: List[List[np.ndarray]],  # N x L (layers)
    expert_params: List[List[np.ndarray]],  # M x L
    cost_fn: Callable[[List[np.ndarray], List[np.ndarray]], float],
    *,
    mode: str = "sinkhorn",
    reg: float = 0.05,
) -> List[List[np.ndarray]]:
    """Update each expert as a convex combination of client params via OT plan.

    Falls back to keeping the previous expert if its column has zero mass
    (only possible numerically). The returned list has the same shapes as
    ``expert_params`` so downstream code can treat it as a drop-in update.
    """
    N, M = len(client_params), len(expert_params)
    if N == 0 or M == 0:
        return [list(map(np.copy, e)) for e in expert_params]

    cost_matrix = np.zeros((N, M), dtype=float)
    for i in range(N):
        for m in range(M):
            cost_matrix[i, m] = float(cost_fn(client_params[i], expert_params[m]))

    T = compute_ot_matrix(cost_matrix, mode=mode, reg=reg)

    new_experts: List[List[np.ndarray]] = []
    for m in range(M):
        col = T[:, m]
        col_sum = float(col.sum())
        if col_sum <= 1e-12:
            new_experts.append([w.copy() for w in expert_params[m]])
            continue
        weights = col / col_sum  # column-normalized: convex combination
        L = len(expert_params[m])
        layers: List[np.ndarray] = []
        for l in range(L):
            agg = sum(weights[i] * client_params[i][l] for i in range(N))
            layers.append(agg)
        new_experts.append(layers)
    return new_experts
