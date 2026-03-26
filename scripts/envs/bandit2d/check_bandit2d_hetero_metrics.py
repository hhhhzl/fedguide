#!/usr/bin/env python3
"""
Check Bandit2D heterogeneity from saved metrics.

This script reads metrics pickle and prints each client's policy peak location
for one or multiple rounds, then compares against expected quadrants:
  client 0 -> right (+x)
  client 1 -> top (+y)
  client 2 -> left (-x)
  client 3 -> bottom (-y)
"""

from __future__ import annotations

import argparse
import pickle
from typing import Dict, List, Tuple


EXPECTED = {
    0: "right",
    1: "top",
    2: "left",
    3: "bottom",
}


def quadrant_from_xy(x: float, y: float) -> str:
    if abs(x) >= abs(y):
        return "right" if x >= 0 else "left"
    return "top" if y >= 0 else "bottom"


def peak_xy(policy_density: List[List[float]], bounds: Tuple[float, float]) -> Tuple[float, float]:
    best = None
    best_i, best_j = 0, 0
    nrows = len(policy_density)
    ncols = len(policy_density[0]) if nrows else 0
    for i, row in enumerate(policy_density):
        for j, value in enumerate(row):
            if best is None or value > best:
                best = value
                best_i, best_j = i, j
    xmin, xmax = bounds
    x = xmin + (best_j / max(ncols - 1, 1)) * (xmax - xmin)
    y = xmin + (best_i / max(nrows - 1, 1)) * (xmax - xmin)
    return x, y


def resolve_round(round_arg: int, history_len: int) -> int:
    if round_arg < 0:
        return max(0, history_len + round_arg)
    return min(round_arg, history_len - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Bandit2D heterogeneity from metrics")
    parser.add_argument("--metrics_path", type=str, required=True, help="Path to bandit2d_metrics.pkl")
    parser.add_argument(
        "--round_num",
        type=int,
        default=-1,
        help="Round index to inspect. -1 means last round.",
    )
    parser.add_argument(
        "--also_round1",
        action="store_true",
        help="Also print round 1 for early-stage divergence check.",
    )
    args = parser.parse_args()

    with open(args.metrics_path, "rb") as f:
        data: Dict = pickle.load(f)

    history: List[Dict] = data.get("metrics_history", [])
    if not history:
        raise ValueError(f"No metrics_history in {args.metrics_path}")

    bounds = tuple(data.get("bounds", (-1.5, 1.5)))
    rounds = [resolve_round(args.round_num, len(history))]
    if args.also_round1 and len(history) > 1:
        rounds = [1, rounds[0]] if rounds[0] != 1 else rounds

    for ridx in rounds:
        metrics = history[ridx]
        client_metrics = metrics.get("client_metrics", {})
        client_ids = sorted(client_metrics.keys())
        print(f"Round {ridx}: client_metrics keys = {client_ids} (count={len(client_ids)})")

        matched = 0
        total = 0
        for cid in client_ids:
            cm = client_metrics.get(cid, {})
            pd = cm.get("policy_density")
            if pd is None:
                print(f"  Client {cid}: missing policy_density")
                continue

            x, y = peak_xy(pd, bounds)
            got = quadrant_from_xy(x, y)
            expect = EXPECTED.get(int(cid), "unknown")
            ok = got == expect
            tag = "OK" if ok else "CHECK"
            print(
                f"  Client {cid}: peak (~{x:+.2f}, ~{y:+.2f}) -> {got} | expected ~{expect} [{tag}]"
            )
            matched += int(ok)
            total += 1
        if total:
            print(f"\nSummary: {matched}/{total} clients match expected quadrant (strict).")
        print()


if __name__ == "__main__":
    main()
