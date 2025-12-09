import pickle
import numpy as np
from flwr.server.history import History


def summarize_history(path: str, label: str):
    """Load a Flower History object and print summary metrics."""

    # Load Flower History
    with open(path, "rb") as f:
        hist: History = pickle.load(f)

    # Flower stores aggregated metrics here:
    metrics = hist.metrics_distributed_fit

    print(f"\n=== Summary for {label} ===")
    print(f"Available metric keys: {list(metrics.keys())}")

    # We care about these two:
    metric_keys = ["train/return", "eval/return"]

    for metric_key in metric_keys:

        if metric_key not in metrics:
            print(f"\n[WARNING] '{metric_key}' not found in metrics!")
            continue

        # Each metric is a list of (round, value)
        pairs = metrics[metric_key]

        if len(pairs) == 0:
            print(f"\n[WARNING] '{metric_key}' exists but contains no data.")
            continue

        rounds = np.array([r for (r, v) in pairs], dtype=np.int32)
        vals = np.array([v for (r, v) in pairs], dtype=np.float32)

        final_round = int(rounds[-1])
        final_val = float(vals[-1])
        best_val = float(vals.max())
        auc = float(vals.mean())  # simple discrete AUC proxy

        print(f"\nMetric: {metric_key}")
        print(f"  Rounds logged:        {len(rounds)} (1 .. {final_round})")
        print(f"  Final value:          {final_val:.6f}")
        print(f"  Best value:           {best_val:.6f}")
        print(f"  AUC (mean over rounds): {auc:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history_path",
        type=str,
        required=True,
        help="Path to training_history.pkl (Flower History)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Run",
        help="Label printed in summary (e.g., FedGuide or FedKL)",
    )

    args = parser.parse_args()
    summarize_history(args.history_path, args.label)
