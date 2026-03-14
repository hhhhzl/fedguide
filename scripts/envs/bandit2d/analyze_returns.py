#!/usr/bin/env python3
"""
Analyze train/return and eval/return from Bandit2D training history pickle files.
Works without flwr dependency by using a custom unpickler.
"""
import pickle
import sys
from io import BytesIO


class FakeHistory:
    """Minimal stub for flwr History when unpickling."""
    pass


def find_class(module, name):
    """Redirect flwr imports to our stub."""
    if module.startswith("flwr."):
        return type("FakeHistory", (), {"metrics_distributed_fit": {}, "metrics_centralized_fit": {}})
    return pickle.loads.__self__  # fallback - will fail, but we try


def load_history_safe(path: str):
    """Load pickle, handling flwr dependency."""
    with open(path, "rb") as f:
        data = f.read()

    # Try normal load first (works if flwr is installed)
    try:
        from flwr.server.history import History
        return pickle.loads(data)
    except Exception:
        pass

    # Pre-register flwr stub modules so pickle can resolve flwr.server.history.History
    import types
    flwr_history_mod = types.ModuleType("flwr.server.history")
    History = type("History", (), {"metrics_distributed_fit": {}, "metrics_centralized_fit": {}})
    flwr_history_mod.History = History

    flwr_server_mod = types.ModuleType("flwr.server")
    flwr_server_mod.history = flwr_history_mod

    flwr_mod = types.ModuleType("flwr")
    flwr_mod.server = flwr_server_mod

    sys.modules["flwr"] = flwr_mod
    sys.modules["flwr.server"] = flwr_server_mod
    sys.modules["flwr.server.history"] = flwr_history_mod

    try:
        obj = pickle.loads(data)
        return obj
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        raise


def summarize_single(hist, label: str, path: str):
    """Print summary for a single history."""
    import numpy as np
    metrics = getattr(hist, "metrics_distributed_fit", None)
    if metrics is None:
        metrics = getattr(hist, "metrics_centralized_fit", {})
    if not metrics:
        metrics = {}

    for metric_key in ["train/return", "eval/return"]:
        if metric_key not in metrics:
            continue
        pairs = metrics[metric_key]
        if not pairs:
            continue
        rounds = np.array([r for (r, v) in pairs], dtype=np.int32)
        vals = np.array([v for (r, v) in pairs], dtype=np.float32)
        final_val = float(vals[-1])
        best_val = float(vals.max())
        return {"final": final_val, "best": best_val, "metric_key": metric_key}
    return None


def main():
    import argparse
    import os
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--history_path", type=str, default=None, help="Path to single training_history.pkl")
    parser.add_argument("--metrics_dir", type=str, default=None, help="Metrics dir (e.g. ./metrics/bandit2d/fedguide); aggregates seed_X if present")
    parser.add_argument("--label", type=str, default="Run")
    args = parser.parse_args()

    path = args.history_path
    metrics_dir = args.metrics_dir
    label = args.label

    if metrics_dir and not path:
        # Multi-seed: look for seed_X/training_history.pkl; fallback to root training_history.pkl
        seed_files = sorted(glob.glob(os.path.join(metrics_dir, "seed_*", "training_history.pkl")))
        root_file = os.path.join(metrics_dir, "training_history.pkl")
        # Prefer root if it's newer (single-seed run) or if no seed dirs
        if seed_files and os.path.isfile(root_file):
            root_mtime = os.path.getmtime(root_file)
            seed_mtimes = [os.path.getmtime(f) for f in seed_files]
            if root_mtime > max(seed_mtimes):
                hist_files = [root_file]  # Use latest single-seed run
            else:
                hist_files = seed_files
        elif seed_files:
            hist_files = seed_files
        else:
            hist_files = [root_file] if os.path.isfile(root_file) else []
        if not hist_files:
            print(f"No history found in {metrics_dir}")
            sys.exit(1)
        import numpy as np
        by_metric = {"train/return": {"final": [], "best": []}, "eval/return": {"final": [], "best": []}}
        for hf in hist_files:
            try:
                hist = load_history_safe(hf)
                m = getattr(hist, "metrics_distributed_fit", None) or getattr(hist, "metrics_centralized_fit", {})
                for mk in ["train/return", "eval/return"]:
                    if mk not in m or not m[mk]:
                        continue
                    pairs = m[mk]
                    vals = np.array([v for (_, v) in pairs], dtype=np.float32)
                    by_metric[mk]["final"].append(float(vals[-1]))
                    by_metric[mk]["best"].append(float(vals.max()))
            except Exception as e:
                print(f"Skip {hf}: {e}")
        n_seeds = len(by_metric["train/return"]["final"]) or len(by_metric["eval/return"]["final"])
        if n_seeds == 0:
            print("No valid histories found")
            sys.exit(1)
        print(f"\n=== Multi-seed Summary for {label} ({n_seeds} seeds) ===")
        for mk in ["train/return", "eval/return"]:
            if not by_metric[mk]["final"]:
                continue
            f = np.array(by_metric[mk]["final"])
            b = np.array(by_metric[mk]["best"])
            print(f"\n{mk}:")
            print(f"  Final: mean={f.mean():.4f} ± {f.std():.4f}")
            print(f"  Best:  mean={b.mean():.4f} ± {b.std():.4f}")
        return

    if not path:
        parser.error("Either --history_path or --metrics_dir required")
    print(f"\n=== Summary for {label} ===")
    print(f"Path: {path}")

    try:
        hist = load_history_safe(path)
    except Exception as e:
        print(f"Load error: {e}")
        print("Tip: Run with fedguide conda env: conda activate fedguide && python ...")
        sys.exit(1)

    metrics = getattr(hist, "metrics_distributed_fit", None)
    if metrics is None:
        metrics = getattr(hist, "metrics_centralized_fit", {})
    if not metrics:
        metrics = {}

    print(f"Available metric keys: {list(metrics.keys())}")

    import numpy as np
    for metric_key in ["train/return", "eval/return"]:
        if metric_key not in metrics:
            print(f"\n[WARNING] '{metric_key}' not found in metrics!")
            continue

        pairs = metrics[metric_key]
        if not pairs:
            print(f"\n[WARNING] '{metric_key}' exists but contains no data.")
            continue

        rounds = np.array([r for (r, v) in pairs], dtype=np.int32)
        vals = np.array([v for (r, v) in pairs], dtype=np.float32)

        final_round = int(rounds[-1])
        final_val = float(vals[-1])
        best_val = float(vals.max())
        auc = float(vals.mean())

        print(f"\nMetric: {metric_key}")
        print(f"  Rounds logged:        {len(rounds)} (1 .. {final_round})")
        print(f"  Final value:          {final_val:.6f}")
        print(f"  Best value:           {best_val:.6f}")
        print(f"  AUC (mean over rounds): {auc:.6f}")


if __name__ == "__main__":
    main()
