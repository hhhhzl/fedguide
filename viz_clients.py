# viz_clients.py
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt

LOGDIR = "/tmp/fedguide"


def load_runs():
    data = {}  # {cid: [(round, traj, passed_gate, reached_goal), ...]}
    for cdir in sorted(glob.glob(os.path.join(LOGDIR, "client_*"))):
        cid = os.path.basename(cdir).split("_")[-1]
        items = []
        for j in sorted(glob.glob(os.path.join(cdir, "round_*_meta.json"))):
            rnd = int(os.path.basename(j).split("_")[1])
            with open(j, "r") as f: meta = json.load(f)
            traj = np.load(os.path.join(cdir, f"round_{rnd}_traj.npy"))
            items.append((rnd, traj, bool(meta.get("passed_gate", False)), bool(meta.get("reached_goal", False))))
        data[cid] = sorted(items, key=lambda x: x[0])
    return data


def plot_maze(ax, grid):
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, grid.shape[0] - 0.5)
    ax.set_ylim(-0.5, grid.shape[1] - 0.5)
    walls = np.argwhere(grid == 0)
    if len(walls) > 0:
        ax.scatter(walls[:, 0], walls[:, 1], s=6, marker="s", alpha=0.4)
    ax.set_title("PointMazeNarrow (latest trajectories)")


def main():
    data = load_runs()
    size = 10
    grid = np.ones((size, size), dtype=int)
    mid = size // 2
    grid[mid, :] = 0
    grid[mid, (size // 2 - 1):(size // 2 + 1) + 1] = 1

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_maze(ax, grid)
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for k, (cid, items) in enumerate(sorted(data.items())):
        if not items: continue
        rnd, traj, passed, goal = items[-1]
        ax.plot(traj[:, 0], traj[:, 1], color=colors[k % len(colors)],
                label=f"client {cid} r{rnd} ({'✓' if passed else '×'})", lw=1.5, alpha=0.9)
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=30, color=colors[k % len(colors)])
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=40, color=colors[k % len(colors)])
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 4))
    win = 5
    for k, (cid, items) in enumerate(sorted(data.items())):
        rounds = [r for (r, _, _, _) in items]
        flags = [int(p) for (_, _, p, _) in items]
        mv = []
        for i in range(len(flags)):
            a = max(0, i - win + 1)
            mv.append(np.mean(flags[a:i + 1]))
        ax.plot(rounds, mv, color=colors[k % len(colors)], label=f"client {cid}")
    ax.set_xlabel("round")
    ax.set_ylabel("passed_gate (moving avg)")
    ax.set_title("Gate passing per client")
    ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
