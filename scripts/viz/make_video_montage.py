"""Build appendix qualitative rollout montages for the FedGuide family.

The script uses already-rendered videos under plots/<env>/<algo>/seed_<s> and
selects a visualization seed from metrics/<env>/<algo>/seed_<s>.  If
client-level return curves are available, each client is selected independently;
otherwise the best seed for the env/algorithm pair is used for all clients.

Outputs:
    -CORL-FedGuide/figures/videos/<env>_fedguide_family.png
    assets/video_selection.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "-CORL-FedGuide"
DEFAULT_OUT_DIR = PAPER_ROOT / "figures" / "videos"
DEFAULT_SELECTION_JSON = PROJECT_ROOT / "assets" / "video_selection.json"

ALGOS = [
    ("fedguide_a", "FedGuide-A", "#ff7f0e"),
    ("fedguide_p", "FedGuide-P", "#d62728"),
    ("fedguide", "FedGuide", "#1f77b4"),
]

ENVS = [
    {"key": "reacher", "display": "Reacher", "clients": 8, "mode": "strip", "clients_per_row": 2, "episodes": 10},
    {"key": "hopper", "display": "Hopper", "clients": 8, "mode": "strip", "clients_per_row": 2},
    {"key": "walker", "display": "Walker2D", "clients": 8, "mode": "strip", "clients_per_row": 2},
    {"key": "halfcheetah", "display": "HalfCheetah", "clients": 8, "mode": "strip", "clients_per_row": 2},
    {"key": "metaworld", "display": "MetaWorld10", "clients": 10, "mode": "strip", "clients_per_row": 2},
]

STRIP_FRAME_SIZE = (96, 96)
STRIP_N_FRAMES = 5
STRIP_GAP = 4
STRIP_W = STRIP_N_FRAMES * STRIP_FRAME_SIZE[0] + (STRIP_N_FRAMES - 1) * STRIP_GAP
STRIP_H = STRIP_FRAME_SIZE[1]

CELL_SIZE = {
    "ghost": (240, 240),
    "ghost_traj": (400, 200),
    "strip": (STRIP_W, STRIP_H),
}

RENDER_EPISODES = {
    "reacher": 1,
    "hopper": 1,
    "walker": 1,
    "halfcheetah": 1,
    "metaworld": 1,
}

SEEDS = [0, 1, 2, 3, 4]
ROUND = 100
TAIL = 10


def _install_pickle_stubs() -> None:
    """Allow Flower History pickles to load even when Flower is unavailable."""
    flwr = types.ModuleType("flwr")
    server = types.ModuleType("flwr.server")
    history = types.ModuleType("flwr.server.history")

    class History:
        pass

    history.History = History
    sys.modules.setdefault("flwr", flwr)
    sys.modules.setdefault("flwr.server", server)
    sys.modules.setdefault("flwr.server.history", history)
    try:
        import numpy.core as np_core
        import numpy.core.multiarray as np_multiarray
        import numpy.core.numeric as np_numeric
        import numpy.core.umath as np_umath
    except Exception:
        return
    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)
    sys.modules.setdefault("numpy._core.umath", np_umath)


_install_pickle_stubs()


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_COL = _font(24, bold=False)
FONT_ROW = _font(24, bold=False)
FONT_NOTE = _font(18, bold=False)
FONT_TASK = _font(16, bold=False)
FONT_MISSING = _font(18, bold=False)


def _history_path(metrics_root: Path, env: str, algo: str, seed: int) -> Path:
    return metrics_root / env / algo / f"seed_{seed}" / "training_history.pkl"


def _video_path(plots_root: Path, env: str, algo: str, seed: int, client: int, round_id: int) -> Path:
    return plots_root / env / algo / f"seed_{seed}" / f"client_{client}" / f"round_{round_id:04d}.mp4"


def _format_task_label(task: str) -> str:
    words = task.replace("-v3", "").replace("_", "-").split("-")
    words = [w.capitalize() for w in words if w]
    if words[:2] == ["Pick", "Place"]:
        return "Pick Place"
    return " ".join(words)


_COMPACT_ABBREV = {
    "button-press-topdown": "BtnPress",
    "peg-insert-side": "PegInsert",
}


def _compact_task_label(task: str) -> str:
    """Tightly-packed task name with no separators (e.g., 'pick-place-v3' → 'PickPlace').

    Long names are abbreviated so the uniform font size across labels stays readable.
    """
    base = task.replace("-v3", "").replace("_", "-")
    if base in _COMPACT_ABBREV:
        return _COMPACT_ABBREV[base]
    return "".join(w.capitalize() for w in base.split("-") if w)


def _metaworld_task_labels(compact: bool = False) -> List[str]:
    path = PROJECT_ROOT / "data" / "metaworld" / "metadata.json"
    try:
        meta = json.loads(path.read_text())
    except Exception:
        return []
    clients = sorted(meta.get("clients", []), key=lambda item: int(item.get("client_id", 0)))
    formatter = _compact_task_label if compact else _format_task_label
    return [formatter(str(client.get("task", f"client-{idx}"))) for idx, client in enumerate(clients)]


def _project_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def _resolve_project_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_history(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"[warn] could not load {path}: {exc}")
        return None


def _tail_mean(series: Sequence[Tuple[int, Any]], tail: int = TAIL) -> Optional[float]:
    if not series:
        return None
    vals = []
    for _, val in series[-tail:]:
        try:
            vals.append(float(val))
        except Exception:
            continue
    if not vals:
        return None
    return float(np.mean(vals))


def _metric_containers(history: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for attr in ("metrics_distributed_fit", "metrics_distributed", "metrics_centralized"):
        container = getattr(history, attr, None)
        if isinstance(container, dict):
            yield attr, container


def _client_metric_keys(client: int) -> List[str]:
    return [
        f"client_{client}/eval/return",
        f"client_{client}:eval/return",
        f"client_{client}.eval/return",
        f"client/{client}/eval/return",
        f"eval/return/client_{client}",
        f"eval/return/client/{client}",
        f"client_eval/return_{client}",
    ]


def _score_seed(history: Any, client: int) -> Tuple[Optional[float], str, str]:
    """Return (score, source, mode) for a seed and client."""
    for attr, container in _metric_containers(history):
        for key in _client_metric_keys(client):
            score = _tail_mean(container.get(key, []))
            if score is not None:
                return score, f"{attr}:{key}:tail{TAIL}", "client_specific"
    for attr, container in _metric_containers(history):
        score = _tail_mean(container.get("eval/return", []))
        if score is not None:
            return score, f"{attr}:eval/return:tail{TAIL}", "env_algo_average"
    return None, "missing", "missing"


def _choose_seed(
    metrics_root: Path,
    plots_root: Path,
    env: str,
    algo: str,
    client: int,
    round_id: int,
) -> Dict[str, Any]:
    candidates = []
    for seed in SEEDS:
        video = _video_path(plots_root, env, algo, seed, client, round_id)
        if not video.exists():
            continue
        history = _load_history(_history_path(metrics_root, env, algo, seed))
        if history is None:
            continue
        score, source, mode = _score_seed(history, client)
        if score is None:
            continue
        candidates.append(
            {
                "seed": seed,
                "score": score,
                "score_source": source,
                "selection_mode": mode,
                "video": _project_rel(video),
                "episode": 0,
            }
        )
    if not candidates:
        fallback = None
        for seed in SEEDS:
            video = _video_path(plots_root, env, algo, seed, client, round_id)
            if video.exists():
                fallback = {
                    "seed": seed,
                    "score": None,
                    "score_source": "video_exists_only",
                    "selection_mode": "fallback",
                    "video": _project_rel(video),
                    "episode": 0,
                }
                break
        if fallback is None:
            fallback = {
                "seed": None,
                "score": None,
                "score_source": "missing",
                "selection_mode": "missing",
                "video": None,
                "episode": 0,
            }
        return fallback
    return max(candidates, key=lambda item: item["score"])


def _episode_bounds(count: int, episode: int, total_episodes: int) -> Tuple[int, int]:
    total = max(1, int(total_episodes))
    ep = max(0, min(int(episode), total - 1))
    start = int(round(ep * count / total))
    stop = int(round((ep + 1) * count / total)) - 1
    return max(0, start), max(start, min(count - 1, stop))


def _read_frames(
    path: Path,
    n_frames: int,
    start: float = 0.08,
    end: float = 0.92,
    episode: int = 0,
    total_episodes: int = 1,
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        raise RuntimeError(f"video has no frames: {path}")
    ep_lo, ep_hi = _episode_bounds(count, episode, total_episodes)
    span = max(1, ep_hi - ep_lo)
    lo = ep_lo + max(0, min(span, int(round(start * span))))
    hi = ep_lo + max(0, min(span, int(round(end * span))))
    hi = max(lo, min(ep_hi, hi))
    indices = np.linspace(lo, hi, num=n_frames, dtype=int)
    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read sampled frames: {path}")
    return frames


def _fit_frame(frame: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    image = Image.fromarray(frame)
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _strip_cell(path: Path, env_key: str, episode: int) -> Image.Image:
    total_episodes = RENDER_EPISODES.get(env_key, 1)
    frames = _read_frames(
        path,
        n_frames=STRIP_N_FRAMES,
        start=0.08,
        end=0.92,
        episode=episode,
        total_episodes=total_episodes,
    )
    if env_key == "reacher":
        frames = _autocrop_letterbox(frames)
    out = Image.new("RGB", (STRIP_W, STRIP_H), "white")
    for i, frame in enumerate(frames):
        out.paste(_fit_frame(frame, STRIP_FRAME_SIZE), (i * (STRIP_FRAME_SIZE[0] + STRIP_GAP), 0))
    return out


def _ghost_window(env_key: str) -> Tuple[float, float, int]:
    if env_key == "reacher":
        return 0.10, 0.95, 5
    if env_key == "metaworld":
        return 0.08, 0.92, 7
    # ghost_traj envs sample more frames to expose the trajectory
    if env_key in {"hopper", "walker", "halfcheetah"}:
        return 0.05, 0.95, 6
    return 0.12, 0.88, 7


def _autocrop_letterbox(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Strip uniform-black top/bottom borders that some envs render (e.g. Reacher)."""
    sample = frames[0]
    h, w, _ = sample.shape
    row_lum = sample.mean(axis=(1, 2))
    threshold = 8.0
    top = 0
    while top < h and row_lum[top] < threshold:
        top += 1
    bot = h - 1
    while bot > top and row_lum[bot] < threshold:
        bot -= 1
    if top == 0 and bot == h - 1:
        return frames
    return [f[top : bot + 1, :, :] for f in frames]


def _read_dense_frames(
    path: Path,
    n_frames: int,
    start: float,
    end: float,
    episode: int,
    total_episodes: int,
) -> List[np.ndarray]:
    """Densely sample frames inside an episode window for median background estimation."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        raise RuntimeError(f"video has no frames: {path}")
    ep_lo, ep_hi = _episode_bounds(count, episode, total_episodes)
    span = max(1, ep_hi - ep_lo)
    lo = ep_lo + max(0, min(span, int(round(start * span))))
    hi = ep_lo + max(0, min(span, int(round(end * span))))
    hi = max(lo, min(ep_hi, hi))
    indices = np.linspace(lo, hi, num=n_frames, dtype=int)
    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read dense frames: {path}")
    return frames


def _bg_subtraction_mask(frame: np.ndarray, bg: np.ndarray, threshold: float) -> np.ndarray:
    diff = np.abs(frame.astype(np.float32) - bg.astype(np.float32)).mean(axis=2)
    mask = (diff > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    return mask.astype(bool)


def _per_frame_agent_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Per-frame agent silhouette via floor/sky color exclusion.

    MuJoCo Gym shares a common palette: cyan checker floor (hue ~75-110, high
    saturation) with dark grid lines, and a dark blue-gray sky.  Anything that
    is neither floor nor sky is treated as agent.  Largest central connected
    component is kept to reject color-similar noise outside the body.
    """
    hsv = cv2.cvtColor(np.clip(frame_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    sky = (val < 110) & (sat < 90)
    floor_tile = (hue >= 70) & (hue <= 115) & (sat > 50)
    floor_dark = (val < 55) & (sat < 110)
    not_bg = ~sky & ~floor_tile & ~floor_dark

    u8 = not_bg.astype(np.uint8) * 255
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))

    h, w = u8.shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8)
    if n_labels > 1:
        cx_target = w / 2.0
        best = 0
        best_score = -1.0
        for i in range(1, n_labels):
            area = float(stats[i, cv2.CC_STAT_AREA])
            cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2.0
            score = area / (1.0 + 0.6 * abs(cx - cx_target))
            if score > best_score:
                best_score = score
                best = i
        u8 = ((labels == best).astype(np.uint8)) * 255

    u8 = cv2.dilate(u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return u8.astype(bool)


def _floor_strip_backdrop(dense_frames: List[np.ndarray], target_h: int, target_w: int) -> np.ndarray:
    """Backdrop: average row-color profile from agent-free edge columns, replicated horizontally.

    Computes per-row median color from the leftmost and rightmost edge strips of
    several dense frames (camera tracks the agent → those columns are reliably
    background).  Smooths vertically, then replicates the single profile column
    across the target width — produces a clean horizon scene with no seams.
    """
    h, w, _ = dense_frames[0].shape
    edge_w = max(16, w // 6)
    edge_pixels: List[np.ndarray] = []
    for f in dense_frames:
        m = _per_frame_agent_mask(f)
        left = f[:, :edge_w, :].astype(np.float32)
        right = f[:, -edge_w:, :].astype(np.float32)
        left_valid = ~m[:, :edge_w]
        right_valid = ~m[:, -edge_w:]
        edge_pixels.append((left, left_valid))
        edge_pixels.append((right, right_valid))
    profile = np.zeros((h, 3), dtype=np.float32)
    for y in range(h):
        vals = []
        for arr, valid in edge_pixels:
            row_valid = valid[y]
            if row_valid.any():
                vals.append(arr[y][row_valid])
        if vals:
            stacked = np.concatenate(vals, axis=0)
            profile[y] = np.median(stacked, axis=0)
    profile = cv2.GaussianBlur(profile.reshape(h, 1, 3), (1, 7), sigmaX=0, sigmaY=3).reshape(h, 3)
    backdrop = np.broadcast_to(profile[:, None, :], (h, target_w, 3)).copy().astype(np.float32)
    if backdrop.shape[0] != target_h:
        backdrop = cv2.resize(backdrop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return backdrop


def _ghost_cell(path: Path, env_key: str, episode: int) -> Image.Image:
    """Fixed-camera ghost overlay: median background + bg subtraction + supersample."""
    start, end, n_frames = _ghost_window(env_key)
    total_episodes = RENDER_EPISODES.get(env_key, 1)
    cell_w, cell_h = CELL_SIZE["ghost"]
    super_w, super_h = cell_w * 2, cell_h * 2

    bg_frames = _read_dense_frames(path, n_frames=16, start=start, end=end, episode=episode, total_episodes=total_episodes)
    sample_frames = _read_frames(path, n_frames=n_frames, start=start, end=end, episode=episode, total_episodes=total_episodes)

    if env_key == "reacher":
        bg_frames = _autocrop_letterbox(bg_frames)
        sample_frames = _autocrop_letterbox(sample_frames)

    def _resize_super(arr: np.ndarray) -> np.ndarray:
        return np.asarray(ImageOps.fit(Image.fromarray(arr), (super_w, super_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))).astype(np.float32)

    bg_stack = np.stack([_resize_super(f) for f in bg_frames], axis=0)
    median_bg = np.median(bg_stack, axis=0)
    canvas = median_bg.copy()

    threshold = 15.0 if env_key == "reacher" else 22.0
    fitted = [_resize_super(f) for f in sample_frames]
    n = len(fitted)
    for idx, frame in enumerate(fitted):
        mask = _bg_subtraction_mask(frame, median_bg, threshold)
        if not np.any(mask):
            continue
        alpha = 0.45 + 0.50 * idx / max(1, n - 1)
        canvas[mask] = (1.0 - alpha) * canvas[mask] + alpha * frame[mask]

    out_img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    return out_img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)


def _camera_track_x(path: Path, episode: int, total_episodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (frame_indices, cumulative_camera_dx_px) over the episode.

    Uses dense pairwise phaseCorrelate on the floor region (bottom half of frame).
    Small inter-frame shifts compose cleanly; sparse correlate fails on periodic
    floor textures.  Units are source-frame pixels.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        raise RuntimeError(f"video has no frames: {path}")
    ep_lo, ep_hi = _episode_bounds(count, episode, total_episodes)
    step = 2
    indices = list(range(ep_lo, ep_hi + 1, step))
    if not indices or indices[-1] != ep_hi:
        indices.append(ep_hi)
    cam_x: List[float] = []
    prev_floor: Optional[np.ndarray] = None
    cum = 0.0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            cam_x.append(cum)
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h = gray.shape[0]
        floor = gray[h // 2 : int(h * 0.97), :].copy()
        floor -= float(floor.mean())
        if prev_floor is not None and prev_floor.shape == floor.shape:
            try:
                (dx, _), _ = cv2.phaseCorrelate(prev_floor, floor)
            except cv2.error:
                dx = 0.0
            # Reject obvious aliasing failures (>1/4 width per step is unphysical at step=2)
            if abs(dx) > floor.shape[1] * 0.25:
                dx = 0.0
            cum += float(dx)
        cam_x.append(cum)
        prev_floor = floor
    cap.release()
    return np.array(indices, dtype=np.int64), np.array(cam_x, dtype=np.float32)


def _ghost_traj_cell(path: Path, env_key: str, episode: int) -> Image.Image:
    """Camera-compensated trajectory ghost.

    Uses dense floor phase-correlation for camera tracking, temporal-variance
    template for clean agent silhouettes, and a synthesized uniform backdrop
    (no floor stitching) to avoid alignment seams.
    """
    start, end, n_frames = _ghost_window(env_key)
    total_episodes = RENDER_EPISODES.get(env_key, 1)
    cell_w, cell_h = CELL_SIZE["ghost_traj"]
    super_w, super_h = cell_w * 2, cell_h * 2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count <= 0:
        raise RuntimeError(f"video has no frames: {path}")

    ep_lo, ep_hi = _episode_bounds(count, episode, total_episodes)
    span = max(1, ep_hi - ep_lo)
    lo = ep_lo + max(0, min(span, int(round(start * span))))
    hi = ep_lo + max(0, min(span, int(round(end * span))))
    hi = max(lo, min(ep_hi, hi))
    sample_idx = np.linspace(lo, hi, num=n_frames, dtype=np.int64)

    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"too few sample frames: {path}")

    dense_idx = np.linspace(lo, hi, num=24, dtype=np.int64)
    cap = cv2.VideoCapture(str(path))
    dense_frames: List[np.ndarray] = []
    for idx in dense_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok:
            dense_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    track_idx, track_cam_x = _camera_track_x(path, episode, total_episodes)
    cam_x_at_sample = np.interp(sample_idx.astype(np.float32), track_idx.astype(np.float32), track_cam_x)
    world_x_src = -cam_x_at_sample

    src_h, src_w, _ = frames[0].shape
    scale_x = super_w / float(src_w)
    world_x = world_x_src * scale_x

    fitted = [cv2.resize(f, (super_w, super_h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) for f in frames]
    masks = [_per_frame_agent_mask(f.astype(np.uint8)) for f in fitted]

    world_x = world_x - world_x.min()
    canvas_w_inner = int(super_w + world_x.max())
    pad = super_w // 16
    canvas_total_w = canvas_w_inner + 2 * pad

    canvas = _floor_strip_backdrop(dense_frames if len(dense_frames) >= 4 else frames, super_h, canvas_total_w)
    edge_color = tuple(int(c) for c in canvas[0, 0])

    n = len(fitted)
    for idx in range(n):
        offset = pad + int(round(world_x[idx]))
        frame = fitted[idx]
        mask = masks[idx]
        x0 = offset
        x1 = offset + super_w
        if x1 <= 0 or x0 >= canvas_total_w:
            continue
        sx0 = max(0, -x0)
        sx1 = super_w - max(0, x1 - canvas_total_w)
        cx0 = x0 + sx0
        cx1 = x0 + sx1
        sub_mask = mask[:, sx0:sx1]
        if not np.any(sub_mask):
            continue
        alpha = 0.42 + 0.53 * idx / max(1, n - 1)
        region = canvas[:, cx0:cx1, :].copy()
        frame_region = frame[:, sx0:sx1, :]
        region[sub_mask] = (1.0 - alpha) * region[sub_mask] + alpha * frame_region[sub_mask]
        canvas[:, cx0:cx1, :] = region

    out_img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    return ImageOps.fit(out_img, (cell_w, cell_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _missing_cell(size: Tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(out)
    text = "Missing"
    box = draw.textbbox((0, 0), text, font=FONT_MISSING)
    draw.text(((size[0] - (box[2] - box[0])) / 2, (size[1] - (box[3] - box[1])) / 2), text, fill=(120, 120, 120), font=FONT_MISSING)
    return out


def _make_cell(entry: Dict[str, Any], mode: str, env_key: str) -> Image.Image:
    target_size = CELL_SIZE.get(mode, CELL_SIZE["ghost"])
    video_path = _resolve_project_path(entry.get("video"))
    episode = int(entry.get("episode", 0) or 0)
    if video_path is None:
        return _missing_cell(target_size)
    try:
        if mode == "ghost":
            return _ghost_cell(video_path, env_key, episode=episode)
        if mode == "ghost_traj":
            return _ghost_traj_cell(video_path, env_key, episode=episode)
        return _strip_cell(video_path, env_key, episode=episode)
    except Exception as exc:
        print(f"[warn] failed to render {video_path}: {exc}")
        return _missing_cell(target_size)


def _draw_centered(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, fill: str | Tuple[int, int, int]) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    x0, y0, x1, y1 = xy
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, fill=fill, font=font)


def _render_rotated_label(
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] | str,
    angle: int = 90,
) -> Image.Image:
    """Render text on transparent canvas and rotate by `angle` degrees CCW.

    angle=90 → text reads bottom-up (left-side label);
    angle=270 → text reads top-down (right-side label).
    """
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = measure.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    padding = 2
    img = Image.new("RGBA", (tw + 2 * padding, th + 2 * padding), (255, 255, 255, 0))
    ImageDraw.Draw(img).text((padding - bbox[0], padding - bbox[1]), text, fill=fill, font=font)
    return img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)


def _fit_rotated_label(
    text: str,
    max_height: int,
    fill: Tuple[int, int, int] | str,
    angle: int = 90,
    max_size: int = 20,
    min_size: int = 8,
) -> Image.Image:
    """Auto-shrink font so the *rendered* rotated label height ≤ max_height."""
    last = None
    for size in range(max_size, min_size - 1, -1):
        img = _render_rotated_label(text, _font(size, bold=False), fill, angle=angle)
        last = img
        if img.height <= max_height:
            return img
    return last if last is not None else _render_rotated_label(text, _font(min_size, bold=False), fill, angle=angle)


def _fit_rotated_label_uniform(
    texts: Sequence[str],
    max_height: int,
    fill: Tuple[int, int, int] | str,
    angle: int = 90,
    max_size: int = 20,
    min_size: int = 8,
) -> List[Image.Image]:
    """Pick one font size for ALL `texts` so each rotated label fits `max_height`.

    Uses the largest size where every text fits — keeps typography consistent
    across the figure (the longest label dictates the size).
    """
    for size in range(max_size, min_size - 1, -1):
        font = _font(size, bold=False)
        imgs = [_render_rotated_label(t, font, fill, angle=angle) for t in texts]
        if all(img.height <= max_height for img in imgs):
            return imgs
    font = _font(min_size, bold=False)
    return [_render_rotated_label(t, font, fill, angle=angle) for t in texts]


def _pick_uniform_font_size(
    texts: Sequence[str],
    max_height: int,
    angle: int,
    max_size: int = 20,
    min_size: int = 8,
) -> int:
    """Largest font size where every text's rotated height fits `max_height` at the given angle."""
    fill = (0, 0, 0)
    for size in range(max_size, min_size - 1, -1):
        font = _font(size, bold=False)
        if all(_render_rotated_label(t, font, fill, angle=angle).height <= max_height for t in texts):
            return size
    return min_size


def _draw_singleline_fit(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str | Tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = xy
    max_w = x1 - x0 - 2
    fitted_font = font
    for size in range(16, 9, -1):
        candidate = _font(size, bold=False)
        box = draw.textbbox((0, 0), text, font=candidate)
        if box[2] - box[0] <= max_w:
            fitted_font = candidate
            break
    _draw_centered(draw, xy, text, fitted_font, fill)


def _compose_env(
    env: Dict[str, Any],
    selections: Dict[str, Dict[str, Dict[str, Any]]],
    out_dir: Path,
) -> Path:
    env_key = env["key"]
    mode = env["mode"]
    n_clients = int(env["clients"])
    clients_per_row = int(env.get("clients_per_row", 2))
    cell_w, cell_h = CELL_SIZE.get(mode, CELL_SIZE["strip"])
    n_rows_per_algo = (n_clients + clients_per_row - 1) // clients_per_row

    is_metaworld = env_key == "metaworld"
    task_labels = _metaworld_task_labels(compact=is_metaworld) if is_metaworld else []
    font_algo = _font(24, bold=True)

    client_texts: List[str] = []
    for client in range(n_clients):
        if task_labels and client < len(task_labels):
            client_texts.append(f"C{client}:{task_labels[client]}")
        else:
            client_texts.append(f"Client {client}")

    # MetaWorld → tilted labels at 30° off vertical (= 60° rotation downward).
    # Other envs → 90° vertical labels.
    angle_left, angle_right = (-60, -60) if is_metaworld else (90, 270)

    label_extent = max(32, cell_h - 4)
    if is_metaworld:
        # Auto-fit uniform font for all client labels at the rotation angle, so
        # every label fits in the strip vertical bounds without enlarging row gap.
        client_label_imgs_left = _fit_rotated_label_uniform(client_texts, label_extent, (50, 50, 50), angle=angle_left)
        client_label_imgs_right = _fit_rotated_label_uniform(client_texts, label_extent, (50, 50, 50), angle=angle_right)
    else:
        client_label_imgs_left = [
            _fit_rotated_label(t, label_extent, (50, 50, 50), angle=angle_left) for t in client_texts
        ]
        client_label_imgs_right = [
            _fit_rotated_label(t, label_extent, (50, 50, 50), angle=angle_right) for t in client_texts
        ]

    algo_label_imgs: List[Image.Image] = [
        _render_rotated_label(label, font_algo, color) for _, label, color in ALGOS
    ]

    pad = 18
    cli_label_gap = 6           # gap between rotated client label and its strip
    pair_gap_x = 22             # gap between adjacent (label+strip) pairs in a row
    algo_gap_y = 26             # gap between algorithm blocks
    algo_block_gap_x = 12       # gap between algo label and the first client column

    # Row gap must keep labels from overlapping when they're taller than the strip
    # (happens for steep metaworld rotations with fixed font size).
    max_cli_label_h = max(img.height for img in client_label_imgs_left + client_label_imgs_right)
    label_overflow = max(0, max_cli_label_h - cell_h)
    row_gap_y = max(6, label_overflow + 6)

    algo_label_w = max(img.width for img in algo_label_imgs)
    client_label_w = max(
        max(img.width for img in client_label_imgs_left),
        max(img.width for img in client_label_imgs_right),
    )

    col_w = client_label_w + cli_label_gap + cell_w  # width of one (label+strip) unit
    inner_w = clients_per_row * col_w + (clients_per_row - 1) * pair_gap_x
    width = pad + algo_label_w + algo_block_gap_x + inner_w + pad

    inner_h_per_algo = n_rows_per_algo * cell_h + (n_rows_per_algo - 1) * row_gap_y
    inner_h = len(ALGOS) * inner_h_per_algo + (len(ALGOS) - 1) * algo_gap_y
    height = pad + inner_h + pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    col_base_x = pad + algo_label_w + algo_block_gap_x
    col_step = col_w + pair_gap_x

    for algo_idx, (algo, _, color) in enumerate(ALGOS):
        algo_y0 = pad + algo_idx * (inner_h_per_algo + algo_gap_y)
        algo_y1 = algo_y0 + inner_h_per_algo

        algo_img = algo_label_imgs[algo_idx]
        ax = pad + (algo_label_w - algo_img.width) // 2
        ay = algo_y0 + (inner_h_per_algo - algo_img.height) // 2
        canvas.paste(algo_img, (ax, ay), algo_img)

        bar_x = pad + algo_label_w + 3
        draw.rectangle((bar_x, algo_y0, bar_x + 3, algo_y1), fill=color)

        for client in range(n_clients):
            row_idx = client // clients_per_row
            col_idx = client % clients_per_row
            y_strip = algo_y0 + row_idx * (cell_h + row_gap_y)
            col_origin_x = col_base_x + col_idx * col_step

            if col_idx == 0:
                # left col: [client_label][strip]
                cli_label_img = client_label_imgs_left[client]
                cli_x = col_origin_x + (client_label_w - cli_label_img.width) // 2
                strip_x = col_origin_x + client_label_w + cli_label_gap
            else:
                # right col (2nd column): [strip][client_label rotated 270°]
                cli_label_img = client_label_imgs_right[client]
                strip_x = col_origin_x
                cli_x = strip_x + cell_w + cli_label_gap + (client_label_w - cli_label_img.width) // 2

            cli_y = y_strip + (cell_h - cli_label_img.height) // 2
            canvas.paste(cli_label_img, (cli_x, cli_y), cli_label_img)

            cell = _make_cell(selections[algo][str(client)], mode, env_key)
            paste_x = strip_x + (cell_w - cell.width) // 2
            canvas.paste(cell, (paste_x, y_strip))
            draw.rectangle(
                (paste_x, y_strip, paste_x + cell.width, y_strip + cell.height),
                outline=(215, 215, 215),
                width=1,
            )

    out_path = out_dir / f"{env_key}_fedguide_family.png"
    canvas.save(out_path, optimize=True)
    return out_path


def _selection_json_path(args: argparse.Namespace, out_dir: Path) -> Path:
    if args.selection_json:
        path = Path(args.selection_json)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_SELECTION_JSON


def _make_manifest(args: argparse.Namespace, metrics_root: Path, plots_root: Path) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "round": args.round,
        "seed_selection": (
            "client_specific when client-level eval/return metrics exist; "
            "otherwise env_algo_average using tail-10 metrics_distributed_fit eval/return"
        ),
        "algorithms": [label for _, label, _ in ALGOS],
        "environments": {},
    }
    for env in ENVS:
        key = env["key"]
        manifest["environments"][key] = {
            "display": env["display"],
            "mode": env["mode"],
            "clients": {},
        }
        for client in range(int(env["clients"])):
            manifest["environments"][key]["clients"][str(client)] = {}
            for algo, label, _ in ALGOS:
                selected = _choose_seed(metrics_root, plots_root, key, algo, client, args.round)
                selected["algorithm"] = label
                selected["client"] = client
                manifest["environments"][key]["clients"][str(client)][algo] = selected
    return manifest


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ok] wrote {path.relative_to(PROJECT_ROOT)}")


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _render_selections_from_manifest(
    manifest: Dict[str, Any],
    env: Dict[str, Any],
    plots_root: Path,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    key = env["key"]
    round_id = int(manifest.get("round", ROUND))
    clients = manifest["environments"][key]["clients"]
    selections: Dict[str, Dict[str, Dict[str, Any]]] = {algo: {} for algo, _, _ in ALGOS}
    for client in range(int(env["clients"])):
        client_block = clients[str(client)]
        for algo, _, _ in ALGOS:
            entry = dict(client_block[algo])
            seed = entry.get("seed")
            if seed is not None:
                entry["video"] = _project_rel(_video_path(plots_root, key, algo, int(seed), client, round_id))
            entry.setdefault("episode", 0)
            selections[algo][str(client)] = entry
    return selections


def build(args: argparse.Namespace) -> List[Path]:
    metrics_root = PROJECT_ROOT / args.metrics_root
    plots_root = PROJECT_ROOT / args.plots_root
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _selection_json_path(args, out_dir)

    if args.refresh_selection or not manifest_path.exists():
        manifest = _make_manifest(args, metrics_root, plots_root)
        _write_manifest(manifest_path, manifest)
    else:
        manifest = _load_manifest(manifest_path)
        print(f"[ok] loaded {manifest_path.relative_to(PROJECT_ROOT)}")

    if args.selection_only:
        return []

    outputs: List[Path] = []
    for env in ENVS:
        if args.env and env["key"] not in set(args.env):
            continue
        selections = _render_selections_from_manifest(manifest, env, plots_root)
        out_path = _compose_env(env, selections, out_dir)
        outputs.append(out_path)
        print(f"[ok] wrote {out_path.relative_to(PROJECT_ROOT)}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", default="metrics")
    parser.add_argument("--plots-root", default="plots")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--round", type=int, default=ROUND)
    parser.add_argument("--selection-json", default=str(DEFAULT_SELECTION_JSON.relative_to(PROJECT_ROOT)))
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--refresh-selection", action="store_true")
    parser.add_argument("--env", action="append", choices=[env["key"] for env in ENVS])
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
