#!/usr/bin/env python
"""Render representative images for each federated environment used in the paper.

For every env (bandit2d, reacher, halfcheetah, walker, hopper, metaworld),
produce two figures under ``assets/envs/<env>/``:

  * ``single.png``         — one canonical render (representative client)
  * ``heterogeneity.png``  — multi-panel grid that visualises cross-client
                             variation (different tasks / dynamics presets /
                             goal regions, etc.)

The script is designed to be robust: failures for a single env are caught and
logged, so running this script on a partially-installed machine still produces
images for the envs that are available.

Headless rendering uses a platform-specific MuJoCo backend for Gymnasium MuJoCo
envs (``glfw`` on macOS, ``egl`` elsewhere).

Usage:
    python scripts/generate_data/generate_envs.py                  # all envs
    python scripts/generate_data/generate_envs.py --envs reacher halfcheetah
    python scripts/generate_data/generate_envs.py --out_dir assets/envs --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
# NOTE: do *not* set SDL_VIDEODRIVER=dummy globally. highway_env.EnvViewer
# disables rendering whenever that variable equals "dummy" (see
# highway_env/envs/common/graphics.py: ``if SDL_VIDEODRIVER == "dummy": enabled=False``),
# producing all-black frames. Headless rendering for highway is achieved via
# the env's own ``offscreen_rendering=True`` config instead.

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT = REPO_ROOT / "assets" / "envs"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    path = FONT_PATH if bold else FONT_PATH_REG
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        return ImageFont.load_default()


def _to_pil(img: np.ndarray) -> Image.Image:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return Image.fromarray(a)


def _square_pad(img: Image.Image, size: int, bg=(255, 255, 255)) -> Image.Image:
    """Letterbox an image into a square canvas."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def _grid(
    tiles: Sequence[Image.Image],
    captions: Sequence[str],
    *,
    cols: int,
    tile_size: int = 320,
    tile_w: Optional[int] = None,
    tile_h: Optional[int] = None,
    cap_h: Optional[int] = None,
    cap_font_size: int = 16,
    title: Optional[str] = None,
    pad: int = 10,
    title_h: int = 56,
    fit: str = "letterbox",  # "letterbox" | "fill"
) -> Image.Image:
    """Compose a labelled grid of images.

    Tiles default to square ``tile_size``×``tile_size`` (letterboxed). Pass
    ``tile_w`` / ``tile_h`` to render rectangular tiles instead; with
    ``fit="fill"`` images are stretched/cropped to the exact tile dimensions
    (use this when the source already matches the desired aspect ratio).
    """
    assert len(tiles) == len(captions)
    n = len(tiles)
    rows = (n + cols - 1) // cols
    tw_px = tile_w if tile_w is not None else tile_size
    th_px = tile_h if tile_h is not None else tile_size
    cap_h_px = cap_h if cap_h is not None else (cap_font_size + 12)

    cell_w = tw_px + 2 * pad
    cell_h = th_px + cap_h_px + 2 * pad
    total_w = cols * cell_w
    total_h = rows * cell_h + (title_h if title else 0)
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if title:
        f = _font(28, bold=True)
        tw = draw.textlength(title, font=f)
        draw.text(((total_w - tw) / 2, (title_h - 28) / 2), title, fill=(20, 20, 20), font=f)

    cap_font = _font(cap_font_size, bold=True)
    for i, (tile, cap) in enumerate(zip(tiles, captions)):
        r, c = divmod(i, cols)
        x0 = c * cell_w + pad
        y0 = r * cell_h + pad + (title_h if title else 0)
        if fit == "fill":
            tile_img = tile.resize((tw_px, th_px), Image.LANCZOS)
        elif tw_px == th_px:
            tile_img = _square_pad(tile, tw_px, bg=(245, 245, 245))
        else:
            # Letterbox into rectangular cell.
            scale = min(tw_px / tile.size[0], th_px / tile.size[1])
            nw, nh = int(round(tile.size[0] * scale)), int(round(tile.size[1] * scale))
            resized = tile.resize((nw, nh), Image.LANCZOS)
            tile_img = Image.new("RGB", (tw_px, th_px), (245, 245, 245))
            tile_img.paste(resized, ((tw_px - nw) // 2, (th_px - nh) // 2))
        canvas.paste(tile_img, (x0, y0))
        if cap:
            tw = draw.textlength(cap, font=cap_font)
            cap_x = x0 + (tw_px - tw) / 2
            cap_y = y0 + th_px + 6
            draw.text((cap_x, cap_y), cap, fill=(20, 20, 20), font=cap_font)
    return canvas


def _save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"  saved {_display_path(path)} ({img.size[0]}x{img.size[1]})")


def _display_path(path: Path) -> Path:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


# ---------------------------------------------------------------------------
# Mujoco render helpers
# ---------------------------------------------------------------------------

def _step_warmup(env, n: int, gymnasium_api: bool = True):
    """Take a few low-amplitude steps to escape the static-pose start frame."""
    for _ in range(n):
        try:
            a = env.action_space.sample() * 0.0  # zero action — env shows a natural pose
        except Exception:
            return
        try:
            out = env.step(a)
        except Exception:
            return
        if gymnasium_api:
            if len(out) == 5:
                _, _, term, trunc, _ = out
                if term or trunc:
                    try:
                        env.reset()
                    except Exception:
                        return
            else:
                _, _, done, _ = out
                if done:
                    try:
                        env.reset()
                    except Exception:
                        return


def _render_mujoco_renderer(model, data, height=480, width=480, camera=None) -> np.ndarray:
    """Render via the mujoco.Renderer API (works for d4rl legacy-gym envs)."""
    import mujoco  # type: ignore
    r = mujoco.Renderer(model, height=height, width=width)
    try:
        if camera is not None:
            r.update_scene(data, camera=camera)
        else:
            r.update_scene(data)
        img = r.render()
    finally:
        try:
            r.close()
        except Exception:
            pass
    return img


def _mujoco_env_core(env):
    """Return the innermost object carrying MuJoCo model/data."""
    cur = env
    seen = set()
    for _ in range(16):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        unwrapped = getattr(cur, "unwrapped", cur)
        if hasattr(unwrapped, "model") and hasattr(unwrapped, "data"):
            return unwrapped
        if hasattr(cur, "model") and hasattr(cur, "data"):
            return cur
        cur = getattr(cur, "env", None)
        if cur is None:
            break
    raise RuntimeError("could not find MuJoCo model/data inside environment wrapper")


def _mujoco_camera(model, preferred: str = "track"):
    names = []
    for i in range(getattr(model, "ncam", 0)):
        try:
            names.append(model.camera(i).name)
        except Exception:
            names.append("")
    return preferred if preferred in names else None


def _render_mujoco_env(env, height=480, width=480) -> np.ndarray:
    inner = _mujoco_env_core(env)
    return _render_mujoco_renderer(
        inner.model,
        inner.data,
        height=height,
        width=width,
        camera=_mujoco_camera(inner.model),
    )


# ---------------------------------------------------------------------------
# Per-env renderers
# ---------------------------------------------------------------------------

def render_bandit2d(meta_path: Path, out_dir: Path, seed: int) -> None:
    """2-D bandit: scatter peaks + per-client weight heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(meta_path) as f:
        meta = json.load(f)
    mu = np.array(meta["mu"])           # (K, 2)
    sigma = float(meta["sigma"])
    n_clients = int(meta["n_clients"])
    K = int(meta.get("K", mu.shape[0]))

    # --- single: global reward landscape with K peaks
    xs = np.linspace(-1.5, 1.5, 200)
    ys = np.linspace(-1.5, 1.5, 200)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX, YY], axis=-1)
    R = np.zeros_like(XX)
    for k in range(K):
        d = np.linalg.norm(pts - mu[k], axis=-1)
        R = np.maximum(R, np.exp(-d ** 2 / (2 * sigma ** 2)))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    im = ax.contourf(XX, YY, R, levels=30, cmap="viridis")
    ax.scatter(mu[:, 0], mu[:, 1], c="red", s=80, edgecolors="white", linewidths=1.5, zorder=5)
    for k in range(K):
        ax.annotate(f"$\\mu_{k}$", mu[k] * 1.18, color="red", fontsize=12,
                    ha="center", va="center", fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="reward")
    fig.tight_layout()
    fig.savefig(out_dir / "single.png", dpi=160)
    plt.close(fig)
    print(f"  saved {_display_path(out_dir / 'single.png')}")

    # --- heterogeneity: per-client weighted reward landscape (K panels)
    fig, axes = plt.subplots(1, K, figsize=(3.4 * K, 3.6), dpi=160)
    if K == 1:
        axes = [axes]
    for i in range(K):
        w = np.ones(K) * 0.1
        w[i] = 1.0
        Ri = np.zeros_like(XX)
        for k in range(K):
            d = np.linalg.norm(pts - mu[k], axis=-1)
            Ri = np.maximum(Ri, w[k] * np.exp(-d ** 2 / (2 * sigma ** 2)))
        ax = axes[i]
        ax.contourf(XX, YY, Ri, levels=30, cmap="viridis", vmin=0, vmax=1)
        ax.scatter(mu[:, 0], mu[:, 1], c="white", s=30, edgecolors="black", linewidths=1.0)
        ax.scatter([mu[i, 0]], [mu[i, 1]], c="red", s=110, edgecolors="white", linewidths=1.5, zorder=5)
        ax.set_title(f"client {i} — peak $\\mu_{i}$", fontsize=11)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Bandit2D heterogeneity — {n_clients} clients × {K} peaks (one preferred per client)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "heterogeneity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {_display_path(out_dir / 'heterogeneity.png')}")


def _gymnasium_locomotion_render(env_name: str, **mkw) -> np.ndarray:
    import gymnasium as gym
    env = gym.make(env_name, **mkw)
    try:
        env.reset(seed=0)
        _step_warmup(env, 12)
        if env_name.lower().startswith("walker"):
            _set_walker_render_pose(env)
        return _render_mujoco_env(env)
    finally:
        env.close()


def _set_walker_render_pose(env) -> None:
    """Use an open biped stance for Walker2D renders.

    The default reset pose has both legs nearly overlapping in the side-view
    camera, which makes Walker2D look like Hopper in small paper figures. This
    only changes the rendered pose, not the benchmark metadata or
    training environments.
    """
    try:
        inner = _mujoco_env_core(env)
        qpos = np.array(inner.data.qpos, dtype=np.float64).copy()
        qvel = np.zeros_like(np.array(inner.data.qvel, dtype=np.float64))
        if qpos.shape[0] < 9:
            return
        qpos[1] = 1.22   # torso height
        qpos[2] = 0.00   # torso pitch
        qpos[3:6] = np.array([0.45, -0.85, 0.35])   # front leg
        qpos[6:9] = np.array([-0.50, -0.75, 0.25])  # rear leg
        inner.set_state(qpos, qvel)
    except Exception:
        return


def render_locomotion(env_short: str, meta_path: Path, out_dir: Path, seed: int) -> None:
    """HalfCheetah / Walker / Hopper / Ant — show the 4 dynamics presets + a
    parameter scatter that makes the cross-client variation legible (a static
    reset-pose render alone barely shows mass / damping / friction shifts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(meta_path) as f:
        meta = json.load(f)
    clients: List[Dict[str, Any]] = meta["clients"]
    env_name = str(meta.get("env_name", clients[0].get("env_name")))
    is_ant = env_name.lower().startswith("ant")

    # --- single representative (nominal client)
    nominal_idx = next((i for i, c in enumerate(clients) if c.get("dynamics_preset") == "nominal"), 0)
    cfg0 = clients[nominal_idx]
    mkw = dict(ctrl_cost_weight=float(cfg0["ctrl_cost_weight"]),
               reset_noise_scale=float(cfg0["reset_noise_scale"]))
    if not is_ant:
        mkw["forward_reward_weight"] = float(cfg0["forward_reward_weight"])
    if is_ant:
        mkw["use_contact_forces"] = True
        mkw["healthy_reward"] = 0.0
    img = _gymnasium_locomotion_render(env_name, **mkw)
    _save(_to_pil(img), out_dir / "single.png")

    # --- heterogeneity: pick one client per dynamics preset and render it.
    seen: Dict[str, int] = {}
    for i, c in enumerate(clients):
        ds = c.get("dynamics_preset") or "default"
        if ds not in seen:
            seen[ds] = i
    preset_order = ["nominal", "high_damping_friction", "weak_actuator", "heavy_body"]
    order = [p for p in preset_order if p in seen] + [p for p in seen if p not in preset_order]

    rendered: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
    for preset in order:
        c = clients[seen[preset]]
        try:
            from fedguide.envs.mujoco_locomotion_hetero import (
                make_hetero_locomotion_env_from_metadata,
            )
            from fedguide.envs.halfcheetah_hetero import (
                make_hetero_halfcheetah_env_from_metadata,
            )
            if env_short == "halfcheetah":
                env = make_hetero_halfcheetah_env_from_metadata(
                    str(meta_path), seen[preset], seed=seed,
                    render_mode="rgb_array", render_eval=True,
                )
            else:
                env = make_hetero_locomotion_env_from_metadata(
                    str(meta_path), seen[preset], seed=seed,
                    render_mode="rgb_array", render_eval=True,
                )
            try:
                env.reset()
            except Exception:
                pass
            _step_warmup(env, 12)
            if env_short == "walker":
                _set_walker_render_pose(env)
            frame = _render_mujoco_env(env)
            env.close()
        except Exception as e:
            print(f"  [{env_short}] preset={preset} fallback (standard Gymnasium): {e}")
            mkw_c = dict(ctrl_cost_weight=float(c["ctrl_cost_weight"]),
                         reset_noise_scale=float(c["reset_noise_scale"]))
            if not is_ant:
                mkw_c["forward_reward_weight"] = float(c["forward_reward_weight"])
            if is_ant:
                mkw_c["use_contact_forces"] = True
                mkw_c["healthy_reward"] = 0.0
            frame = _gymnasium_locomotion_render(env_name, **mkw_c)
        rendered.append((preset, frame, c))

    # Compose: top row = renders, bottom row = three parameter scatter plots.
    n_panels = len(rendered)
    fig = plt.figure(figsize=(3.4 * max(n_panels, 4), 7.6), dpi=140)
    gs = fig.add_gridspec(2, max(n_panels, 4), hspace=0.18, wspace=0.12,
                          height_ratios=[1.4, 1.0])

    for i, (preset, frame, c) in enumerate(rendered):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.asarray(frame))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"{preset}\nm×{c['mass_scale']:.2f}  d×{c['damping_scale']:.2f}  "
            f"μ×{c.get('ground_friction', 1.0):.2f}\n"
            f"gain×{c.get('action_gain', 1.0):.2f}  "
            f"pref={c.get('preference_preset', '—')}",
            fontsize=10,
        )

    preset_color = {
        "nominal": "#1f77b4",
        "high_damping_friction": "#ff7f0e",
        "weak_actuator": "#2ca02c",
        "heavy_body": "#d62728",
    }
    colors = [preset_color.get(c.get("dynamics_preset"), "gray") for c in clients]

    def _ax_xy(slot, x_key, y_key, x_label, y_label):
        ax = fig.add_subplot(gs[1, slot])
        xs = [float(c[x_key]) for c in clients]
        ys = [float(c[y_key]) for c in clients]
        ax.scatter(xs, ys, c=colors, s=25, alpha=0.85, edgecolors="white", linewidths=0.5)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

    # Use only as many bottom-row slots as we have presets (≤4).
    cols = max(n_panels, 4)
    plots = [("mass_scale", "damping_scale", "mass_scale", "damping_scale"),
             ("ground_friction", "action_gain", "ground_friction", "action_gain"),
             ("forward_reward_weight", "ctrl_cost_weight",
              "forward_reward_weight", "ctrl_cost_weight"),
             ("reset_noise_scale", "unstable_cost_weight",
              "reset_noise_scale", "unstable_cost_weight")]
    for i, p in enumerate(plots[:cols]):
        try:
            _ax_xy(i, *p)
        except KeyError:
            ax = fig.add_subplot(gs[1, i])
            ax.axis("off")

    # Legend across the bottom.
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=preset_color[p], markersize=8, label=p)
               for p in preset_order if p in seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

    n_pref = len(set(c.get("preference_preset") for c in clients))
    n_dyn = len(set(c.get("dynamics_preset") for c in clients))
    fig.suptitle(
        f"{env_short.capitalize()} — {len(clients)} clients × {n_dyn} dynamics presets "
        f"× {n_pref} reward preferences",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_dir / "heterogeneity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {_display_path(out_dir / 'heterogeneity.png')}")


def render_metaworld(meta_path: Path, out_dir: Path, seed: int) -> None:
    """MetaWorld ML10 — render one frame per task."""
    import metaworld  # type: ignore

    with open(meta_path) as f:
        meta = json.load(f)
    clients = meta["clients"]
    task_names = [c["task"] for c in clients]
    ml10 = metaworld.ML10(seed=int(meta.get("seed", seed)))

    def _render_task(task_name: str) -> Optional[np.ndarray]:
        try:
            env_cls = ml10.train_classes.get(task_name) or ml10.test_classes.get(task_name)
            if env_cls is None:
                return None
            env = env_cls(render_mode="rgb_array")
            tasks = [t for t in (ml10.train_tasks + ml10.test_tasks) if t.env_name == task_name]
            if not tasks:
                env.close()
                return None
            env.set_task(tasks[0])
            env.reset()
            _step_warmup(env, 5)
            img = env.render()
            env.close()
            return img
        except Exception as e:
            print(f"  [metaworld] {task_name} fail: {e}")
            return None

    # --- single
    img = _render_task(task_names[0])
    if img is not None:
        _save(_to_pil(img), out_dir / "single.png")

    # --- heterogeneity grid (2 rows × 5 cols for 10 tasks)
    tiles, caps = [], []
    for tn in task_names:
        img = _render_task(tn)
        if img is None:
            tiles.append(Image.new("RGB", (480, 480), (230, 230, 230)))
        else:
            tiles.append(_to_pil(img))
        caps.append(tn.replace("-v3", ""))
    cols = 5 if len(tiles) >= 5 else len(tiles)
    # Capitalised "Reach", "Push", … for the paper figure.
    caps = [c.replace("-", " ").title() for c in caps]
    grid = _grid(
        tiles, caps,
        cols=cols,
        tile_size=360,
        cap_font_size=28,
        pad=14,
    )
    _save(grid, out_dir / "heterogeneity.png")


def render_reacher(meta_path: Path, out_dir: Path, seed: int) -> None:
    """Reacher — render env + overlay heterogeneous goal regions / variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with open(meta_path) as f:
        meta = json.load(f)
    clients = meta["clients"]

    # --- single: a clean mujoco render of Reacher-v4
    import gymnasium as gym
    try:
        env = gym.make("Reacher-v4", render_mode="rgb_array")
        env.reset(seed=seed)
        for _ in range(5):
            env.step(np.zeros(2, dtype=np.float32))
        img = env.render()
        env.close()
        _save(_to_pil(img), out_dir / "single.png")
    except Exception as e:
        print(f"  [reacher] single render fail: {e}")

    # --- heterogeneity: split into two panels
    #   (a) the goal-region 8×8 grid coloured by dataset variant
    #   (b) overlaid per-client action-noise magnitude
    variants = meta.get("variants") or sorted({c["variant"] for c in clients})
    var_color = {v: c for v, c in zip(variants,
                                      ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), dpi=160)
    ax = axes[0]
    for c in clients:
        (lx, hx), (ly, hy) = c["qpos_high_low"]
        ax.add_patch(Rectangle((lx, ly), hx - lx, hy - ly,
                               facecolor=var_color.get(c["variant"], "gray"),
                               edgecolor="white", alpha=0.85, linewidth=1.0))
    ax.add_patch(plt.Circle((0, 0), 0.25, fill=False, edgecolor="black",
                            linewidth=1.2, linestyle="--", label="reach radius"))
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect("equal")
    ax.set_title(f"Goal-region grid ({len(clients)} clients) — color = D4RL variant")
    handles = [plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=var_color[v], markersize=12, label=v)
               for v in variants]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    ax = axes[1]
    rs = np.array([c["reward_scale"] for c in clients])
    an = np.array([np.linalg.norm(c["action_noise"]) for c in clients])
    sc = ax.scatter(rs, an, c=[var_color[c["variant"]] for c in clients], s=40, alpha=0.85,
                    edgecolors="white", linewidths=0.6)
    ax.set_xlabel("reward_scale")
    ax.set_ylabel("||action_noise||")
    ax.set_title("Per-client reward × action noise heterogeneity")
    fig.suptitle(
        f"Reacher heterogeneity — {len(clients)} clients across "
        f"{len(variants)} dataset variants × init-state grid × noise/reward shift",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "heterogeneity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {_display_path(out_dir / 'heterogeneity.png')}")


# ---------------------------------------------------------------------------
# Appendix metadata figures
# ---------------------------------------------------------------------------

LOCOMOTION_LABELS = {
    "hopper": "Hopper",
    "walker": "Walker2D",
    "halfcheetah": "HalfCheetah",
}
LOCOMOTION_SHORT_LABELS = {
    "hopper": "Ho",
    "walker": "W",
    "halfcheetah": "HC",
}

METADATA_FEATURES: Sequence[Tuple[str, str, float]] = (
    ("mass_scale", "Mass", 1.0),
    ("damping_scale", "Damping", 1.0),
    ("ground_friction", "Friction", 1.0),
    ("action_gain", "Actuator", 1.0),
    ("forward_reward_weight", "Forward", 1.0),
    ("ctrl_cost_weight", "Log Ctrl", 1e-3),
    ("unstable_cost_weight", "Unstable", 0.0),
    ("reset_noise_scale", "Reset", 0.0),
)


def _representative_indices(n: int, max_items: int = 8) -> List[int]:
    if n <= max_items:
        return list(range(n))
    return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, max_items)))


def _load_locomotion_metadata(
    data_root: Path,
    *,
    metadata_name: str = "metadata.json",
    representative: bool = False,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for env_short in ("hopper", "walker", "halfcheetah"):
        meta_path = data_root / env_short / metadata_name
        if not meta_path.is_file():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        clients = list(meta.get("clients", []))
        indices = _representative_indices(len(clients)) if representative else list(range(len(clients)))
        for idx in indices:
            c = dict(clients[idx])
            c["_env_short"] = env_short
            c["_env_label"] = LOCOMOTION_LABELS[env_short]
            c["_metadata"] = metadata_name
            records.append(c)
    return records


def _feature_value(record: Dict[str, Any], key: str, default: float) -> float:
    value = float(record.get(key, default))
    if key == "ctrl_cost_weight":
        value = float(np.log10(max(value, 1e-8)))
    return value


def _metadata_matrix(records: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    labels = [label for _, label, _ in METADATA_FEATURES]
    rows = []
    for record in records:
        rows.append([_feature_value(record, key, default)
                     for key, _, default in METADATA_FEATURES])
    return np.asarray(rows, dtype=np.float64), labels


def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    lo = np.min(x, axis=0, keepdims=True)
    hi = np.max(x, axis=0, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-9)


METAWORLD_SEMANTIC_COLUMNS = [
    "EE-Only",
    "Free Object",
    "Lift/Place",
    "Push/Sweep",
    "Articulated",
    "Press",
    "Insert",
    "Precision",
    "Contact-Rich",
    "Multi-Stage",
]

METAWORLD_TASK_SEMANTICS: Dict[str, Sequence[float]] = {
    "reach-v3":                 [1, 0, 0, 0, 0, 0, 0, 0.5, 0, 0],
    "push-v3":                  [0, 1, 0, 1, 0, 0, 0, 0.5, 1, 0],
    "pick-place-v3":            [0, 1, 1, 0, 0, 0, 0, 0.7, 1, 1],
    "door-open-v3":             [0, 0, 0, 0, 1, 0, 0, 0.6, 1, 0.5],
    "drawer-close-v3":          [0, 0, 0, 0.5, 1, 0, 0, 0.5, 1, 0],
    "button-press-topdown-v3":  [0, 0, 0, 0, 0, 1, 0, 1.0, 1, 0],
    "peg-insert-side-v3":       [0, 1, 0.5, 0, 0, 0, 1, 1.0, 1, 1],
    "window-open-v3":           [0, 0, 0, 0.5, 1, 0, 0, 0.5, 1, 0],
    "sweep-v3":                 [0, 1, 0, 1, 0, 0, 0, 0.4, 1, 0.5],
    "basketball-v3":            [0, 1, 1, 0, 0, 0, 0, 0.8, 1, 1],
}


def _apply_appendix_plot_style(plt) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "font.weight": "normal",
        "axes.titlesize": 22,
        "axes.titleweight": "normal",
        "axes.labelsize": 18,
        "axes.labelweight": "normal",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 18,
        "axes.linewidth": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.9,
        "savefig.bbox": "tight",
    })


def _load_clients_from(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return list(json.load(f).get("clients", []))


def _select_records(records: Sequence[Dict[str, Any]], max_items: int = 8) -> List[Dict[str, Any]]:
    return [records[i] for i in _representative_indices(len(records), max_items=max_items)]


def _reacher_value(record: Dict[str, Any], key: str, default: float = 0.0) -> float:
    if key == "goal_x":
        return float(np.mean(record["qpos_high_low"][0]))
    if key == "goal_y":
        return float(np.mean(record["qpos_high_low"][1]))
    if key == "action_noise_norm":
        return float(np.linalg.norm(record.get("action_noise", [0.0, 0.0])))
    return float(record.get(key, default))


def _matrix_from_specs(
    records: Sequence[Dict[str, Any]],
    specs: Sequence[Tuple[str, str, float]],
    *,
    getter: Callable[[Dict[str, Any], str, float], float],
) -> Tuple[np.ndarray, List[str]]:
    rows = [
        [getter(record, key, default) for key, _, default in specs]
        for record in records
    ]
    return np.asarray(rows, dtype=np.float64), [label for _, label, _ in specs]


def _draw_heatmap_panel(ax, matrix, row_labels, col_labels, title, *, cmap="YlGnBu"):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, pad=8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0, width=1.2)
    for y in range(matrix.shape[0] + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.8)
    for x in range(matrix.shape[1] + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    return im


def _metaworld_semantic_matrix(records: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[str]]:
    rows = []
    labels = []
    for c in records:
        task = str(c["task"])
        rows.append(list(METAWORLD_TASK_SEMANTICS.get(
            task, [0.0] * len(METAWORLD_SEMANTIC_COLUMNS)
        )))
        clean = task.replace("-v3", "").replace("-", " ")
        labels.append(f"C{int(c.get('client_id', len(labels)))} {clean.title()}")
    return np.asarray(rows, dtype=np.float64), labels, METAWORLD_SEMANTIC_COLUMNS


def render_heterogeneity_heatmaps(data_root: Path, out_root: Path) -> None:
    """Render five aligned heatmaps: Hopper, Walker2D, HalfCheetah, Reacher, MetaWorld10."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_appendix_plot_style(plt)
    panels = []

    for env_short, title, metadata_name in (
        ("hopper", "Hopper", "metadata.json"),
        ("walker", "Walker2D", "metadata.json"),
        ("halfcheetah", "HalfCheetah", "metadata_mild64.json"),
    ):
        all_records = _load_clients_from(data_root / env_short / metadata_name)
        records = all_records[:8] if env_short == "halfcheetah" else _select_records(all_records, 8)
        raw, labels = _metadata_matrix(records)
        matrix = _minmax_normalize(raw)
        row_prefix = LOCOMOTION_SHORT_LABELS[env_short]
        row_labels = [f"{row_prefix}{int(r.get('client_id', i))}" for i, r in enumerate(records)]
        panels.append((title + " (Main)", matrix, row_labels, labels))

    reacher_specs = (
        ("goal_x", "Goal X", 0.0),
        ("goal_y", "Goal Y", 0.0),
        ("action_noise_norm", "|Noise|", 0.0),
        ("reward_scale", "Reward", 1.0),
        ("angle_noise", "Angle", 0.0),
    )
    reacher_records = _load_clients_from(data_root / "reacher" / "metadata.json")
    raw, labels = _matrix_from_specs(reacher_records, reacher_specs, getter=_reacher_value)
    row_labels = [f"R{int(r.get('client_id', i))}" for i, r in enumerate(reacher_records)]
    panels.append(("Reacher (Main)", _minmax_normalize(raw), row_labels, labels))

    metaworld_records = _load_clients_from(data_root / "metaworld" / "metadata.json")
    matrix, row_labels, labels = _metaworld_semantic_matrix(metaworld_records)
    panels.append(("MetaWorld10", matrix, row_labels, labels))

    fig = plt.figure(figsize=(20.0, 9.2), dpi=180)
    gs = fig.add_gridspec(2, 12, hspace=0.62, wspace=1.15, height_ratios=[1.0, 1.08])
    axes = [
        fig.add_subplot(gs[0, 0:4]),
        fig.add_subplot(gs[0, 4:8]),
        fig.add_subplot(gs[0, 8:12]),
        fig.add_subplot(gs[1, 0:5]),
        fig.add_subplot(gs[1, 7:12]),
    ]

    ims = []
    for ax, (title, matrix, row_labels, labels) in zip(axes, panels):
        ims.append(_draw_heatmap_panel(ax, matrix, row_labels, labels, title))

    cbar = fig.colorbar(ims[-1], ax=axes, fraction=0.018, pad=0.015)
    cbar.set_label("Normalized Value / Semantic Intensity")
    cbar.ax.tick_params(labelsize=13, width=1.2, length=4)
    out_path = out_root / "heterogeneity_heatmaps.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {_display_path(out_path)}")


def _range_feature_value(record: Dict[str, Any], key: str, default: float) -> float:
    if key == "goal_x":
        return float(np.mean(record["qpos_high_low"][0]))
    if key == "goal_y":
        return float(np.mean(record["qpos_high_low"][1]))
    if key == "action_noise_norm":
        return float(np.linalg.norm(record.get("action_noise", [0.0, 0.0])))
    if key == "ctrl_cost_weight":
        return float(np.log10(max(float(record.get(key, default)), 1e-8)))
    return float(record.get(key, default))


def _range_widths(main_records, hard_records, specs) -> Tuple[np.ndarray, np.ndarray]:
    main_widths = []
    hard_widths = []
    for key, _label, default in specs:
        main_vals = np.asarray([_range_feature_value(r, key, default) for r in main_records], dtype=np.float64)
        hard_vals = np.asarray([_range_feature_value(r, key, default) for r in hard_records], dtype=np.float64)
        pooled_lo = min(float(main_vals.min()), float(hard_vals.min()))
        pooled_hi = max(float(main_vals.max()), float(hard_vals.max()))
        denom = max(pooled_hi - pooled_lo, 1e-9)
        main_widths.append((float(main_vals.max()) - float(main_vals.min())) / denom)
        hard_widths.append((float(hard_vals.max()) - float(hard_vals.min())) / denom)
    return np.asarray(main_widths), np.asarray(hard_widths)


def _draw_range_panel(ax, env_title, main_records, hard_records, specs):
    main_color = "#1f77b4"
    hard_color = "#ff7f0e"
    rng = np.random.default_rng(2026)
    main_data = []
    hard_data = []
    feature_labels = []
    for key, label, default in specs:
        main_vals = np.asarray([_range_feature_value(r, key, default) for r in main_records], dtype=np.float64)
        hard_vals = np.asarray([_range_feature_value(r, key, default) for r in hard_records], dtype=np.float64)
        pooled_lo = min(float(main_vals.min()), float(hard_vals.min()))
        pooled_hi = max(float(main_vals.max()), float(hard_vals.max()))
        denom = max(pooled_hi - pooled_lo, 1e-9)
        main_data.append((main_vals - pooled_lo) / denom)
        hard_data.append((hard_vals - pooled_lo) / denom)
        feature_labels.append(label)

    base = np.arange(len(specs), dtype=float)
    offset = 0.19
    main_pos = base - offset
    hard_pos = base + offset

    def _violin_ready(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if np.ptp(values) < 1e-8:
            values = values + rng.normal(0.0, 0.004, size=values.shape)
        return np.clip(values, 0.0, 1.0)

    main_plot = [_violin_ready(v) for v in main_data]
    hard_plot = [_violin_ready(v) for v in hard_data]
    parts = ax.violinplot(
        main_plot + hard_plot,
        positions=np.concatenate([main_pos, hard_pos]),
        widths=0.32,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    colors = [main_color] * len(specs) + [hard_color] * len(specs)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("0.18")
        body.set_linewidth(1.8)
        body.set_alpha(0.92)

    box = ax.boxplot(
        main_data + hard_data,
        positions=np.concatenate([main_pos, hard_pos]),
        widths=0.06,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        zorder=4,
    )
    for patch in box["boxes"]:
        patch.set_facecolor("0.25")
        patch.set_edgecolor("0.18")
        patch.set_linewidth(1.2)
        patch.set_alpha(0.9)
    for item in box["whiskers"] + box["caps"]:
        item.set_color("0.18")
        item.set_linewidth(1.2)
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(2.0)

    for positions, data in ((main_pos, main_data), (hard_pos, hard_data)):
        for pos, values in zip(positions, data):
            jitter = rng.uniform(-0.035, 0.035, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=12,
                facecolor="white",
                edgecolor="0.2",
                linewidth=0.5,
                alpha=0.78,
                zorder=5,
            )

    ax.set_title(env_title, pad=8)
    ax.set_xlim(-0.62, len(specs) - 0.38)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks(base)
    ax.set_xticklabels(feature_labels, rotation=32, ha="right")
    ax.set_ylabel("Normalized Value")
    ax.grid(axis="y", alpha=0.28, linewidth=0.9)
    ax.tick_params(width=1.3, length=5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.4)


def render_range_expansion(data_root: Path, out_root: Path) -> None:
    """Render Main-vs-Hard metadata range expansion for Reacher and MuJoCo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _apply_appendix_plot_style(plt)
    locomotion_specs = tuple(METADATA_FEATURES)
    reacher_specs = (
        ("goal_x", "Goal X", 0.0),
        ("goal_y", "Goal Y", 0.0),
        ("action_noise_norm", "|Noise|", 0.0),
        ("reward_scale", "Reward", 1.0),
        ("angle_noise", "Angle", 0.0),
        ("mass_scale", "Mass", 1.0),
        ("damping_scale", "Damping", 1.0),
        ("gear_scale", "Gear", 1.0),
    )
    configs = [
        ("Reacher", data_root / "reacher" / "metadata.json",
         data_root / "reacher" / "metadata_C.json", reacher_specs),
        ("Hopper", data_root / "hopper" / "metadata.json",
         data_root / "hopper" / "metadata_hard.json", locomotion_specs),
        ("Walker2D", data_root / "walker" / "metadata.json",
         data_root / "walker" / "metadata_hard.json", locomotion_specs),
        ("HalfCheetah", data_root / "halfcheetah" / "metadata_mild64.json",
         data_root / "halfcheetah" / "metadata.json", locomotion_specs),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.2), dpi=180, sharey=True)
    for ax, (title, main_path, hard_path, specs) in zip(axes.flat, configs):
        _draw_range_panel(
            ax, title,
            _load_clients_from(main_path),
            _load_clients_from(hard_path),
            specs,
        )
    fig.legend(
        handles=[
            Patch(facecolor="#1f77b4", edgecolor="0.18", label="Main"),
            Patch(facecolor="#ff7f0e", edgecolor="0.18", label="Hard"),
        ],
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="0.3",
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path = out_root / "range_expansion.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {_display_path(out_path)}")


def render_appendix_metadata_figures(data_root: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    render_heterogeneity_heatmaps(data_root, out_root)
    render_range_expansion(data_root, out_root)



# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

RENDERERS: Dict[str, Callable[[Path, Path, int], None]] = {
    "bandit2d":    render_bandit2d,
    "reacher":     render_reacher,
    "halfcheetah": lambda mp, od, s: render_locomotion("halfcheetah", mp, od, s),
    "walker":      lambda mp, od, s: render_locomotion("walker", mp, od, s),
    "hopper":      lambda mp, od, s: render_locomotion("hopper", mp, od, s),
    "metaworld":   render_metaworld,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=list(RENDERERS.keys()),
                    choices=list(RENDERERS.keys()),
                    help="which envs to render (default: all)")
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--appendix_figures", action="store_true",
                    help="also render compact metadata and hard-range appendix figures")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    data_root = Path(args.data_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, str] = {}
    for env in args.envs:
        meta_path = data_root / env / "metadata.json"
        if not meta_path.is_file():
            print(f"\n=== {env} === SKIP (no metadata at {meta_path})")
            results[env] = "skipped"
            continue
        print(f"\n=== {env} ===")
        out_dir = out_root / env
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            RENDERERS[env](meta_path, out_dir, args.seed)
            results[env] = "ok"
        except Exception as e:
            print(f"  ERROR rendering {env}: {e}")
            traceback.print_exc()
            results[env] = f"error: {e}"

    print("\n=== summary ===")
    for env, status in results.items():
        print(f"  {env:12s} → {status}")

    if args.appendix_figures:
        print("\n=== appendix metadata figures ===")
        render_appendix_metadata_figures(data_root, out_root)


if __name__ == "__main__":
    main()
