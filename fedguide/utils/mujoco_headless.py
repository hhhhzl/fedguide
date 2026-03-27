"""Headless MuJoCo rendering (Docker / SSH without X11).

MuJoCo defaults to GLFW + X11; without DISPLAY this raises GLFW errors.
Set MUJOCO_GL=egl before the first MuJoCo load for GPU offscreen rendering.
If EGL is unavailable, set MUJOCO_GL=osmesa (CPU, slower) before running.
"""

from __future__ import annotations

import os


def ensure_mujoco_headless_gl_if_needed() -> None:
    """
    MuJoCo requires MUJOCO_GL in {glfw, egl, osmesa}; an empty string is invalid.

    - No DISPLAY: use EGL offscreen (GPU). Override with MUJOCO_GL=osmesa if needed.
    - With DISPLAY: default empty/unset to glfw for windowed rendering.
    """
    gl = (os.environ.get("MUJOCO_GL") or "").strip()
    if gl in ("egl", "osmesa"):
        return
    if gl == "glfw":
        return
    if not gl:
        os.environ["MUJOCO_GL"] = "glfw" if os.environ.get("DISPLAY") else "egl"
