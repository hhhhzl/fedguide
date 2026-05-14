"""Space checks that work with both Gym and Gymnasium Box types."""

from __future__ import annotations

from typing import Any


def is_box1d(space: Any) -> bool:
    """True if space is a 1-D Box (either gym.spaces.Box or gymnasium.spaces.Box)."""
    from gym.spaces import Box as BoxGym

    try:
        from gymnasium.spaces import Box as BoxGymnasium

        box_types = (BoxGym, BoxGymnasium)
    except Exception:
        box_types = (BoxGym,)
    return isinstance(space, box_types) and len(getattr(space, "shape", ())) == 1
