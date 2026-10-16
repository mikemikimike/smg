"""The readers the engines' configuration views share.

Each engine package reads its engine's configuration objects through typed
sections (``<engine>/config_view.py``); the sections differ, the way an
attribute is read does not. The readers here are lenient: an absent object
or attribute reads as the default named, and a count is a count only when it
is a positive integer (a bool is not one). Engine-free: nothing from any
engine is imported.
"""

from __future__ import annotations

from typing import Any


def positive_int(value: Any) -> int | None:
    """``value`` as a count above zero, else ``None`` (a bool is not a count)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def hf_str(hf_config: Any, name: str) -> str:
    """A text attribute of the HF config; ``""`` when the config or the attribute is absent."""
    return str((getattr(hf_config, name, "") or "") if hf_config else "")


def hf_list(hf_config: Any, name: str) -> list:
    """A list attribute of the HF config, copied; ``[]`` when the config or the attribute is absent."""
    return list((getattr(hf_config, name, []) or []) if hf_config else [])


def hf_int(hf_config: Any, name: str) -> int:
    """An integer attribute of the HF config; ``0`` when the config or the attribute is absent."""
    return int((getattr(hf_config, name, 0) or 0) if hf_config else 0)
