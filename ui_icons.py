"""Icon helpers — maps design icon names to FontAwesome 6 (via qtawesome)."""

from __future__ import annotations

import qtawesome as qta
from PyQt6.QtGui import QIcon

from ui_theme import TOKENS

# Map design icon names to Font Awesome 6 'solid' names.
_NAME_MAP: dict[str, str] = {
    "wave": "fa6s.wave-square",
    "mic": "fa6s.microphone",
    "speaker": "fa6s.volume-high",
    "cpu": "fa6s.microchip",
    "sparkles": "fa6s.wand-magic-sparkles",
    "settings": "fa6s.sliders",
    "play": "fa6s.play",
    "stop": "fa6s.stop",
    "check": "fa6s.check",
    "alert": "fa6s.triangle-exclamation",
    "spinner": "fa6s.spinner",
    "chevron": "fa6s.chevron-down",
    "folder": "fa6s.folder-open",
    "upload": "fa6s.upload",
    "download": "fa6s.download",
    "trash": "fa6s.trash",
    "external": "fa6s.arrow-up-right-from-square",
    "copy": "fa6s.copy",
    "close": "fa6s.xmark",
    "minimize": "fa6s.minus",
    "maximize": "fa6s.expand",
}


def icon(name: str, color: str | None = None) -> QIcon:
    """Return a QIcon for a design-named icon."""
    qta_name = _NAME_MAP.get(name)
    if qta_name is None:
        raise KeyError(f"unknown icon name: {name}")
    return qta.icon(qta_name, color=color or TOKENS["text"])
