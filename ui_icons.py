"""Icon helpers — maps design icon names to FontAwesome 6 (via qtawesome)."""

from __future__ import annotations

import qtawesome as qta
from PySide6.QtGui import QIcon

from ui_theme import TOKENS

# Map design icon names to Material Design Icons 6 (qtawesome bundled).
_NAME_MAP: dict[str, str] = {
    "wave": "mdi6.waveform",
    "mic": "mdi6.microphone",
    "speaker": "mdi6.volume-high",
    "cpu": "mdi6.chip",
    "sparkles": "mdi6.auto-fix",
    "settings": "mdi6.tune",
    "play": "mdi6.play",
    "stop": "mdi6.stop",
    "check": "mdi6.check",
    "alert": "mdi6.alert",
    "spinner": "mdi6.loading",
    "chevron": "mdi6.chevron-down",
    "folder": "mdi6.folder-open",
    "upload": "mdi6.upload",
    "download": "mdi6.download",
    "trash": "mdi6.trash-can",
    "external": "mdi6.open-in-new",
    "copy": "mdi6.content-copy",
    "close": "mdi6.close",
    "minimize": "mdi6.minus",
    "maximize": "mdi6.fullscreen",
}


def icon(name: str, color: str | None = None) -> QIcon:
    """Return a QIcon for a design-named icon."""
    qta_name = _NAME_MAP.get(name)
    if qta_name is None:
        raise KeyError(f"unknown icon name: {name}")
    return qta.icon(qta_name, color=color or TOKENS["text"])
