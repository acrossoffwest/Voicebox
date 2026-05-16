"""Theme tokens + small color helpers for the PySide6 UI.

Dark theme only in v1. The single accent is amber (#FF9F45)."""

from __future__ import annotations

ACCENT = "#FF9F45"
ACCENT_DARK = "#E08938"  # shade(ACCENT, -10)

# Status colors
COLOR_OK = "#34D399"
COLOR_ERR = "#FF5A4E"
COLOR_WARN = "#F0B95B"
COLOR_INFO = "#7AB8FF"

# Core tokens
TOKENS = {
    "bg": "#141418",
    "surface": "#1C1C21",
    "surface2": "#22222A",
    "surface3": "#26262E",
    "border": "rgba(255, 255, 255, 0.07)",
    "border_strong": "rgba(255, 255, 255, 0.12)",
    "text": "#EEEEF2",
    "text_sub": "rgba(238, 238, 242, 0.62)",
    "text_dim": "rgba(238, 238, 242, 0.38)",
    "hover": "rgba(255, 255, 255, 0.04)",
    "log_bg": "#0A0A0D",
    "log_text": "#C5C5CD",
    "sidebar_bg": "#1C1C20",  # solid approximation of rgba(28,28,32,0.78) over panel bg
    "toolbar_bg": "#16161A",  # solid approximation of toolbar
    "panel_bg_top": "#16161B",
    "panel_bg_bot": "#131316",
}

# Font stacks
FONT_UI = '"Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont, sans-serif'
FONT_MONO = '"JetBrains Mono", "Menlo", "ui-monospace", monospace'


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_alpha(hex_str: str, alpha: float) -> str:
    """Return an `rgba(...)` string usable in QSS."""
    r, g, b = hex_to_rgb(hex_str)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def shade(hex_str: str, pct: int) -> str:
    """Lighten (pct > 0) or darken (pct < 0) a hex color by an amount in 0..100."""
    r, g, b = hex_to_rgb(hex_str)
    target = 0 if pct < 0 else 255
    p = abs(pct) / 100.0
    r = round((target - r) * p + r)
    g = round((target - g) * p + g)
    b = round((target - b) * p + b)
    return f"#{r:02X}{g:02X}{b:02X}"
