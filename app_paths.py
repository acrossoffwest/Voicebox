"""Single source of truth for where the app stores its data and finds its
bundled resources. Works both when running from source (`./venv/bin/python ui.py`)
and from a PyInstaller-packaged `.app` bundle."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running from a PyInstaller-built bundle."""
    return getattr(sys, "frozen", False)


def resource_dir() -> Path:
    """Read-only directory shipped inside the bundle (or repo root when running
    from source). Holds `rmvpe.pt` and other small read-only assets."""
    if is_frozen():
        # PyInstaller exposes the bundle root via sys._MEIPASS.
        base = Path(getattr(sys, "_MEIPASS", "."))
        return base
    return Path(__file__).resolve().parent


def user_data_dir() -> Path:
    """Writable per-user directory. macOS-only path; we don't target Linux/Win."""
    if is_frozen():
        root = Path.home() / "Library" / "Application Support" / "Voicebox"
    else:
        root = Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    return root


def base_models_dir() -> Path:
    """`models/base/` location. Resolved to the writable user dir at runtime —
    HuBERT / ContentVec / rmvpe live here. The bundle seeds rmvpe.pt on first
    launch by copying it out of the read-only resource dir."""
    d = user_data_dir() / "models" / "base"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rvc_models_dir() -> Path:
    """`models/rvc/` location — user's voice models."""
    d = user_data_dir() / "models" / "rvc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def settings_path() -> Path:
    """Where the UI saves its settings.json."""
    if is_frozen():
        d = user_data_dir() / "config"
    else:
        d = Path.home() / ".config" / "microphone"
    d.mkdir(parents=True, exist_ok=True)
    return d / "ui.json"


def seed_bundled_assets() -> None:
    """Copy read-only bundled assets (rmvpe.pt) into the user's writable
    base-models dir if they aren't already there. Idempotent — call on every
    app launch."""
    if not is_frozen():
        return
    src = resource_dir() / "models" / "base" / "rmvpe.pt"
    dst = base_models_dir() / "rmvpe.pt"
    if src.is_file() and not dst.is_file():
        import shutil
        shutil.copy2(src, dst)
