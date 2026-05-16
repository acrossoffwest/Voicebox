"""System requirement checks for the Voicebox UI Setup screen.

Each function returns a `Check` carrying a status (`ok` / `todo` / `error`)
and a human-readable detail string. Pure-ish: no Qt, no audio, only stdlib
and small standard tools (`brew`, `pkgutil`, filesystem)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Status = Literal["ok", "todo", "error", "pending"]


@dataclass
class Check:
    key: str
    label: str
    status: Status
    detail: str
    action: str | None = None  # opaque hint the UI may use
    required: bool = True


# ── primitives ─────────────────────────────────────────────────


def _run(cmd: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


_HOMEBREW_PATHS = ("/opt/homebrew/bin/brew", "/usr/local/bin/brew")


def _find_brew() -> str | None:
    """GUI .app bundles have a stripped PATH; check the standard install
    locations directly instead of relying on `shutil.which`."""
    for p in _HOMEBREW_PATHS:
        if Path(p).is_file() and os.access(p, os.X_OK):
            return p
    return shutil.which("brew")


def check_homebrew() -> Check:
    import app_paths
    brew = _find_brew()
    # In a packaged .app Homebrew is only needed if the user wants to install
    # BlackHole via `brew install blackhole-2ch`. The .pkg installer works
    # without brew, so we don't block on it.
    required = not app_paths.is_frozen()
    if brew is None:
        status = "todo" if app_paths.is_frozen() else "error"
        return Check(
            "homebrew",
            "Homebrew",
            status,
            "Optional. Useful for installing BlackHole via `brew install blackhole-2ch`.",
            action="install_brew",
            required=required,
        )
    try:
        out = _run([brew, "--version"])
        version = out.stdout.splitlines()[0] if out.stdout else "installed"
    except Exception as exc:
        return Check("homebrew", "Homebrew", "error", str(exc), required=required)
    prefix = "/opt/homebrew" if brew.startswith("/opt") else "/usr/local"
    return Check(
        "homebrew", "Homebrew", "ok", f"Package manager · {version} · {prefix}", required=required
    )


def _find_python310() -> str | None:
    candidates = (
        "/opt/homebrew/bin/python3.10",
        "/opt/homebrew/opt/python@3.10/bin/python3.10",
        "/usr/local/bin/python3.10",
        "/usr/local/opt/python@3.10/bin/python3.10",
    )
    for p in candidates:
        if Path(p).is_file() and os.access(p, os.X_OK):
            return p
    return shutil.which("python3.10")


def check_python310() -> Check:
    py = _find_python310()
    # Inside the packaged .app, the bundled Python interpreter is what's
    # actually running the pipeline — Python 3.10 on the host is only needed
    # when developing from source. We surface this as "ok" if the embedded
    # interpreter is sufficient.
    import app_paths
    if py is None and app_paths.is_frozen():
        return Check(
            "python",
            "Python 3.10 (bundled)",
            "ok",
            "Bundled interpreter ships inside Voicebox.app",
            required=False,
        )
    if py is None:
        return Check(
            "python",
            "Python 3.10",
            "error",
            "Not installed. Run `brew install python@3.10`.",
            action="install_python310",
        )
    try:
        out = _run([py, "--version"])
        version = (out.stdout or out.stderr).strip()
    except Exception as exc:
        return Check("python", "Python 3.10", "error", str(exc))
    required = not app_paths.is_frozen()
    return Check("python", f"{version}", "ok", f"Detected at {py}", required=required)


def check_blackhole() -> Check:
    driver_path = Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver")
    if driver_path.is_dir():
        return Check(
            "blackhole",
            "BlackHole 2ch driver",
            "ok",
            "Virtual audio loopback (Existential Audio) · driver installed",
        )
    # Maybe brew cask?
    try:
        out = _run(["brew", "list", "--cask", "blackhole-2ch"])
        if out.returncode == 0:
            return Check("blackhole", "BlackHole 2ch driver", "ok", "Installed via Homebrew cask")
    except Exception:
        pass
    return Check(
        "blackhole",
        "BlackHole 2ch driver",
        "error",
        "Not installed. Run `brew install blackhole-2ch`.",
        action="install_blackhole",
    )


def reload_sounddevice_devices() -> None:
    """Force PortAudio to re-enumerate CoreAudio devices so freshly-created
    Multi-Output / Aggregate devices show up without restarting the process.

    DANGEROUS while audio streams are open — only call from explicit user
    actions (e.g. user finished a Multi-Output setup), never from passive
    refresh loops. Wrapped in try/except so it cannot crash callers."""
    try:
        import sounddevice as sd

        try:
            sd._terminate()
        except Exception:
            pass
        try:
            sd._initialize()
        except Exception:
            pass
    except Exception:
        pass


def check_multi_output_device() -> Check:
    """Detect if any CoreAudio aggregate device exists that combines BlackHole
    with at least one other output. Heuristic: scan `sounddevice.query_devices()`
    for an entry whose name contains 'Multi-Output' or 'Aggregate'.

    Does NOT re-initialize PortAudio — see `reload_sounddevice_devices()`."""
    try:
        import sounddevice as sd

        for info in sd.query_devices():
            name = info["name"]
            if (
                ("Multi-Output" in name or "Aggregate" in name)
                and info["max_output_channels"] >= 2
            ):
                return Check(
                    "multiOutput",
                    "Multi-Output Device",
                    "ok",
                    f"{name} · combines BlackHole with normal output",
                    required=False,
                )
    except Exception as exc:
        return Check("multiOutput", "Multi-Output Device", "error", str(exc), required=False)
    return Check(
        "multiOutput",
        "Multi-Output Device",
        "todo",
        "Optional. Without it, you won't hear yourself while audio is routed to BlackHole.",
        action="open_midi_setup",
        required=False,
    )


def check_base_models(base_dir: Path | None = None) -> Check:
    if base_dir is None:
        import app_paths
        base_dir = app_paths.base_models_dir()
    hubert = base_dir / "hubert_base.pt"
    rmvpe = base_dir / "rmvpe.pt"
    missing = [p.name for p in (hubert, rmvpe) if not p.is_file()]
    if missing:
        return Check(
            "baseModels",
            "Base models",
            "todo",
            f"Missing {', '.join(missing)} — click Download to fetch from HuggingFace.",
            action="download_base_models",
        )
    total_mb = (hubert.stat().st_size + rmvpe.stat().st_size) / (1024 * 1024)
    return Check(
        "baseModels",
        "Base models",
        "ok",
        f"hubert_base.pt · rmvpe.pt — {total_mb:.0f} MB",
    )


HF_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
BASE_MODEL_URLS = {
    "hubert_base.pt": f"{HF_BASE}/hubert_base.pt",
    "rmvpe.pt": f"{HF_BASE}/rmvpe.pt",
}


def download_base_models(
    base_dir: Path | None = None,
    on_progress=None,
) -> list[Path]:
    """Download hubert_base.pt + rmvpe.pt into `base_dir`. Returns list of
    paths actually downloaded (already-present files are skipped)."""
    if base_dir is None:
        import app_paths
        base_dir = app_paths.base_models_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    import requests

    saved: list[Path] = []
    for fname, url in BASE_MODEL_URLS.items():
        target = base_dir / fname
        if target.is_file() and target.stat().st_size > 0:
            continue
        if on_progress:
            on_progress(f"Downloading {fname}…")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    f.write(chunk)
        saved.append(target)
        if on_progress:
            mb = target.stat().st_size / (1024 * 1024)
            on_progress(f"Saved {fname} ({mb:.0f} MB)")
    return saved


_mic_granted: bool | None = None
_mic_denied: bool = False


def check_mic_permission() -> Check:
    """Status comes from a cached probe in `request_mic_permission()`. We do
    NOT auto-probe here because that would trigger the macOS prompt on every
    refresh. The UI exposes a 'Request access' button that calls the probe."""
    if _mic_granted is True:
        return Check(
            "micPermission",
            "Microphone permission",
            "ok",
            "Access granted · CoreAudio input available",
            required=False,
        )
    if _mic_denied:
        return Check(
            "micPermission",
            "Microphone permission",
            "error",
            "Access denied · enable in System Settings → Privacy & Security → Microphone",
            action="open_privacy",
            required=False,
        )
    return Check(
        "micPermission",
        "Microphone permission",
        "todo",
        "Optional here — Start in Voice Pipeline triggers the macOS prompt the first time.",
        action="request_mic",
        required=False,
    )


def request_mic_permission(timeout_s: float = 1.5) -> bool:
    """Briefly open a CoreAudio input stream so macOS shows the permission
    prompt. Returns True if granted (stream opened), False if denied or
    timed out. Updates the cached check status."""
    global _mic_granted, _mic_denied
    try:
        import numpy as np
        import sounddevice as sd

        try:
            with sd.InputStream(channels=1, samplerate=16000, blocksize=256):
                # Hold the stream briefly so CoreAudio commits the auth state.
                import time

                time.sleep(timeout_s)
        except Exception as exc:
            _mic_granted = False
            _mic_denied = "permission" in str(exc).lower() or "-50" in str(exc)
            return False
    except Exception:
        _mic_granted = False
        _mic_denied = True
        return False
    _mic_granted = True
    _mic_denied = False
    return True


def run_all(base_dir: Path | None = None) -> list[Check]:
    """Run every check in the order they appear in the Setup screen."""
    return [
        check_homebrew(),
        check_python310(),
        check_blackhole(),
        check_multi_output_device(),
        check_base_models(base_dir),
        check_mic_permission(),
    ]


# ── actions invoked by Setup screen ────────────────────────────


def open_audio_midi_setup() -> None:
    subprocess.run(["open", "-a", "Audio MIDI Setup"], check=False)


def open_privacy_microphone() -> None:
    subprocess.run(
        ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"],
        check=False,
    )


def install_command(action: str) -> str | None:
    """Returns the shell command that the UI should expose under a 'Copy' button."""
    return {
        "install_brew": '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
        "install_python310": "brew install python@3.10",
        "install_blackhole": "brew install blackhole-2ch",
        "run_setup": "./setup.sh",
    }.get(action)
