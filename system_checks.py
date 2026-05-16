"""System requirement checks for the Voicebox UI Setup screen.

Each function returns a `Check` carrying a status (`ok` / `todo` / `error`)
and a human-readable detail string. Pure-ish: no Qt, no audio, only stdlib
and small standard tools (`brew`, `pkgutil`, filesystem)."""

from __future__ import annotations

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


def check_homebrew() -> Check:
    if shutil.which("brew") is None:
        return Check(
            "homebrew",
            "Homebrew",
            "error",
            "Not installed. Install from https://brew.sh and re-run.",
            action="install_brew",
        )
    try:
        out = _run(["brew", "--version"])
        version = out.stdout.splitlines()[0] if out.stdout else "installed"
    except Exception as exc:
        return Check("homebrew", "Homebrew", "error", str(exc))
    prefix = "/opt/homebrew" if Path("/opt/homebrew/bin/brew").exists() else "/usr/local"
    return Check("homebrew", "Homebrew", "ok", f"Package manager · {version} · {prefix}")


def check_python310() -> Check:
    if shutil.which("python3.10") is None and not Path("/opt/homebrew/bin/python3.10").exists():
        return Check(
            "python",
            "Python 3.10",
            "error",
            "Not installed. Run `brew install python@3.10`.",
            action="install_python310",
        )
    py = "/opt/homebrew/bin/python3.10"
    if not Path(py).exists():
        py = shutil.which("python3.10") or "python3.10"
    try:
        out = _run([py, "--version"])
        version = (out.stdout or out.stderr).strip()
    except Exception as exc:
        return Check("python", "Python 3.10", "error", str(exc))
    return Check("python", f"{version}", "ok", f"Detected at {py}")


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


def check_base_models(base_dir: Path = Path("models/base")) -> Check:
    hubert = base_dir / "hubert_base.pt"
    rmvpe = base_dir / "rmvpe.pt"
    missing = [p.name for p in (hubert, rmvpe) if not p.is_file()]
    if missing:
        return Check(
            "baseModels",
            "Base models",
            "error",
            f"Missing: {', '.join(missing)} · run ./setup.sh to download.",
            action="run_setup",
        )
    total_mb = (hubert.stat().st_size + rmvpe.stat().st_size) / (1024 * 1024)
    return Check(
        "baseModels",
        "Base models",
        "ok",
        f"hubert_base.pt · rmvpe.pt — {total_mb:.0f} MB",
    )


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
        )
    if _mic_denied:
        return Check(
            "micPermission",
            "Microphone permission",
            "error",
            "Access denied · enable in System Settings → Privacy & Security → Microphone",
            action="open_privacy",
        )
    return Check(
        "micPermission",
        "Microphone permission",
        "todo",
        "Click Request access to trigger the macOS prompt",
        action="request_mic",
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


def run_all(base_dir: Path = Path("models/base")) -> list[Check]:
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
