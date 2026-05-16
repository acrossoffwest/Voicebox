"""Voicebox PySide6 UI entry point.

Usage: `./venv/bin/python ui.py`
"""

from __future__ import annotations

import os
import sys

# Some torch ops used by rmvpe (e.g. aten::_fft_r2c) aren't implemented for
# the MPS backend yet. Fall back to CPU for unsupported ops instead of
# crashing the audio processor.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

import app_paths
from ui_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    # Copy bundled rmvpe.pt into the user data dir on first launch (no-op
    # when running from source).
    app_paths.seed_bundled_assets()

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Voicebox")
    app.setStyle("Fusion")
    # Force dark palette regardless of system setting
    from PySide6.QtGui import QPalette, QColor

    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#0E0E11"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#EEEEF2"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#1C1C21"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#22222A"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#EEEEF2"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#2A2A32"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#EEEEF2"))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
