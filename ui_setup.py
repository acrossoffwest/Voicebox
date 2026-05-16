"""Setup screen: system checks, voice models, run-setup, log."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import QProcess, Qt, pyqtSignal
from PyQt6.QtGui import QClipboard, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import models_manager
import system_checks
from ui_theme import (
    ACCENT,
    COLOR_ERR,
    COLOR_OK,
    FONT_UI,
    TOKENS,
    hex_alpha,
)
from ui_widgets import (
    Button,
    Card,
    CardTitle,
    DropZone,
    LogView,
    ModelRow,
    Pill,
    StatusRow,
)

EXAMPLES: list[tuple[str, str, str]] = [
    # (label, .pth URL, .index URL or "")
    # placeholder — user-editable in v2 via UI; for now we ship with empty list and
    # surface the input field as the primary path. The buttons stay disabled if the
    # list is empty, with a "configure examples" hint.
]


class SetupScreen(QWidget):
    state_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self._qss())
        self._process: QProcess | None = None
        self._status_rows: dict[str, StatusRow] = {}
        self._build()
        self.refresh()

    def _qss(self) -> str:
        return f"""
        QWidget {{ background: transparent; color: {TOKENS['text']}; font-family: {FONT_UI}; }}
        QLineEdit {{
            background: #16161A;
            color: {TOKENS['text']};
            border: 1px solid {TOKENS['border_strong']};
            border-radius: 7px;
            padding: 0 12px;
            font-size: 12px;
            min-height: 28px;
        }}
        """

    def _build(self) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(24, 24, 24, 24)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        grid.addWidget(self._build_system_check_card(), 0, 0, 1, 1)
        right = QVBoxLayout()
        right.setSpacing(16)
        right.addWidget(self._build_models_card())
        right.addWidget(self._build_log_card(), 1)
        rwrap = QWidget()
        rwrap.setLayout(right)
        grid.addWidget(rwrap, 0, 1, 1, 1)
        grid.setRowStretch(0, 1)

    # ── System check card ───────────────────────────────────────

    def _build_system_check_card(self) -> Card:
        card = Card(padding=0)
        card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        header = QFrame()
        hlay = QVBoxLayout(header)
        hlay.setContentsMargins(18, 16, 18, 12)
        self._system_pill = Pill("Action needed", tone="accent", dot=True)
        self._system_title = CardTitle("System check", sub="…", right=self._system_pill)
        hlay.addWidget(self._system_title)
        card.add(header)

        rows_wrap = QFrame()
        rwrap = QVBoxLayout(rows_wrap)
        rwrap.setContentsMargins(18, 0, 18, 6)
        rwrap.setSpacing(0)
        for c in system_checks.run_all():
            row = StatusRow(
                status=c.status,
                label=c.label,
                sub=c.detail,
                action=self._action_for(c),
            )
            self._status_rows[c.key] = row
            rwrap.addWidget(row)
        # last row no bottom border
        if rows_wrap.layout().count() > 0:
            last = rows_wrap.layout().itemAt(rows_wrap.layout().count() - 1).widget()
            last.setStyleSheet(last.styleSheet() + " QFrame#StatusRow { border-bottom: none; }")
        card.add(rows_wrap)

        footer = QFrame()
        footer.setStyleSheet(
            f"background: rgba(255,255,255,0.015); border-top: 1px solid {TOKENS['border']};"
            f" border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;"
        )
        flay = QHBoxLayout(footer)
        flay.setContentsMargins(18, 12, 18, 12)
        flay.setSpacing(10)
        col = QVBoxLayout()
        col.setSpacing(1)
        title = QLabel("Run setup script")
        title.setStyleSheet(f"font-size: 12px; color: {TOKENS['text']}; font-weight: 600;")
        col.addWidget(title)
        sub = QLabel("Installs Homebrew, Python, BlackHole, and base models")
        sub.setStyleSheet(f"font-size: 11px; color: {TOKENS['text_sub']};")
        col.addWidget(sub)
        flay.addLayout(col, 1)
        self._run_btn = Button("Run setup.sh", variant="primary", size="md", icon_name="play")
        self._run_btn.clicked.connect(self._run_setup)
        flay.addWidget(self._run_btn)
        card.add(footer)

        return card

    def _action_for(self, c: system_checks.Check) -> QWidget | None:
        if c.action == "open_midi_setup":
            btn = Button("Open MIDI Setup", variant="secondary", size="sm", icon_name="external")
            btn.clicked.connect(system_checks.open_audio_midi_setup)
            return btn
        if c.action == "open_privacy":
            btn = Button("Open Privacy", variant="secondary", size="sm", icon_name="external")
            btn.clicked.connect(system_checks.open_privacy_microphone)
            return btn
        if c.action in ("install_brew", "install_python310", "install_blackhole", "run_setup"):
            cmd = system_checks.install_command(c.action)
            if cmd:
                btn = Button("Copy", variant="secondary", size="sm", icon_name="copy")
                btn.clicked.connect(lambda _checked=False, command=cmd: self._copy(command))
                return btn
        return None

    def _copy(self, text: str) -> None:
        QGuiApplication.clipboard().setText(text)
        self._log_append(f"Copied to clipboard: {text}", level="info")

    # ── Voice models card ───────────────────────────────────────

    def _build_models_card(self) -> Card:
        card = Card(padding=18)
        self._models_pill = Pill("Folder", tone="neutral")
        folder_btn = Button("Folder", variant="secondary", size="sm", icon_name="folder")
        folder_btn.clicked.connect(models_manager.open_models_folder)
        self._models_title = CardTitle("Voice models", sub="…", right=folder_btn)
        card.add(self._models_title)

        self._models_wrap = QFrame()
        self._models_lay = QVBoxLayout(self._models_wrap)
        self._models_lay.setContentsMargins(0, 0, 0, 0)
        self._models_lay.setSpacing(6)
        card.add(self._models_wrap)

        self._dropzone = DropZone()
        self._dropzone.files_dropped.connect(self._on_drop)
        card.add(self._dropzone)

        url_row = QFrame()
        urow = QHBoxLayout(url_row)
        urow.setContentsMargins(0, 10, 0, 0)
        urow.setSpacing(8)
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("Paste model URL (HuggingFace / Drive)")
        urow.addWidget(self._url_edit, 1)
        dl_btn = Button("Download", variant="secondary", size="md", icon_name="download")
        dl_btn.clicked.connect(self._download_url)
        urow.addWidget(dl_btn)
        card.add(url_row)

        # Examples row (placeholder — see EXAMPLES constant)
        if EXAMPLES:
            ex_row = QHBoxLayout()
            ex_row.setSpacing(6)
            for label, pth_url, idx_url in EXAMPLES:
                btn = Button(f"Download {label}", variant="secondary", size="sm", icon_name="download")
                btn.clicked.connect(
                    lambda _checked=False, p=pth_url, i=idx_url, l=label: self._download_example(l, p, i)
                )
                ex_row.addWidget(btn)
            ex_row.addStretch(1)
            ew = QFrame()
            ew.setLayout(ex_row)
            card.add(ew)
        else:
            hint = QLabel("Tip: paste a HuggingFace `.pth` URL above or drag files in.")
            hint.setStyleSheet(f"font-size: 11px; color: {TOKENS['text_dim']}; margin-top: 6px;")
            card.add(hint)

        return card

    def _refresh_models(self) -> None:
        # clear
        for i in reversed(range(self._models_lay.count())):
            w = self._models_lay.itemAt(i).widget()
            if w is not None:
                w.setParent(None)
        models = models_manager.list_voice_models()
        if not models:
            empty = QLabel("No voice models yet. Drop files below or download from a URL.")
            empty.setStyleSheet(f"font-size: 11.5px; color: {TOKENS['text_dim']};")
            self._models_lay.addWidget(empty)
        else:
            for m in models:
                row = ModelRow(m.name, m.files_label, m.size_label, m.full)
                row.removed.connect(self._on_remove_model)
                self._models_lay.addWidget(row)
        self._models_title.set_sub(f"{len(models)} models installed · ./models/rvc")
        self.state_changed.emit()

    def _on_remove_model(self, name: str) -> None:
        reply = QMessageBox.question(
            self,
            "Remove voice model",
            f"Remove '{name}' from models/rvc/?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            models_manager.remove_model(name)
            self._log_append(f"Removed model: {name}", level="warn")
        except Exception as exc:
            self._log_append(f"Failed to remove {name}: {exc}", level="err")
        self._refresh_models()

    def _on_drop(self, paths: list[Path]) -> None:
        try:
            moved = models_manager.accept_drop(paths)
        except Exception as exc:
            self._log_append(f"Drop failed: {exc}", level="err")
            return
        for p in moved:
            self._log_append(f"Added {p.relative_to(Path.cwd()) if p.is_absolute() else p}", level="ok")
        self._refresh_models()

    # ── Log card ────────────────────────────────────────────────

    def _build_log_card(self) -> Card:
        card = Card(padding=0)
        header = QFrame()
        hlay = QVBoxLayout(header)
        hlay.setContentsMargins(18, 16, 18, 10)
        self._log_pill = Pill("Idle", tone="neutral", dot=True)
        self._log_title = CardTitle("Setup log", sub="stdout / stderr stream", right=self._log_pill)
        hlay.addWidget(self._log_title)
        card.add(header)
        self._log_view = LogView()
        card.add(self._log_view)
        # seed with initial check summary
        for c in system_checks.run_all():
            lvl = "ok" if c.status == "ok" else "warn" if c.status == "todo" else "err"
            mark = "✓" if c.status == "ok" else "!" if c.status == "todo" else "✗"
            self._log_append(f"{mark} {c.label} — {c.detail}", level=lvl)
        return card

    def _log_append(self, text: str, level: str = "info") -> None:
        ts = time.strftime("%H:%M:%S")
        self._log_view.append(text, level=level, timestamp=ts)

    # ── Run setup ───────────────────────────────────────────────

    def _run_setup(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            return
        self._log_pill.set_text("Running")
        self._log_pill.set_tone("accent")
        self._run_btn.setEnabled(False)
        self._log_append("$ bash ./setup.sh", level="info")

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_setup_stdout)
        self._process.finished.connect(self._on_setup_finished)
        self._process.start("bash", ["./setup.sh"])

    def _on_setup_stdout(self) -> None:
        assert self._process is not None
        data = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        for line in data.splitlines():
            level = "info"
            stripped = line.strip()
            if stripped.startswith("==>") or stripped.startswith("→"):
                level = "info"
            elif stripped.startswith("ERROR") or "error" in stripped.lower():
                level = "err"
            elif stripped.startswith("WARN") or "warning" in stripped.lower():
                level = "warn"
            elif stripped.startswith("✓"):
                level = "ok"
            if line.strip():
                self._log_append(line.rstrip(), level=level)

    def _on_setup_finished(self, code: int, status) -> None:
        ok = code == 0
        self._log_append(
            f"setup.sh exited with code {code}",
            level="ok" if ok else "err",
        )
        self._log_pill.set_text("Idle")
        self._log_pill.set_tone("neutral")
        self._run_btn.setEnabled(True)
        self._process = None
        self.refresh()

    # ── URL download ────────────────────────────────────────────

    def _download_url(self) -> None:
        url = self._url_edit.text().strip()
        if not url:
            return
        if not (url.startswith("http://") or url.startswith("https://")):
            self._log_append(f"Bad URL: {url}", level="err")
            return
        self._log_append(f"Downloading {url}…", level="info")
        try:
            self._do_download(url)
            self._url_edit.clear()
            self.refresh()
        except Exception as exc:
            self._log_append(f"Download failed: {exc}", level="err")

    def _do_download(self, url: str) -> None:
        import requests

        suffix = ".pth" if url.endswith(".pth") else (".index" if url.endswith(".index") else "")
        with tempfile.NamedTemporaryFile(prefix="voicebox-dl-", suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    tmp.write(chunk)
        if suffix == "":
            ctype = ""
            try:
                ctype = requests.head(url, allow_redirects=True, timeout=10).headers.get("content-type", "")
            except Exception:
                pass
            if "octet-stream" in ctype or url.endswith(".bin"):
                renamed = tmp_path.with_suffix(".pth")
                tmp_path.rename(renamed)
                tmp_path = renamed
        if tmp_path.suffix.lower() not in (".pth", ".index"):
            renamed = tmp_path.with_suffix(".pth")
            tmp_path.rename(renamed)
            tmp_path = renamed
        moved = models_manager.accept_drop([tmp_path])
        for p in moved:
            self._log_append(f"Saved {p}", level="ok")

    def _download_example(self, label: str, pth_url: str, idx_url: str) -> None:
        try:
            self._do_download(pth_url)
            if idx_url:
                self._do_download(idx_url)
            self._log_append(f"Example {label} ready", level="ok")
            self.refresh()
        except Exception as exc:
            self._log_append(f"Example {label} failed: {exc}", level="err")

    # ── refresh ────────────────────────────────────────────────

    def refresh(self) -> None:
        checks = system_checks.run_all()
        ok_count = 0
        for c in checks:
            row = self._status_rows.get(c.key)
            if row is None:
                continue
            row.set_status(c.status)
            row.set_label(c.label)
            row.set_sub(c.detail)
            if c.status == "ok":
                ok_count += 1
        total = len(checks)
        self._system_title.set_sub(f"{ok_count} of {total} requirements satisfied")
        if ok_count == total:
            self._system_pill.set_text("Ready")
            self._system_pill.set_tone("success")
        else:
            self._system_pill.set_text("Action needed")
            self._system_pill.set_tone("accent")
        self._refresh_models()
