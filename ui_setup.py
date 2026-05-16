"""Setup screen: system checks, voice models, run-setup, log."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import threading

from PySide6.QtCore import QObject, QProcess, Qt, Signal
from PySide6.QtGui import QClipboard, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
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
from PySide6.QtCore import Qt

import app_paths
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
    # (label, .pth URL, .index URL — empty string if not available)
    # Tested-public URLs only. Many "sail-rvc/*" repos are gated and 401.
    ("Default test voice", "https://huggingface.co/PhoenixStormJr/RVC-V2-default-voice/resolve/main/default.pth", ""),
]

BROWSE_LINKS: list[tuple[str, str]] = [
    ("Browse HuggingFace", "https://huggingface.co/models?other=rvc"),
    ("Browse weights.gg", "https://www.weights.gg/"),
]


class _DLSignals(QObject):
    log = Signal(str, str)             # (text, level)
    progress = Signal(str, str)        # (row_key, sub_text)
    finished = Signal(str, bool, str)  # (row_key, success, message)


class SetupScreen(QWidget):
    state_changed = Signal()
    ready_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self._qss())
        self._process: QProcess | None = None
        self._status_rows: dict[str, StatusRow] = {}
        self._ready = False
        self._downloading: set[str] = set()
        self._dl = _DLSignals(self)
        self._dl.log.connect(lambda msg, lvl: self._log_append(msg, level=lvl))
        self._dl.progress.connect(self._on_dl_progress)
        self._dl.finished.connect(self._on_dl_finished)
        self._build()
        self.refresh()

    def is_ready(self) -> bool:
        return self._ready

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
        checks = system_checks.run_all()
        last_row: StatusRow | None = None
        for c in checks:
            row = StatusRow(
                status=c.status,
                label=c.label,
                sub=c.detail,
                action=self._action_for(c),
            )
            row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            self._status_rows[c.key] = row
            rwrap.addWidget(row)
            last_row = row
        rwrap.addStretch(1)
        # last row no bottom border
        if last_row is not None:
            last_row.setStyleSheet(last_row.styleSheet() + " QFrame#StatusRow { border-bottom: none; }")
        card.add(rows_wrap)

        # The shell-script setup makes sense only when running from the repo —
        # the .app has no setup.sh and brew on the host. Inside the bundle we
        # rely on the per-row Download/Install buttons.
        if app_paths.is_frozen():
            card.add_stretch()
            return card

        footer = QFrame()
        footer.setObjectName("SetupFooter")
        footer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        footer.setStyleSheet(
            f"""
            QFrame#SetupFooter {{
                background: rgba(255, 255, 255, 0.015);
                border: none;
                border-top: 1px solid {TOKENS['border']};
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }}
            QFrame#SetupFooter QLabel {{ border: none; background: transparent; }}
            """
        )
        flay = QHBoxLayout(footer)
        flay.setContentsMargins(18, 14, 18, 14)
        flay.setSpacing(12)
        flay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Run setup script")
        title.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 12px; color: {TOKENS['text']}; font-weight: 600;"
        )
        col.addWidget(title)
        sub = QLabel("Installs Homebrew, Python, BlackHole, and base models")
        sub.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 11px; color: {TOKENS['text_sub']};"
        )
        sub.setWordWrap(True)
        col.addWidget(sub)
        flay.addLayout(col, 1)
        self._run_btn = Button("Run setup.sh", variant="primary", size="md", icon_name="play")
        self._run_btn.clicked.connect(self._run_setup)
        flay.addWidget(self._run_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        card.add(footer)

        return card

    def _action_for(self, c: system_checks.Check) -> QWidget | None:
        if c.action == "open_midi_setup":
            btn = Button("Open MIDI Setup", variant="secondary", size="sm", icon_name="external")
            btn.clicked.connect(self._open_midi_setup_with_help)
            return btn
        if c.action == "open_privacy":
            btn = Button("Open Privacy", variant="secondary", size="sm", icon_name="external")
            btn.clicked.connect(system_checks.open_privacy_microphone)
            return btn
        if c.action == "request_mic":
            btn = Button("Request access", variant="primary", size="sm", icon_name="mic")
            btn.clicked.connect(self._request_mic)
            return btn
        if c.action == "download_base_models":
            btn = Button("Download", variant="primary", size="sm", icon_name="download")
            btn.clicked.connect(self._download_base_models)
            return btn
        if c.action in ("install_brew", "install_python310", "install_blackhole", "run_setup"):
            cmd = system_checks.install_command(c.action)
            if cmd:
                btn = Button("Copy", variant="secondary", size="sm", icon_name="copy")
                btn.clicked.connect(lambda _checked=False, command=cmd: self._copy(command))
                return btn
        return None

    def _open_midi_setup_with_help(self) -> None:
        # Non-modal stay-on-top guide so the user can follow along while
        # working inside Audio MIDI Setup.
        if getattr(self, "_midi_dlg", None) is not None and self._midi_dlg.isVisible():
            self._midi_dlg.raise_()
            self._midi_dlg.activateWindow()
            system_checks.open_audio_midi_setup()
            return

        dlg = QDialog(self)
        self._midi_dlg = dlg
        dlg.setWindowTitle("Create a Multi-Output Device")
        dlg.setModal(False)
        dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.setMinimumWidth(420)
        dlg.setStyleSheet(
            f"""
            QDialog {{
                background: {TOKENS['surface']};
                color: {TOKENS['text']};
                font-family: {FONT_UI};
            }}
            QLabel {{ color: {TOKENS['text']}; }}
            QLabel#title {{ font-size: 14px; font-weight: 600; }}
            QLabel#sub {{ color: {TOKENS['text_sub']}; font-size: 12px; }}
            QLabel#step {{ font-size: 12px; color: {TOKENS['text']}; }}
            """
        )
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(10)

        title = QLabel("Create a Multi-Output Device")
        title.setObjectName("title")
        lay.addWidget(title)
        sub = QLabel(
            "Combine BlackHole 2ch with your normal output so you hear yourself while routing audio to Discord / Zoom / OBS."
        )
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        steps = [
            "Click the + in the bottom-left of Audio MIDI Setup → Create Multi-Output Device.",
            "In the device list, tick BOTH \"BlackHole 2ch\" AND your normal output (MacBook Pro Speakers / headphones).",
            "Right-click the new Multi-Output Device → \"Use This Device For Sound Output\".",
            "Come back to Voicebox — the System check refreshes automatically.",
        ]
        for i, text in enumerate(steps, start=1):
            row = QHBoxLayout()
            row.setSpacing(8)
            num = QLabel(f"{i}.")
            num.setStyleSheet(f"color: {ACCENT}; font-weight: 700; font-size: 12px; min-width: 14px;")
            row.addWidget(num, 0, Qt.AlignmentFlag.AlignTop)
            body = QLabel(text)
            body.setObjectName("step")
            body.setWordWrap(True)
            row.addWidget(body, 1)
            wrap = QFrame()
            wrap.setLayout(row)
            lay.addWidget(wrap)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch(1)
        open_btn = Button("Open Audio MIDI Setup", variant="primary", size="md", icon_name="external")
        open_btn.clicked.connect(self._on_midi_open_clicked)
        btn_row.addWidget(open_btn)
        close_btn = Button("I'm done", variant="secondary", size="md")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(close_btn)
        wrap_btns = QFrame()
        wrap_btns.setLayout(btn_row)
        lay.addWidget(wrap_btns)

        dlg.destroyed.connect(self._on_midi_dlg_destroyed)
        dlg.show()
        # Open MIDI Setup immediately so the user has both windows at once.
        self._on_midi_open_clicked()

    def _on_midi_open_clicked(self) -> None:
        system_checks.open_audio_midi_setup()
        self._log_append("Opened Audio MIDI Setup — follow the steps in the floating guide.", level="info")

    def _on_midi_dlg_destroyed(self) -> None:
        self._midi_dlg = None
        # User just finished Audio MIDI Setup work — force PortAudio to
        # re-enumerate devices so a freshly-created Multi-Output Device shows
        # up. Safe here because no audio stream is open from the UI on the
        # Setup screen.
        system_checks.reload_sounddevice_devices()
        self.refresh()

    def _request_mic(self) -> None:
        self._log_append("Requesting microphone permission via CoreAudio…", level="info")
        ok = system_checks.request_mic_permission()
        if ok:
            self._log_append("Microphone access granted", level="ok")
        else:
            self._log_append(
                "Microphone access not granted. Open System Settings → Privacy & Security → "
                "Microphone and enable Python / Voicebox there, then return.",
                level="err",
            )
            system_checks.open_privacy_microphone()
        self.refresh()

    def _copy(self, text: str) -> None:
        QGuiApplication.clipboard().setText(text)
        self._log_append(f"Copied to clipboard: {text}", level="info")

    def _download_base_models(self) -> None:
        if "baseModels" in self._downloading:
            return
        self._downloading.add("baseModels")
        self._dl.log.emit("Downloading base models from HuggingFace…", "info")

        sig = self._dl

        def worker():
            try:
                def _log(msg, lvl="info"):
                    sig.log.emit(msg, lvl)

                def _bytes(idx, total_files, done, total):
                    if total > 0:
                        pct = 100.0 * done / total
                        sig.progress.emit(
                            "baseModels",
                            f"Downloading file {idx + 1}/{total_files} — {pct:.0f}% ({done / (1024*1024):.0f} / {total / (1024*1024):.0f} MB)",
                        )
                    else:
                        sig.progress.emit(
                            "baseModels",
                            f"Downloading file {idx + 1}/{total_files} — {done / (1024*1024):.0f} MB",
                        )

                saved = system_checks.download_base_models(on_log=_log, on_bytes=_bytes)
                msg = (
                    "Base models ready (" + ", ".join(p.name for p in saved) + ")"
                    if saved else "Base models already present."
                )
                sig.finished.emit("baseModels", True, msg)
            except Exception as exc:
                sig.finished.emit("baseModels", False, str(exc))

        threading.Thread(target=worker, name="dl-base-models", daemon=True).start()

    def _on_dl_progress(self, row_key: str, sub_text: str) -> None:
        row = self._status_rows.get(row_key)
        if row is not None:
            row.set_sub(sub_text)
            row.set_status("pending")
        if row_key == "contentVec":
            self._encoder_status.setText(sub_text)
            self._encoder_status.setStyleSheet(
                f"font-size: 11px; color: {ACCENT}; margin-top: 2px;"
            )

    def _on_dl_finished(self, row_key: str, success: bool, message: str) -> None:
        self._downloading.discard(row_key)
        self._dl.log.emit(message, "ok" if success else "err")
        self.refresh()
        self._refresh_encoder_status()

    def _refresh_encoder_status(self) -> None:
        content_vec = app_paths.base_models_dir() / "content_vec.pt"
        hubert = app_paths.base_models_dir() / "hubert_base.pt"
        if content_vec.is_file():
            self._encoder_status.setText(
                f"Active: ContentVec ({content_vec.stat().st_size / (1024*1024):.0f} MB) — better for non-English speech"
            )
            self._encoder_status.setStyleSheet(
                f"font-size: 11px; color: {COLOR_OK}; margin-top: 2px;"
            )
        elif hubert.is_file():
            self._encoder_status.setText(
                "Active: HuBERT (English) — replace with ContentVec for Russian / other non-English voices"
            )
            self._encoder_status.setStyleSheet(
                f"font-size: 11px; color: {TOKENS['text_dim']}; margin-top: 2px;"
            )
        else:
            self._encoder_status.setText("No encoder installed.")
            self._encoder_status.setStyleSheet(
                f"font-size: 11px; color: {COLOR_ERR}; margin-top: 2px;"
            )

    def _download_content_vec(self) -> None:
        if "contentVec" in self._downloading:
            return
        # `lengyue233/content-vec-best` no longer exposes the checkpoint_best_legacy
        # filename (their repo now ships `pytorch_model.bin` in HF Transformers
        # format, which fairseq can't load directly). The w-okada vcclient repo
        # mirrors the original ContentVec legacy checkpoint that rvc-python /
        # fairseq expects.
        url = "https://huggingface.co/wok000/vcclient_modules/resolve/main/contentvec/checkpoint_best_legacy_500.pt"
        target = app_paths.base_models_dir() / "content_vec.pt"
        if target.is_file():
            self._log_append("ContentVec already installed.", level="ok")
            return
        self._downloading.add("contentVec")
        self._dl.log.emit(f"Downloading ContentVec from {url} …", "info")

        sig = self._dl

        def worker():
            try:
                def _log(msg, lvl="info"):
                    sig.log.emit(msg, lvl)

                def _bytes(done, total):
                    if total > 0:
                        pct = 100.0 * done / total
                        sig.progress.emit(
                            "contentVec",
                            f"Downloading content_vec.pt — {pct:.0f}% ({done / (1024*1024):.0f} / {total / (1024*1024):.0f} MB)",
                        )

                system_checks.download_file(url, target, on_log=_log, on_bytes=_bytes)
                sig.finished.emit("contentVec", True, "ContentVec installed. Restart the pipeline to use it.")
            except Exception as exc:
                try:
                    if target.exists():
                        target.unlink()
                except Exception:
                    pass
                sig.finished.emit("contentVec", False, f"ContentVec download failed: {exc}")

        threading.Thread(target=worker, name="dl-content-vec", daemon=True).start()

    def _open_url(self, url: str) -> None:
        import subprocess

        subprocess.run(["open", url], check=False)
        self._log_append(f"Opened {url}", level="info")

    # ── Voice models card ───────────────────────────────────────

    def _build_models_card(self) -> Card:
        card = Card(padding=18)
        self._models_pill = Pill("Folder", tone="neutral")
        folder_btn = Button("Folder", variant="secondary", size="sm", icon_name="folder")
        folder_btn.clicked.connect(models_manager.open_models_folder)
        self._models_title = CardTitle("Voice models", sub="…", right=folder_btn)
        card.add(self._models_title)

        # No QScrollArea — its viewport on macOS PySide6 lets child widgets
        # render past the supposed clip rect. Just stack rows in the card and
        # let the column expand naturally. With many models the card grows;
        # the right-column log card below absorbs less stretch — acceptable.
        self._models_wrap = QFrame()
        self._models_wrap.setObjectName("ModelsWrap")
        self._models_wrap.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._models_wrap.setStyleSheet(
            f"QFrame#ModelsWrap {{ background: transparent; border: none; }}"
        )
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

        # Examples row + browse buttons
        ex_header = QLabel("Try an example or browse community models")
        ex_header.setStyleSheet(
            f"font-size: 11px; color: {TOKENS['text_sub']}; font-weight: 600;"
            f" letter-spacing: 0.3px; text-transform: uppercase; margin-top: 10px;"
        )
        card.add(ex_header)

        ex_grid = QGridLayout()
        ex_grid.setHorizontalSpacing(6)
        ex_grid.setVerticalSpacing(6)
        ex_grid.setContentsMargins(0, 6, 0, 0)
        all_buttons: list[QWidget] = []
        for label, pth_url, idx_url in EXAMPLES:
            btn = Button(label, variant="secondary", size="sm", icon_name="download")
            btn.clicked.connect(
                lambda _checked=False, p=pth_url, i=idx_url, l=label: self._download_example(l, p, i)
            )
            all_buttons.append(btn)
        for label, url in BROWSE_LINKS:
            btn = Button(label, variant="ghost", size="sm", icon_name="external")
            btn.clicked.connect(lambda _checked=False, u=url: self._open_url(u))
            all_buttons.append(btn)
        for i, btn in enumerate(all_buttons):
            r, c = divmod(i, 2)
            ex_grid.addWidget(btn, r, c)
        ex_grid.setColumnStretch(0, 1)
        ex_grid.setColumnStretch(1, 1)
        ew = QFrame()
        ew.setLayout(ex_grid)
        card.add(ew)

        hint = QLabel(
            "Models from weights.gg or HuggingFace usually ship as a folder with a .pth + .index. "
            "Drop both files above — they're sorted by name automatically."
        )
        hint.setStyleSheet(f"font-size: 11px; color: {TOKENS['text_dim']}; margin-top: 8px;")
        hint.setWordWrap(True)
        card.add(hint)

        # Encoder (HuBERT / ContentVec) row
        enc_header = QLabel("Encoder")
        enc_header.setStyleSheet(
            f"font-size: 11px; color: {TOKENS['text_sub']}; font-weight: 600;"
            f" letter-spacing: 0.3px; text-transform: uppercase; margin-top: 14px;"
        )
        card.add(enc_header)
        self._encoder_status = QLabel("…")
        self._encoder_status.setStyleSheet(
            f"font-size: 11px; color: {TOKENS['text_dim']}; margin-top: 2px;"
        )
        self._encoder_status.setWordWrap(True)
        card.add(self._encoder_status)
        enc_btn = Button("Download ContentVec (multilingual)", variant="secondary", size="sm", icon_name="download")
        enc_btn.clicked.connect(self._download_content_vec)
        enc_wrap = QFrame()
        enc_lay = QHBoxLayout(enc_wrap)
        enc_lay.setContentsMargins(0, 6, 0, 0)
        enc_lay.addWidget(enc_btn)
        enc_lay.addStretch(1)
        card.add(enc_wrap)
        self._refresh_encoder_status()

        return card

    def _refresh_models(self) -> None:
        # Clear everything (widgets AND any leftover stretch items).
        while self._models_lay.count():
            item = self._models_lay.takeAt(0)
            w = item.widget()
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
        # Trailing stretch keeps rows anchored to the top of the scroll area.
        self._models_lay.addStretch(1)
        self._models_title.set_sub(f"{len(models)} models installed · {app_paths.rvc_models_dir()}")
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

    def _do_download(self, url: str, voice_name: str | None = None) -> None:
        import re
        import requests

        # Derive a sane voice folder name from the URL (e.g. ".../default.pth"
        # → "default"). Fall back to caller-provided label.
        url_basename = url.rsplit("/", 1)[-1].split("?")[0]
        url_stem = url_basename.rsplit(".", 1)[0] if "." in url_basename else url_basename
        name = voice_name or url_stem or "model"
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "model"
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
                suffix = ".pth"
        if not suffix:
            suffix = ".pth"
        # Rename tmp file to <name><suffix> so accept_drop creates a proper folder.
        renamed = tmp_path.with_name(f"{name}{suffix}")
        try:
            tmp_path.rename(renamed)
        except OSError:
            renamed = tmp_path
        moved = models_manager.accept_drop([renamed])
        for p in moved:
            self._log_append(f"Saved {p}", level="ok")

    def _download_example(self, label: str, pth_url: str, idx_url: str) -> None:
        import re

        voice_name = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-") or "example"
        try:
            self._do_download(pth_url, voice_name=voice_name)
            if idx_url:
                self._do_download(idx_url, voice_name=voice_name)
            self._log_append(f"Example {label} ready", level="ok")
            self.refresh()
        except Exception as exc:
            self._log_append(f"Example {label} failed: {exc}", level="err")

    # ── refresh ────────────────────────────────────────────────

    def refresh(self) -> None:
        checks = system_checks.run_all()
        required_total = sum(1 for c in checks if c.required)
        required_ok = sum(1 for c in checks if c.required and c.status == "ok")
        ok_count = sum(1 for c in checks if c.status == "ok")
        for c in checks:
            row = self._status_rows.get(c.key)
            if row is None:
                continue
            row.set_status(c.status)
            label = c.label + ("  (optional)" if not c.required else "")
            row.set_label(label)
            row.set_sub(c.detail)
            row.set_action(self._action_for(c))
        total = len(checks)
        self._system_title.set_sub(
            f"{required_ok} of {required_total} required satisfied · {ok_count}/{total} total"
        )
        ready = required_ok == required_total
        if ready:
            self._system_pill.set_text("Ready")
            self._system_pill.set_tone("success")
        else:
            self._system_pill.set_text("Action needed")
            self._system_pill.set_tone("accent")
        if ready != self._ready:
            self._ready = ready
            self.ready_changed.emit(ready)
        self._refresh_models()
