"""Pipeline screen: signal chain, routing, voice tuning, transport, telemetry."""

from __future__ import annotations

from pathlib import Path

import threading

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import models_manager
from engine import Engine, EngineConfig
from ui_theme import ACCENT, COLOR_ERR, COLOR_OK, FONT_MONO, FONT_UI, TOKENS, hex_alpha, shade
from ui_widgets import (
    Button,
    Card,
    CardTitle,
    LatencyBars,
    MeterBar,
    Pill,
    Select,
    SignalChain,
    Slider,
    Toggle,
)


def _list_input_devices() -> list[tuple[int, str]]:
    import sounddevice as sd

    return [
        (i, info["name"])
        for i, info in enumerate(sd.query_devices())
        if info["max_input_channels"] > 0
    ]


def _list_output_devices() -> list[tuple[int, str]]:
    import sounddevice as sd

    return [
        (i, info["name"])
        for i, info in enumerate(sd.query_devices())
        if info["max_output_channels"] > 0
    ]


class _LoadSignals(QObject):
    ready = pyqtSignal()
    failed = pyqtSignal(str)


class PipelineScreen(QWidget):
    config_changed = pyqtSignal()

    def __init__(self, settings_get=lambda k, default=None: default,
                 settings_set=lambda k, v: None, parent=None):
        super().__init__(parent)
        self._engine: Engine | None = None
        self._loading = False
        self._load_signals = _LoadSignals(self)
        self._load_signals.ready.connect(self._on_engine_ready)
        self._load_signals.failed.connect(self._on_engine_failed)
        self._get = settings_get
        self._set = settings_set
        self._input_devices = _list_input_devices()
        self._output_devices = _list_output_devices()
        self._models = models_manager.list_voice_models()
        self.setStyleSheet(
            f"QWidget {{ background: transparent; color: {TOKENS['text']}; font-family: {FONT_UI}; }}"
        )
        self._build()
        self._refresh_devices_and_models()

        self._poll = QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self._poll_stats)
        self._poll.start()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 16, 24, 24)
        outer.setSpacing(12)

        outer.addWidget(self._build_blackhole_banner())

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        grid.setColumnStretch(0, 14)
        grid.setColumnStretch(1, 10)

        left = QVBoxLayout()
        left.setSpacing(16)
        left.addWidget(self._build_signal_chain_card())
        left.addWidget(self._build_routing_card())
        left.addWidget(self._build_tuning_card())
        left.addStretch(1)
        lwrap = QWidget()
        lwrap.setLayout(left)
        grid.addWidget(lwrap, 0, 0)

        right = QVBoxLayout()
        right.setSpacing(16)
        right.addWidget(self._build_transport_card())
        right.addWidget(self._build_telemetry_card())
        right.addStretch(1)
        rwrap = QWidget()
        rwrap.setLayout(right)
        grid.addWidget(rwrap, 0, 1)
        outer.addLayout(grid, 1)

    # ── BlackHole input banner ──────────────────────────────────

    def _build_blackhole_banner(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("BHBanner")
        frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        frame.setStyleSheet(
            f"""
            QFrame#BHBanner {{
                background: {hex_alpha(ACCENT, 0.10)};
                border: 1px solid {hex_alpha(ACCENT, 0.30)};
                border-radius: 10px;
            }}
            """
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)
        ic = QLabel("🎙️")
        ic.setStyleSheet(f"font-size: 16px;")
        lay.addWidget(ic)
        col = QVBoxLayout()
        col.setSpacing(1)
        title = QLabel("Set BlackHole 2ch as the microphone input in your target app")
        title.setStyleSheet(f"font-family: {FONT_UI}; font-size: 12px; font-weight: 600; color: {TOKENS['text']};")
        col.addWidget(title)
        sub = QLabel("Discord, Zoom, OBS, QuickTime, Google Meet — pick BlackHole 2ch as the mic source so they receive the processed voice.")
        sub.setStyleSheet(f"font-family: {FONT_UI}; font-size: 11px; color: {TOKENS['text_sub']};")
        sub.setWordWrap(True)
        col.addWidget(sub)
        lay.addLayout(col, 1)
        from ui_widgets import Button

        help_btn = Button("How?", variant="secondary", size="sm", icon_name="external")
        help_btn.clicked.connect(self._show_target_app_guide)
        lay.addWidget(help_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        return frame

    def _show_target_app_guide(self) -> None:
        if getattr(self, "_target_dlg", None) is not None and self._target_dlg.isVisible():
            self._target_dlg.raise_()
            self._target_dlg.activateWindow()
            return
        dlg = QDialog(self)
        self._target_dlg = dlg
        dlg.setWindowTitle("Use BlackHole 2ch in other apps")
        dlg.setModal(False)
        dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.setMinimumWidth(460)
        dlg.setStyleSheet(
            f"""
            QDialog {{
                background: {TOKENS['surface']};
                color: {TOKENS['text']};
                font-family: {FONT_UI};
            }}
            QLabel#title {{ font-size: 14px; font-weight: 600; color: {TOKENS['text']}; }}
            QLabel#sub {{ font-size: 12px; color: {TOKENS['text_sub']}; }}
            QLabel#app {{ font-size: 12px; font-weight: 600; color: {ACCENT}; padding-top: 6px; }}
            QLabel#step {{ font-size: 12px; color: {TOKENS['text']}; }}
            """
        )
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(8)

        title = QLabel("Send your voice through Voicebox")
        title.setObjectName("title")
        lay.addWidget(title)
        sub = QLabel(
            "Voicebox emits the processed audio into BlackHole 2ch. Any app that "
            "selects BlackHole as its microphone input will receive your voice with "
            "the denoise + RVC effect applied."
        )
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        sections = [
            ("Discord", "Settings → Voice & Video → Input Device → BlackHole 2ch."),
            ("Zoom", "Settings → Audio → Microphone → BlackHole 2ch."),
            ("OBS", "Sources → + → Audio Input Capture → Device: BlackHole 2ch."),
            ("Google Meet (browser)", "Meeting settings → Microphone → BlackHole 2ch."),
            ("QuickTime Player (test)", "File → New Audio Recording → ▼ next to ⏺ → BlackHole 2ch."),
            ("System-wide (rare)", "System Settings → Sound → Input → BlackHole 2ch (only do this if you want every app to receive your processed voice by default)."),
        ]
        for app, instr in sections:
            app_lbl = QLabel(app)
            app_lbl.setObjectName("app")
            lay.addWidget(app_lbl)
            step = QLabel(instr)
            step.setObjectName("step")
            step.setWordWrap(True)
            lay.addWidget(step)

        note = QLabel("Tip: keep the output (speakers/headphones) on your normal device, not BlackHole — otherwise you create a feedback loop.")
        note.setStyleSheet(f"font-size: 11px; color: {TOKENS['text_dim']}; padding-top: 8px;")
        note.setWordWrap(True)
        lay.addWidget(note)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        from ui_widgets import Button

        copy_btn = Button("Copy device name", variant="secondary", size="md", icon_name="copy")
        copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText("BlackHole 2ch"))
        btn_row.addWidget(copy_btn)
        close_btn = Button("Close", variant="primary", size="md")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(close_btn)
        wrap_btns = QFrame()
        wrap_btns.setLayout(btn_row)
        lay.addWidget(wrap_btns)

        dlg.destroyed.connect(self._on_target_dlg_destroyed)
        dlg.show()

    def _on_target_dlg_destroyed(self) -> None:
        self._target_dlg = None

    # ── Signal chain ────────────────────────────────────────────

    def _build_signal_chain_card(self) -> Card:
        card = Card(padding=0)
        head = QFrame()
        hlay = QVBoxLayout(head)
        hlay.setContentsMargins(18, 16, 18, 8)
        hlay.addWidget(CardTitle("Signal chain", sub="Real-time audio routing"))
        card.add(head)
        body = QFrame()
        blay = QVBoxLayout(body)
        blay.setContentsMargins(18, 4, 18, 18)
        self._chain = SignalChain()
        blay.addWidget(self._chain)
        card.add(body)
        return card

    # ── Routing card ────────────────────────────────────────────

    def _build_routing_card(self) -> Card:
        card = Card(padding=18)
        card.add(CardTitle("Routing"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        grid.setColumnMinimumWidth(0, 110)
        grid.setColumnStretch(1, 1)

        self._input_select = Select(icon_name="mic")
        self._input_select.changed.connect(self._on_input_changed)
        self._output_select = Select(icon_name="speaker")
        self._output_select.changed.connect(self._on_output_changed)
        self._model_select = Select(icon_name="cpu")
        self._model_select.changed.connect(self._on_model_changed)

        grid.addWidget(self._label("Input device"), 0, 0)
        grid.addWidget(self._input_select, 0, 1)
        grid.addWidget(self._label("Output device"), 1, 0)
        grid.addWidget(self._output_select, 1, 1)
        grid.addWidget(self._label("RVC model"), 2, 0)
        grid.addWidget(self._model_select, 2, 1)
        card.add_layout(grid)
        return card

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"font-size: 12px; color: {TOKENS['text_sub']}; font-weight: 500;"
        )
        return lbl

    # ── Voice tuning card ───────────────────────────────────────

    def _build_tuning_card(self) -> Card:
        card = Card(padding=18)
        card.add(CardTitle("Voice tuning", sub="Adjust before going live"))

        row = QGridLayout()
        row.setHorizontalSpacing(14)
        row.setVerticalSpacing(0)
        row.setColumnMinimumWidth(0, 110)
        row.setColumnStretch(1, 1)
        row.addWidget(self._label("Pitch shift"), 0, 0)
        self._pitch_slider = Slider(
            value=int(self._get("pitch", 0)),
            minimum=-24,
            maximum=24,
            step=1,
            fmt=lambda v: f"{v:+d} st" if v != 0 else "0 st",
        )
        self._pitch_slider.changed.connect(self._on_pitch_changed)
        row.addWidget(self._pitch_slider, 0, 1)
        card.add_layout(row)

        ctrls = QHBoxLayout()
        ctrls.setContentsMargins(0, 12, 0, 0)
        ctrls.setSpacing(16)
        self._denoise_toggle = Toggle(on=bool(self._get("denoise", True)), label="Denoise (DeepFilterNet)")
        self._denoise_toggle.toggled.connect(self._on_denoise_changed)
        ctrls.addWidget(self._denoise_toggle)
        self._bypass_toggle = Toggle(on=bool(self._get("bypass", False)), label="Bypass model")
        self._bypass_toggle.toggled.connect(self._on_bypass_changed)
        ctrls.addWidget(self._bypass_toggle)
        ctrls.addStretch(1)
        self._block_info = QLabel("Block: 480 · idle")
        self._block_info.setStyleSheet(f"font-size: 11px; color: {TOKENS['text_sub']}; font-weight: 500;")
        ctrls.addWidget(self._block_info)
        card.add_layout(ctrls)
        return card

    # ── Transport card ──────────────────────────────────────────

    def _build_transport_card(self) -> Card:
        card = Card(padding=18)
        self._state_pill = Pill("Stopped", tone="neutral", dot=True)
        card.add(CardTitle("Pipeline", right=self._state_pill))

        self._transport_btn = _TransportButton()
        self._transport_btn.clicked.connect(self._toggle_pipeline)
        card.add(self._transport_btn)

        gap = QFrame()
        gap.setFixedHeight(12)
        card.add(gap)

        self._in_meter = MeterBar("Input level", color=COLOR_OK)
        card.add(self._in_meter)
        gap2 = QFrame()
        gap2.setFixedHeight(10)
        card.add(gap2)
        self._out_meter = MeterBar("Output level", color=ACCENT)
        card.add(self._out_meter)
        return card

    # ── Telemetry card ──────────────────────────────────────────

    def _build_telemetry_card(self) -> Card:
        card = Card(padding=18)
        card.add(CardTitle("Telemetry", sub="End-to-end latency budget"))

        big_row = QHBoxLayout()
        big_row.setSpacing(8)
        self._total_lbl = QLabel("0")
        self._total_lbl.setStyleSheet(
            f"font-family: {FONT_MONO}; font-size: 38px; font-weight: 700; color: {TOKENS['text']};"
            f" letter-spacing: -1px;"
        )
        big_row.addWidget(self._total_lbl)
        unit = QLabel("ms total")
        unit.setStyleSheet(f"font-size: 13px; color: {TOKENS['text_sub']}; font-weight: 600;")
        unit.setAlignment(Qt.AlignmentFlag.AlignBottom)
        big_row.addWidget(unit)
        big_row.addStretch(1)
        self._budget_pill = Pill("—", tone="neutral")
        big_row.addWidget(self._budget_pill, 0, Qt.AlignmentFlag.AlignBottom)
        card.add_layout(big_row)
        card.add_spacing(4)

        self._lat_bars = LatencyBars()
        card.add(self._lat_bars)

        card.add_spacing(14)
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {TOKENS['border']};")
        card.add(divider)
        card.add_spacing(14)

        stats = QGridLayout()
        stats.setHorizontalSpacing(16)
        stats.setVerticalSpacing(10)
        self._stat_dfn = _Stat("Denoise")
        self._stat_rvc = _Stat("Model")
        self._stat_under = _Stat("Underruns")
        self._stat_over = _Stat("Overruns")
        stats.addWidget(self._stat_dfn, 0, 0)
        stats.addWidget(self._stat_rvc, 0, 1)
        stats.addWidget(self._stat_under, 1, 0)
        stats.addWidget(self._stat_over, 1, 1)
        card.add_layout(stats)
        return card

    # ── Population / wiring ─────────────────────────────────────

    def _refresh_devices_and_models(self) -> None:
        self._input_devices = _list_input_devices()
        self._output_devices = _list_output_devices()
        self._models = models_manager.list_voice_models()

        in_names = [name for _, name in self._input_devices] or ["(no input devices)"]
        out_names = [name for _, name in self._output_devices] or ["(no output devices)"]
        model_names = ["<empty>"] + [m.name for m in self._models]

        self._input_select.set_options(in_names)
        self._output_select.set_options(out_names)
        self._model_select.set_options(model_names)

        # restore from settings
        saved_in = self._get("input_device_name")
        if saved_in and saved_in in in_names:
            self._input_select.set_value(saved_in)
        saved_out = self._get("output_device_name")
        if saved_out and saved_out in out_names:
            self._output_select.set_value(saved_out)
        else:
            for _, n in self._output_devices:
                if "BlackHole" in n:
                    self._output_select.set_value(n)
                    break
        saved_model = self._get("rvc_model") or "<empty>"
        if saved_model in model_names:
            self._model_select.set_value(saved_model)
        self._update_chain()

    def refresh_models(self) -> None:
        self._refresh_devices_and_models()

    def _input_index(self) -> int | None:
        name = self._input_select.value()
        for i, n in self._input_devices:
            if n == name:
                return i
        return None

    def _output_index(self) -> int | None:
        name = self._output_select.value()
        for i, n in self._output_devices:
            if n == name:
                return i
        return None

    # ── handlers ────────────────────────────────────────────────

    def _on_input_changed(self, value: str) -> None:
        self._set("input_device_name", value)
        self._update_chain()
        self._restart_if_running()

    def _on_output_changed(self, value: str) -> None:
        self._set("output_device_name", value)
        self._update_chain()
        self._restart_if_running()

    def _on_model_changed(self, value: str) -> None:
        self._set("rvc_model", "" if value == "<empty>" else value)
        self._update_chain()
        self._restart_if_running()

    def _on_pitch_changed(self, value: int) -> None:
        self._set("pitch", value)
        # Live: mutate RVC pitch in place if available; no restart.
        if self._engine is not None:
            try:
                if self._engine._rvc is not None:
                    self._engine._rvc.pitch_shift_semitones = value
            except Exception:
                pass

    def _on_denoise_changed(self, on: bool) -> None:
        self._set("denoise", on)
        # Live toggle on Pipeline; no restart required.
        if self._engine is not None and self._engine._pipeline is not None:
            self._engine._pipeline.denoise = on

    def _on_bypass_changed(self, on: bool) -> None:
        self._set("bypass", on)
        if self._engine is not None and self._engine._pipeline is not None:
            self._engine._pipeline.bypass = on

    def _update_chain(self) -> None:
        running = self._engine.is_running() if self._engine else False
        model = self._model_select.value()
        self._chain.set_state(
            self._input_select.value(),
            "No model" if model == "<empty>" else model,
            self._output_select.value(),
            running,
        )

    # ── transport ───────────────────────────────────────────────

    def _toggle_pipeline(self) -> None:
        if self._engine is not None and self._engine.is_running():
            self._stop_engine()
        else:
            self._start_engine()

    def _build_config(self) -> EngineConfig:
        import torch
        import platform

        use_mps = platform.machine() == "arm64" and torch.backends.mps.is_available()
        model = self._model_select.value()
        rvc_dir = None
        if model and model != "<empty>":
            rvc_dir = Path("models/rvc") / model
        return EngineConfig(
            input_device=self._input_index(),
            output_device=self._output_index(),
            sample_rate=48000,
            blocksize=480,
            device="mps" if use_mps else "cpu",
            denoise=self._denoise_toggle.isOn(),
            bypass=self._bypass_toggle.isOn(),
            rvc_model_dir=rvc_dir,
            rvc_base_dir=Path("models/base"),
            pitch_shift=self._pitch_slider.value(),
            window_ms=384,
            crossfade_ms=128,
        )

    def _build_or_replace_engine(self, cfg: EngineConfig) -> None:
        """Reuse the current Engine if its heavy config (denoise on, rvc model,
        device) matches; otherwise build a new one. Light flags (denoise on/off,
        bypass, pitch_shift) update in-place at runtime, so they don't trigger
        a rebuild here."""
        prev = self._engine
        if prev is not None:
            same_model = prev.config.rvc_model_dir == cfg.rvc_model_dir
            same_device = prev.config.device == cfg.device
            same_denoise_loaded = prev.config.denoise == cfg.denoise
            same_bypass = prev.config.bypass == cfg.bypass
            if same_model and same_device and same_denoise_loaded and same_bypass:
                # Reuse: just update config knobs that AudioIO needs at next start.
                prev.config.input_device = cfg.input_device
                prev.config.output_device = cfg.output_device
                prev.config.pitch_shift = cfg.pitch_shift
                if prev._rvc is not None:
                    try:
                        prev._rvc.pitch_shift_semitones = cfg.pitch_shift
                    except Exception:
                        pass
                if prev._pipeline is not None:
                    prev._pipeline.denoise = cfg.denoise
                    prev._pipeline.bypass = cfg.bypass
                return
        # Heavy change — rebuild.
        if prev is not None:
            prev.stop()
        self._engine = Engine(cfg)

    def _start_engine(self) -> None:
        if self._loading or (self._engine is not None and self._engine.is_running()):
            return
        cfg = self._build_config()
        try:
            self._build_or_replace_engine(cfg)
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Engine error", str(exc))
            return

        self._loading = True
        self._transport_btn.set_loading(True)
        self._state_pill.set_text("Loading…")
        self._state_pill.set_tone("accent")
        self._block_info.setText("Loading models…")

        eng = self._engine

        def worker():
            try:
                eng.prepare()
                eng.start()
            except Exception as exc:
                self._load_signals.failed.emit(str(exc))
                return
            self._load_signals.ready.emit()

        t = threading.Thread(target=worker, name="engine-load", daemon=True)
        t.start()

    def _on_engine_ready(self) -> None:
        self._loading = False
        self._transport_btn.set_loading(False)
        self._transport_btn.set_running(True)
        self._state_pill.set_text("Live")
        self._state_pill.set_tone("success")
        self._block_info.setText("Block: 480 · 48 kHz")
        self._update_chain()

    def _on_engine_failed(self, msg: str) -> None:
        self._loading = False
        self._transport_btn.set_loading(False)
        self._transport_btn.set_running(False)
        self._state_pill.set_text("Stopped")
        self._state_pill.set_tone("neutral")
        self._block_info.setText("Block: 480 · idle")
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.critical(self, "Engine error", msg)

    def _stop_engine(self) -> None:
        if self._engine is None:
            return
        self._engine.stop()
        # Engine kept alive — models stay loaded, no permission re-ask, fast restart.
        self._transport_btn.set_running(False)
        self._state_pill.set_text("Stopped")
        self._state_pill.set_tone("neutral")
        self._block_info.setText("Block: 480 · idle")
        self._update_chain()

    def _restart_if_running(self) -> None:
        if self._engine is not None and self._engine.is_running():
            self._stop_engine()
            self._start_engine()

    # ── polling ─────────────────────────────────────────────────

    def _poll_stats(self) -> None:
        # Detect crashed processor: state says Live but worker died → revert UI.
        if (
            self._engine is not None
            and self._transport_btn._running
            and not self._engine.is_running()
            and not self._loading
        ):
            self._stop_engine()
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Pipeline stopped",
                "The processor thread crashed — likely an RVC model or audio error. "
                "Check the Setup log for details.",
            )
            return
        if self._engine is None:
            self._in_meter.set_value(0)
            self._out_meter.set_value(0)
            self._total_lbl.setText("0")
            self._budget_pill.set_text("—")
            self._budget_pill.set_tone("neutral")
            self._lat_bars.set_values(0, 0, 0)
            self._stat_dfn.set_value("0 ms")
            self._stat_rvc.set_value("0 ms")
            self._stat_under.set_value("0")
            self._stat_over.set_value("0")
            return
        s = self._engine.stats()
        in_pct = s["in_fill"] * 100
        out_pct = s["out_fill"] * 100
        self._in_meter.set_value(in_pct, color=COLOR_ERR if in_pct > 85 else COLOR_OK)
        self._out_meter.set_value(out_pct, color=COLOR_ERR if out_pct > 85 else ACCENT)
        total = int(s["total_ms"])
        self._total_lbl.setText(str(total))
        if total < 80:
            self._budget_pill.set_text("Good")
            self._budget_pill.set_tone("success")
        elif total < 150:
            self._budget_pill.set_text("OK")
            self._budget_pill.set_tone("accent")
        else:
            self._budget_pill.set_text("High")
            self._budget_pill.set_tone("danger")
        io = max(0.0, s["total_ms"] - s["denoise_ms"] - s["rvc_ms"])
        self._lat_bars.set_values(s["denoise_ms"], s["rvc_ms"], io)
        self._stat_dfn.set_value(f"{int(s['denoise_ms'])} ms")
        self._stat_rvc.set_value(f"{int(s['rvc_ms'])} ms")
        self._stat_under.set_value(str(s["underruns"]), danger=s["underruns"] > 0)
        self._stat_over.set_value(str(s["overruns"]), danger=s["overruns"] > 0)


class _TransportButton(QFrame):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._loading = False
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lbl = QLabel("▶  Start pipeline")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._lbl)

    def set_running(self, on: bool) -> None:
        self._running = on
        self._refresh_label()
        self._apply()

    def set_loading(self, on: bool) -> None:
        self._loading = on
        self.setCursor(
            Qt.CursorShape.WaitCursor if on else Qt.CursorShape.PointingHandCursor
        )
        self._refresh_label()
        self._apply()

    def _refresh_label(self) -> None:
        if self._loading:
            self._lbl.setText("⏳  Loading models…")
        elif self._running:
            self._lbl.setText("■  Stop pipeline")
        else:
            self._lbl.setText("▶  Start pipeline")

    def _apply(self) -> None:
        if self._loading:
            grad = f"qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {shade(ACCENT, -20)}, stop:1 {shade(ACCENT, -40)})"
        elif self._running:
            grad = f"qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #FF5A4E, stop:1 #D4382C)"
        else:
            grad = f"qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {ACCENT}, stop:1 {shade(ACCENT, -15)})"
        self.setStyleSheet(
            f"""
            QFrame {{
                background: {grad};
                border-radius: 12px;
                border: none;
            }}
            QFrame:hover {{ background: {grad}; }}
            QLabel {{
                color: #fff;
                font-family: {FONT_UI};
                font-size: 15px;
                font-weight: 700;
                letter-spacing: -0.2px;
            }}
            """
        )

    def mousePressEvent(self, ev):
        if self._loading:
            return
        self.clicked.emit()


class _Stat(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        self._lbl = QLabel(label.upper())
        self._lbl.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 10.5px; font-weight: 500; color: {TOKENS['text_sub']};"
            f" letter-spacing: 0.4px;"
        )
        lay.addWidget(self._lbl)
        self._val = QLabel("—")
        self._val.setStyleSheet(self._value_style(False))
        lay.addWidget(self._val)

    def set_value(self, text: str, danger: bool = False) -> None:
        self._val.setText(text)
        self._val.setStyleSheet(self._value_style(danger))

    @staticmethod
    def _value_style(danger: bool) -> str:
        color = COLOR_ERR if danger else TOKENS["text"]
        return (
            f"font-family: {FONT_MONO}; font-size: 14px; font-weight: 600;"
            f" color: {color}; letter-spacing: -0.2px;"
        )
