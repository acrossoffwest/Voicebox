"""Pipeline screen: signal chain, routing, voice tuning, transport, telemetry."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
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


class PipelineScreen(QWidget):
    config_changed = pyqtSignal()

    def __init__(self, settings_get=lambda k, default=None: default,
                 settings_set=lambda k, v: None, parent=None):
        super().__init__(parent)
        self._engine: Engine | None = None
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
        grid = QGridLayout(self)
        grid.setContentsMargins(24, 24, 24, 24)
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
        self._restart_if_running()

    def _on_denoise_changed(self, on: bool) -> None:
        self._set("denoise", on)
        self._restart_if_running()

    def _on_bypass_changed(self, on: bool) -> None:
        self._set("bypass", on)
        self._restart_if_running()

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
            window_ms=256,
            crossfade_ms=64,
        )

    def _start_engine(self) -> None:
        cfg = self._build_config()
        try:
            self._engine = Engine(cfg)
            self._engine.prepare()
            self._engine.start()
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Engine error", str(exc))
            self._engine = None
            return
        self._transport_btn.set_running(True)
        self._state_pill.set_text("Live")
        self._state_pill.set_tone("success")
        self._block_info.setText("Block: 480 · 48 kHz")
        self._update_chain()

    def _stop_engine(self) -> None:
        if self._engine is None:
            return
        self._engine.stop()
        self._engine = None
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
        self._lbl.setText("■  Stop pipeline" if on else "▶  Start pipeline")
        self._apply()

    def _apply(self) -> None:
        if self._running:
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
