"""Voicebox UI primitives — Card, Button, Pill, Toggle, Select, Slider,
StatusRow, MeterBar, LatencyBars, SignalChain, ModelRow, DropZone, LogView.

All widgets are dark-themed only. QSS where it suffices; QPainter where it
doesn't (Slider center-fill, MeterBar gradient with glow, LatencyBars stacked
proportional fill, SignalChain arrows with marching dashes)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Property,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QTextCursor,
    QTextOption,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui_theme import (
    ACCENT,
    ACCENT_DARK,
    COLOR_ERR,
    COLOR_INFO,
    COLOR_OK,
    COLOR_WARN,
    FONT_MONO,
    FONT_UI,
    TOKENS,
    hex_alpha,
    shade,
)
from ui_icons import icon


# ── Card ────────────────────────────────────────────────────────


class Card(QFrame):
    def __init__(self, padding: int = 18, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setStyleSheet(self._qss())
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(0)

    def add(self, w: QWidget) -> None:
        self._layout.addWidget(w)

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)

    def add_spacing(self, px: int) -> None:
        self._layout.addSpacing(px)

    def add_stretch(self) -> None:
        self._layout.addStretch(1)

    def set_padding(self, padding: int) -> None:
        self._layout.setContentsMargins(padding, padding, padding, padding)

    def _qss(self) -> str:
        return f"""
        QFrame#Card {{
            background-color: {TOKENS['surface']};
            border: 1px solid {TOKENS['border']};
            border-radius: 12px;
        }}
        """


class CardTitle(QWidget):
    def __init__(self, text: str, sub: str = "", right: QWidget | None = None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(10)

        col = QVBoxLayout()
        col.setSpacing(2)
        title = QLabel(text)
        title.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; font-weight: 600;"
            f" color: {TOKENS['text']}; letter-spacing: -0.1px;"
        )
        col.addWidget(title)
        if sub:
            sub_label = QLabel(sub)
            sub_label.setObjectName("CardTitleSub")
            sub_label.setStyleSheet(
                f"font-family: {FONT_UI}; font-size: 11px; font-weight: 500;"
                f" color: {TOKENS['text_sub']};"
            )
            col.addWidget(sub_label)
            self._sub_label = sub_label
        else:
            self._sub_label = None
        layout.addLayout(col, 1)
        if right is not None:
            layout.addWidget(right, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

    def set_sub(self, text: str) -> None:
        if self._sub_label is not None:
            self._sub_label.setText(text)


# ── Button ──────────────────────────────────────────────────────


class Button(QPushButton):
    def __init__(
        self,
        text: str = "",
        variant: str = "secondary",
        size: str = "md",
        icon_name: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(text, parent)
        self.variant = variant
        self.size_ = size
        self._icon_name = icon_name
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._configure_size()
        self._apply_style()
        if icon_name:
            color = "#fff" if variant == "primary" else TOKENS["text"]
            if variant == "danger":
                color = COLOR_ERR
            self.setIcon(icon(icon_name, color=color))
            self.setIconSize(QSize(12, 12))
        if variant == "primary":
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(16)
            shadow.setOffset(0, 4)
            shadow.setColor(QColor(255, 159, 69, 90))
            self.setGraphicsEffect(shadow)

    def _configure_size(self) -> None:
        sizes = {
            "sm": (24, 10, 11),
            "md": (30, 14, 12),
            "lg": (38, 18, 13),
        }
        h, padx, fs = sizes[self.size_]
        self.setFixedHeight(h)
        self._padx = padx
        self._fs = fs
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        fm = QFontMetrics(QFont("Inter", self._fs))
        text_w = fm.horizontalAdvance(self.text())
        icon_w = 18 if self._icon_name else 0
        self.setMinimumWidth(text_w + icon_w + 2 * self._padx)

    def _apply_style(self) -> None:
        radius = {"sm": 6, "md": 7, "lg": 9}[self.size_]
        if self.variant == "primary":
            bg = f"qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {ACCENT}, stop:1 {ACCENT_DARK})"
            color = "#FFFFFF"
            border = "transparent"
        elif self.variant == "ghost":
            bg = "transparent"
            color = TOKENS["text"]
            border = "transparent"
        elif self.variant == "danger":
            bg = hex_alpha(COLOR_ERR, 0.14)
            color = COLOR_ERR
            border = hex_alpha(COLOR_ERR, 0.30)
        else:
            bg = "#2A2A32"
            color = TOKENS["text"]
            border = TOKENS["border_strong"]
        self.setStyleSheet(
            f"""
            QPushButton {{
                background: {bg};
                color: {color};
                border: 1px solid {border};
                border-radius: {radius}px;
                padding: 0 {self._padx}px;
                font-family: {FONT_UI};
                font-size: {self._fs}px;
                font-weight: 600;
                letter-spacing: -0.1px;
                text-align: center;
            }}
            QPushButton:hover {{ background: {shade(ACCENT, -5) if self.variant == 'primary' else '#34343C'}; }}
            QPushButton:disabled {{ opacity: 0.45; color: {TOKENS['text_dim']}; }}
            """
        )


# ── Pill ────────────────────────────────────────────────────────


class Pill(QFrame):
    TONES = {
        "neutral": {"bg": "rgba(255,255,255,0.06)", "fg": TOKENS["text_sub"], "dot": TOKENS["text_dim"]},
        "success": {"bg": hex_alpha(COLOR_OK, 0.16), "fg": COLOR_OK, "dot": COLOR_OK},
        "danger": {"bg": hex_alpha(COLOR_ERR, 0.16), "fg": COLOR_ERR, "dot": COLOR_ERR},
        "accent": {"bg": hex_alpha(ACCENT, 0.18), "fg": ACCENT, "dot": ACCENT},
    }

    def __init__(self, text: str = "", tone: str = "neutral", dot: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("Pill")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._tone = tone
        self._dot = dot
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(5)
        if dot:
            self._dot_widget = QFrame()
            self._dot_widget.setFixedSize(6, 6)
            layout.addWidget(self._dot_widget)
        else:
            self._dot_widget = None
        self._label = QLabel(text)
        layout.addWidget(self._label)
        self._apply()

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def set_tone(self, tone: str) -> None:
        self._tone = tone
        self._apply()

    def _apply(self) -> None:
        s = self.TONES[self._tone]
        self._label.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 11px; font-weight: 600;"
            f" color: {s['fg']}; letter-spacing: -0.05px;"
        )
        self.setStyleSheet(
            f"QFrame#Pill {{ background: {s['bg']}; border-radius: 999px; }}"
        )
        if self._dot_widget is not None:
            self._dot_widget.setStyleSheet(
                f"background: {s['dot']}; border-radius: 3px;"
            )


# ── Toggle ──────────────────────────────────────────────────────


class Toggle(QWidget):
    toggled = Signal(bool)

    def __init__(self, on: bool = False, label: str = "", parent=None):
        super().__init__(parent)
        self._on = on
        self._knob_pos = 14.0 if on else 2.0
        self._anim = QPropertyAnimation(self, b"knob_pos")
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._switch = _ToggleSwitch(self)
        layout.addWidget(self._switch)
        if label:
            lbl = QLabel(label)
            lbl.setStyleSheet(
                f"font-family: {FONT_UI}; font-size: 12px; font-weight: 500; color: {TOKENS['text']};"
            )
            layout.addWidget(lbl)
        layout.addStretch(1)
        self.setFixedHeight(20)

    def isOn(self) -> bool:
        return self._on

    def setOn(self, on: bool) -> None:
        if on == self._on:
            return
        self._on = on
        self._anim.stop()
        self._anim.setStartValue(self._knob_pos)
        self._anim.setEndValue(14.0 if on else 2.0)
        self._anim.start()
        self.toggled.emit(on)

    def toggle(self) -> None:
        self.setOn(not self._on)

    def _get_knob_pos(self) -> float:
        return self._knob_pos

    def _set_knob_pos(self, v: float) -> None:
        self._knob_pos = v
        self._switch.update()

    knob_pos = Property(float, fget=_get_knob_pos, fset=_set_knob_pos)


class _ToggleSwitch(QWidget):
    def __init__(self, parent: Toggle):
        super().__init__(parent)
        self._parent: Toggle = parent
        self.setFixedSize(30, 18)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def mousePressEvent(self, ev):
        self._parent.toggle()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, 30, 18)
        on = self._parent.isOn()
        bg = QColor(ACCENT) if on else QColor("#2D2D36")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(rect, 9, 9)
        knob = QColor("#FFFFFF")
        p.setBrush(knob)
        p.drawEllipse(QPointF(self._parent._knob_pos + 7, 9), 7, 7)


# ── StatusRow ───────────────────────────────────────────────────


class StatusRow(QFrame):
    STATUS_STYLES = {
        "ok":      {"bg": hex_alpha(COLOR_OK, 0.16), "fg": COLOR_OK, "icon": "check"},
        "pending": {"bg": hex_alpha(ACCENT, 0.18),  "fg": ACCENT,   "icon": "spinner"},
        "error":   {"bg": hex_alpha(COLOR_ERR, 0.16),"fg": COLOR_ERR,"icon": "alert"},
        "todo":    {"bg": "rgba(255,255,255,0.05)",  "fg": TOKENS["text_dim"], "icon": None},
    }

    def __init__(self, status: str, label: str, sub: str = "", action: QWidget | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusRow")
        self.setStyleSheet(
            f"QFrame#StatusRow {{ border-bottom: 1px solid {TOKENS['border']}; }}"
        )
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(4, 10, 4, 10)
        self._layout.setSpacing(12)

        self._badge = QLabel()
        self._badge.setFixedSize(22, 22)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._badge)

        col = QVBoxLayout()
        col.setSpacing(1)
        self._label = QLabel(label)
        self._label.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 12px; font-weight: 500;"
            f" color: {TOKENS['text']}; letter-spacing: -0.1px;"
        )
        col.addWidget(self._label)
        self._sub = QLabel(sub)
        self._sub.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 11px; color: {TOKENS['text_sub']};"
        )
        self._sub.setWordWrap(True)
        col.addWidget(self._sub)
        self._layout.addLayout(col, 1)
        self._action: QWidget | None = None
        self.set_action(action)
        self.set_status(status)

    def set_action(self, action: QWidget | None) -> None:
        if self._action is not None:
            self._layout.removeWidget(self._action)
            self._action.setParent(None)
            self._action.deleteLater()
            self._action = None
        if action is not None:
            self._layout.addWidget(action, 0, Qt.AlignmentFlag.AlignRight)
            self._action = action

    def set_status(self, status: str) -> None:
        s = self.STATUS_STYLES[status]
        if s["icon"]:
            self._badge.setPixmap(icon(s["icon"], color=s["fg"]).pixmap(11, 11))
        else:
            self._badge.clear()
        self._badge.setStyleSheet(
            f"background: {s['bg']}; border-radius: 7px; color: {s['fg']};"
        )

    def set_label(self, label: str) -> None:
        self._label.setText(label)

    def set_sub(self, sub: str) -> None:
        self._sub.setText(sub)


# ── Select (button + QMenu popup) ───────────────────────────────


class Select(QPushButton):
    changed = Signal(str)

    def __init__(self, value: str = "", options: list[str] | None = None, icon_name: str | None = None, parent=None):
        super().__init__(parent)
        self._value = value
        self._options = options or []
        self._icon_name = icon_name
        self.setFixedHeight(30)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet(self._qss())
        self.setText(self._render_text())
        if icon_name:
            self.setIcon(icon(icon_name, color=TOKENS["text_sub"]))
            self.setIconSize(QSize(12, 12))
        self.clicked.connect(self._open_menu)

    def _qss(self) -> str:
        return f"""
        QPushButton {{
            background: {TOKENS['surface2']};
            color: {TOKENS['text']};
            border: 1px solid {TOKENS['border_strong']};
            border-radius: 7px;
            padding: 0 10px;
            text-align: left;
            font-family: {FONT_UI};
            font-size: 12px;
            font-weight: 500;
        }}
        QPushButton:hover {{ background: {TOKENS['surface3']}; }}
        QMenu {{
            background: #28282F;
            border: 1px solid {TOKENS['border_strong']};
            border-radius: 8px;
            padding: 4px;
            font-family: {FONT_UI};
            color: {TOKENS['text']};
        }}
        QMenu::item {{
            padding: 6px 10px;
            border-radius: 5px;
        }}
        QMenu::item:selected {{ background: {hex_alpha(ACCENT, 0.18)}; color: {ACCENT}; }}
        """

    def _render_text(self) -> str:
        return self._value or "<empty>"

    def set_options(self, options: list[str]) -> None:
        self._options = options
        if self._value not in options and options:
            self.set_value(options[0])

    def set_value(self, value: str) -> None:
        if value == self._value:
            return
        self._value = value
        self.setText(self._render_text())
        self.changed.emit(value)

    def value(self) -> str:
        return self._value

    def _open_menu(self) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(self._qss())
        for opt in self._options:
            act = menu.addAction(opt)
            act.triggered.connect(lambda _checked=False, v=opt: self.set_value(v))
        menu.exec(self.mapToGlobal(QPoint(0, self.height() + 4)))


# ── Slider (center-zero fill, custom paint) ─────────────────────


class Slider(QWidget):
    changed = Signal(int)

    def __init__(self, value: int, minimum: int, maximum: int, step: int = 1,
                 fmt: Callable[[int], str] | None = None, parent=None):
        super().__init__(parent)
        self._value = value
        self._min = minimum
        self._max = maximum
        self._step = step
        self._fmt = fmt or (lambda v: str(v))
        self.setFixedHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self._track = _SliderTrack(self)
        layout.addWidget(self._track, 1)
        self._label = QLabel(self._fmt(value))
        self._label.setFixedWidth(48)
        self._label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._label.setStyleSheet(
            f"font-family: {FONT_MONO}; font-size: 12px; font-weight: 600; color: {TOKENS['text']};"
        )
        layout.addWidget(self._label)

    def value(self) -> int:
        return self._value

    def set_value(self, value: int) -> None:
        value = max(self._min, min(self._max, value))
        if value == self._value:
            return
        self._value = value
        self._label.setText(self._fmt(value))
        self._track.update()
        self.changed.emit(value)


class _SliderTrack(QWidget):
    def __init__(self, parent: Slider):
        super().__init__(parent)
        self._owner: Slider = parent
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMouseTracking(True)
        self._dragging = False

    def mousePressEvent(self, ev: QMouseEvent):
        self._dragging = True
        self._drag(ev.position().x())

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self._dragging:
            self._drag(ev.position().x())

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self._dragging = False

    def _drag(self, x: float) -> None:
        w = max(1.0, self.width())
        pct = max(0.0, min(1.0, x / w))
        raw = self._owner._min + pct * (self._owner._max - self._owner._min)
        snapped = round(raw / self._owner._step) * self._owner._step
        self._owner.set_value(int(snapped))

    def paintEvent(self, ev: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        cy = h / 2

        track_rect = QRectF(0, cy - 2, w, 4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#26262E"))
        p.drawRoundedRect(track_rect, 2, 2)

        mn, mx, val = self._owner._min, self._owner._max, self._owner._value
        center_pct = (-mn / (mx - mn)) if mn < 0 < mx else 0.0
        val_pct = (val - mn) / (mx - mn)

        # center tick
        if mn < 0 < mx:
            cx = center_pct * w
            p.setBrush(QColor(TOKENS["text_dim"][:-1] + ",128)") if "rgba" in TOKENS["text_dim"] else QColor("#888"))
            p.setBrush(QColor(170, 170, 170, 128))
            p.drawRect(QRectF(cx - 1, cy - 4, 2, 8))

        # fill (from center if bipolar)
        p.setBrush(QColor(ACCENT))
        if mn < 0 < mx:
            cx = center_pct * w
            vx = val_pct * w
            if val >= 0:
                p.drawRoundedRect(QRectF(cx, cy - 2, max(0, vx - cx), 4), 2, 2)
            else:
                p.drawRoundedRect(QRectF(vx, cy - 2, max(0, cx - vx), 4), 2, 2)
        else:
            p.drawRoundedRect(QRectF(0, cy - 2, val_pct * w, 4), 2, 2)

        # knob
        kx = val_pct * w
        p.setPen(QPen(QColor(0, 0, 0, 26), 1))
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QPointF(kx, cy), 8, 8)


# ── MeterBar ────────────────────────────────────────────────────


class MeterBar(QWidget):
    def __init__(self, label: str, color: str, value: float = 0.0, parent=None):
        super().__init__(parent)
        self._value = value
        self._color = color
        self._label_text = label
        self.setFixedHeight(28)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)
        self._lbl = QLabel(label)
        self._lbl.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 11px; color: {TOKENS['text_sub']}; font-weight: 500;"
        )
        top.addWidget(self._lbl)
        top.addStretch(1)
        self._val_lbl = QLabel("0.0%")
        self._val_lbl.setStyleSheet(
            f"font-family: {FONT_MONO}; font-size: 11px; color: {TOKENS['text']};"
        )
        top.addWidget(self._val_lbl)
        layout.addLayout(top)
        self._bar = _MeterBarTrack(self)
        layout.addWidget(self._bar)

    def set_value(self, value: float, value_label: str | None = None, color: str | None = None) -> None:
        self._value = max(0.0, min(100.0, value))
        if color is not None and color != self._color:
            self._color = color
        self._val_lbl.setText(value_label if value_label is not None else f"{self._value:.1f}%")
        self._bar.update()


class _MeterBarTrack(QWidget):
    def __init__(self, parent: MeterBar):
        super().__init__(parent)
        self._owner: MeterBar = parent
        self.setFixedHeight(8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, ev: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#1F1F26"))
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
        pct = self._owner._value / 100.0
        if pct <= 0:
            return
        color = QColor(self._owner._color)
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0, color)
        grad.setColorAt(1, QColor(shade(self._owner._color, 10)))
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(QRectF(0, 0, w * pct, h), h / 2, h / 2)


# ── LatencyBars ─────────────────────────────────────────────────


class LatencyBars(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dfn = 0.0
        self._rvc = 0.0
        self._io = 0.0
        self._total = 1.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._bar = _LatencyBarTrack(self)
        layout.addWidget(self._bar)
        legend = QHBoxLayout()
        legend.setSpacing(12)
        for label, color in (("I/O", COLOR_INFO), ("Denoise", COLOR_OK), ("Model", ACCENT)):
            box = QFrame()
            box.setFixedSize(8, 8)
            box.setStyleSheet(f"background: {color}; border-radius: 2px;")
            row = QHBoxLayout()
            row.setSpacing(5)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(box)
            lbl = QLabel(label)
            lbl.setStyleSheet(
                f"font-family: {FONT_UI}; font-size: 11px; color: {TOKENS['text_sub']}; font-weight: 500;"
            )
            row.addWidget(lbl)
            wrap = QWidget()
            wrap.setLayout(row)
            legend.addWidget(wrap)
        legend.addStretch(1)
        layout.addLayout(legend)

    def set_values(self, dfn: float, rvc: float, io: float) -> None:
        self._dfn = max(0.0, dfn)
        self._rvc = max(0.0, rvc)
        self._io = max(0.0, io)
        self._total = max(1.0, self._dfn + self._rvc + self._io)
        self._bar.update()


class _LatencyBarTrack(QWidget):
    def __init__(self, parent: LatencyBars):
        super().__init__(parent)
        self._owner: LatencyBars = parent
        self.setFixedHeight(10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, ev: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#1F1F26"))
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
        x = 0.0
        for val, color in (
            (self._owner._io, COLOR_INFO),
            (self._owner._dfn, COLOR_OK),
            (self._owner._rvc, ACCENT),
        ):
            seg = (val / self._owner._total) * w
            if seg <= 0:
                continue
            p.setBrush(QColor(color))
            p.drawRect(QRectF(x, 0, seg, h))
            x += seg


# ── SignalChain ─────────────────────────────────────────────────


class SignalChain(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._input = ""
        self._model = ""
        self._output = ""
        self._running = False
        self._dash_offset = 0.0
        self.setFixedHeight(120)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def set_state(self, input_: str, model: str, output: str, running: bool) -> None:
        self._input = input_
        self._model = model
        self._output = output
        was = self._running
        self._running = running
        if running and not was:
            self._timer.start()
        elif not running and was:
            self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._dash_offset = (self._dash_offset + 1.5) % 12.0
        self.update()

    def paintEvent(self, ev: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        # Layout: 3 nodes evenly spaced with arrows between
        node_w = (w - 48) / 3
        gap = 24
        node_h = h - 8
        names = [self._input, self._model, self._output]
        labels = ["INPUT", "MODEL", "OUTPUT"]
        icon_names = ["mic", "sparkles", "speaker"]
        hot = [self._running, self._running and self._model != "" and self._model != "<empty>", self._running]

        for i in range(3):
            x = i * (node_w + gap)
            self._draw_node(p, x, 4, node_w, node_h, labels[i], names[i] or "—", hot[i])

        # Arrows between
        for i in range(2):
            x1 = (i + 1) * node_w + i * gap
            y = h / 2
            self._draw_arrow(p, x1, y, x1 + gap, y, self._running)

    def _draw_node(self, p: QPainter, x: float, y: float, w: float, h: float, label: str, name: str, hot: bool):
        p.setPen(Qt.PenStyle.NoPen)
        rect = QRectF(x, y, w, h)
        bg = QColor(255, 255, 255, 6) if not hot else QColor(255, 159, 69, 30)
        p.setBrush(bg)
        path = QPainterPath()
        path.addRoundedRect(rect, 10, 10)
        p.drawPath(path)
        p.setPen(QPen(QColor(255, 255, 255, 18), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # icon tile
        tile_x = x + (w - 38) / 2
        tile_y = y + 14
        tile_rect = QRectF(tile_x, tile_y, 38, 38)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(ACCENT) if hot else QColor(255, 255, 255, 16))
        if hot:
            c = QColor(ACCENT)
            c.setAlpha(50)
            p.setBrush(c)
        p.drawRoundedRect(tile_rect, 9, 9)

        # icon (centered using qtawesome pixmap)
        try:
            ic_color = ACCENT if hot else TOKENS["text"]
            pix = icon({"INPUT": "mic", "MODEL": "sparkles", "OUTPUT": "speaker"}[label], color=ic_color).pixmap(18, 18)
            p.drawPixmap(int(tile_x + 10), int(tile_y + 10), pix)
        except Exception:
            pass

        # uppercase label
        p.setPen(QColor(TOKENS["text_sub"][:-1] + ",1)") if False else QColor(180, 180, 188))
        p.setPen(QColor(180, 180, 188))
        font = QFont("Inter", 8)
        font.setWeight(QFont.Weight.Medium)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
        p.setFont(font)
        p.drawText(QRectF(x, y + 60, w, 14), Qt.AlignmentFlag.AlignCenter, label)

        # name (truncated)
        p.setPen(QColor(TOKENS["text"]))
        font2 = QFont("Inter", 10)
        font2.setWeight(QFont.Weight.DemiBold)
        p.setFont(font2)
        fm = QFontMetrics(font2)
        text = fm.elidedText(name, Qt.TextElideMode.ElideRight, int(w - 12))
        p.drawText(QRectF(x, y + 78, w, 16), Qt.AlignmentFlag.AlignCenter, text)

    def _draw_arrow(self, p: QPainter, x1: float, y: float, x2: float, y2: float, running: bool):
        color = QColor(ACCENT) if running else QColor(160, 160, 168, 90)
        pen = QPen(color, 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if running:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([3, 3])
            pen.setDashOffset(self._dash_offset)
        p.setPen(pen)
        p.drawLine(QPointF(x1 + 4, y), QPointF(x2 - 8, y))
        # arrowhead
        p.setPen(QPen(color, 2))
        p.drawLine(QPointF(x2 - 8, y), QPointF(x2 - 14, y - 5))
        p.drawLine(QPointF(x2 - 8, y), QPointF(x2 - 14, y + 5))


# ── ModelRow ────────────────────────────────────────────────────


class ModelRow(QFrame):
    removed = Signal(str)

    def __init__(self, name: str, files_label: str, size_label: str, full: bool, parent=None):
        super().__init__(parent)
        self.name = name
        self.setObjectName("ModelRow")
        # Lock height so the scroll viewport can't stretch rows via spare space.
        self.setFixedHeight(48)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            f"""
            QFrame#ModelRow {{
                background: rgba(255,255,255,0.03);
                border: 1px solid {TOKENS['border']};
                border-radius: 8px;
            }}
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        ic = QLabel()
        ic.setFixedSize(26, 26)
        ic.setStyleSheet(
            f"background: {hex_alpha(ACCENT, 0.18)}; border-radius: 6px;"
        )
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic.setPixmap(icon("cpu", color=ACCENT).pixmap(13, 13))
        layout.addWidget(ic)

        col = QVBoxLayout()
        col.setSpacing(1)
        title = QLabel(name)
        title.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 12px; font-weight: 600; color: {TOKENS['text']};"
        )
        col.addWidget(title)
        meta = QLabel(f"{files_label} · {size_label}")
        meta.setStyleSheet(
            f"font-family: {FONT_MONO}; font-size: 10px; color: {TOKENS['text_sub']};"
        )
        col.addWidget(meta)
        layout.addLayout(col, 1)

        pill = Pill(files_label, tone="success" if full else "neutral")
        layout.addWidget(pill)

        btn = QPushButton()
        btn.setIcon(icon("trash", color=TOKENS["text_dim"]))
        btn.setIconSize(QSize(12, 12))
        btn.setFixedSize(22, 22)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 5px; }"
            "QPushButton:hover { background: rgba(255,90,78,0.16); }"
        )
        btn.clicked.connect(lambda: self.removed.emit(self.name))
        layout.addWidget(btn)


# ── DropZone ────────────────────────────────────────────────────


class DropZone(QFrame):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self._over = False
        self._apply_style()
        self.setAcceptDrops(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        ic = QLabel()
        ic.setFixedSize(30, 30)
        ic.setStyleSheet(f"background: {hex_alpha(ACCENT, 0.16)}; border-radius: 7px;")
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic.setPixmap(icon("upload", color=ACCENT).pixmap(14, 14))
        layout.addWidget(ic)
        col = QVBoxLayout()
        col.setSpacing(1)
        head = QLabel(
            'Drop <code style="background:#26262C;padding:0 4px;border-radius:3px;">.pth</code> '
            'or <code style="background:#26262C;padding:0 4px;border-radius:3px;">.index</code> here'
        )
        head.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 12px; font-weight: 600; color: {TOKENS['text']};"
        )
        head.setTextFormat(Qt.TextFormat.RichText)
        col.addWidget(head)
        sub = QLabel("Files are sorted into voice folders automatically")
        sub.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 11px; color: {TOKENS['text_sub']};"
        )
        col.addWidget(sub)
        layout.addLayout(col, 1)

    def _apply_style(self) -> None:
        if self._over:
            border = ACCENT
            bg = hex_alpha(ACCENT, 0.08)
        else:
            border = TOKENS["border_strong"]
            bg = "rgba(255,255,255,0.02)"
        self.setStyleSheet(
            f"""
            QFrame#DropZone {{
                background: {bg};
                border: 2px dashed {border};
                border-radius: 9px;
            }}
            """
        )

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            self._over = True
            self._apply_style()
            ev.acceptProposedAction()

    def dragLeaveEvent(self, ev):
        self._over = False
        self._apply_style()

    def dropEvent(self, ev):
        urls = ev.mimeData().urls()
        paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        paths = [p for p in paths if p.suffix.lower() in (".pth", ".index")]
        if paths:
            self.files_dropped.emit(paths)
        self._over = False
        self._apply_style()


# ── LogView ─────────────────────────────────────────────────────


class LogView(QFrame):
    LEVEL_COLORS = {
        "ok": COLOR_OK,
        "warn": COLOR_WARN,
        "err": COLOR_ERR,
        "info": TOKENS["log_text"],
        "default": TOKENS["log_text"],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LogView")
        self.setStyleSheet(
            f"""
            QFrame#LogView {{
                background: {TOKENS['log_bg']};
                border: none;
                border-top: 1px solid {TOKENS['border']};
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(2)
        self._view = QPlainTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setFrameStyle(QFrame.Shape.NoFrame)
        self._view.setStyleSheet(
            f"background: transparent; color: {TOKENS['log_text']};"
            f" font-family: {FONT_MONO}; font-size: 11px; line-height: 17px;"
        )
        self._view.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        layout.addWidget(self._view)
        # blinking $ prompt
        prompt = QLabel()
        prompt.setStyleSheet(
            f"color: {COLOR_OK}; font-family: {FONT_MONO}; font-size: 11px;"
        )
        prompt.setText("$ ▮")
        self._prompt = prompt
        layout.addWidget(prompt)
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._blink)
        self._blink_timer.start()
        self._cursor_visible = True
        self.setMaximumHeight(220)

    def _blink(self) -> None:
        self._cursor_visible = not self._cursor_visible
        self._prompt.setText("$ ▮" if self._cursor_visible else "$ ")

    def append(self, line: str, level: str = "default", timestamp: str | None = None) -> None:
        color = self.LEVEL_COLORS.get(level, TOKENS["log_text"])
        ts = f'<span style="color: rgba(255,255,255,0.28); margin-right: 8px;">{timestamp}</span> ' if timestamp else ""
        html = f'<div><span style="color: {color};">{ts}{self._escape(line)}</span></div>'
        self._view.appendHtml(html)
        cur = self._view.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        self._view.setTextCursor(cur)

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def clear(self) -> None:
        self._view.clear()
