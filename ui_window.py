"""Main window: frameless 1080x720, sidebar + toolbar + screen switcher."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

import app_paths
from ui_pipeline import PipelineScreen
from ui_setup import SetupScreen
from ui_theme import ACCENT, FONT_UI, TOKENS, hex_alpha, shade
from ui_widgets import Pill
from ui_icons import icon
from PySide6.QtGui import QIcon

def _settings_file() -> Path:
    import app_paths
    return app_paths.settings_path()


def _load_settings() -> dict:
    p = _settings_file()
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_settings(data: dict) -> None:
    p = _settings_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


class _TrafficLights(QWidget):
    """Three macOS-style buttons. Show ×/−/+ glyphs when the group is hovered."""

    def __init__(self, on_close, on_min, on_max, parent=None):
        super().__init__(parent)
        self.setFixedSize(76, 22)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 5, 0, 5)
        lay.setSpacing(8)
        self._buttons: list[_TrafficLightBtn] = []
        for color, cb, glyph in (
            ("#ff5f57", on_close, "close"),
            ("#febc2e", on_min, "minimize"),
            ("#28c840", on_max, "maximize"),
        ):
            btn = _TrafficLightBtn(color, glyph)
            btn.clicked.connect(cb)
            lay.addWidget(btn)
            self._buttons.append(btn)
        lay.addStretch(1)
        self.setMouseTracking(True)

    def enterEvent(self, ev):
        for b in self._buttons:
            b.set_group_hover(True)

    def leaveEvent(self, ev):
        for b in self._buttons:
            b.set_group_hover(False)


class _TrafficLightBtn(QPushButton):
    GLYPHS = {
        "close": "✕",
        "minimize": "−",
        "maximize": "+",
    }

    def __init__(self, color: str, glyph: str, parent=None):
        super().__init__(parent)
        self._color = color
        self._glyph = glyph
        self._group_hover = False
        self.setFixedSize(12, 12)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_group_hover(self, hov: bool) -> None:
        if hov == self._group_hover:
            return
        self._group_hover = hov
        self.update()

    def paintEvent(self, ev):
        from PySide6.QtCore import QRectF
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, 11.0, 11.0)
        p.setBrush(QColor(self._color))
        p.setPen(QPen(QColor(0, 0, 0, 64), 0.5))
        p.drawEllipse(rect)
        if self._group_hover:
            p.setPen(QPen(QColor(0, 0, 0, 200), 1.2))
            font = QFont("Helvetica", 8)
            font.setWeight(QFont.Weight.Bold)
            p.setFont(font)
            p.drawText(QRectF(0, 0, 12, 12), Qt.AlignmentFlag.AlignCenter, self.GLYPHS[self._glyph])


class Sidebar(QFrame):
    selected = Signal(str)

    def __init__(self, on_close, on_min, on_max, parent=None):
        super().__init__(parent)
        self.setFixedWidth(230)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"""
            Sidebar {{
                background: {TOKENS['sidebar_bg']};
                border: none;
                border-right: 1px solid {TOKENS['border']};
                border-top-left-radius: 12px;
                border-bottom-left-radius: 12px;
            }}
            """
        )
        self._drag_origin: QPoint | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # traffic lights
        tl_wrap = QFrame()
        tl_lay = QHBoxLayout(tl_wrap)
        tl_lay.setContentsMargins(16, 12, 16, 8)
        tl_lay.addWidget(_TrafficLights(on_close, on_min, on_max))
        tl_lay.addStretch(1)
        lay.addWidget(tl_wrap)

        # brand
        brand_wrap = QFrame()
        brand_lay = QHBoxLayout(brand_wrap)
        brand_lay.setContentsMargins(16, 6, 16, 14)
        brand_lay.setSpacing(10)
        app_icon = QLabel()
        app_icon.setFixedSize(28, 28)
        app_icon.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            f" stop:0 {ACCENT}, stop:1 {shade(ACCENT, -25)});"
            f" border-radius: 7px;"
        )
        app_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_icon.setPixmap(icon("wave", color="#FFFFFF").pixmap(15, 15))
        brand_lay.addWidget(app_icon)
        col = QVBoxLayout()
        col.setSpacing(1)
        name = QLabel("Voicebox")
        name.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; font-weight: 700; color: {TOKENS['text']};"
            f" letter-spacing: -0.2px;"
        )
        col.addWidget(name)
        sub = QLabel("RVC Studio · 0.4.1")
        sub.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 10.5px; font-weight: 500; color: {TOKENS['text_sub']};"
        )
        col.addWidget(sub)
        brand_lay.addLayout(col, 1)
        lay.addWidget(brand_wrap)

        # workspace header
        wh = QLabel("WORKSPACE")
        wh.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 10.5px; font-weight: 600;"
            f" color: {hex_alpha('#EEEEF2', 0.42)}; padding: 4px 22px;"
            f" letter-spacing: 0.6px;"
        )
        lay.addWidget(wh)

        # nav items — Voice Pipeline first per UX intent
        self._nav_wrap = QFrame()
        nav_lay = QVBoxLayout(self._nav_wrap)
        nav_lay.setContentsMargins(10, 2, 10, 2)
        nav_lay.setSpacing(1)
        self._nav_buttons: dict[str, _NavButton] = {}
        for key, label, icon_name in (
            ("pipeline", "Voice Pipeline", "wave"),
            ("setup", "Setup", "settings"),
        ):
            btn = _NavButton(label, icon_name, key)
            btn.clicked.connect(lambda _checked=False, k=key: self.selected.emit(k))
            self._nav_buttons[key] = btn
            nav_lay.addWidget(btn)
        lay.addWidget(self._nav_wrap)

        lay.addStretch(1)

        # footer
        self._footer = _SidebarFooter()
        lay.addWidget(self._footer)

    def set_active(self, key: str) -> None:
        for k, btn in self._nav_buttons.items():
            btn.set_active(k == key)

    def set_badge(self, key: str, text: str | None, tone: str | None) -> None:
        btn = self._nav_buttons.get(key)
        if btn is not None:
            btn.set_badge(text, tone)

    def set_locked(self, key: str, locked: bool) -> None:
        btn = self._nav_buttons.get(key)
        if btn is not None:
            btn.set_locked(locked)

    def set_footer_state(self, running: bool, detail: str) -> None:
        self._footer.set_state(running, detail)

    # window drag support
    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = ev.globalPosition().toPoint()

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self._drag_origin is None or self.window() is None:
            return
        if not (ev.buttons() & Qt.MouseButton.LeftButton):
            return
        gp = ev.globalPosition().toPoint()
        delta = gp - self._drag_origin
        self.window().move(self.window().pos() + delta)
        self._drag_origin = gp

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self._drag_origin = None


class _NavButton(QPushButton):
    def __init__(self, label: str, icon_name: str, key: str, parent=None):
        super().__init__(label, parent)
        self._key = key
        self._icon_name = icon_name
        self._label_text = label
        self._active = False
        self._locked = False
        self._badge: str | None = None
        self._badge_tone: str | None = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(30)
        self._apply()
        self.setIcon(icon(icon_name, color=TOKENS["text_sub"]))
        self.setIconSize(QSize(14, 14))

    def set_active(self, active: bool) -> None:
        self._active = active
        self._apply()
        color = ACCENT if active else (TOKENS["text_dim"] if self._locked else TOKENS["text_sub"])
        self.setIcon(icon(self._icon_name, color=color))

    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        # keep clickable so we can show a confirm dialog; only style changes
        self.setCursor(
            Qt.CursorShape.ForbiddenCursor if locked else Qt.CursorShape.PointingHandCursor
        )
        self._apply()
        color = TOKENS["text_dim"] if locked else (
            ACCENT if self._active else TOKENS["text_sub"]
        )
        self.setIcon(icon(self._icon_name, color=color))

    def set_badge(self, text: str | None, tone: str | None) -> None:
        self._badge = text
        self._badge_tone = tone
        self._apply()
        if text:
            self.setText(f"{self.text().split('  ')[0]}  {text}")

    def _apply(self) -> None:
        bg = hex_alpha(ACCENT, 0.16) if self._active else "transparent"
        color = ACCENT if self._active else TOKENS["text"]
        weight = 600 if self._active else 500
        self.setStyleSheet(
            f"""
            QPushButton {{
                background: {bg};
                color: {color};
                border: none;
                border-radius: 7px;
                padding: 7px 10px;
                text-align: left;
                font-family: {FONT_UI};
                font-size: 13px;
                font-weight: {weight};
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.04); }}
            """
        )


class _SidebarFooter(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background: transparent; border-top: 1px solid {TOKENS['border']}; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)
        self._dot = QFrame()
        self._dot.setFixedSize(8, 8)
        self._dot.setStyleSheet(f"background: {TOKENS['text_dim']}; border-radius: 4px;")
        lay.addWidget(self._dot)
        col = QVBoxLayout()
        col.setSpacing(0)
        self._head = QLabel("Pipeline idle")
        self._head.setStyleSheet(
            f"font-size: 11.5px; font-weight: 600; color: {TOKENS['text']};"
        )
        col.addWidget(self._head)
        self._sub = QLabel("Press Start")
        self._sub.setStyleSheet(
            f"font-size: 10.5px; color: {TOKENS['text_sub']};"
        )
        col.addWidget(self._sub)
        lay.addLayout(col, 1)

    def set_state(self, running: bool, detail: str) -> None:
        if running:
            self._dot.setStyleSheet("background: #34D399; border-radius: 4px;")
            self._head.setText("Pipeline live")
        else:
            self._dot.setStyleSheet(f"background: {TOKENS['text_dim']}; border-radius: 4px;")
            self._head.setText("Pipeline idle")
        self._sub.setText(detail or ("Press Start" if not running else ""))


class TopBar(QFrame):
    """Unified top bar: traffic lights, brand, segmented tabs, right-side status."""

    selected = Signal(str)

    def __init__(self, on_close, on_minimize, on_fullscreen, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"""
            TopBar {{
                background: {TOKENS['toolbar_bg']};
                border: none;
                border-bottom: 1px solid {TOKENS['border']};
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }}
            """
        )
        self._drag_origin: QPoint | None = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 18, 0)
        lay.setSpacing(14)

        lay.addWidget(_TrafficLights(on_close, on_minimize, on_fullscreen))

        # Brand block
        app_icon = QLabel()
        app_icon.setFixedSize(22, 22)
        app_icon.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            f" stop:0 {ACCENT}, stop:1 {shade(ACCENT, -25)});"
            f" border-radius: 6px;"
        )
        app_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_icon.setPixmap(icon("wave", color="#FFFFFF").pixmap(13, 13))
        lay.addWidget(app_icon)

        brand_col = QVBoxLayout()
        brand_col.setSpacing(0)
        name_lbl = QLabel("Voicebox")
        name_lbl.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 13px; font-weight: 700;"
            f" color: {TOKENS['text']}; letter-spacing: -0.2px;"
        )
        brand_col.addWidget(name_lbl)
        ver_lbl = QLabel("RVC Studio · 0.4.1")
        ver_lbl.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 10.5px; color: {TOKENS['text_sub']};"
        )
        brand_col.addWidget(ver_lbl)
        lay.addLayout(brand_col)

        # Segmented tab buttons in the middle.
        lay.addStretch(1)
        tabs_wrap = QFrame()
        tabs_wrap.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        tabs_wrap.setStyleSheet(
            f"QFrame {{ background: rgba(255,255,255,0.04); border-radius: 9px; }}"
        )
        tabs_lay = QHBoxLayout(tabs_wrap)
        tabs_lay.setContentsMargins(3, 3, 3, 3)
        tabs_lay.setSpacing(2)
        self._tab_buttons: dict[str, _SegmentedTabButton] = {}
        for key, label in (("setup", "Setup"), ("pipeline", "Voice Pipeline")):
            btn = _SegmentedTabButton(label, key)
            btn.clicked.connect(lambda _checked=False, k=key: self.selected.emit(k))
            self._tab_buttons[key] = btn
            tabs_lay.addWidget(btn)
        lay.addWidget(tabs_wrap)
        lay.addStretch(1)

        self._right = QLabel("")
        self._right.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 11px; color: {TOKENS['text_sub']};"
        )
        lay.addWidget(self._right)

    def set_active(self, key: str) -> None:
        for k, btn in self._tab_buttons.items():
            btn.set_active(k == key)

    def set_locked(self, key: str, locked: bool) -> None:
        btn = self._tab_buttons.get(key)
        if btn is not None:
            btn.set_locked(locked)

    def set_right(self, text: str) -> None:
        self._right.setText(text)

    # window drag via the top bar
    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = ev.globalPosition().toPoint()

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self._drag_origin is None or self.window() is None:
            return
        if not (ev.buttons() & Qt.MouseButton.LeftButton):
            return
        gp = ev.globalPosition().toPoint()
        delta = gp - self._drag_origin
        self.window().move(self.window().pos() + delta)
        self._drag_origin = gp

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self._drag_origin = None

    def mouseDoubleClickEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            win = self.window()
            if win is not None:
                if win.isMaximized():
                    win.showNormal()
                else:
                    win.showMaximized()


class _SegmentedTabButton(QPushButton):
    def __init__(self, label: str, key: str, parent=None):
        super().__init__(label, parent)
        self._key = key
        self._active = False
        self._locked = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(28)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._apply()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._apply()

    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        self.setCursor(
            Qt.CursorShape.ForbiddenCursor if locked else Qt.CursorShape.PointingHandCursor
        )
        self._apply()

    def _apply(self) -> None:
        if self._active:
            bg = TOKENS["surface2"]
            color = ACCENT
            weight = 600
        else:
            bg = "transparent"
            color = TOKENS["text_dim"] if self._locked else TOKENS["text"]
            weight = 500
        self.setStyleSheet(
            f"""
            QPushButton {{
                background: {bg};
                color: {color};
                border: none;
                border-radius: 7px;
                padding: 0 14px;
                font-family: {FONT_UI};
                font-size: 12px;
                font-weight: {weight};
                letter-spacing: -0.05px;
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.06); }}
            """
        )


class Toolbar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_origin: QPoint | None = None
        self.setFixedHeight(56)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"""
            Toolbar {{
                background: {TOKENS['toolbar_bg']};
                border: none;
                border-bottom: 1px solid {TOKENS['border']};
                border-top-right-radius: 12px;
            }}
            """
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 12, 24, 10)
        lay.setSpacing(10)
        lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        col = QVBoxLayout()
        col.setSpacing(2)
        self._title = QLabel("")
        self._title.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 14px; font-weight: 600;"
            f" color: {TOKENS['text']}; letter-spacing: -0.1px;"
        )
        col.addWidget(self._title)
        self._sub = QLabel("")
        self._sub.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 11.5px; font-weight: 500;"
            f" color: {TOKENS['text_sub']};"
        )
        col.addWidget(self._sub)
        lay.addLayout(col, 1)
        self._right = QLabel("")
        self._right.setStyleSheet(
            f"font-family: {FONT_UI}; font-size: 11px; color: {TOKENS['text_sub']};"
        )
        lay.addWidget(self._right)

    def set_title(self, title: str, subtitle: str = "") -> None:
        self._title.setText(title)
        self._sub.setText(subtitle)

    def set_right(self, text: str) -> None:
        self._right.setText(text)

    # window drag support — same handler as Sidebar
    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = ev.globalPosition().toPoint()

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self._drag_origin is None or self.window() is None:
            return
        if not (ev.buttons() & Qt.MouseButton.LeftButton):
            return
        gp = ev.globalPosition().toPoint()
        delta = gp - self._drag_origin
        self.window().move(self.window().pos() + delta)
        self._drag_origin = gp

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self._drag_origin = None

    def mouseDoubleClickEvent(self, ev: QMouseEvent):
        if ev.button() == Qt.MouseButton.LeftButton:
            win = self.window()
            if win is not None:
                if win.isMaximized():
                    win.showNormal()
                else:
                    win.showMaximized()


class MainWindow(QMainWindow):
    MIN_W = 1242  # 1080 * 1.15
    MIN_H = 805   # 700 * 1.15
    EDGE = 6  # px hot zone for edge resize

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(self.MIN_W, self.MIN_H)
        self.resize(self.MIN_W, self.MIN_H)
        self.setMouseTracking(True)
        self._resize_edge: str | None = None
        self._resize_origin = None
        self._resize_geom = None

        self._settings = _load_settings()
        self._save_timer = QTimer(self)
        self._save_timer.setInterval(500)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._do_save)

        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)

        panel = _Panel()
        self._panel = panel
        root_lay.addWidget(panel)

        # Resize grip in bottom-right corner of the central widget.
        self._size_grip = QSizeGrip(root)
        self._size_grip.setFixedSize(16, 16)
        self._size_grip.setStyleSheet(
            "QSizeGrip { background: transparent; border: none; }"
        )
        self._size_grip.raise_()

        outer_col = QVBoxLayout(panel)
        outer_col.setContentsMargins(0, 0, 0, 0)
        outer_col.setSpacing(0)

        self._topbar = TopBar(
            on_close=self.close,
            on_minimize=self.showMinimized,
            on_fullscreen=self._toggle_fullscreen,
        )
        outer_col.addWidget(self._topbar)

        self._stack = QStackedWidget()
        self._stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._stack.setStyleSheet(
            f"""
            QStackedWidget {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {TOKENS['panel_bg_top']}, stop:1 {TOKENS['panel_bg_bot']});
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }}
            """
        )

        self._setup_screen = SetupScreen()
        self._pipeline_screen = PipelineScreen(
            settings_get=self._settings.get,
            settings_set=self._on_setting_change,
        )
        self._setup_screen.state_changed.connect(self._pipeline_screen.refresh_models)
        self._stack.addWidget(self._setup_screen)
        self._stack.addWidget(self._pipeline_screen)

        outer_col.addWidget(self._stack, 1)

        self._topbar.selected.connect(self._on_select_tab)
        self._setup_screen.ready_changed.connect(self._on_ready_changed)
        # initial lock state based on current readiness
        self._on_ready_changed(self._setup_screen.is_ready())
        last = self._settings.get("last_tab", "setup")
        if last == "pipeline" and not self._setup_screen.is_ready():
            last = "setup"
        self._on_select_tab(last)

        self._footer_timer = QTimer(self)
        self._footer_timer.setInterval(500)
        self._footer_timer.timeout.connect(self._update_footer)
        self._footer_timer.start()

        self._force_quit = False
        self._build_tray_icon()

    def _on_ready_changed(self, ready: bool) -> None:
        self._topbar.set_locked("pipeline", not ready)
        if not ready and self._stack.currentWidget() is self._pipeline_screen:
            self._on_select_tab("setup")

    def _on_setting_change(self, key, value) -> None:
        self._settings[key] = value
        self._save_timer.start()

    def _do_save(self) -> None:
        try:
            _save_settings(self._settings)
        except Exception:
            pass

    def _on_select_tab(self, key: str) -> None:
        if key == "pipeline" and not self._setup_screen.is_ready():
            self._show_locked_dialog()
            self._topbar.set_active("setup")
            return
        self._topbar.set_active(key)
        if key == "setup":
            self._stack.setCurrentWidget(self._setup_screen)
            self._topbar.set_right("")
        else:
            self._stack.setCurrentWidget(self._pipeline_screen)
            self._refresh_pipeline_toolbar()
        self._on_setting_change("last_tab", key)

    def _show_locked_dialog(self) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Setup incomplete")
        box.setText("Finish the Setup checklist first.")
        box.setInformativeText(
            "Voice Pipeline becomes available once all system requirements show green checks. "
            "Open Setup, satisfy the remaining items, then come back here."
        )
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _refresh_pipeline_toolbar(self) -> None:
        try:
            engine = self._pipeline_screen._engine
            if engine is None:
                self._topbar.set_right("Idle")
                return
            s = engine.stats()
            self._topbar.set_right(
                f"in {s['in_fill'] * 100:.1f}% · out {s['out_fill'] * 100:.1f}% · {int(s['total_ms'])} ms"
            )
        except Exception:
            self._topbar.set_right("")

    def _update_footer(self) -> None:
        if self._stack.currentWidget() is self._pipeline_screen:
            self._refresh_pipeline_toolbar()

    def _toggle_max(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if hasattr(self, "_size_grip"):
            w, h = self.width(), self.height()
            self._size_grip.move(w - self._size_grip.width() - 4, h - self._size_grip.height() - 4)

    def _build_tray_icon(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            return
        # Use a dedicated monochrome waveform glyph for the menu bar — that's
        # the macOS convention (Slack, Spotify, etc.). The colorful cube .icns
        # stays the Dock / window icon. Idle = neutral, Live = accent red.
        self._tray_icon_idle = icon("wave", color="#E6E6EA")
        self._tray_icon_live = icon("wave", color="#FF3B30")
        self._tray = QSystemTrayIcon(self._tray_icon_idle, self)
        self._tray.setToolTip("Voicebox")

        menu = QMenu()
        act_show = menu.addAction("Show Voicebox")
        act_show.triggered.connect(self._show_from_tray)
        self._act_toggle_pipe = menu.addAction("Start pipeline")
        self._act_toggle_pipe.triggered.connect(self._tray_toggle_pipeline)
        menu.addSeparator()
        act_quit = menu.addAction("Quit Voicebox")
        act_quit.triggered.connect(self._quit_from_tray)
        self._tray.setContextMenu(menu)
        # Don't connect `activated` — on macOS clicking the icon already
        # opens the context menu via setContextMenu. Window only shows via
        # the explicit "Show Voicebox" item.
        # Keep the tray menu's pipeline label in sync with state.
        self._tray_label_timer = QTimer(self)
        self._tray_label_timer.setInterval(500)
        self._tray_label_timer.timeout.connect(self._refresh_tray_label)
        self._tray_label_timer.start()
        self._tray.show()

    def _show_from_tray(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _tray_toggle_pipeline(self) -> None:
        try:
            if self._pipeline_screen._engine is not None and self._pipeline_screen._engine.is_running():
                self._pipeline_screen._stop_engine()
            else:
                self._pipeline_screen._start_engine()
        except Exception:
            pass

    def _refresh_tray_label(self) -> None:
        if not getattr(self, "_act_toggle_pipe", None):
            return
        running = (
            self._pipeline_screen._engine is not None
            and self._pipeline_screen._engine.is_running()
        )
        self._act_toggle_pipe.setText("Stop pipeline" if running else "Start pipeline")
        if self._tray is None:
            return
        target = self._tray_icon_live if running else self._tray_icon_idle
        if getattr(self, "_tray_state_running", None) != running:
            self._tray_state_running = running
            self._tray.setIcon(target)
            self._tray.setToolTip("Voicebox — Live" if running else "Voicebox")

    def _make_live_tray_icon(self, base: QIcon) -> QIcon:
        """Return a copy of `base` with a red badge painted ON TOP of the
        cube. We hard-code position via ratios because the icon has a dark
        opaque squircle background, so alpha-bounds returns the whole canvas
        and gives no useful anchor for "where the cube is"."""
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QPainter, QPixmap, QColor

        out = QIcon()
        for size in (16, 22, 32, 44, 64, 128, 256, 512, 1024):
            src = base.pixmap(QSize(size, size))
            if src.isNull():
                continue
            w, h = src.width(), src.height()
            pm = QPixmap(w, h)
            pm.fill(QColor(0, 0, 0, 0))
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.drawPixmap(0, 0, src)
            # Place badge over the cube's lower-right (the cube occupies
            # roughly the central 15..85% of the icon; 0.72/0.72 lands inside
            # the orange artwork, not in the dark squircle padding).
            d = max(6, int(w * 0.34))
            cx = int(w * 0.72)
            cy = int(h * 0.72)
            p.setPen(Qt.PenStyle.NoPen)
            # Soft halo for contrast on the orange surface and over wallpapers.
            p.setBrush(QColor(0, 0, 0, 140))
            p.drawEllipse(cx - d // 2 - 2, cy - d // 2 - 2, d + 4, d + 4)
            p.setBrush(QColor("#FF3B30"))
            p.drawEllipse(cx - d // 2, cy - d // 2, d, d)
            p.end()
            out.addPixmap(pm)
        return out

    def _quit_from_tray(self) -> None:
        self._force_quit = True
        try:
            if self._pipeline_screen._engine is not None:
                self._pipeline_screen._engine.stop()
        except Exception:
            pass
        self._do_save()
        if self._tray is not None:
            self._tray.hide()
        QApplication.quit()

    def closeEvent(self, ev):
        # Hide-on-close so the app keeps running in the menu bar tray.
        if not self._force_quit and self._tray is not None and self._tray.isVisible():
            ev.ignore()
            self.hide()
            self._do_save()
            # Brief tray hint the first time the user hides via the red button.
            if not getattr(self, "_close_hint_shown", False):
                self._close_hint_shown = True
                try:
                    self._tray.showMessage(
                        "Voicebox is still running",
                        "The pipeline keeps working in the background. "
                        "Click the menu-bar icon to bring the window back, or pick Quit Voicebox to fully exit.",
                        QSystemTrayIcon.MessageIcon.Information,
                        4000,
                    )
                except Exception:
                    pass
            return
        try:
            if self._pipeline_screen._engine is not None:
                self._pipeline_screen._engine.stop()
        except Exception:
            pass
        self._do_save()
        super().closeEvent(ev)


CORNER_RADIUS = 12


class _Panel(QFrame):
    """Outer rounded panel. Background painted antialiased via paintEvent so
    corners stay smooth (QSS border-radius + WA_TranslucentBackground parent)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # No background here; children paint their own rounded backgrounds.

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(TOKENS["bg"]))
        from PySide6.QtCore import QRectF
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.drawRoundedRect(rect, CORNER_RADIUS, CORNER_RADIUS)
