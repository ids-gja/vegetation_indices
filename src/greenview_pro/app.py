"""Qt user interface for IDS GreenView Pro."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import tomllib

import cv2
import numpy as np
from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from processing import HIGH_DEFAULT, LOW_DEFAULT, CameraWorker

APP_NAME = "IDS GreenView Pro"
VERSION = "3.1"
DATA_DIR = Path(__file__).resolve().parent / "resources"
ICON_DIR = DATA_DIR / "icons"
COLOR_IMAGE_FILE = DATA_DIR / "images" / "color_image.jpg"
CONFIG_FILE = DATA_DIR / "config.toml"
MESSAGE_DIR = DATA_DIR / "messages"
SNAPSHOT_DIR = Path.cwd() / "snapshots"
GUI_INTERVAL_MS = 66
MESSAGE_PROGRESS_INTERVAL_MS = 16
VIEW_NAMES = ("REFERENCE IMAGE", "RAW", "NDVI", "CVI")


def load_icon(name: str) -> QIcon:
    path = ICON_DIR / name
    return QIcon(str(path)) if path.exists() else QIcon()


class ImageCard(QFrame):
    IMAGE_MARGIN = 12

    def __init__(self, title, subtitle="", smooth=True):
        super().__init__()
        self.setObjectName("imageCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.image = QLabel("Waiting for image...")
        self.image.setObjectName("imageArea")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image.setMinimumSize(0, 0)
        layout.addWidget(self.image, 0, Qt.AlignCenter)
        self.overlay = QLabel(title, self)
        self.overlay.setObjectName("imageOverlay")
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._array = None
        self._smooth = smooth
        self.aspect_ratio = 1.0

    def set_image(self, array):
        self._array = array
        self._refresh()

    def clear_image(self):
        self._array = None
        self.image.setPixmap(QPixmap())
        self.image.setText("Waiting for image...")
        self.overlay.hide()

    def set_aspect_ratio(self, ratio):
        self.aspect_ratio = ratio
        self._size_image()

    def _refresh(self):
        if self._array is None or self.image.width() < 2 or self.image.height() < 2:
            return
        a = np.ascontiguousarray(self._array)
        h, w = a.shape[:2]
        q = QImage(a.data, w, h, a.strides[0], QImage.Format_BGR888).copy()
        mode = Qt.SmoothTransformation if self._smooth else Qt.FastTransformation
        pix = QPixmap.fromImage(q).scaled(self.image.size(), Qt.KeepAspectRatio, mode)
        self.image.setPixmap(pix)
        self.image.setText("")
        pos = self.image.mapTo(self, self.image.rect().topLeft())
        xo = max(0, (self.image.width() - pix.width()) // 2)
        yo = max(0, (self.image.height() - pix.height()) // 2)
        self.overlay.adjustSize()
        self.overlay.move(
            pos.x() + xo + self.IMAGE_MARGIN, pos.y() + yo + self.IMAGE_MARGIN
        )
        self.overlay.show()
        self.overlay.raise_()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._size_image()

    def _size_image(self):
        width = self.contentsRect().width()
        height = self.contentsRect().height()
        image_size = QSize(
            min(width, round(height * self.aspect_ratio)),
            min(height, round(width / self.aspect_ratio)),
        )
        if (
            min(image_size.width(), image_size.height()) >= 2
            and self.image.size() != image_size
        ):
            self.image.setFixedSize(image_size)
        self._refresh()


class PercentControl(QFrame):
    changed = Signal(int)

    def __init__(self, value, minimum, maximum):
        super().__init__()
        self.setObjectName("percent")
        self._value, self._minimum, self._maximum = value, minimum, maximum
        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 2, 2, 2)
        layout.setSpacing(2)
        self.label = QLabel()
        self.label.setMinimumWidth(42)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        arrows = QVBoxLayout()
        arrows.setSpacing(0)
        arrows.setContentsMargins(0, 0, 0, 0)
        self.up = QToolButton()
        self.down = QToolButton()
        for button in (self.up, self.down):
            button.setObjectName("arrow")
            button.setFixedSize(20, 15)
        self.up.setIcon(load_icon("arrow_up.svg"))
        self.down.setIcon(load_icon("arrow_down.svg"))
        self.up.setIconSize(QSize(14, 9))
        self.down.setIconSize(QSize(14, 9))
        self.up.clicked.connect(self.increase)
        self.down.clicked.connect(self.decrease)
        arrows.addWidget(self.up)
        arrows.addWidget(self.down)
        layout.addWidget(self.label)
        layout.addLayout(arrows)
        self._refresh()

    def value(self):
        return self._value

    def set_limits(self, minimum, maximum):
        self._minimum, self._maximum = minimum, maximum
        self._value = max(minimum, min(self._value, maximum))
        self._refresh()

    def increase(self):
        if self._value < self._maximum:
            self._value += 1
            self._refresh()
            self.changed.emit(self._value)

    def decrease(self):
        if self._value > self._minimum:
            self._value -= 1
            self._refresh()
            self.changed.emit(self._value)

    def _refresh(self):
        self.label.setText(f"{self._value} %")
        self.up.setEnabled(self._value < self._maximum)
        self.down.setEnabled(self._value > self._minimum)


class InfoRow(QWidget):
    def __init__(self, name, value="-"):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        label = QLabel(name)
        label.setObjectName("subtle")
        self.value = QLabel(value)
        self.value.setObjectName("info")
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(self.value)

    def set(self, text):
        self.value.setText(text)


class MessageProgressIndicator(QWidget):
    """A ring showing time elapsed until the next message."""

    def __init__(self, color):
        super().__init__()
        self._color = QColor(color)
        self._progress = 0.0
        self.setFixedSize(20, 20)

    def set_progress(self, progress):
        self._progress = max(0.0, min(float(progress), 1.0))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        bounds = self.rect().adjusted(2, 2, -2, -2)
        painter.setPen(QPen(QColor("#D9E2E3"), 3))
        painter.drawEllipse(bounds)
        painter.setPen(QPen(self._color, 3))
        painter.drawArc(
            bounds,
            90 * 16,
            -round(self._progress * 360 * 16),
        )


class ViewProgressIndicator(QWidget):
    """Subpixel-accurate elapsed-time bar for the image slideshow."""

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._progress = 0.0
        self.setFixedHeight(6)

    def set_progress(self, progress):
        self._progress = max(0.0, min(float(progress), 1.0))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#D9E2E3"))
        painter.fillRect(
            0.0, 0.0, self.width() * self._progress, float(self.height()), self._color
        )


class MainWindow(QMainWindow):
    worker_class = CameraWorker
    start_requested = Signal()
    close_requested = Signal()
    exposure_requested = Signal(int)
    percentiles_requested = Signal(int, int)
    preview_size_requested = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.resize(1600, 950)
        self.demo_mode = False
        self._normal_geometry = None
        self._was_maximized = False
        self._last_images = None
        self._last_gui_version = -1
        self._marketing_index = 0
        self._display_view_index = 0
        self._animation = None
        self._last_preview_size = None
        self.marketing_settings, self.marketing_messages = self.load_marketing()
        self._view_spacing = max(0, int(self.marketing_settings.get("VIEW_SPACING", 1)))
        self._message_cycle_started = None
        self._view_cycle_started = None
        self._demo_aspect_ratio = None
        self.color_image = cv2.imread(str(COLOR_IMAGE_FILE))
        self._build()
        self._style()
        self.worker = self.worker_class()
        self.worker.state_changed.connect(self.update_state)
        self.worker.exposure_range.connect(self.configure_exposure)
        self.start_requested.connect(self.worker.start_camera)
        self.close_requested.connect(self.worker.close_camera)
        self.exposure_requested.connect(self.worker.set_exposure)
        self.percentiles_requested.connect(self.worker.set_percentiles)
        self.preview_size_requested.connect(self.worker.set_preview_size)
        self.worker.recoverable_error.connect(self.show_camera_error)
        self.worker.start()
        self.gui_timer = QTimer(self)
        self.gui_timer.setInterval(GUI_INTERVAL_MS)
        self.gui_timer.timeout.connect(self.pull_latest)
        self.gui_timer.start()
        self.marketing_timer = QTimer(self)
        self.marketing_timer.setInterval(
            int(float(self.marketing_settings.get("DISPLAY_TIME", 12)) * 1000)
        )
        self.marketing_timer.timeout.connect(self.next_message)
        self.message_progress_timer = QTimer(self)
        self.message_progress_timer.setInterval(MESSAGE_PROGRESS_INTERVAL_MS)
        self.message_progress_timer.setTimerType(Qt.PreciseTimer)
        self.message_progress_timer.timeout.connect(self._update_message_progress)
        self.message_progress_timer.timeout.connect(self._update_view_progress)
        self.view_timer = QTimer(self)
        self.view_timer.setInterval(
            int(float(self.marketing_settings.get("VIEW_ROTATION_SECONDS", 5)) * 1000)
        )
        self.view_timer.timeout.connect(self.next_view)

    def _build(self):
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.main_page, self.demo_page = QWidget(), QWidget()
        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.demo_page)
        self._build_main()
        self._build_demo()

    def _make_cards(self):
        cards = [
            ImageCard("REFERENCE IMAGE", smooth=True),
            ImageCard("RAW", smooth=False),
            ImageCard("NDVI", smooth=True),
            ImageCard("CVI", smooth=True),
        ]
        if self.color_image is not None:
            cards[0].set_image(self.color_image)
        else:
            cards[0].image.setText("color_image.jpg missing")
        return cards

    @staticmethod
    def _add_grid(layout, cards, margin=0, spacing=4):
        grid = QGridLayout()
        grid.setContentsMargins(margin, margin, margin, margin)
        grid.setSpacing(spacing)
        for card, (row, col) in zip(cards, ((0, 0), (0, 1), (1, 0), (1, 1))):
            grid.addWidget(card, row, col)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, 1)

    def _build_main(self):
        root = QVBoxLayout(self.main_page)
        root.setContentsMargins(18, 14, 18, 8)
        root.setSpacing(8)
        controls = QFrame()
        controls.setObjectName("controls")
        bar = QHBoxLayout(controls)
        logo = QLabel()
        logo.setFixedSize(140, 44)
        logo_path = ICON_DIR / "ids-logo_black_rgb.png"
        if logo_path.exists():
            logo.setPixmap(
                QPixmap(str(logo_path)).scaled(
                    logo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        else:
            logo.setText("IDS")
        title = QLabel("GreenView Pro")
        title.setObjectName("title")
        bar.addWidget(logo)
        bar.addWidget(title)
        bar.addSpacing(16)
        bar.addWidget(QLabel("Exposure"))
        self.exposure = QSlider(Qt.Horizontal)
        self.exposure.setEnabled(False)
        self.exposure.valueChanged.connect(self.change_exposure)
        self.exposure_label = QLabel("- us")
        bar.addWidget(self.exposure, 1)
        bar.addWidget(self.exposure_label)
        bar.addWidget(QLabel("Contrast"))
        self.low = PercentControl(LOW_DEFAULT, 0, 98)
        self.high = PercentControl(HIGH_DEFAULT, 1, 100)
        self.low.changed.connect(self.change_percentiles)
        self.high.changed.connect(self.change_percentiles)
        bar.addWidget(self.low)
        bar.addWidget(QLabel("to"))
        bar.addWidget(self.high)
        self.snapshot_button = QToolButton()
        self.snapshot_button.setIcon(load_icon("save.svg"))
        self.snapshot_button.setIconSize(QSize(22, 22))
        self.snapshot_button.setFixedSize(36, 36)
        self.snapshot_button.setToolTip("Save snapshot")
        self.snapshot_button.setAccessibleName("Save snapshot")
        self.snapshot_button.clicked.connect(self.save_snapshot)
        bar.addWidget(self.snapshot_button)
        self.camera_status = QLabel("Connecting camera...")
        bar.addWidget(self.camera_status)
        self.demo_button = QToolButton()
        self.demo_button.setIcon(load_icon("fullscreen.svg"))
        self.demo_button.setIconSize(QSize(22, 22))
        self.demo_button.setFixedSize(36, 36)
        self.demo_button.setToolTip("Demo Preview (fullscreen)")
        self.demo_button.setAccessibleName("Demo Preview (fullscreen)")
        self.demo_button.clicked.connect(self.enter_demo)
        bar.addWidget(self.demo_button)
        root.addWidget(controls)
        self.main_cards = self._make_cards()
        self._add_grid(
            root,
            self.main_cards,
            margin=self._view_spacing,
            spacing=self._view_spacing,
        )

    def _build_demo(self):
        root = QVBoxLayout(self.demo_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.demo_header = QWidget()
        header = QGridLayout(self.demo_header)
        header.setContentsMargins(28, 8, 28, 8)
        header.setHorizontalSpacing(20)
        self.demo_logo = QLabel()
        self.demo_logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        logo_path = ICON_DIR / "ids-logo_black_rgb.png"
        self.demo_logo_source = (
            QPixmap(str(logo_path)) if logo_path.exists() else QPixmap()
        )
        header.addWidget(self.demo_logo, 0, 0, Qt.AlignLeft | Qt.AlignVCenter)
        message_container = QWidget()
        message_container_layout = QHBoxLayout(message_container)
        message_container_layout.setContentsMargins(0, 0, 0, 0)
        message_container_layout.setSpacing(self._view_spacing)
        self.message_box = QWidget()
        message = QVBoxLayout(self.message_box)
        message.setContentsMargins(0, 0, 0, 0)
        message.setSpacing(3)
        self.demo_title = QLabel()
        self.demo_title.setObjectName("demoTitle")
        self.demo_title.setAlignment(Qt.AlignCenter)
        self.demo_title.setWordWrap(True)
        self.demo_subtitle = QLabel()
        self.demo_subtitle.setObjectName("demoSubtitle")
        self.demo_subtitle.setAlignment(Qt.AlignCenter)
        self.demo_subtitle.setWordWrap(True)
        message.addStretch()
        message.addWidget(self.demo_title)
        message.addWidget(self.demo_subtitle)
        message.addStretch()
        self.message_effect = QGraphicsOpacityEffect(self.message_box)
        self.message_box.setGraphicsEffect(self.message_effect)
        self.message_progress = MessageProgressIndicator(
            self.marketing_settings.get("ACCENT_COLOR", "#008A96")
        )
        message_container_layout.addWidget(self.message_box, 1)
        message_container_layout.addWidget(
            self.message_progress, 0, Qt.AlignRight | Qt.AlignVCenter
        )
        header.addWidget(message_container, 0, 1, Qt.AlignVCenter)
        self.demo_header_balance = QWidget()
        header.addWidget(self.demo_header_balance, 0, 2)
        header.setColumnStretch(1, 1)
        root.addWidget(self.demo_header)
        self.demo_middle = QWidget()
        middle = QHBoxLayout(self.demo_middle)
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(self._view_spacing)
        self.demo_large = ImageCard(VIEW_NAMES[self._display_view_index], smooth=True)
        middle.addWidget(self.demo_large)
        self.view_progress = ViewProgressIndicator(
            self.marketing_settings.get("ACCENT_COLOR", "#008A96"),
            self.demo_large,
        )
        self.view_progress.setObjectName("viewProgress")
        self.view_progress.setAccessibleName("Image slideshow progress")
        self.demo_reference = QWidget()
        refs = QVBoxLayout(self.demo_reference)
        refs.setContentsMargins(0, 0, 0, 0)
        refs.setSpacing(self._view_spacing)
        self.demo_cards = [
            ImageCard(name, smooth=name != "RAW") for name in VIEW_NAMES[1:]
        ]
        for card in self.demo_cards:
            refs.addWidget(card, 1)
        middle.addWidget(self.demo_reference)
        root.addWidget(self.demo_middle, 0, Qt.AlignHCenter | Qt.AlignVCenter)
        self.demo_footer = QWidget()
        bottom = QVBoxLayout(self.demo_footer)
        bottom.setContentsMargins(24, 8, 24, 12)
        self.facts = QWidget()
        facts = QHBoxLayout(self.facts)
        facts.setContentsMargins(0, 0, 0, 0)
        facts.setSpacing(28)
        self.fact_blocks = []
        for _ in range(3):
            block = QWidget()
            bl = QVBoxLayout(block)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(2)
            h = QLabel()
            h.setObjectName("demoFact")
            h.setAlignment(Qt.AlignCenter)
            h.setWordWrap(True)
            d = QLabel()
            d.setObjectName("demoFactDetail")
            d.setAlignment(Qt.AlignCenter)
            d.setWordWrap(True)
            bl.addWidget(h)
            bl.addWidget(d)
            facts.addWidget(block, 1)
            self.fact_blocks.append((h, d))
        self.facts_effect = QGraphicsOpacityEffect(self.facts)
        self.facts.setGraphicsEffect(self.facts_effect)
        bottom.addWidget(self.facts)
        root.addWidget(self.demo_footer)
        self.apply_message()
        QTimer.singleShot(0, self._size_demo_layout)

    def _size_demo_layout(self):
        screen = self.screen() or QApplication.primaryScreen()
        size = screen.size() if screen else self.size()
        sw = max(1, size.width())
        sh = max(1, size.height())
        aspect = self._demo_image_aspect()
        self._demo_aspect_ratio = aspect
        header_h = 130
        footer_h = 140
        reserve = 12
        available = max(1, sh - reserve - header_h - footer_h)
        gap = max(12, self._view_spacing)
        raster_h = min(
            available,
            max(1, int((sw - 24 - gap) * 3 / (4 * aspect))),
        )
        while raster_h > 1:
            large_w = round(raster_h * aspect)
            sidebar_h = max(1, (raster_h - 2 * self._view_spacing) // 3)
            sidebar_w = round(sidebar_h * aspect)
            if large_w + gap + sidebar_w <= sw - 24:
                break
            raster_h -= 1
        large_w = round(raster_h * aspect)
        sidebar_h = max(1, (raster_h - 2 * self._view_spacing) // 3)
        sidebar_w = round(sidebar_h * aspect)
        self.demo_header.setFixedHeight(header_h)
        self.demo_footer.setFixedHeight(footer_h)
        self.demo_middle.layout().setSpacing(gap)
        self.demo_middle.setFixedSize(large_w + gap + sidebar_w, raster_h)
        self.demo_large.set_aspect_ratio(aspect)
        self.demo_large.setFixedSize(large_w, raster_h)
        self.demo_reference.setFixedSize(sidebar_w, raster_h)
        for card in self.demo_cards:
            card.set_aspect_ratio(aspect)
        self.view_progress.setGeometry(
            0,
            raster_h - self.view_progress.height(),
            large_w,
            self.view_progress.height(),
        )
        self.view_progress.raise_()
        logo_w, logo_h = 285, 93
        self.demo_logo.setFixedSize(logo_w, logo_h)
        self.demo_header_balance.setFixedWidth(logo_w)
        if not self.demo_logo_source.isNull():
            self.demo_logo.setPixmap(
                self.demo_logo_source.scaled(
                    self.demo_logo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        self.demo_title.setStyleSheet(
            "font-size: 52px; font-weight: 700"
        )
        self.demo_subtitle.setStyleSheet(
            "font-size: 25px; font-weight: 600"
        )
        for h, d in self.fact_blocks:
            h.setStyleSheet("font-size: 30px; font-weight: 700")
            d.setStyleSheet("font-size: 28px; font-weight: 600")
        for card in self.demo_cards + [self.demo_large]:
            card.IMAGE_MARGIN = 12
        self._request_preview_size()

    def _demo_image_aspect(self):
        raw = self._last_images[0] if self._last_images is not None else None
        image = raw if raw is not None else self.color_image
        return image.shape[1] / image.shape[0] if image is not None else 4 / 3

    def _request_preview_size(self):
        cards = [self.demo_large] if self.demo_mode else self.main_cards[1:]
        card = max(
            cards,
            key=lambda candidate: candidate.image.width() * candidate.image.height(),
        )
        size = card.image.size()
        preview_size = (size.width(), size.height())
        if min(preview_size) < 2 or preview_size == self._last_preview_size:
            return
        self._last_preview_size = preview_size
        self.preview_size_requested.emit(*preview_size)

    def _style(self):
        s = self.marketing_settings
        bg = s.get("BACKGROUND_COLOR", "#F5F5F5")
        QApplication.instance().setFont(
            QFont(s.get("FONT_FAMILY", "Source Sans Pro"), 10)
        )
        self.setStyleSheet(f"""
            QWidget {{
                background: {bg};
                color: #343434
            }}
            QLabel {{
                background: transparent
            }}
            #title {{
                font-size: 22px;
                font-weight: 700
            }}
            QFrame#imageCard {{
                background: {bg};
                border: none
            }}
            #imageArea {{
                background: {bg};
                color: #777777;
                border: none
            }}
            #imageOverlay {{
                color: rgba(225, 225, 225, 215);
                background: rgba(32, 58, 64, 100);
                border-radius: 4px;
                padding: 5px 9px;
                font-size: 18px;
                font-weight: 700
            }}
            QFrame#controls {{
                background: white;
                border: 1px solid #DEDEDE;
                border-radius: 9px
            }}
            QFrame#percent {{
                background: white;
                border: 1px solid #CCCCCC;
                border-radius: 5px
            }}
            QPushButton {{
                background: #008A96;
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 9px 12px;
                font-weight: 600
            }}
            #demoTitle {{
                color: {s.get("TITLE_COLOR", "#203A40")};
                font-size: {s.get("TITLE_FONT_SIZE", "52")}px;
                font-weight: 700
            }}
            #demoSubtitle {{
                color: {s.get("SUBTITLE_COLOR", "#4F4F4F")};
                font-size: {s.get("SUBTITLE_FONT_SIZE", "25")}px;
                font-weight: 600
            }}
            #demoFact {{
                color: {s.get("ACCENT_COLOR", "#008A96")};
                font-size: {s.get("FACT_TITLE_FONT_SIZE", "30")}px;
                font-weight: 700
            }}
            #demoFactDetail {{
                color: {s.get("DETAIL_COLOR", "#4F4F4F")};
                font-size: {s.get("FACT_DETAIL_FONT_SIZE", "28")}px;
                font-weight: 600
            }}
        """)

    def load_marketing(self):
        settings = {}
        if CONFIG_FILE.exists():
            with CONFIG_FILE.open("rb") as config_file:
                settings = tomllib.load(config_file).get("settings", {})

        messages = []
        for message_file in sorted(MESSAGE_DIR.glob("*.toml")):
            with message_file.open("rb") as source_file:
                messages.append(self.message_from(tomllib.load(source_file)))

        fallback = [
            {
                "title": "MULTISPECTRAL INSIGHT",
                "subtitle": "Live vegetation analysis",
                "facts": [
                    ("20 MP", "High spatial resolution"),
                    ("NDVI", "Vegetation vitality"),
                    ("CVI", "Chlorophyll-related differences"),
                ],
            }
        ]
        return settings, messages or fallback

    @staticmethod
    def message_from(data):
        facts = data.get("facts", [])
        if len(facts) != 3:
            raise ValueError("Each marketing message must define exactly three facts")
        return {
            "title": data["title"],
            "subtitle": data["subtitle"],
            "large_view": data.get("large_view", "").strip().upper(),
            "facts": [(fact["title"], fact["detail"]) for fact in facts],
        }

    def pull_latest(self):
        result, version = self.worker.latest()
        if result is None or version == self._last_gui_version:
            return
        self._last_gui_version = version
        self._last_images = result
        if self.demo_mode:
            if self._demo_aspect_ratio != self._demo_image_aspect():
                self._size_demo_layout()
            self._refresh_demo_views()
        else:
            for card, image in zip(self.main_cards[1:], result):
                card.set_image(image)

    @Slot(int, int, int)
    def configure_exposure(self, minimum, maximum, current):
        self.exposure.setRange(minimum, maximum)
        self.exposure.setValue(current)
        self.exposure.setEnabled(True)
        self.exposure_label.setText(f"{current} us")

    def change_exposure(self, value):
        self.exposure_label.setText(f"{value} us")
        self.exposure_requested.emit(value)

    def change_percentiles(self, _):
        low, high = self.low.value(), self.high.value()
        self.low.set_limits(0, high - 1)
        self.high.set_limits(low + 1, 100)
        self.percentiles_requested.emit(low, high)

    @Slot(str)
    def show_camera_error(self, message):
        self.camera_status.setText(message)

    @Slot(str, bool)
    def update_state(self, text, connected):
        self.camera_status.setText(text)

    def enter_demo(self):
        self.demo_mode = True
        self._was_maximized = self.isMaximized()
        self._normal_geometry = self.normalGeometry()
        self.stack.setCurrentWidget(self.demo_page)
        self.showFullScreen()
        self.marketing_timer.start()
        self._restart_view_progress()
        self.view_timer.start()
        self._restart_message_progress()
        self.message_progress_timer.start()
        QTimer.singleShot(0, self._size_demo_layout)
        QTimer.singleShot(120, self._refresh_demo_views)

    def exit_demo(self):
        self.demo_mode = False
        self.marketing_timer.stop()
        self.message_progress_timer.stop()
        self.view_timer.stop()
        self.message_progress.set_progress(0)
        self.view_progress.set_progress(0)
        self.stack.setCurrentWidget(self.main_page)
        if self._was_maximized:
            self.showMaximized()
        else:
            self.showNormal()
        if (
            not self._was_maximized
            and self._normal_geometry
            and self._normal_geometry.isValid()
        ):
            self.setGeometry(self._normal_geometry)
        QTimer.singleShot(100, self.refresh_visible)

    def _view_images(self):
        raw, ndvi, cvi = self._last_images or (None, None, None)
        return {
            "REFERENCE IMAGE": self.color_image,
            "RAW": raw,
            "NDVI": ndvi,
            "CVI": cvi,
        }

    def _refresh_demo_views(self):
        images = self._view_images()
        selected_name = VIEW_NAMES[self._display_view_index]
        self.demo_large.overlay.setText(self.view_title(selected_name))
        selected_image = images[selected_name]
        if selected_image is not None:
            self.demo_large.set_image(selected_image)

        side_names = [name for name in VIEW_NAMES if name != selected_name]
        for card, name in zip(self.demo_cards, side_names):
            card.overlay.setText(self.view_title(name))
            image = images[name]
            if image is not None:
                card.set_image(image)

    def view_title(self, name):
        return name

    def next_view(self):
        self._display_view_index = (self._display_view_index + 1) % len(VIEW_NAMES)
        self._refresh_demo_views()
        self._restart_view_progress()

    def _restart_view_progress(self):
        self._view_cycle_started = time.monotonic()
        self.view_progress.set_progress(0)

    def _update_view_progress(self):
        if self._view_cycle_started is None:
            return
        duration = self.view_timer.interval() / 1000
        self.view_progress.set_progress(
            (time.monotonic() - self._view_cycle_started) / duration
        )

    def apply_message(self):
        message = self.marketing_messages[self._marketing_index]
        self.demo_title.setText(message["title"])
        self.demo_subtitle.setText(message["subtitle"])
        for i, (h, d) in enumerate(self.fact_blocks):
            h.setText(message["facts"][i][0])
            d.setText(message["facts"][i][1])
        self._refresh_demo_views()

    def make_animation(self, effect, start, end, seconds):
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setDuration(int(seconds * 1000))
        animation.setEasingCurve(QEasingCurve.InOutCubic)
        return animation

    def next_message(self):
        if self._animation is not None:
            return
        fade = float(self.marketing_settings.get("FADE_OUT_TIME", 1))
        self.message_progress.set_progress(1)
        group = QParallelAnimationGroup(self)
        for effect in (self.message_effect, self.facts_effect):
            group.addAnimation(self.make_animation(effect, 1, 0, fade))

        def swap():
            self._marketing_index = (self._marketing_index + 1) % len(
                self.marketing_messages
            )
            self.apply_message()
            self._restart_message_progress()
            inside = QParallelAnimationGroup(self)
            for effect in (self.message_effect, self.facts_effect):
                inside.addAnimation(
                    self.make_animation(
                        effect,
                        0,
                        1,
                        float(self.marketing_settings.get("FADE_IN_TIME", 1)),
                    )
                )
            inside.finished.connect(lambda: setattr(self, "_animation", None))
            self._animation = inside
            inside.start()

        group.finished.connect(swap)
        self._animation = group
        group.start()

    def _restart_message_progress(self):
        self._message_cycle_started = time.monotonic()
        self.message_progress.set_progress(0)

    def _update_message_progress(self):
        if self._message_cycle_started is None:
            return
        display_time = float(self.marketing_settings.get("DISPLAY_TIME", 12))
        self.message_progress.set_progress(
            (time.monotonic() - self._message_cycle_started) / display_time
        )

    def refresh_visible(self):
        for card in (
            (self.demo_cards + [self.demo_large]) if self.demo_mode else self.main_cards
        ):
            card._refresh()
        if self.demo_mode:
            self._size_demo_layout()

    def save_snapshot(self):
        if self._last_images is None:
            return
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for name, image in zip(
            ("color_image", "raw", "ndvi", "cvi"),
            (self.color_image,) + self._last_images,
        ):
            if image is not None:
                cv2.imwrite(str(SNAPSHOT_DIR / f"{stamp}_{name}.png"), image)

    def keyPressEvent(self, event):
        if self.demo_mode and event.key() == Qt.Key_Escape:
            self.exit_demo()
            return
        if self.demo_mode and event.key() == Qt.Key_Space:
            self.next_message()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.demo_mode:
            QTimer.singleShot(0, self._size_demo_layout)
        else:
            QTimer.singleShot(0, self._request_preview_size)

    def closeEvent(self, event):
        self.gui_timer.stop()
        self.marketing_timer.stop()
        self.message_progress_timer.stop()
        self.view_timer.stop()
        if self._animation:
            self._animation.stop()
        self.worker.stop_worker()
        if not self.worker.wait(3500):
            self.worker.terminate()
            self.worker.wait()
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description="IDS GreenView Pro live demo")
    parser.add_argument(
        "--calibration", action="store_true", help="Enable white-target calibration"
    )
    args = parser.parse_args(argv)
    if args.calibration:
        from calibration.ui import CalibrationWindow

        window_class = CalibrationWindow
    else:
        window_class = MainWindow
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    window = window_class()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
