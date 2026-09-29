"""Qt user interface for IDS GreenView Pro."""

import argparse
from datetime import datetime
import math
from pathlib import Path
import re
import sys
import time
import tomllib

import cv2
import numpy as np
from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QSaveFile,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from greenview_pro.contrast_control import ContrastRange
from greenview_pro.processing import HIGH_DEFAULT, LOW_DEFAULT, CameraWorker

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
ROTATING_VIEWS = ("NDVI", "CVI")
VIEW_NAMES = ("REFERENCE IMAGE", "RAW", *ROTATING_VIEWS)
RAW_GAMMA_LUT = np.rint(
    255 * (np.arange(256, dtype=np.float32) / 255) ** (1 / 2.2)
).astype(np.uint8)


def load_icon(name: str) -> QIcon:
    path = ICON_DIR / name
    return QIcon(str(path)) if path.exists() else QIcon()


def update_settings_text(document, updates):
    header = re.search(r"(?m)^\[settings\][ \t]*(?:#[^\r\n]*)?$", document)
    if header is None:
        if document and not document.endswith("\n"):
            document += "\n"
        document += "[settings]\n"
        header = re.search(r"(?m)^\[settings\]$", document)
    following = re.search(r"(?m)^\[", document[header.end():])
    end = header.end() + following.start() if following else len(document)
    section = document[header.end():end]
    for name, value in updates.items():
        encoded = (
            "true" if value is True else "false" if value is False else str(value)
        )
        pattern = rf"(?m)^([ \t]*){re.escape(name)}[ \t]*=[^\r\n]*"
        section, count = re.subn(
            pattern, lambda match: f"{match.group(1)}{name} = {encoded}",
            section, count=1,
        )
        if not count:
            if section and not section.endswith("\n"):
                section += "\n"
            section += f"{name} = {encoded}\n"
    return document[:header.end()] + section + document[end:]


class ImageCard(QFrame):
    IMAGE_MARGIN = 12

    def __init__(self, title, subtitle="", smooth=True, raw_gamma=False):
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
        self._raw_gamma = raw_gamma
        self.aspect_ratio = 1.0

    def set_image(self, array, *, raw_gamma=None, smooth=None):
        if raw_gamma is not None:
            self._raw_gamma = raw_gamma
        if smooth is not None:
            self._smooth = smooth
        self._array = cv2.LUT(array, RAW_GAMMA_LUT) if self._raw_gamma else array
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
        self.setFixedSize(40, 40)

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


class MainWindow(QMainWindow):
    worker_class = CameraWorker
    start_requested = Signal()
    close_requested = Signal()
    exposure_requested = Signal(int)
    percentiles_requested = Signal(str, int, int)
    temporal_requested = Signal(bool, float)
    preview_size_requested = Signal(object)

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
        self._source_aspect_ratio = None
        self.marketing_settings, self.marketing_messages = self.load_marketing()
        self.temporal_enabled = self.marketing_settings["TEMPORAL_ENABLED"]
        self.temporal_weight = float(self.marketing_settings["TEMPORAL_WEIGHT"])
        self._view_spacing = max(0, int(self.marketing_settings.get("VIEW_SPACING", 1)))
        self._message_cycle_started = None
        self._demo_aspect_ratio = None
        self.color_image = cv2.imread(str(COLOR_IMAGE_FILE))
        self._build()
        self._style()
        self.worker = self.worker_class()
        self.worker.state_changed.connect(self.update_state)
        self.worker.frame_dimensions_changed.connect(self.update_frame_dimensions)
        self.worker.exposure_range.connect(self.configure_exposure)
        self.start_requested.connect(self.worker.start_camera)
        self.close_requested.connect(self.worker.close_camera)
        self.exposure_requested.connect(self.worker.set_exposure)
        self.percentiles_requested.connect(self.worker.set_percentiles)
        self.temporal_requested.connect(self.worker.set_temporal_filter)
        self.preview_size_requested.connect(self.worker.set_preview_sizes)
        self.worker.recoverable_error.connect(self.show_camera_error)
        self.worker.set_temporal_filter(self.temporal_enabled, self.temporal_weight)
        for index in ("ndvi", "cvi"):
            self.worker.set_percentiles(
                index, *getattr(self, f"{index}_contrast").values()
            )
        if "EXPOSURE_US" in self.marketing_settings:
            self.worker.set_exposure(self.marketing_settings["EXPOSURE_US"])
        self.worker.start()
        self.gui_timer = QTimer(self)
        self.gui_timer.setInterval(GUI_INTERVAL_MS)
        self.gui_timer.timeout.connect(self.pull_latest)
        self.gui_timer.start()
        self.marketing_timer = QTimer(self)
        self.marketing_timer.setInterval(
            int(float(self.marketing_settings.get("DISPLAY_TIME", 12)) * 1000)
        )
        self.marketing_timer.setSingleShot(True)
        self.marketing_timer.timeout.connect(self.next_message)
        self.message_progress_timer = QTimer(self)
        self.message_progress_timer.setInterval(MESSAGE_PROGRESS_INTERVAL_MS)
        self.message_progress_timer.setTimerType(Qt.PreciseTimer)
        self.message_progress_timer.timeout.connect(self._update_message_progress)

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
            ImageCard("RAW", smooth=False, raw_gamma=True),
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
        grid.setAlignment(Qt.AlignCenter)
        for card, (row, col) in zip(cards, ((0, 0), (0, 1), (1, 0), (1, 1))):
            grid.addWidget(card, row, col)
        layout.addLayout(grid, 1)

    def _build_main(self):
        root = QVBoxLayout(self.main_page)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        controls = QFrame()
        self.controls = controls
        controls.setObjectName("controls")
        panel = QVBoxLayout(controls)
        panel.setSpacing(2)
        bar = QHBoxLayout()
        panel.addLayout(bar)
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
        self.exposure.setMinimumWidth(130)
        self.exposure.setEnabled(False)
        self.exposure.valueChanged.connect(self.change_exposure)
        self.exposure_label = QLabel("- us")
        self.exposure_label.setFixedWidth(80)
        self.exposure_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bar.addWidget(self.exposure, 1)
        bar.addWidget(self.exposure_label)
        bar.addSpacing(16)
        contrast_rows = QVBoxLayout()
        contrast_rows.setContentsMargins(0, 0, 0, 0)
        contrast_rows.setSpacing(0)
        for name in ("NDVI", "CVI"):
            row = QHBoxLayout()
            title = QLabel(name)
            title.setFixedWidth(44)
            row.addWidget(title)
            low = self.marketing_settings[f"{name}_LOW_PERCENTILE"]
            high = self.marketing_settings[f"{name}_HIGH_PERCENTILE"]
            control = ContrastRange(low, high)
            control.setAccessibleName(f"{name} contrast percentile range")
            control.changed.connect(
                lambda low, high, index=name.lower(): self.change_percentiles(
                    index, low, high
                )
            )
            row.addWidget(control, 1)
            label = QLabel(f"{low}-{high} %")
            label.setFixedWidth(78)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(label)
            setattr(self, f"{name.lower()}_contrast", control)
            setattr(self, f"{name.lower()}_contrast_label", label)
            contrast_rows.addLayout(row)
        bar.addLayout(contrast_rows, 2)
        self.save_config_button = QToolButton()
        self.save_config_button.setIcon(load_icon("save.svg"))
        self.save_config_button.setIconSize(QSize(22, 22))
        self.save_config_button.setFixedSize(36, 36)
        self.save_config_button.setToolTip("Save current configuration")
        self.save_config_button.setAccessibleName("Save config")
        self.save_config_button.clicked.connect(self.save_config)
        bar.addWidget(self.save_config_button)
        self.snapshot_button = QToolButton()
        self.snapshot_button.setIcon(load_icon("save.svg"))
        self.snapshot_button.setIconSize(QSize(22, 22))
        self.snapshot_button.setFixedSize(36, 36)
        self.snapshot_button.setStyleSheet(
            "QToolButton { border: 1px solid #008A96; "
            "border-radius: 4px; background: white; }"
        )
        self.snapshot_button.setToolTip("Save snapshot")
        self.snapshot_button.setAccessibleName("Save snapshot")
        self.snapshot_button.clicked.connect(self.save_snapshot)
        bar.addWidget(self.snapshot_button)
        self.settings_button = QToolButton()
        self.settings_button.setIcon(load_icon("gear.svg"))
        self.settings_button.setIconSize(QSize(22, 22))
        self.settings_button.setFixedSize(36, 36)
        self.settings_button.setToolTip("Settings")
        self.settings_button.setAccessibleName("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        bar.addWidget(self.settings_button)
        self.demo_button = QToolButton()
        self.demo_button.setIcon(load_icon("fullscreen.svg"))
        self.demo_button.setIconSize(QSize(22, 22))
        self.demo_button.setFixedSize(36, 36)
        self.demo_button.setToolTip("Demo Preview (fullscreen)")
        self.demo_button.setAccessibleName("Demo Preview (fullscreen)")
        self.demo_button.clicked.connect(self.enter_demo)
        bar.addWidget(self.demo_button)
        self.status_layout = QVBoxLayout()
        self.status_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_status = QLabel("Connecting camera...")
        self.camera_status.setWordWrap(True)
        self.status_layout.addWidget(self.camera_status)
        panel.addLayout(self.status_layout)
        root.addWidget(controls)
        self.main_cards = self._make_cards()
        self._add_grid(
            root,
            self.main_cards,
            margin=0,
            spacing=self._view_spacing,
        )

    def _size_main_layout(self):
        layout = self.main_page.layout()
        margins = layout.contentsMargins()
        width = self.main_page.width() - margins.left() - margins.right()
        height = (
            self.main_page.height()
            - margins.top()
            - margins.bottom()
            - self.controls.height()
            - layout.spacing()
        )
        spacing = self._view_spacing
        max_width = (width - spacing) // 2
        max_height = (height - spacing) // 2
        if min(max_width, max_height) < 2:
            return
        aspect = self._demo_image_aspect()
        card_height = min(max_height, max(2, round(max_width / aspect)))
        card_width = min(max_width, max(2, round(card_height * aspect)))
        for card in self.main_cards:
            card.set_aspect_ratio(aspect)
            card.setFixedSize(card_width, card_height)
        self._request_preview_size()

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
        self.message_progress.setVisible(
            self.marketing_settings["SHOW_PROGRESS_CIRCLE"]
        )
        header.addWidget(self.message_box, 0, 1, Qt.AlignVCenter)
        self.demo_header_balance = QWidget()
        balance_layout = QHBoxLayout(self.demo_header_balance)
        balance_layout.setContentsMargins(0, 0, 0, 0)
        balance_layout.addWidget(
            self.message_progress, 0, Qt.AlignRight | Qt.AlignTop
        )
        header.addWidget(self.demo_header_balance, 0, 2)
        header.setColumnStretch(1, 1)
        root.addWidget(self.demo_header)
        self.demo_middle = QWidget()
        middle = QHBoxLayout(self.demo_middle)
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(self._view_spacing)
        self.demo_large = ImageCard(
            ROTATING_VIEWS[self._display_view_index], smooth=True
        )
        middle.addWidget(self.demo_large)
        self.demo_reference = QWidget()
        refs = QGridLayout(self.demo_reference)
        refs.setContentsMargins(0, 0, 0, 0)
        refs.setSpacing(self._view_spacing)
        self.demo_cards = [
            ImageCard(name, smooth=name != "RAW", raw_gamma=name == "RAW")
            for name in VIEW_NAMES
        ]
        for index, card in enumerate(self.demo_cards):
            refs.addWidget(card, index // 2, index % 2)
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
            max(1, int((sw - 24 - gap - self._view_spacing) / (2 * aspect))),
        )
        while raster_h > 1:
            large_w = round(raster_h * aspect)
            sidebar_h = max(1, (raster_h - self._view_spacing + 1) // 2)
            sidebar_w = round(sidebar_h * aspect)
            if large_w + gap + 2 * sidebar_w + self._view_spacing <= sw - 24:
                break
            raster_h -= 1
        large_w = round(raster_h * aspect)
        sidebar_h = max(1, (raster_h - self._view_spacing + 1) // 2)
        bottom_h = max(1, raster_h - self._view_spacing - sidebar_h)
        sidebar_w = round(sidebar_h * aspect)
        grid_width = 2 * sidebar_w + self._view_spacing
        self.demo_header.setFixedHeight(header_h)
        self.demo_footer.setFixedHeight(footer_h)
        self.demo_middle.layout().setSpacing(gap)
        self.demo_middle.setFixedSize(large_w + gap + grid_width, raster_h)
        self.demo_large.set_aspect_ratio(aspect)
        self.demo_large.setFixedSize(large_w, raster_h)
        self.demo_reference.setFixedSize(grid_width, raster_h)
        for index, card in enumerate(self.demo_cards):
            card.set_aspect_ratio(aspect)
            card.setFixedSize(sidebar_w, sidebar_h if index < 2 else bottom_h)
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
        if self._source_aspect_ratio is not None:
            return self._source_aspect_ratio
        raw = self._last_images[0] if self._last_images is not None else None
        image = raw if raw is not None else self.color_image
        return image.shape[1] / image.shape[0] if image is not None else 4 / 3

    def _request_preview_size(self):
        if self.demo_mode:
            raw_card = self.demo_cards[1].image.size()
            large = self.demo_large.image.size()
            raw = (raw_card.width(), raw_card.height())
            large_size = (large.width(), large.height())
            ndvi_card = self.demo_cards[2].image.size()
            cvi_card = self.demo_cards[3].image.size()
            ndvi = large_size if self._display_view_index == 0 else (
                ndvi_card.width(), ndvi_card.height()
            )
            cvi = large_size if self._display_view_index == 1 else (
                cvi_card.width(), cvi_card.height()
            )
            preview_sizes = (raw, ndvi, cvi)
        else:
            cards = self.main_cards[1:]
            preview_sizes = tuple(
                (card.image.width(), card.image.height()) for card in cards
            )
        if (
            any(min(size) < 2 for size in preview_sizes)
            or preview_sizes == self._last_preview_size
        ):
            return
        self._last_preview_size = preview_sizes
        self.preview_size_requested.emit(preview_sizes)

    def open_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Processing settings")
        layout = QVBoxLayout(dialog)
        enabled = QCheckBox("Temporal filter")
        enabled.setChecked(self.temporal_enabled)
        layout.addWidget(enabled)
        row = QHBoxLayout()
        row.addWidget(QLabel("New frame weight"))
        weight = QSpinBox()
        weight.setRange(1, 100)
        weight.setSuffix(" %")
        weight.setValue(round(100 * self.temporal_weight))
        weight.setEnabled(self.temporal_enabled)
        row.addWidget(weight)
        layout.addLayout(row)

        def change_filter():
            self.temporal_enabled = enabled.isChecked()
            self.temporal_weight = weight.value() / 100
            weight.setEnabled(self.temporal_enabled)
            self.temporal_requested.emit(self.temporal_enabled, self.temporal_weight)

        enabled.toggled.connect(change_filter)
        weight.valueChanged.connect(change_filter)
        self._add_mode_settings(layout)
        dialog.exec()

    def _add_mode_settings(self, layout):
        pass

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
        settings.setdefault("SHOW_PROGRESS_CIRCLE", False)
        if not isinstance(settings["SHOW_PROGRESS_CIRCLE"], bool):
            raise ValueError("SHOW_PROGRESS_CIRCLE must be true or false")
        settings.setdefault("TEMPORAL_ENABLED", True)
        settings.setdefault("TEMPORAL_WEIGHT", 0.25)
        if not isinstance(settings["TEMPORAL_ENABLED"], bool):
            raise ValueError("TEMPORAL_ENABLED must be true or false")
        weight = settings["TEMPORAL_WEIGHT"]
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or not 0 < weight <= 1):
            raise ValueError("TEMPORAL_WEIGHT must be in (0, 1]")
        for index in ("NDVI", "CVI"):
            low_key, high_key = f"{index}_LOW_PERCENTILE", f"{index}_HIGH_PERCENTILE"
            settings.setdefault(low_key, LOW_DEFAULT)
            settings.setdefault(high_key, HIGH_DEFAULT)
            low, high = settings[low_key], settings[high_key]
            if type(low) is not int or type(high) is not int or not 0 <= low <= high - 2 <= 98:
                raise ValueError(f"{index} percentiles require a gap of at least 2")
        if "EXPOSURE_US" in settings and (
            type(settings["EXPOSURE_US"]) is not int
            or not 1 <= settings["EXPOSURE_US"] <= 2_147_483_647
        ):
            raise ValueError("EXPOSURE_US must be a positive camera exposure")

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
        with QSignalBlocker(self.exposure):
            self.exposure.setRange(minimum, maximum)
            self.exposure.setValue(current)
        self.exposure.setEnabled(True)
        self.exposure_label.setText(f"{current} us")

    def change_exposure(self, value):
        self.exposure_label.setText(f"{value} us")
        self.exposure_requested.emit(value)

    def change_percentiles(self, index, low, high):
        getattr(self, f"{index}_contrast_label").setText(f"{low}-{high} %")
        self.percentiles_requested.emit(index, low, high)

    @Slot(int, int)
    def update_frame_dimensions(self, width, height):
        aspect = width / height
        if aspect != self._source_aspect_ratio:
            self._source_aspect_ratio = aspect
            if self.demo_mode:
                self._size_demo_layout()
            else:
                self._size_main_layout()

    @Slot(str)
    def show_camera_error(self, message):
        self.camera_status.setText(message)

    @Slot(str, bool)
    def update_state(self, text, connected):
        self.camera_status.setText(text)

    def enter_demo(self):
        self.demo_mode = True
        self._display_view_index = 0
        self._was_maximized = self.isMaximized()
        self._normal_geometry = self.normalGeometry()
        self.stack.setCurrentWidget(self.demo_page)
        self.showFullScreen()
        self._restart_message_progress()
        if self.marketing_settings["SHOW_PROGRESS_CIRCLE"]:
            self.message_progress_timer.start()
        QTimer.singleShot(0, self._size_demo_layout)
        QTimer.singleShot(120, self._refresh_demo_views)

    def exit_demo(self):
        self.demo_mode = False
        self.marketing_timer.stop()
        self.message_progress_timer.stop()
        self.message_progress.set_progress(0)
        if self._animation is not None:
            self._animation.stop()
            self._animation = None
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
        selected_name = ROTATING_VIEWS[self._display_view_index]
        self.demo_large.overlay.setText(self.view_title(selected_name))
        selected_image = images[selected_name]
        if selected_image is not None:
            self.demo_large.set_image(selected_image)

        for card, name in zip(self.demo_cards, VIEW_NAMES):
            card.overlay.setText(self.view_title(name))
            image = images[name]
            if image is not None:
                card.set_image(
                    image, raw_gamma=name == "RAW", smooth=name != "RAW"
                )

    def view_title(self, name):
        return name

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
            if self.demo_mode and not self.marketing_timer.isActive():
                self.marketing_timer.start()
            return
        self.marketing_timer.stop()
        fade = float(self.marketing_settings.get("FADE_OUT_TIME", 1))
        self.message_progress.set_progress(1)
        group = QParallelAnimationGroup(self)
        for effect in (self.message_effect, self.facts_effect):
            group.addAnimation(self.make_animation(effect, 1, 0, fade))

        def swap():
            self._marketing_index = (self._marketing_index + 1) % len(
                self.marketing_messages
            )
            self._display_view_index = (
                self._display_view_index + 1
            ) % len(ROTATING_VIEWS)
            self._request_preview_size()
            self.apply_message()
            if self.demo_mode:
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
        if self.marketing_settings["SHOW_PROGRESS_CIRCLE"]:
            self._message_cycle_started = time.monotonic()
            self.message_progress.set_progress(0)
        self.marketing_timer.start()

    def _update_message_progress(self):
        if self._message_cycle_started is None:
            return
        display_time = float(self.marketing_settings.get("DISPLAY_TIME", 12))
        self.message_progress.set_progress(
            (time.monotonic() - self._message_cycle_started) / display_time
        )

    def refresh_visible(self):
        if not self.demo_mode:
            self._size_main_layout()
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
            ("raw", "ndvi", "cvi"),
            self._last_images,
        ):
            if image is not None:
                cv2.imwrite(str(SNAPSHOT_DIR / f"{stamp}_{name}.png"), image)

    def save_config(self):
        updates = {
            "TEMPORAL_ENABLED": self.temporal_enabled,
            "TEMPORAL_WEIGHT": self.temporal_weight,
        }
        for index in ("NDVI", "CVI"):
            low, high = getattr(self, f"{index.lower()}_contrast").values()
            updates[f"{index}_LOW_PERCENTILE"] = low
            updates[f"{index}_HIGH_PERCENTILE"] = high
        exposure = (
            self.exposure.value()
            if self.exposure.isEnabled()
            else self.worker._saved_exposure
        )
        if exposure is not None:
            updates["EXPOSURE_US"] = exposure
        try:
            document = (
                CONFIG_FILE.read_text(encoding="utf-8")
                if CONFIG_FILE.exists() else "[settings]\n"
            )
            tomllib.loads(document)
            content = update_settings_text(document, updates).encode("utf-8")
            output = QSaveFile(str(CONFIG_FILE))
            if not output.open(QSaveFile.WriteOnly):
                raise OSError(output.errorString())
            if output.write(content) != len(content) or not output.commit():
                raise OSError(output.errorString())
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            QMessageBox.critical(self, "Cannot save configuration", str(exc))

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
            QTimer.singleShot(0, self._size_main_layout)

    def closeEvent(self, event):
        self.gui_timer.stop()
        self.marketing_timer.stop()
        self.message_progress_timer.stop()
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
