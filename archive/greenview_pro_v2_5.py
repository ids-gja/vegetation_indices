#!/usr/bin/env python3
"""IDS GreenView Pro v2.2 - RAW, NDVI, CVI and TVI live visualization."""

import copy
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QCloseEvent, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSlider, QToolButton, QVBoxLayout, QWidget,
)
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension

APP_NAME = "IDS GreenView Pro"
VERSION = "2.5"
BASE_DIR = Path(__file__).resolve().parent
ICON_DIR = BASE_DIR / "icons"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
LOW_DEFAULT = 5
HIGH_DEFAULT = 99
RECONNECT_SECONDS = 2.0

CCM = np.array([
    [0.116014, -0.017513, -0.016800],
    [-0.000626, 0.077999, -0.013066],
    [-0.092187, -0.045365, 0.129469],
], dtype=np.float32)

_cmap = matplotlib.colormaps.get_cmap("RdYlGn").resampled(256)
_colors = (_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
COLORMAP = np.ascontiguousarray(_colors.reshape(256, 1, 3)[..., ::-1])


def load_icon(name):
    path = ICON_DIR / name
    return QIcon(str(path)) if path.exists() else QIcon()


def normalized(image, low, high):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    result = (image - lo) / (hi - lo)
    return np.nan_to_num(np.clip(result, 0, 1), nan=0, posinf=1, neginf=0).astype(np.float32)


def heatmap(image):
    return cv2.applyColorMap(np.ascontiguousarray((image * 255).astype(np.uint8)), COLORMAP)


def stats(image):
    return float(np.min(image)), float(np.max(image)), float(np.mean(image))


class CameraWorker(QThread):
    frames_ready = Signal(object, object, object, object, dict)
    state_changed = Signal(str, bool)
    exposure_range = Signal(int, int, int)
    recoverable_error = Signal(str)
    white_balance_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._shutdown = False
        self._wanted = True
        self._low = LOW_DEFAULT
        self._high = HIGH_DEFAULT
        # Session settings survive disconnect/reconnect and manual close/start.
        self._saved_exposure = None
        self._pending_exposure = None
        # Host-side auto white balance. This is independent of the camera node
        # BalanceWhiteAuto and therefore also works when that node is unavailable.
        self._white_balance_mode = "Off"
        self._pending_white_balance = None
        self._wb_gains = np.ones(3, dtype=np.float32)  # red, green, blue
        self._wb_once_frames_remaining = 0
        self._wb_frame_counter = 0
        self._library_open = False
        self._device = None
        self._remote = None
        self._stream = None

    def start_camera(self):
        self._wanted = True

    @Slot()
    def close_camera(self):
        self._wanted = False
        self._close_camera()
        self.state_changed.emit("Camera stopped", False)

    @Slot(int, int)
    def set_percentiles(self, low, high):
        if 0 <= low < high <= 100:
            self._low, self._high = low, high

    @Slot(int)
    def set_exposure(self, value):
        self._saved_exposure = int(value)
        self._pending_exposure = int(value)

    @Slot(str)
    def set_white_balance(self, mode):
        if mode in ("Off", "Once", "Continuous"):
            self._pending_white_balance = mode

    def stop_worker(self):
        self._shutdown = True
        self._wanted = False
        self.requestInterruption()

    def _open_camera(self):
        if not self._library_open:
            ids_peak.Library.Initialize()
            self._library_open = True
        manager = ids_peak.DeviceManager.Instance()
        manager.Update()
        devices = manager.Devices()
        if len(devices) == 0:
            raise RuntimeError("No IDS camera found")

        self._device = devices[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        self._remote = self._device.RemoteDevice().NodeMaps()[0]
        payload = self._remote.FindNode("PayloadSize").Value()
        exposure = self._remote.FindNode("ExposureTime")
        exposure_minimum = int(np.ceil(exposure.Minimum()))
        exposure_maximum = int(np.floor(exposure.Maximum()))
        if self._saved_exposure is None:
            self._saved_exposure = int(exposure.Value())
        else:
            self._saved_exposure = max(
                exposure_minimum, min(self._saved_exposure, exposure_maximum)
            )
            exposure.SetValue(float(self._saved_exposure))
        self._pending_exposure = None
        self.exposure_range.emit(
            exposure_minimum, exposure_maximum, self._saved_exposure
        )

        # Host white balance state is retained across reconnects.
        self.white_balance_changed.emit(self._white_balance_mode)

        self._stream = self._device.DataStreams()[0].OpenDataStream()
        try:
            self._stream.NodeMaps()[0].FindNode("StreamBufferHandlingMode").SetCurrentEntry("NewestOnly")
        except Exception:
            pass
        for _ in range(self._stream.NumBuffersAnnouncedMinRequired()):
            buffer = self._stream.AllocAndAnnounceBuffer(payload)
            self._stream.QueueBuffer(buffer)
        self._stream.StartAcquisition()
        node = self._remote.FindNode("AcquisitionStart")
        node.Execute()
        node.WaitUntilDone()
        self.state_changed.emit("Camera connected", True)

    def _close_camera(self):
        try:
            if self._remote is not None:
                node = self._remote.FindNode("AcquisitionStop")
                node.Execute()
                node.WaitUntilDone()
        except Exception:
            pass
        try:
            if self._stream is not None:
                self._stream.StopAcquisition()
                self._stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                for buffer in self._stream.AnnouncedBuffers():
                    self._stream.RevokeBuffer(buffer)
        except Exception:
            pass
        self._stream = None
        self._remote = None
        self._device = None

    def _close_library(self):
        self._close_camera()
        if self._library_open:
            try:
                ids_peak.Library.Close()
            except Exception:
                pass
            self._library_open = False

    def _update_host_white_balance(self, red_raw, green_raw, blue_raw):
        """Update gray-world gains from the central 80 percent of the Bayer image."""
        height, width = red_raw.shape
        y0, y1 = int(height * 0.1), max(int(height * 0.9), 1)
        x0, x1 = int(width * 0.1), max(int(width * 0.9), 1)
        channels = (
            red_raw[y0:y1, x0:x1],
            green_raw[y0:y1, x0:x1],
            blue_raw[y0:y1, x0:x1],
        )
        means = np.array(
            [float(np.mean(channel)) for channel in channels], dtype=np.float32
        )
        means = np.maximum(means, np.float32(1e-6))
        target = float(np.mean(means))
        requested = np.clip(target / means, 0.25, 4.0).astype(np.float32)
        # Smooth updates to avoid visible pumping in continuous operation.
        alpha = np.float32(0.35 if self._white_balance_mode == "Once" else 0.08)
        self._wb_gains = (1.0 - alpha) * self._wb_gains + alpha * requested

    def _process(self, raw, fps):
        if raw.ndim != 2 or min(raw.shape) < 2:
            raise RuntimeError("The camera does not provide the expected 2D RAW Bayer image")
        raw = raw[:raw.shape[0] // 2 * 2, :raw.shape[1] // 2 * 2]
        blue_raw = raw[1::2, 0::2].astype(np.float32)
        red_raw = raw[0::2, 1::2].astype(np.float32)
        green_raw = raw[0::2, 0::2].astype(np.float32)

        self._wb_frame_counter += 1
        if self._white_balance_mode == "Continuous":
            if self._wb_frame_counter % 5 == 0:
                self._update_host_white_balance(red_raw, green_raw, blue_raw)
        elif self._white_balance_mode == "Once" and self._wb_once_frames_remaining > 0:
            self._update_host_white_balance(red_raw, green_raw, blue_raw)
            self._wb_once_frames_remaining -= 1
            if self._wb_once_frames_remaining == 0:
                self._white_balance_mode = "Off"
                self.white_balance_changed.emit("Off")

        red_raw = red_raw * self._wb_gains[0]
        green_raw = green_raw * self._wb_gains[1]
        blue = blue_raw * self._wb_gains[2]

        nir = CCM[2, 0] * red_raw + CCM[2, 1] * green_raw + CCM[2, 2] * blue
        green = CCM[0, 0] * red_raw + CCM[0, 1] * green_raw + CCM[0, 2] * blue
        red = CCM[1, 0] * red_raw + CCM[1, 1] * green_raw + CCM[1, 2] * blue
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ndvi_raw = (nir - red) / (nir + red + np.float32(0.01))
            cvi_raw = (nir * red) / (green * green + np.float32(0.01))
            tvi_raw = 0.5 * (120 * (nir - green) - 200 * (red - green))
        ndvi = normalized(ndvi_raw, self._low, self._high)
        cvi = normalized(cvi_raw, self._low, self._high)
        tvi = normalized(tvi_raw, self._low, self._high)

        if raw.dtype == np.uint8:
            raw8 = raw
        else:
            maximum = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else max(float(np.max(raw)), 1)
            raw8 = np.clip(raw.astype(np.float32) * (255 / maximum), 0, 255).astype(np.uint8)
        raw_bgr = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)
        info = {
            "fps": fps,
            "resolution": f"{raw.shape[1]} x {raw.shape[0]}",
            "raw": stats(raw), "ndvi": stats(ndvi), "cvi": stats(cvi), "tvi": stats(tvi),
        }
        return raw_bgr, heatmap(ndvi), heatmap(cvi), heatmap(tvi), info

    def run(self):
        last_attempt = 0.0
        last_frame = time.perf_counter()
        smooth_fps = 0.0
        connected = False
        try:
            while not self._shutdown and not self.isInterruptionRequested():
                if not self._wanted:
                    if connected:
                        self._close_camera()
                        connected = False
                    self.msleep(100)
                    continue

                if not connected:
                    now = time.monotonic()
                    if now - last_attempt < RECONNECT_SECONDS:
                        self.msleep(100)
                        continue
                    last_attempt = now
                    self.state_changed.emit("Connecting camera...", False)
                    try:
                        self._open_camera()
                        connected = True
                        last_frame = time.perf_counter()
                        smooth_fps = 0.0
                    except Exception:
                        self._close_camera()
                        self.state_changed.emit("Camera disconnected - reconnecting...", False)
                        continue

                try:
                    if self._pending_exposure is not None:
                        self._remote.FindNode("ExposureTime").SetValue(float(self._pending_exposure))
                        self._saved_exposure = int(self._pending_exposure)
                        self._pending_exposure = None
                    if self._pending_white_balance is not None:
                        requested_mode = self._pending_white_balance
                        self._pending_white_balance = None
                        self._white_balance_mode = requested_mode
                        if requested_mode == "Once":
                            self._wb_once_frames_remaining = 12
                        elif requested_mode == "Continuous":
                            self._wb_once_frames_remaining = 0
                        self.white_balance_changed.emit(requested_mode)
                    buffer = self._stream.WaitForFinishedBuffer(ids_peak.Timeout(1000))
                    try:
                        image = ids_peak_ipl_extension.BufferToImage(buffer)
                        raw = copy.deepcopy(image.get_numpy_2D())
                    finally:
                        self._stream.QueueBuffer(buffer)
                    now = time.perf_counter()
                    instant = 1 / max(now - last_frame, 1e-6)
                    smooth_fps = instant if smooth_fps == 0 else 0.9 * smooth_fps + 0.1 * instant
                    last_frame = now
                    self.frames_ready.emit(*self._process(raw, smooth_fps))
                except Exception as exc:
                    connected = False
                    self._close_camera()
                    self.state_changed.emit("Camera disconnected - reconnecting...", False)
                    self.recoverable_error.emit(str(exc))
        finally:
            self._close_library()


class ImageCard(QFrame):
    def __init__(self, title, subtitle):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        header = QHBoxLayout()
        name = QLabel(title); name.setObjectName("cardTitle")
        sub = QLabel(subtitle); sub.setObjectName("subtle")
        header.addWidget(name); header.addStretch(); header.addWidget(sub)
        self.image = QLabel("Waiting for camera image...")
        self.image.setObjectName("imageArea")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(290, 190)
        self.statistics = QLabel("Min: -   Max: -   Mean: -")
        self.statistics.setObjectName("subtle")
        layout.addLayout(header); layout.addWidget(self.image, 1); layout.addWidget(self.statistics)
        self._array = None

    def set_image(self, array):
        self._array = array.copy()
        self._refresh()

    def clear_image(self):
        self._array = None
        self.image.clear()
        self.image.setText("Waiting for camera image...")
        self.statistics.setText("Min: -   Max: -   Mean: -")

    def _refresh(self):
        if self._array is None:
            return
        array = np.ascontiguousarray(self._array)
        h, w = array.shape[:2]
        image = QImage(array.data, w, h, array.strides[0], QImage.Format_BGR888).copy()
        self.image.setPixmap(QPixmap.fromImage(image).scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class PercentControl(QFrame):
    changed = Signal(int)
    def __init__(self, value, minimum, maximum):
        super().__init__()
        self.setObjectName("percent")
        self._value, self._minimum, self._maximum = value, minimum, maximum
        layout = QHBoxLayout(self); layout.setContentsMargins(7, 2, 2, 2); layout.setSpacing(2)
        self.label = QLabel(); self.label.setMinimumWidth(42); self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        arrows = QVBoxLayout(); arrows.setSpacing(0); arrows.setContentsMargins(0, 0, 0, 0)
        self.up = QToolButton(); self.down = QToolButton()
        for button in (self.up, self.down): button.setObjectName("arrow"); button.setFixedSize(20, 15)
        self.up.setIcon(load_icon("arrow_up.png")); self.down.setIcon(load_icon("arrow_down.png"))
        self.up.setIconSize(QSize(14, 9)); self.down.setIconSize(QSize(14, 9))
        self.up.clicked.connect(self.increase); self.down.clicked.connect(self.decrease)
        arrows.addWidget(self.up); arrows.addWidget(self.down)
        layout.addWidget(self.label); layout.addLayout(arrows)
        self._refresh()
    def value(self): return self._value
    def set_limits(self, minimum, maximum):
        self._minimum, self._maximum = minimum, maximum
        self._value = max(minimum, min(self._value, maximum)); self._refresh()
    def increase(self):
        if self._value < self._maximum: self._value += 1; self._refresh(); self.changed.emit(self._value)
    def decrease(self):
        if self._value > self._minimum: self._value -= 1; self._refresh(); self.changed.emit(self._value)
    def _refresh(self):
        self.label.setText(f"{self._value} %")
        self.up.setEnabled(self._value < self._maximum); self.down.setEnabled(self._value > self._minimum)


class InfoRow(QWidget):
    def __init__(self, name, value="-"):
        super().__init__(); layout = QHBoxLayout(self); layout.setContentsMargins(0, 3, 0, 3)
        label = QLabel(name); label.setObjectName("subtle"); self.value = QLabel(value); self.value.setObjectName("info")
        layout.addWidget(label); layout.addStretch(); layout.addWidget(self.value)
    def set(self, text): self.value.setText(text)


class MainWindow(QMainWindow):
    start_requested = Signal()
    close_requested = Signal()
    exposure_requested = Signal(int)
    percentiles_requested = Signal(int, int)
    white_balance_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.resize(1640, 970)
        self._last_images = None
        self._build()
        self._style()
        self.worker = CameraWorker()
        self.worker.frames_ready.connect(self.update_frames)
        self.worker.state_changed.connect(self.update_state)
        self.worker.exposure_range.connect(self.configure_exposure)
        self.worker.recoverable_error.connect(self.show_transient_error)
        self.start_requested.connect(self.worker.start_camera)
        self.close_requested.connect(self.worker.close_camera)
        self.exposure_requested.connect(self.worker.set_exposure)
        self.percentiles_requested.connect(self.worker.set_percentiles)
        self.white_balance_requested.connect(self.worker.set_white_balance)
        self.worker.white_balance_changed.connect(self.update_white_balance)
        self.worker.start()

    def _build(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(24, 18, 24, 10); root.setSpacing(14)
        header = QHBoxLayout()
        self.logo = QLabel(); self.logo.setFixedSize(170, 52); self.logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        logo_path = ICON_DIR / "ids-logo_black_rgb.png"
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            target = self.logo.size() * self.devicePixelRatioF()
            scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled.setDevicePixelRatio(self.devicePixelRatioF())
            self.logo.setPixmap(scaled)
        else: self.logo.setText("IDS")
        title_box = QVBoxLayout(); title_box.setSpacing(0)
        title = QLabel("GreenView Pro"); title.setObjectName("title")
        subtitle = QLabel("Live vegetation index analysis"); subtitle.setObjectName("subtle")
        title_box.addWidget(title); title_box.addWidget(subtitle)
        self.camera_icon = QLabel(); self.camera_icon.setFixedSize(32, 32); self.camera_icon.setAlignment(Qt.AlignCenter)
        self.state_text = QLabel("Connecting camera...")
        self.fps_badge = QLabel("0.0 FPS"); self.fps_badge.setObjectName("badge")
        self.start_button = QPushButton("Camera start")
        self.start_button.setIcon(load_icon("camera.png"))
        self.start_button.setIconSize(QSize(22, 22))
        self.start_button.clicked.connect(self.start_camera)
        self.close_button = QPushButton("Camera close")
        self.close_button.setIcon(load_icon("camera_off.png"))
        self.close_button.setIconSize(QSize(22, 22))
        self.close_button.clicked.connect(self.close_camera)

        header.addWidget(self.logo); header.addSpacing(8); header.addLayout(title_box)
        header.addSpacing(22); header.addWidget(self.start_button); header.addWidget(self.close_button)
        header.addStretch()
        header.addWidget(self.camera_icon); header.addWidget(self.state_text); header.addSpacing(15); header.addWidget(self.fps_badge)
        root.addLayout(header)

        content = QHBoxLayout(); content.setSpacing(14)
        grid = QGridLayout(); grid.setSpacing(14)
        self.raw = ImageCard("RAW", "Sensor data"); self.ndvi = ImageCard("NDVI", "Vegetation vitality")
        self.cvi = ImageCard("CVI", "Chlorophyll index"); self.tvi = ImageCard("TVI", "Transformed vegetation index")
        grid.addWidget(self.raw, 0, 0); grid.addWidget(self.ndvi, 0, 1); grid.addWidget(self.cvi, 1, 0); grid.addWidget(self.tvi, 1, 1)
        content.addLayout(grid, 1)
        side = QFrame(); side.setObjectName("card"); side.setFixedWidth(245); sl = QVBoxLayout(side)
        heading = QLabel("Camera information"); heading.setObjectName("sideTitle"); sl.addWidget(heading)
        self.info_status = InfoRow("Status", "Disconnected"); self.info_fps = InfoRow("Frame rate", "0.0 FPS")
        self.info_exposure = InfoRow("Exposure", "-"); self.info_resolution = InfoRow("Resolution", "-")
        for row in (self.info_status, self.info_fps, self.info_exposure, self.info_resolution): sl.addWidget(row)
        sl.addSpacing(16); heading2 = QLabel("Statistics"); heading2.setObjectName("sideTitle"); sl.addWidget(heading2)
        self.info_ndvi = InfoRow("NDVI mean"); self.info_cvi = InfoRow("CVI mean"); self.info_tvi = InfoRow("TVI mean")
        for row in (self.info_ndvi, self.info_cvi, self.info_tvi): sl.addWidget(row)
        sl.addStretch(); hint = QLabel("Displayed index values use the normalized display range from 0 to 1. Exposure, contrast and host white-balance settings are retained during reconnects.")
        hint.setWordWrap(True); hint.setObjectName("subtle"); sl.addWidget(hint)
        content.addWidget(side); root.addLayout(content, 1)

        controls = QFrame(); controls.setObjectName("controls"); cl = QHBoxLayout(controls)
        cl.addWidget(QLabel("Exposure")); self.exposure = QSlider(Qt.Horizontal); self.exposure.setEnabled(False)
        self.exposure.valueChanged.connect(self.change_exposure); self.exposure_label = QLabel("- us"); self.exposure_label.setMinimumWidth(85)
        cl.addWidget(self.exposure, 1); cl.addWidget(self.exposure_label); cl.addSpacing(12)
        cl.addWidget(QLabel("Contrast range")); self.low = PercentControl(LOW_DEFAULT, 0, 98); self.high = PercentControl(HIGH_DEFAULT, 1, 100)
        self.low.changed.connect(self.change_percentiles); self.high.changed.connect(self.change_percentiles)
        cl.addWidget(self.low); cl.addWidget(QLabel("to")); cl.addWidget(self.high); cl.addSpacing(12)
        cl.addSpacing(12)
        cl.addWidget(QLabel("Auto-Whitebalance"))
        self.wb_once_button = QPushButton("Once")
        self.wb_auto_button = QPushButton("Continuous")
        for button in (self.wb_once_button, self.wb_auto_button):
            button.setObjectName("modeButton")
            button.setCheckable(True)
            button.setMinimumWidth(76)
        self.wb_once_button.clicked.connect(
            lambda: self.request_white_balance("Once")
        )
        self.wb_auto_button.clicked.connect(self.toggle_continuous_white_balance)
        cl.addWidget(self.wb_once_button)
        cl.addWidget(self.wb_auto_button)
        cl.addStretch()

        self.snapshot_button = QPushButton("Save snapshot")
        self.snapshot_button.setIcon(load_icon("save.png"))
        self.snapshot_button.clicked.connect(self.save_snapshot)
        self.fullscreen_button = QPushButton("Fullscreen")
        self.fullscreen_button.setIcon(load_icon("fullscreen.png"))
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        for button in (self.snapshot_button, self.fullscreen_button):
            button.setIconSize(QSize(22, 22))
            cl.addWidget(button)
        root.addWidget(controls)
        footer = QHBoxLayout(); footer.addStretch(); mark = QLabel("AI generated"); mark.setObjectName("watermark"); footer.addWidget(mark); root.addLayout(footer)
        self._camera_icon(False)

    def _style(self):
        font = QFont("Source Sans Pro"); font.setPointSize(10); QApplication.instance().setFont(font)
        self.setStyleSheet("""
            QWidget { background:#F5F5F5; color:#343434; } QLabel { background:transparent; }
            #title { font-size:22px; font-weight:700; } #sideTitle { font-size:15px; font-weight:700; color:#203A40; }
            #subtle { color:#777777; font-size:10px; } #info { font-size:10px; font-weight:600; }
            #badge { background:#203A40; color:white; padding:7px 11px; border-radius:12px; font-weight:600; }
            QFrame#card { background:white; border:1px solid #DEDEDE; border-radius:12px; }
            #cardTitle { color:#203A40; font-size:17px; font-weight:700; }
            #imageArea { background:#203A40; color:#B9E4E8; border-radius:7px; }
            QFrame#controls { background:white; border:1px solid #DEDEDE; border-radius:10px; }
            QFrame#percent { background:white; border:1px solid #CCCCCC; border-radius:5px; }
            QToolButton#arrow { background:transparent; border:0; padding:0; }
            QToolButton#arrow:hover { background:#E5F3F4; } QToolButton#arrow:pressed { background:#B9E4E8; }
            QPushButton { background:#008A96; color:white; border:0; border-radius:6px; padding:9px 12px; font-weight:600; }
            QPushButton:hover { background:#007E88; } QPushButton:pressed { background:#00666E; padding-top:10px; padding-bottom:8px; }
            QPushButton:disabled { background:#B7D8DB; color:#F5F5F5; }
            QPushButton#modeButton { background:#E5F3F4; color:#203A40; padding:8px 10px; }
            QPushButton#modeButton:hover { background:#B9E4E8; }
            QPushButton#modeButton:checked { background:#008A96; color:white; }
            QSlider::groove:horizontal { height:5px; background:#CCCCCC; border-radius:2px; }
            QSlider::sub-page:horizontal { background:#008A96; border-radius:2px; }
            QSlider::handle:horizontal { background:#008A96; width:17px; margin:-6px 0; border-radius:8px; }
            #watermark { color:#A6A6A6; font-size:9px; font-weight:600; }
        """)

    def _camera_icon(self, connected):
        name = "camera.png" if connected else "camera_off.png"; path = ICON_DIR / name
        if path.exists():
            pix = QPixmap(str(path))
            target = self.camera_icon.size() * self.devicePixelRatioF()
            scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled.setDevicePixelRatio(self.devicePixelRatioF())
            self.camera_icon.setPixmap(scaled)
        else: self.camera_icon.setText("●"); self.camera_icon.setStyleSheet("color:#71C090" if connected else "color:#C94C4C")

    @Slot(int, int, int)
    def configure_exposure(self, minimum, maximum, current):
        self.exposure.blockSignals(True); self.exposure.setRange(minimum, maximum); self.exposure.setValue(current); self.exposure.blockSignals(False)
        self.exposure.setEnabled(True); self.exposure_label.setText(f"{current} us"); self.info_exposure.set(f"{current} us")

    def change_exposure(self, value): self.exposure_label.setText(f"{value} us"); self.info_exposure.set(f"{value} us"); self.exposure_requested.emit(value)
    def change_percentiles(self, _):
        low, high = self.low.value(), self.high.value(); self.low.set_limits(0, high - 1); self.high.set_limits(low + 1, 100)
        self.percentiles_requested.emit(low, high)

    def request_white_balance(self, mode):
        self.white_balance_requested.emit(mode)

    def toggle_continuous_white_balance(self):
        mode = "Continuous" if self.wb_auto_button.isChecked() else "Off"
        self.request_white_balance(mode)

    @Slot(str)
    def update_white_balance(self, mode):
        self.wb_once_button.setChecked(mode == "Once")
        self.wb_auto_button.setChecked(mode == "Continuous")
        self.wb_once_button.setEnabled(True)
        self.wb_auto_button.setEnabled(True)

    def start_camera(self):
        self.start_button.setEnabled(False); self.close_button.setEnabled(True); self.state_text.setText("Connecting camera..."); self.start_requested.emit()
    def close_camera(self):
        self.close_requested.emit(); self.start_button.setEnabled(True); self.close_button.setEnabled(False); self.clear_images()

    @Slot(str, bool)
    def update_state(self, text, connected):
        self.state_text.setText(text); self._camera_icon(connected); self.info_status.set("Connected" if connected else ("Stopped" if text == "Camera stopped" else "Disconnected"))
        self.start_button.setEnabled(connected is False and text == "Camera stopped"); self.close_button.setEnabled(text != "Camera stopped")
        if not connected and "disconnected" in text.lower(): self.clear_images()

    @Slot(str)
    def show_transient_error(self, message):
        self.state_text.setToolTip(message)

    def clear_images(self):
        self._last_images = None
        for card in (self.raw, self.ndvi, self.cvi, self.tvi): card.clear_image()
        self.fps_badge.setText("0.0 FPS"); self.info_fps.set("0.0 FPS"); self.info_resolution.set("-")
        self.info_ndvi.set("-"); self.info_cvi.set("-"); self.info_tvi.set("-")

    @Slot(object, object, object, object, dict)
    def update_frames(self, raw, ndvi, cvi, tvi, data):
        self._last_images = tuple(x.copy() for x in (raw, ndvi, cvi, tvi))
        for card, image, key in zip((self.raw, self.ndvi, self.cvi, self.tvi), (raw, ndvi, cvi, tvi), ("raw", "ndvi", "cvi", "tvi")):
            card.set_image(image); mn, mx, mean = data[key]; card.statistics.setText(f"Min: {mn:.3f}   Max: {mx:.3f}   Mean: {mean:.3f}")
        fps = f"{data['fps']:.1f} FPS"; self.fps_badge.setText(fps); self.info_fps.set(fps); self.info_resolution.set(data["resolution"])
        self.info_ndvi.set(f"{data['ndvi'][2]:.3f}"); self.info_cvi.set(f"{data['cvi'][2]:.3f}"); self.info_tvi.set(f"{data['tvi'][2]:.3f}")

    def feedback(self, button, text, duration=900):
        old = button.text(); button.setText(text); button.setStyleSheet("background:#71C090;")
        QTimer.singleShot(duration, lambda: (button.setText(old), button.setStyleSheet("")))

    def save_snapshot(self):
        if self._last_images is None:
            QMessageBox.information(self, "Snapshot", "No camera image is available yet."); return
        SNAPSHOT_DIR.mkdir(exist_ok=True); stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        failed = [name for name, image in zip(("raw", "ndvi", "cvi", "tvi"), self._last_images) if not cv2.imwrite(str(SNAPSHOT_DIR / f"{stamp}_{name}.png"), image)]
        if failed: QMessageBox.warning(self, "Snapshot", "Could not save: " + ", ".join(failed)); return
        self.feedback(self.snapshot_button, "Saved")

    def toggle_fullscreen(self):
        if self.isFullScreen(): self.showNormal(); self.fullscreen_button.setText("Fullscreen")
        else: self.showFullScreen(); self.fullscreen_button.setText("Window mode")
        self.feedback(self.fullscreen_button, self.fullscreen_button.text(), 300)

    def closeEvent(self, event: QCloseEvent):
        self.worker.stop_worker()
        if not self.worker.wait(3500): self.worker.terminate(); self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv); app.setApplicationName(APP_NAME)
    window = MainWindow(); window.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
