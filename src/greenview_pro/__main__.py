#!/usr/bin/env python3
"""IDS GreenView Pro v2.8 - RAW, NDVI, CVI and TVI live visualization."""

import sys
import time
from threading import Lock
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QSize, QPropertyAnimation, QParallelAnimationGroup, QEasingCurve
from PySide6.QtGui import QCloseEvent, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QStackedWidget, QGraphicsOpacityEffect, QSizePolicy,
    QMessageBox, QPushButton, QSlider, QToolButton, QVBoxLayout, QWidget,
)
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension

APP_NAME = "IDS GreenView Pro"
VERSION = "2.8"
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
ICON_DIR = BASE_DIR / "assets" / "icons"
SNAPSHOT_DIR = PROJECT_ROOT / "snapshots"
COLOR_IMAGE_FILE = BASE_DIR / "assets" / "images" / "color_image.jpg"
MARKETING_FILE = BASE_DIR / "config" / "marketing_messages_ndvi_v1_1.txt"
GUI_INTERVAL_MS = 66
LOW_DEFAULT = 5
HIGH_DEFAULT = 99
RECONNECT_SECONDS = 2.0
PREVIEW_MAX_PIXELS = 1280 * 720

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


def preview_size(height, width):
    """Return a bounded preview size without upscaling a camera image."""
    pixels = height * width
    if pixels <= PREVIEW_MAX_PIXELS:
        return width, height
    scale = np.sqrt(PREVIEW_MAX_PIXELS / pixels)
    return max(1, int(width * scale)), max(1, int(height * scale))


def center_crop_16_9(image):
    if image is None: return None
    h,w=image.shape[:2]; target=16/9; current=w/max(h,1)
    if current>target:
        nw=max(1,int(round(h*target))); x=(w-nw)//2; return image[:,x:x+nw]
    nh=max(1,int(round(w/target))); y=(h-nh)//2; return image[y:y+nh,:]

def normalize_with_bounds(image, lo, hi):
    if hi <= lo:return np.zeros_like(image,dtype=np.float32)
    return np.nan_to_num(np.clip((image-np.float32(lo))/np.float32(hi-lo),0,1),nan=0,posinf=1,neginf=0).astype(np.float32)
def full_percentile_bounds(image,low,high):
    finite=image[np.isfinite(image)]
    if finite.size==0:return 0.0,1.0
    return float(np.percentile(finite,low)),float(np.percentile(finite,high))
def stats(image):
    return float(np.min(image)), float(np.max(image)), float(np.mean(image))


class CameraWorker(QThread):
    frames_ready = Signal(object, object, object)
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
        self._latest = None
        self._latest_version = 0
        self._latest_lock = Lock()
        self._normalization_frame=0;self._ndvi_bounds=None;self._cvi_bounds=None;self._bounds_update_interval=10;self._bounds_smoothing=np.float32(0.25)

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
            self._ndvi_bounds = None; self._cvi_bounds = None

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
        output_size = preview_size(*red_raw.shape)
        if output_size != (red_raw.shape[1], red_raw.shape[0]):
            red_raw = cv2.resize(red_raw, output_size, interpolation=cv2.INTER_AREA)
            green_raw = cv2.resize(green_raw, output_size, interpolation=cv2.INTER_AREA)
            blue_raw = cv2.resize(blue_raw, output_size, interpolation=cv2.INTER_AREA)
        # RAW is a grayscale preview: one Bayer phase avoids resizing the full sensor image.
        raw_preview = raw[0::2, 0::2]
        if output_size != (raw_preview.shape[1], raw_preview.shape[0]):
            raw_preview = cv2.resize(raw_preview, output_size, interpolation=cv2.INTER_AREA)

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
            ndvi_raw=(nir-red)/(nir+red+np.float32(0.01)); cvi_raw=(nir*red)/(green*green+np.float32(0.01))
        self._normalization_frame += 1
        if self._ndvi_bounds is None or self._normalization_frame % self._bounds_update_interval == 0:
            ndvi_new=full_percentile_bounds(ndvi_raw,self._low,self._high);cvi_new=full_percentile_bounds(cvi_raw,self._low,self._high)
            if self._ndvi_bounds is None:self._ndvi_bounds,self._cvi_bounds=ndvi_new,cvi_new
            else:
                a=self._bounds_smoothing;self._ndvi_bounds=tuple((1-a)*o+a*n for o,n in zip(self._ndvi_bounds,ndvi_new));self._cvi_bounds=tuple((1-a)*o+a*n for o,n in zip(self._cvi_bounds,cvi_new))
        ndvi=heatmap(normalize_with_bounds(ndvi_raw,*self._ndvi_bounds));cvi=heatmap(normalize_with_bounds(cvi_raw,*self._cvi_bounds))
        maximum = (
            float(np.iinfo(raw.dtype).max)
            if np.issubdtype(raw.dtype, np.integer)
            else max(float(np.max(raw_preview)), 1)
        )
        raw8 = (
            raw_preview
            if raw_preview.dtype == np.uint8
            else cv2.convertScaleAbs(raw_preview, alpha=255 / maximum)
        )
        raw_bgr = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)
        return tuple(
            np.ascontiguousarray(center_crop_16_9(image))
            for image in (raw_bgr, ndvi, cvi)
        )
    def latest(self):
        with self._latest_lock:return self._latest,self._latest_version
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
                        raw = image.get_numpy_2D().copy()
                    finally:
                        self._stream.QueueBuffer(buffer)
                    now = time.perf_counter()
                    instant = 1 / max(now - last_frame, 1e-6)
                    smooth_fps = instant if smooth_fps == 0 else 0.9 * smooth_fps + 0.1 * instant
                    last_frame = now
                    result=self._process(raw, smooth_fps)
                    with self._latest_lock:self._latest=result;self._latest_version+=1
                except Exception as exc:
                    connected = False
                    self._close_camera()
                    self.state_changed.emit("Camera disconnected - reconnecting...", False)
                    self.recoverable_error.emit(str(exc))
        finally:
            self._close_library()


class ImageCard(QFrame):
    IMAGE_MARGIN=12
    def __init__(self,title,subtitle="",smooth=True):
        super().__init__();self.setObjectName("imageCard");layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);self.image=QLabel("Waiting for image...");self.image.setObjectName("imageArea");self.image.setAlignment(Qt.AlignCenter);self.image.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored);self.image.setMinimumSize(0,0);layout.addWidget(self.image,1);self.overlay=QLabel(title,self);self.overlay.setObjectName("imageOverlay");self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents,True);self._array=None;self._smooth=smooth
    def set_image(self,array):self._array=np.ascontiguousarray(center_crop_16_9(array));self._refresh()
    def clear_image(self):self._array=None;self.image.setPixmap(QPixmap());self.image.setText("Waiting for image...");self.overlay.hide()
    def _refresh(self):
        if self._array is None or self.image.width()<2 or self.image.height()<2:return
        h,w=self._array.shape[:2];q=QImage(self._array.data,w,h,self._array.strides[0],QImage.Format_BGR888);mode=Qt.SmoothTransformation if self._smooth else Qt.FastTransformation;pix=QPixmap.fromImage(q).scaled(self.image.size(),Qt.KeepAspectRatio,mode);self.image.setPixmap(pix);self.image.setText("");pos=self.image.mapTo(self,self.image.rect().topLeft());xo=max(0,(self.image.width()-pix.width())//2);yo=max(0,(self.image.height()-pix.height())//2);self.overlay.adjustSize();self.overlay.move(pos.x()+xo+self.IMAGE_MARGIN,pos.y()+yo+self.IMAGE_MARGIN);self.overlay.show();self.overlay.raise_()
    def resizeEvent(self,e):super().resizeEvent(e);self._refresh()
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
        self.resize(1600, 950)
        self.demo_mode = False
        self._normal_geometry = None
        self._last_images = None
        self._last_gui_version = -1
        self._marketing_index = 0
        self._animation = None
        self.marketing_settings, self.marketing_messages = self.load_marketing()
        self.color_image = center_crop_16_9(cv2.imread(str(COLOR_IMAGE_FILE)))
        self._build()
        self._style()
        self.worker = CameraWorker()
        self.worker.state_changed.connect(self.update_state)
        self.worker.exposure_range.connect(self.configure_exposure)
        self.start_requested.connect(self.worker.start_camera)
        self.close_requested.connect(self.worker.close_camera)
        self.exposure_requested.connect(self.worker.set_exposure)
        self.percentiles_requested.connect(self.worker.set_percentiles)
        self.white_balance_requested.connect(self.worker.set_white_balance)
        self.worker.start()
        self.gui_timer = QTimer(self)
        self.gui_timer.setInterval(GUI_INTERVAL_MS)
        self.gui_timer.timeout.connect(self.pull_latest)
        self.gui_timer.start()
        self.marketing_timer = QTimer(self)
        self.marketing_timer.setInterval(int(float(self.marketing_settings.get("DISPLAY_TIME", 12)) * 1000))
        self.marketing_timer.timeout.connect(self.next_message)

    def _build(self):
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.main_page, self.demo_page = QWidget(), QWidget()
        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.demo_page)
        self._build_main()
        self._build_demo()

    def _make_cards(self):
        cards = [ImageCard("COLOR IMAGE",smooth=True), ImageCard("RAW",smooth=False), ImageCard("NDVI",smooth=True), ImageCard("CVI",smooth=True)]
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
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        layout.addLayout(grid, 1)

    def _build_main(self):
        root = QVBoxLayout(self.main_page)
        root.setContentsMargins(18, 14, 18, 8)
        root.setSpacing(8)
        header = QHBoxLayout()
        logo = QLabel(); logo.setFixedSize(170, 52)
        logo_path = ICON_DIR / "ids-logo_black_rgb.png"
        if logo_path.exists(): logo.setPixmap(QPixmap(str(logo_path)).scaled(logo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: logo.setText("IDS")
        title = QLabel("GreenView Pro"); title.setObjectName("title")
        self.demo_button = QPushButton("Demo Preview"); self.demo_button.clicked.connect(self.enter_demo)
        header.addWidget(logo); header.addWidget(title); header.addStretch(); header.addWidget(self.demo_button)
        root.addLayout(header)
        self.main_cards = self._make_cards()
        self._add_grid(root, self.main_cards, spacing=6)
        controls = QFrame(); controls.setObjectName("controls")
        bar = QHBoxLayout(controls)
        bar.addWidget(QLabel("Exposure"))
        self.exposure = QSlider(Qt.Horizontal); self.exposure.setEnabled(False)
        self.exposure.valueChanged.connect(self.change_exposure)
        self.exposure_label = QLabel("- us")
        bar.addWidget(self.exposure, 1); bar.addWidget(self.exposure_label)
        bar.addWidget(QLabel("Contrast"))
        self.low = PercentControl(LOW_DEFAULT, 0, 98); self.high = PercentControl(HIGH_DEFAULT, 1, 100)
        self.low.changed.connect(self.change_percentiles); self.high.changed.connect(self.change_percentiles)
        bar.addWidget(self.low); bar.addWidget(QLabel("to")); bar.addWidget(self.high)
        self.snapshot_button = QPushButton("Save snapshot"); self.snapshot_button.clicked.connect(self.save_snapshot)
        bar.addWidget(self.snapshot_button)
        root.addWidget(controls)

    def _build_demo(self):
        root = QVBoxLayout(self.demo_page)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        top=QWidget();top.setObjectName("demoHeader");header_layout=QHBoxLayout(top);header_layout.setContentsMargins(34,8,34,8);header_layout.setSpacing(28);self.demo_logo=QLabel();self.demo_logo.setFixedSize(285,93);logo_path=ICON_DIR/"ids-logo_black_rgb.png"
        if logo_path.exists():self.demo_logo.setPixmap(QPixmap(str(logo_path)).scaled(self.demo_logo.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        else:self.demo_logo.setText("IDS")
        self.demo_logo.setAlignment(Qt.AlignLeft|Qt.AlignVCenter);header_layout.addWidget(self.demo_logo,0,Qt.AlignVCenter);self.message_box=QWidget();message_layout=QVBoxLayout(self.message_box);message_layout.setContentsMargins(0,0,0,0);message_layout.setSpacing(4);self.demo_title=QLabel();self.demo_title.setObjectName("demoTitle");self.demo_title.setAlignment(Qt.AlignCenter);self.demo_title.setWordWrap(True);self.demo_subtitle=QLabel();self.demo_subtitle.setObjectName("demoSubtitle");self.demo_subtitle.setAlignment(Qt.AlignCenter);self.demo_subtitle.setWordWrap(True);message_layout.addStretch();message_layout.addWidget(self.demo_title);message_layout.addWidget(self.demo_subtitle);message_layout.addStretch();self.message_effect=QGraphicsOpacityEffect(self.message_box);self.message_box.setGraphicsEffect(self.message_effect);header_layout.addWidget(self.message_box,1,Qt.AlignVCenter);top.setMinimumHeight(125);root.addWidget(top)
        self.demo_cards = self._make_cards()
        self._add_grid(root, self.demo_cards, margin=3, spacing=4)
        bottom = QWidget(); bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(20, 2, 20, 4)
        self.facts = QWidget(); facts_layout = QHBoxLayout(self.facts); facts_layout.setContentsMargins(0, 0, 0, 0)
        self.fact_blocks = []
        for _ in range(3):
            block = QWidget(); block_layout = QVBoxLayout(block); block_layout.setContentsMargins(0, 0, 0, 0); block_layout.setSpacing(0)
            headline = QLabel(); headline.setObjectName("demoFact"); headline.setAlignment(Qt.AlignCenter)
            detail = QLabel(); detail.setObjectName("demoFactDetail"); detail.setAlignment(Qt.AlignCenter)
            block_layout.addWidget(headline); block_layout.addWidget(detail); facts_layout.addWidget(block, 1)
            self.fact_blocks.append((headline, detail))
        self.facts_effect = QGraphicsOpacityEffect(self.facts); self.facts.setGraphicsEffect(self.facts_effect)
        bottom_layout.addWidget(self.facts); root.addWidget(bottom)
        self.apply_message()

    def _style(self):
        s = self.marketing_settings
        bg = s.get("BACKGROUND_COLOR", "#F5F5F5")
        QApplication.instance().setFont(QFont(s.get("FONT_FAMILY", "Source Sans Pro"), 10))
        self.setStyleSheet(f"""
            QWidget {{ background:{bg}; color:#343434; }}
            QLabel {{ background:transparent; }}
            #title {{ font-size:22px; font-weight:700; }}
            QFrame#imageCard {{ background:{bg}; border:none; }}
            #imageArea {{ background:{bg}; color:#777777; border:none; }}
            #imageOverlay {{ color:rgba(225,225,225,215); background:rgba(32,58,64,100); border-radius:4px; padding:5px 9px; font-size:18px; font-weight:700; }}
            QFrame#controls {{ background:white; border:1px solid #DEDEDE; border-radius:9px; }}
            QFrame#percent {{ background:white; border:1px solid #CCCCCC; border-radius:5px; }}
            QPushButton {{ background:#008A96; color:white; border:0; border-radius:6px; padding:9px 12px; font-weight:600; }}
            #demoTitle {{ color:{s.get('TITLE_COLOR','#203A40')}; font-size:{s.get('TITLE_FONT_SIZE','52')}px; font-weight:700; }}
            #demoSubtitle {{ color:{s.get('SUBTITLE_COLOR','#4F4F4F')}; font-size:{s.get('SUBTITLE_FONT_SIZE','25')}px; font-weight:600; }}
            #demoFact {{ color:{s.get('ACCENT_COLOR','#008A96')}; font-size:{s.get('FACT_TITLE_FONT_SIZE','30')}px; font-weight:700; }}
            #demoFactDetail {{ color:{s.get('DETAIL_COLOR','#4F4F4F')}; font-size:{s.get('FACT_DETAIL_FONT_SIZE','28')}px; font-weight:600; }}
        """)

    def load_marketing(self):
        settings, messages, section, current = {}, [], "", {}
        if MARKETING_FILE.exists():
            for raw in MARKETING_FILE.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"): continue
                if line.startswith("[") and line.endswith("]"):
                    if section.startswith("MSG") and current: messages.append(self.message_from(current))
                    section, current = line[1:-1].upper(), {}
                    continue
                if "=" not in line: continue
                key, value = [x.strip() for x in line.split("=", 1)]
                if section == "SETTINGS": settings[key.upper()] = value
                elif section.startswith("MSG"): current[key.upper()] = value
            if section.startswith("MSG") and current: messages.append(self.message_from(current))
        fallback = [{"title":"MULTISPECTRAL INSIGHT","subtitle":"Live vegetation analysis","facts":[("20 MP","High spatial resolution"),("NDVI","Vegetation vitality"),("CVI","Chlorophyll-related differences")]}]
        return settings, messages or fallback

    @staticmethod
    def message_from(data):
        return {"title":data["TITLE"], "subtitle":data["SUBTITLE"], "facts":[(data[f"FACT{i}_TITLE"],data[f"FACT{i}_DETAIL"]) for i in range(1,4)]}

    def pull_latest(self):
        result, version = self.worker.latest()
        if result is None or version == self._last_gui_version: return
        self._last_gui_version = version; self._last_images = result
        cards = self.demo_cards if self.demo_mode else self.main_cards
        for card, image in zip(cards[1:], result): card.set_image(image)

    @Slot(int, int, int)
    def configure_exposure(self, minimum, maximum, current):
        self.exposure.setRange(minimum, maximum); self.exposure.setValue(current); self.exposure.setEnabled(True); self.exposure_label.setText(f"{current} us")

    def change_exposure(self, value): self.exposure_label.setText(f"{value} us"); self.exposure_requested.emit(value)
    def change_percentiles(self, _):
        low, high = self.low.value(), self.high.value(); self.low.set_limits(0, high-1); self.high.set_limits(low+1, 100); self.percentiles_requested.emit(low, high)
    @Slot(str, bool)
    def update_state(self, text, connected): pass

    def enter_demo(self):
        self.demo_mode = True; self._normal_geometry = self.normalGeometry(); self.stack.setCurrentWidget(self.demo_page); self.showFullScreen(); self.marketing_timer.start(); QTimer.singleShot(100, self.refresh_visible)
    def exit_demo(self):
        self.demo_mode = False; self.marketing_timer.stop(); self.stack.setCurrentWidget(self.main_page); self.showNormal()
        if self._normal_geometry and self._normal_geometry.isValid(): self.setGeometry(self._normal_geometry)
        QTimer.singleShot(100, self.refresh_visible)

    def apply_message(self):
        message = self.marketing_messages[self._marketing_index]; self.demo_title.setText(message["title"]); self.demo_subtitle.setText(message["subtitle"])
        for index, (headline, detail) in enumerate(self.fact_blocks): headline.setText(message["facts"][index][0]); detail.setText(message["facts"][index][1])

    def make_animation(self, effect, start, end, seconds):
        animation = QPropertyAnimation(effect, b"opacity", self); animation.setStartValue(start); animation.setEndValue(end); animation.setDuration(int(seconds*1000)); animation.setEasingCurve(QEasingCurve.InOutCubic); return animation

    def next_message(self):
        if self._animation is not None: return
        fade = float(self.marketing_settings.get("FADE_OUT_TIME", 1)); group = QParallelAnimationGroup(self)
        for effect in (self.message_effect, self.facts_effect): group.addAnimation(self.make_animation(effect, 1, 0, fade))
        def swap():
            self._marketing_index = (self._marketing_index+1) % len(self.marketing_messages); self.apply_message(); inside = QParallelAnimationGroup(self)
            for effect in (self.message_effect, self.facts_effect): inside.addAnimation(self.make_animation(effect, 0, 1, float(self.marketing_settings.get("FADE_IN_TIME",1))))
            inside.finished.connect(lambda: setattr(self, "_animation", None)); self._animation = inside; inside.start()
        group.finished.connect(swap); self._animation = group; group.start()

    def refresh_visible(self):
        for card in (self.demo_cards if self.demo_mode else self.main_cards): card._refresh()

    def save_snapshot(self):
        if self._last_images is None: return
        SNAPSHOT_DIR.mkdir(exist_ok=True); stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for name, image in zip(("color_image","raw","ndvi","cvi"),(self.color_image,)+self._last_images):
            if image is not None: cv2.imwrite(str(SNAPSHOT_DIR/f"{stamp}_{name}.png"), image)

    def keyPressEvent(self, event):
        if self.demo_mode and event.key() == Qt.Key_Escape: self.exit_demo(); return
        if self.demo_mode and event.key() == Qt.Key_Space: self.next_message(); return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.gui_timer.stop(); self.marketing_timer.stop()
        if self._animation: self._animation.stop()
        self.worker.stop_worker()
        if not self.worker.wait(3500): self.worker.terminate(); self.worker.wait()
        event.accept()

def main():
    app = QApplication(sys.argv); app.setApplicationName(APP_NAME)
    window = MainWindow(); window.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
