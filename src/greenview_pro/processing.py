"""Camera acquisition and vegetation-index image processing."""

import time
from dataclasses import replace
from threading import Lock

import cv2
import matplotlib
import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension
from PySide6.QtCore import QThread, Signal, Slot
from vegetation_indices import (
    CameraImageParameters,
    calculate_vegetation_indices,
)

LOW_DEFAULT = 5
HIGH_DEFAULT = 99
RECONNECT_SECONDS = 2.0
DEFAULT_PREVIEW_SIZE = (1280, 1280)

_cmap = matplotlib.colormaps.get_cmap("RdYlGn").resampled(256)
_colors = (_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
COLORMAP = np.ascontiguousarray(_colors.reshape(256, 1, 3)[..., ::-1])


def normalized(image, low, high):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    result = (image - lo) / (hi - lo)
    return np.nan_to_num(np.clip(result, 0, 1), nan=0, posinf=1, neginf=0).astype(
        np.float32
    )


def heatmap(image):
    return cv2.applyColorMap(
        np.ascontiguousarray((image * 255).astype(np.uint8)), COLORMAP
    )


def normalize_with_bounds(image, lo, hi):
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.nan_to_num(
        np.clip((image - np.float32(lo)) / np.float32(hi - lo), 0, 1),
        nan=0,
        posinf=1,
        neginf=0,
    ).astype(np.float32)


def full_percentile_bounds(image, low, high):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, low)), float(np.percentile(finite, high))


def stats(image):
    return float(np.min(image)), float(np.max(image)), float(np.mean(image))


def preview_bayer(raw, maximum_size=DEFAULT_PREVIEW_SIZE):
    """Copy a Bayer frame sized for a display without mixing color sites."""
    if raw.ndim != 2:
        raise ValueError("A Bayer preview requires a two-dimensional image")

    height, width = raw.shape
    maximum_width, maximum_height = maximum_size
    if maximum_width < 2 or maximum_height < 2:
        raise ValueError("Preview dimensions must be at least two pixels")

    scale = min(1.0, maximum_width / width, maximum_height / height)
    target_width = max(2, int(width * scale) // 2 * 2)
    target_height = max(2, int(height * scale) // 2 * 2)
    if target_width == width and target_height == height:
        return raw.copy()

    target_shape = (target_width // 2, target_height // 2)
    preview = np.empty((target_height, target_width), dtype=raw.dtype)
    preview[0::2, 1::2] = cv2.resize(
        raw[0::2, 1::2], target_shape, interpolation=cv2.INTER_AREA
    )
    preview[0::2, 0::2] = cv2.resize(
        raw[0::2, 0::2], target_shape, interpolation=cv2.INTER_AREA
    )
    preview[1::2, 0::2] = cv2.resize(
        raw[1::2, 0::2], target_shape, interpolation=cv2.INTER_AREA
    )
    preview[1::2, 1::2] = cv2.resize(
        raw[1::2, 1::2], target_shape, interpolation=cv2.INTER_AREA
    )
    return preview


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
        self._normalization_frame = 0
        self._ndvi_bounds = None
        self._cvi_bounds = None
        self._bounds_update_interval = 10
        self._bounds_smoothing = np.float32(0.25)
        self._preview_size = DEFAULT_PREVIEW_SIZE
        self._image_parameters = CameraImageParameters()

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
            self._ndvi_bounds = None
            self._cvi_bounds = None

    @Slot(int)
    def set_exposure(self, value):
        self._saved_exposure = int(value)
        self._pending_exposure = int(value)

    @Slot(str)
    def set_white_balance(self, mode):
        if mode in ("Off", "Once", "Continuous"):
            self._pending_white_balance = mode

    @Slot(int, int)
    def set_preview_size(self, width, height):
        if width >= 2 and height >= 2:
            self._preview_size = (width, height)

    def set_image_parameters(self, parameters):
        """Apply calibration and Bayer metadata read from the active camera."""
        self._image_parameters = parameters

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
            self._stream.NodeMaps()[0].FindNode(
                "StreamBufferHandlingMode"
            ).SetCurrentEntry("NewestOnly")
        except Exception:
            pass
        buffer_count = max(self._stream.NumBuffersAnnouncedMinRequired(), 10)
        for _ in range(buffer_count):
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
            raise RuntimeError(
                "The camera does not provide the expected 2D RAW Bayer image"
            )
        raw = raw[: raw.shape[0] // 2 * 2, : raw.shape[1] // 2 * 2]
        self._wb_frame_counter += 1
        if self._white_balance_mode == "Continuous":
            if self._wb_frame_counter % 5 == 0:
                blue_raw = raw[1::2, 0::2].astype(np.float32)
                red_raw = raw[0::2, 1::2].astype(np.float32)
                green_raw = raw[0::2, 0::2].astype(np.float32)
                self._update_host_white_balance(red_raw, green_raw, blue_raw)
        elif self._white_balance_mode == "Once" and self._wb_once_frames_remaining > 0:
            blue_raw = raw[1::2, 0::2].astype(np.float32)
            red_raw = raw[0::2, 1::2].astype(np.float32)
            green_raw = raw[0::2, 0::2].astype(np.float32)
            self._update_host_white_balance(red_raw, green_raw, blue_raw)
            self._wb_once_frames_remaining -= 1
            if self._wb_once_frames_remaining == 0:
                self._white_balance_mode = "Off"
                self.white_balance_changed.emit("Off")

        parameters = replace(
            self._image_parameters,
            red_gain=float(self._wb_gains[0]),
            green_gain=float(self._wb_gains[1]),
            blue_gain=float(self._wb_gains[2]),
        )
        indices = calculate_vegetation_indices(raw, parameters)
        ndvi_raw = indices.ndvi
        cvi_raw = indices.cvi
        self._normalization_frame += 1
        if (
            self._ndvi_bounds is None
            or self._normalization_frame % self._bounds_update_interval == 0
        ):
            ndvi_new = full_percentile_bounds(ndvi_raw, self._low, self._high)
            cvi_new = full_percentile_bounds(cvi_raw, self._low, self._high)
            if self._ndvi_bounds is None:
                self._ndvi_bounds, self._cvi_bounds = ndvi_new, cvi_new
            else:
                a = self._bounds_smoothing
                self._ndvi_bounds = tuple(
                    (1 - a) * o + a * n for o, n in zip(self._ndvi_bounds, ndvi_new)
                )
                self._cvi_bounds = tuple(
                    (1 - a) * o + a * n for o, n in zip(self._cvi_bounds, cvi_new)
                )
        ndvi = heatmap(normalize_with_bounds(ndvi_raw, *self._ndvi_bounds))
        cvi = heatmap(normalize_with_bounds(cvi_raw, *self._cvi_bounds))
        maximum = (
            float(np.iinfo(raw.dtype).max)
            if np.issubdtype(raw.dtype, np.integer)
            else max(float(np.max(raw)), 1)
        )
        raw8 = (
            raw
            if raw.dtype == np.uint8
            else np.clip(raw.astype(np.float32) * (255 / maximum), 0, 255).astype(
                np.uint8
            )
        )
        raw_bgr = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)
        return raw_bgr, ndvi, cvi

    def latest(self):
        with self._latest_lock:
            return self._latest, self._latest_version

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
                        self.state_changed.emit(
                            "Camera disconnected - reconnecting...", False
                        )
                        continue

                try:
                    if self._pending_exposure is not None:
                        self._remote.FindNode("ExposureTime").SetValue(
                            float(self._pending_exposure)
                        )
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
                        raw = preview_bayer(image.get_numpy_2D(), self._preview_size)
                    finally:
                        self._stream.QueueBuffer(buffer)
                    now = time.perf_counter()
                    instant = 1 / max(now - last_frame, 1e-6)
                    smooth_fps = (
                        instant if smooth_fps == 0 else 0.9 * smooth_fps + 0.1 * instant
                    )
                    last_frame = now
                    result = self._process(raw, smooth_fps)
                    with self._latest_lock:
                        self._latest = result
                        self._latest_version += 1
                except Exception as exc:
                    connected = False
                    self._close_camera()
                    self.state_changed.emit(
                        "Camera disconnected - reconnecting...", False
                    )
                    self.recoverable_error.emit(str(exc))
        finally:
            self._close_library()
