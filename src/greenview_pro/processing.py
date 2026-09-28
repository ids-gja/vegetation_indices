"""Camera acquisition and vegetation-index image processing."""

import time
from importlib.resources import files
from threading import Lock

import cv2
import matplotlib
import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension
from PySide6.QtCore import QThread, Signal, Slot
from ndvi_processing import NDVIProcessor, SensorConfig

LOW_DEFAULT = 5
HIGH_DEFAULT = 99
RECONNECT_SECONDS = 2.0
DEFAULT_PREVIEW_SIZE = (1280, 1280)

_cmap = matplotlib.colormaps.get_cmap("RdYlGn").resampled(256)
_colors = (_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
COLORMAP = np.ascontiguousarray(_colors.reshape(256, 1, 3)[..., ::-1])


def maximize_frame_rate(nodemap):
    throughput = nodemap.FindNode("DeviceLinkThroughputLimit")
    exposure = nodemap.FindNode("ExposureTime")
    frame_rate = nodemap.FindNode("AcquisitionFrameRate")
    throughput.SetValue(throughput.Maximum())
    exposure.SetValue(exposure.Minimum())

    link_limit = nodemap.FindNode("DeviceLinkAcquisitionFrameRateLimit")
    target_fps = min(float(link_limit.Value()), float(frame_rate.Maximum()))
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError(f"Camera reported an invalid maximum frame rate: {target_fps}")
    frame_rate.SetValue(target_fps)
    actual_fps = float(frame_rate.Value())
    if not np.isfinite(actual_fps) or actual_fps <= 0:
        raise ValueError(f"Camera reported an invalid frame rate: {actual_fps}")

    minimum = int(np.ceil(exposure.Minimum()))
    maximum = int(np.floor(min(exposure.Maximum(), 1_000_000 / actual_fps)))
    if maximum < minimum:
        raise ValueError("Maximum frame rate leaves no usable exposure range")
    for _ in range(8):
        exposure.SetValue(float(maximum))
        supported_fps = min(float(frame_rate.Maximum()), float(link_limit.Value()))
        if supported_fps >= actual_fps * (1 - 1e-5):
            frame_rate.SetValue(min(actual_fps, float(frame_rate.Maximum())))
            if min(float(frame_rate.Value()), float(link_limit.Value())) >= actual_fps * (
                1 - 1e-5
            ):
                break
        if not np.isfinite(supported_fps) or supported_fps <= 0 or maximum <= minimum:
            raise ValueError("Camera cannot sustain the selected frame rate")
        shortfall_us = int(
            np.ceil(1_000_000 / supported_fps - 1_000_000 / actual_fps)
        )
        maximum = max(minimum, maximum - max(1, shortfall_us + 1))
    else:
        raise ValueError("Camera frame rate did not stabilize at maximum exposure")
    return minimum, maximum, actual_fps


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

    def __init__(self):
        super().__init__()
        self._shutdown = False
        self._wanted = True
        self._low = LOW_DEFAULT
        self._high = HIGH_DEFAULT
        # Session settings survive disconnect/reconnect and manual close/start.
        self._saved_exposure = None
        self._pending_exposure = None
        self._exposure_min = None
        self._exposure_cap = None
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
        qe = files("ndvi_processing").joinpath("resources", "sensor_AR2020.csv")
        self._default_processor = NDVIProcessor(SensorConfig("AR2020", "GRBG", qe))
        self._processor = self._default_processor

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
        requested = int(value)
        if self._exposure_cap is not None:
            requested = max(self._exposure_min, min(requested, self._exposure_cap))
        self._saved_exposure = requested
        self._pending_exposure = requested

    @Slot(int, int)
    def set_preview_size(self, width, height):
        if width >= 2 and height >= 2:
            self._preview_size = (width, height)

    def _raw_indices(self, raw):
        return self._processor.process_raw(raw).indices

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
        exposure_minimum, exposure_maximum, _ = maximize_frame_rate(self._remote)
        self._exposure_min = exposure_minimum
        self._exposure_cap = exposure_maximum
        payload = self._remote.FindNode("PayloadSize").Value()
        exposure = self._remote.FindNode("ExposureTime")
        if self._saved_exposure is None:
            self._saved_exposure = exposure_maximum
        else:
            self._saved_exposure = max(
                exposure_minimum, min(self._saved_exposure, exposure_maximum)
            )
            exposure.SetValue(float(self._saved_exposure))
        self._pending_exposure = None
        self.exposure_range.emit(
            exposure_minimum, exposure_maximum, self._saved_exposure
        )

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
        self._camera_opened()
        self.state_changed.emit("Camera connected", True)

    def _camera_opened(self):
        pass

    def _frame_received(self, raw):
        pass

    def _check_processor(self):
        pass

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
        self._camera_closed()

    def _camera_closed(self):
        pass

    def _close_library(self):
        self._close_camera()
        if self._library_open:
            try:
                ids_peak.Library.Close()
            except Exception:
                pass
            self._library_open = False

    def _process(self, raw, fps):
        if raw.ndim != 2 or min(raw.shape) < 2:
            raise RuntimeError(
                "The camera does not provide the expected 2D RAW Bayer image"
            )
        raw = raw[: raw.shape[0] // 2 * 2, : raw.shape[1] // 2 * 2]
        indices = self._raw_indices(raw)
        ndvi_raw = indices["ndvi"]
        cvi_raw = indices["cvi"]
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
                    except Exception as exc:
                        self._close_camera()
                        self.state_changed.emit(
                            "Camera disconnected - reconnecting...", False
                        )
                        self.recoverable_error.emit(str(exc))
                        continue

                try:
                    if self._pending_exposure is not None:
                        self._remote.FindNode("ExposureTime").SetValue(
                            float(self._pending_exposure)
                        )
                        self._saved_exposure = int(self._pending_exposure)
                        self._pending_exposure = None
                    buffer = self._stream.WaitForFinishedBuffer(ids_peak.Timeout(1000))
                    try:
                        image = ids_peak_ipl_extension.BufferToImage(buffer)
                        full_raw = image.get_numpy_2D()
                        self._frame_received(full_raw)
                        raw = preview_bayer(full_raw, self._preview_size)
                    finally:
                        self._stream.QueueBuffer(buffer)
                    now = time.perf_counter()
                    instant = 1 / max(now - last_frame, 1e-6)
                    smooth_fps = (
                        instant if smooth_fps == 0 else 0.9 * smooth_fps + 0.1 * instant
                    )
                    last_frame = now
                    self._check_processor()
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
