"""Qt worker adapting headless camera acquisition and image processing."""

import time
from threading import Lock

import numpy as np
from PySide6.QtCore import QThread, Signal, Slot
from greenview_pro.auto_exposure import ExposureSearch
from greenview_pro.camera import (
    BLACK_LEVEL_DN,
    INITIAL_EXPOSURE_US,
    CameraSession,
    configure_camera_exposure,
    ids_peak,
    maximize_frame_rate,
)
from greenview_pro.image_processing import (
    DEFAULT_PREVIEW_SIZE,
    HIGH_DEFAULT,
    LOW_DEFAULT,
    FrameProcessor,
    preview_bayer,
    validate_temporal_filter,
)

RECONNECT_SECONDS = 2.0


class CameraWorker(QThread):
    frames_ready = Signal(object, object, object)
    frame_dimensions_changed = Signal(int, int)
    state_changed = Signal(str, bool)
    exposure_range = Signal(int, int, int)
    recoverable_error = Signal(str)

    def __init__(self):
        super().__init__()
        self._shutdown = False
        self._wanted = True
        self._saved_exposure = None
        self._pending_exposure = None
        self._exposure_min = None
        self._exposure_cap = None
        self._initial_exposure = INITIAL_EXPOSURE_US
        self._black_level = BLACK_LEVEL_DN
        self._auto_exposure = None
        self._auto_exposure_available = True
        self._auto_settle_frames = 0
        self._camera = CameraSession()
        self._latest = None
        self._latest_version = 0
        self._latest_lock = Lock()
        self._settings_lock = Lock()
        self._pending_temporal = None
        self._preview_size = DEFAULT_PREVIEW_SIZE
        self._source_dimensions = None
        self._frame_processor = FrameProcessor()
        self._default_processor = self._frame_processor.default_processor
        self._processor = self._default_processor
        self._temporal_warmup_frames = 0
        self._temporal_reset_after_process = False

    @property
    def _remote(self):
        return self._camera.remote

    @_remote.setter
    def _remote(self, value):
        self._camera.remote = value

    @property
    def _ndvi_bounds(self):
        return self._frame_processor._ndvi_bounds

    @_ndvi_bounds.setter
    def _ndvi_bounds(self, value):
        self._frame_processor._ndvi_bounds = value

    @property
    def _cvi_bounds(self):
        return self._frame_processor._cvi_bounds

    @_cvi_bounds.setter
    def _cvi_bounds(self, value):
        self._frame_processor._cvi_bounds = value

    def start_camera(self):
        self._wanted = True

    @Slot()
    def close_camera(self):
        self._wanted = False
        self._close_camera()
        self.state_changed.emit("Camera stopped", False)

    @Slot(str, int, int)
    def set_percentiles(self, index, low, high):
        self._frame_processor.set_percentiles(index, low, high)

    @Slot(bool, float)
    def set_temporal_filter(self, enabled, weight):
        validate_temporal_filter(enabled, weight)
        if not self.isRunning():
            self._frame_processor.set_temporal_filter(enabled, weight)
        else:
            with self._settings_lock:
                self._pending_temporal = (enabled, weight)

    @Slot(int)
    def set_exposure(self, value):
        requested = int(value)
        if self._exposure_cap is not None:
            requested = max(self._exposure_min, min(requested, self._exposure_cap))
        self._saved_exposure = requested
        self._pending_exposure = requested
        self._auto_exposure = None
        self._auto_exposure_available = False
        self._auto_settle_frames = 0
        self._temporal_warmup_frames = 2

    @Slot(int, int)
    def set_preview_size(self, width, height):
        if width >= 2 and height >= 2:
            self._preview_size = (width, height)

    @Slot(object)
    def set_preview_sizes(self, sizes):
        if len(sizes) != 3 or any(min(size) < 2 for size in sizes):
            raise ValueError("Expected three valid preview sizes")
        self._preview_size = tuple(tuple(size) for size in sizes)

    def _raw_indices(self, raw):
        self._frame_processor.processor = self._processor
        self._frame_processor.black_level = self._black_level
        return self._frame_processor.indices(raw)

    def stop_worker(self):
        self._shutdown = True
        self._wanted = False
        self.requestInterruption()

    def _open_camera(self):
        requested_exposure = (
            self._initial_exposure
            if self._auto_exposure_available
            else self._saved_exposure
        )
        exposure_minimum, exposure_maximum, self._saved_exposure = self._camera.open(
            requested_exposure
        )
        self._frame_processor.reset_temporal()
        self._temporal_warmup_frames = 0
        self._temporal_reset_after_process = False
        self._exposure_min = exposure_minimum
        self._exposure_cap = exposure_maximum
        self._black_level = self._camera.black_level
        self._auto_exposure = (
            ExposureSearch(
                exposure_minimum, exposure_maximum, self._saved_exposure,
                white_level=self._camera.white_level,
            )
            if self._auto_exposure_available
            else None
        )
        self._auto_settle_frames = 2 if self._auto_exposure is not None else 0
        self._pending_exposure = None
        self.exposure_range.emit(
            exposure_minimum, exposure_maximum, self._saved_exposure
        )

        self._camera_opened()
        self._auto_exposure_available = False
        self.state_changed.emit("Camera connected", True)

    def _camera_opened(self):
        pass

    def _frame_received(self, raw):
        pass

    def _check_processor(self):
        pass

    def _advance_auto_exposure(self, raw):
        search = self._auto_exposure
        if search is None or search.done:
            return False
        if self._auto_settle_frames:
            self._auto_settle_frames -= 1
            return False
        requested = search.observe(raw)
        if search.done and search.safe is None:
            self.recoverable_error.emit(
                "Autoexposure: image is still saturated at minimum exposure"
            )
        if requested is None:
            return False
        actual = self._camera.set_exposure(requested)
        search.exposure = actual
        self._saved_exposure = actual
        self._auto_settle_frames = 2
        self._temporal_reset_after_process = True
        self.exposure_range.emit(self._exposure_min, self._exposure_cap, actual)
        return True

    def _close_camera(self):
        self._auto_exposure = None
        self._auto_settle_frames = 0
        self._frame_processor.reset_temporal()
        self._temporal_warmup_frames = 0
        self._temporal_reset_after_process = False
        self._camera.close()
        self._source_dimensions = None
        self._camera_closed()

    def _camera_closed(self):
        pass

    def _note_frame_dimensions(self, raw):
        if raw.ndim != 2 or min(raw.shape) < 2:
            raise RuntimeError(
                "The camera does not provide the expected 2D RAW Bayer image"
            )
        height, width = raw.shape
        dimensions = (width, height)
        if dimensions != self._source_dimensions:
            self._source_dimensions = dimensions
            self.frame_dimensions_changed.emit(width, height)

    def _close_library(self):
        self._close_camera()
        self._camera.shutdown()

    def _process(self, raw, fps):
        self._frame_processor.processor = self._processor
        self._frame_processor.black_level = self._black_level
        with self._settings_lock:
            pending_temporal = self._pending_temporal
            self._pending_temporal = None
        if pending_temporal is not None:
            self._frame_processor.set_temporal_filter(*pending_temporal)
        if self._temporal_warmup_frames:
            self._frame_processor.reset_temporal()
            self._temporal_warmup_frames -= 1
        result = self._frame_processor.render(raw, self._preview_size)
        if self._temporal_reset_after_process:
            self._frame_processor.reset_temporal()
            self._temporal_warmup_frames = 2
            self._temporal_reset_after_process = False
        return result

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
                        self._saved_exposure = self._camera.set_exposure(
                            self._pending_exposure
                        )
                        self._pending_exposure = None
                    full_raw = self._camera.read()
                    self._note_frame_dimensions(full_raw)
                    self._advance_auto_exposure(full_raw)
                    self._frame_received(full_raw)
                    now = time.perf_counter()
                    instant = 1 / max(now - last_frame, 1e-6)
                    smooth_fps = (
                        instant if smooth_fps == 0 else 0.9 * smooth_fps + 0.1 * instant
                    )
                    last_frame = now
                    self._check_processor()
                    result = self._process(full_raw, smooth_fps)
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
