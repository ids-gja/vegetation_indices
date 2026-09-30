"""Qt-independent IDS camera acquisition and device configuration."""

from contextlib import contextmanager
import re

import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension

from greenview_pro.camera_gains import reset_color_gains

INITIAL_EXPOSURE_US = 3500
EXPOSURE_MIN_US = 10_000
EXPOSURE_MAX_US = 150_000
BLACK_LEVEL_DN = 0


def sensor_white_level(nodemap):
    try:
        pixel_format = nodemap.FindNode("PixelFormat").CurrentEntry().SymbolicValue()
    except (KeyError, ids_peak.Exception):
        return 255
    match = re.search(r"(?:Bayer[A-Z]{2}|Mono)(\d+)", pixel_format)
    if match is None:
        raise ValueError(f"Unsupported camera pixel format: {pixel_format}")
    return (1 << int(match.group(1))) - 1


def maximize_frame_rate(nodemap):
    throughput = nodemap.FindNode("DeviceLinkThroughputLimit")
    exposure = nodemap.FindNode("ExposureTime")
    frame_rate = nodemap.FindNode("AcquisitionFrameRate")
    current_exposure = float(exposure.Value())
    previous_fps = float(frame_rate.Value())
    previous_throughput = float(throughput.Value())
    if not np.isfinite(current_exposure) or current_exposure <= 0:
        raise ValueError(f"Camera reported an invalid exposure: {current_exposure}")
    try:
        throughput.SetValue(throughput.Maximum())
        link_limit = nodemap.FindNode("DeviceLinkAcquisitionFrameRateLimit")
        target_fps = min(
            float(link_limit.Value()),
            float(frame_rate.Maximum()),
            1_000_000 / current_exposure,
        )
        if not np.isfinite(target_fps) or target_fps <= 0:
            raise ValueError(f"Camera reported an invalid maximum frame rate: {target_fps}")
        frame_rate.SetValue(target_fps)
        actual_fps = float(frame_rate.Value())
        if not np.isfinite(actual_fps) or actual_fps <= 0:
            raise ValueError(f"Camera reported an invalid frame rate: {actual_fps}")
        if not np.isclose(exposure.Value(), current_exposure, rtol=0, atol=1e-6):
            raise RuntimeError("Camera changed exposure while maximizing frame rate")
        return actual_fps
    except (ValueError, RuntimeError, ids_peak.Exception):
        if not np.isclose(throughput.Value(), previous_throughput, rtol=0, atol=1e-6):
            throughput.SetValue(previous_throughput)
        if not np.isclose(frame_rate.Value(), previous_fps, rtol=0, atol=1e-6):
            frame_rate.SetValue(previous_fps)
        if not np.isclose(exposure.Value(), current_exposure, rtol=0, atol=1e-6):
            exposure.SetValue(current_exposure)
        raise


def _needs_slower_frame_rate(nodemap, value):
    exposure = nodemap.FindNode("ExposureTime")
    frame_rate = nodemap.FindNode("AcquisitionFrameRate")
    previous_exposure = float(exposure.Value())
    current_fps = float(frame_rate.Value())
    if not np.isfinite(current_fps) or current_fps <= 0:
        raise ValueError(f"Camera reported an invalid frame rate: {current_fps}")
    if current_fps > 1_000_000 / value:
        return True
    try:
        exposure.SetValue(float(value))
    except (ValueError, ids_peak.Exception):
        return True
    if np.isclose(exposure.Value(), value, rtol=0, atol=1):
        return False
    exposure.SetValue(previous_exposure)
    return True


def _set_exposure_with_slower_frame_rate(nodemap, value):
    exposure = nodemap.FindNode("ExposureTime")
    frame_rate = nodemap.FindNode("AcquisitionFrameRate")
    previous_exposure = float(exposure.Value())
    previous_fps = float(frame_rate.Value())
    frame_rate.SetValue(frame_rate.Minimum())
    try:
        exposure.SetValue(float(value))
        target = min(
            previous_fps,
            float(frame_rate.Maximum()),
            float(nodemap.FindNode("DeviceLinkAcquisitionFrameRateLimit").Value()),
            1_000_000 / float(exposure.Value()),
        )
        if not np.isfinite(target) or target <= 0:
            raise ValueError(f"Camera reported an invalid frame rate: {target}")
        frame_rate.SetValue(target)
        if not np.isclose(exposure.Value(), value, rtol=0, atol=1):
            raise RuntimeError("Camera changed exposure while adjusting frame rate")
    except (ValueError, RuntimeError, ids_peak.Exception):
        if not np.isclose(exposure.Value(), previous_exposure, rtol=0, atol=1e-6):
            exposure.SetValue(previous_exposure)
        frame_rate.SetValue(previous_fps)
        raise
    return int(round(exposure.Value()))


def configure_camera_exposure(
    nodemap, minimum, maximum, requested_us=None
):
    black_level = nodemap.FindNode("BlackLevel")
    black_level.SetValue(BLACK_LEVEL_DN)
    if not np.isclose(black_level.Value(), BLACK_LEVEL_DN, rtol=0, atol=1e-6):
        raise ValueError(f"camera BlackLevel must be {BLACK_LEVEL_DN} DN")

    exposure = nodemap.FindNode("ExposureTime")
    if requested_us is None:
        requested_us = exposure.Value()
    selected = float(np.clip(requested_us, minimum, maximum))
    if not np.isclose(exposure.Value(), selected, rtol=0, atol=1e-6):
        if _needs_slower_frame_rate(nodemap, selected):
            return _set_exposure_with_slower_frame_rate(nodemap, selected)
    return int(round(exposure.Value()))


class CameraSession:
    """Own an IDS stream; read() returns independent full-resolution RAW frames."""

    def __init__(self):
        self.library_open = False
        self.device = None
        self.remote = None
        self.stream = None
        self.exposure_min = None
        self.exposure_cap = None
        self.exposure = None
        self.black_level = None
        self.white_level = 255
        self.configured_exposure_min = EXPOSURE_MIN_US
        self.configured_exposure_max = EXPOSURE_MAX_US

    def set_exposure_limits(self, minimum, maximum):
        if (
            type(minimum) is not int or type(maximum) is not int
            or minimum < 1 or minimum >= maximum or maximum > 2_147_483_647
        ):
            raise ValueError("Exposure limits must be increasing positive microseconds")
        self.configured_exposure_min = minimum
        self.configured_exposure_max = maximum

    def _supported_exposure_range(self):
        exposure = self.remote.FindNode("ExposureTime")
        frame_rate = self.remote.FindNode("AcquisitionFrameRate")
        current_fps = float(frame_rate.Value())
        current_exposure = float(exposure.Value())
        minimum = max(self.configured_exposure_min, int(np.ceil(exposure.Minimum())))
        maximum = int(np.floor(exposure.Maximum()))
        if maximum < self.configured_exposure_max:
            frame_rate.SetValue(frame_rate.Minimum())
            try:
                maximum = int(np.floor(exposure.Maximum()))
            finally:
                frame_rate.SetValue(current_fps)
        maximum = min(self.configured_exposure_max, maximum)
        if not np.isclose(exposure.Value(), current_exposure, rtol=0, atol=1e-6):
            exposure.SetValue(current_exposure)
            raise RuntimeError("Camera changed exposure while reading its limits")
        if minimum > maximum:
            raise ValueError("Configured exposure range is unsupported by the camera")
        return minimum, maximum

    def open(self, requested_exposure=None):
        if self.stream is not None:
            raise RuntimeError("Camera is already open")
        if not self.library_open:
            ids_peak.Library.Initialize()
            self.library_open = True
        try:
            manager = ids_peak.DeviceManager.Instance()
            manager.Update()
            devices = manager.Devices()
            if len(devices) == 0:
                raise RuntimeError("No IDS camera found")

            self.device = devices[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            self.remote = self.device.RemoteDevice().NodeMaps()[0]
            self.white_level = sensor_white_level(self.remote)
            self.exposure_min, self.exposure_cap = self._supported_exposure_range()
            payload = self.remote.FindNode("PayloadSize").Value()
            self.exposure = configure_camera_exposure(
                self.remote, self.exposure_min, self.exposure_cap, requested_exposure
            )
            self.black_level = float(self.remote.FindNode("BlackLevel").Value())
            reset_color_gains(self.remote)

            self.stream = self.device.DataStreams()[0].OpenDataStream()
            try:
                self.stream.NodeMaps()[0].FindNode(
                    "StreamBufferHandlingMode"
                ).SetCurrentEntry("NewestOnly")
            except Exception:
                pass
            buffer_count = max(self.stream.NumBuffersAnnouncedMinRequired(), 10)
            for _ in range(buffer_count):
                buffer = self.stream.AllocAndAnnounceBuffer(payload)
                self.stream.QueueBuffer(buffer)
            self.stream.StartAcquisition()
            node = self.remote.FindNode("AcquisitionStart")
            node.Execute()
            node.WaitUntilDone()
            return self.exposure_min, self.exposure_cap, self.exposure
        except Exception:
            self.shutdown()
            raise

    def read(self, timeout_ms=1000):
        if self.stream is None:
            raise RuntimeError("Camera is not open")
        buffer = self.stream.WaitForFinishedBuffer(ids_peak.Timeout(timeout_ms))
        try:
            image = ids_peak_ipl_extension.BufferToImage(buffer)
            return image.get_numpy_2D().copy()
        finally:
            self.stream.QueueBuffer(buffer)

    def set_exposure(self, value):
        if self.remote is None:
            raise RuntimeError("Camera is not open")
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Exposure must be positive and finite")
        if _needs_slower_frame_rate(self.remote, value):
            with self._paused_acquisition():
                self.exposure = _set_exposure_with_slower_frame_rate(
                    self.remote, value
                )
        else:
            self.exposure = int(round(self.remote.FindNode("ExposureTime").Value()))
        return self.exposure

    @contextmanager
    def _paused_acquisition(self):
        if self.remote is None or self.stream is None:
            raise RuntimeError("Camera is not acquiring")
        stop = self.remote.FindNode("AcquisitionStop")
        stop.Execute()
        stop.WaitUntilDone()
        try:
            yield
        finally:
            start = self.remote.FindNode("AcquisitionStart")
            start.Execute()
            start.WaitUntilDone()

    def maximize_fps_for_exposure(self):
        with self._paused_acquisition():
            return maximize_frame_rate(self.remote)

    def close(self):
        try:
            if self.remote is not None:
                node = self.remote.FindNode("AcquisitionStop")
                node.Execute()
                node.WaitUntilDone()
        except Exception:
            pass
        try:
            if self.stream is not None:
                self.stream.StopAcquisition()
                self.stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                for buffer in self.stream.AnnouncedBuffers():
                    self.stream.RevokeBuffer(buffer)
        except Exception:
            pass
        self.stream = None
        self.remote = None
        self.device = None

    def shutdown(self):
        self.close()
        if self.library_open:
            try:
                ids_peak.Library.Close()
            finally:
                self.library_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
