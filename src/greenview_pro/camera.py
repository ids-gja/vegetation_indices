"""Qt-independent IDS camera acquisition and device configuration."""

import re

import numpy as np
from ids_peak import ids_peak, ids_peak_ipl_extension

from greenview_pro.camera_gains import reset_color_gains

INITIAL_EXPOSURE_US = 3500
BLACK_LEVEL_DN = 2.0


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


def configure_camera_exposure(
    nodemap, minimum, maximum, requested_us=INITIAL_EXPOSURE_US
):
    black_level = nodemap.FindNode("BlackLevel")
    black_level.SetValue(BLACK_LEVEL_DN)
    if not np.isclose(black_level.Value(), BLACK_LEVEL_DN, rtol=0, atol=1e-6):
        raise ValueError(f"camera BlackLevel must be {BLACK_LEVEL_DN} DN")

    exposure = nodemap.FindNode("ExposureTime")
    exposure.SetValue(float(np.clip(requested_us, minimum, maximum)))
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

    def open(self, requested_exposure=INITIAL_EXPOSURE_US):
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
            self.exposure_min, self.exposure_cap, _ = maximize_frame_rate(self.remote)
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
        node = self.remote.FindNode("ExposureTime")
        node.SetValue(float(value))
        self.exposure = int(round(node.Value()))
        return self.exposure

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
