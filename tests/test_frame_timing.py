from types import SimpleNamespace

import numpy as np
import pytest

import greenview_pro.processing as processing
from greenview_pro.processing import CameraWorker, maximize_frame_rate
from test_calibration_flow import FakeMap


class FakeNode:
    def __init__(self, value, minimum, maximum, events, name):
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.events = events
        self.name = name

    def Value(self):
        return self.value

    def Minimum(self):
        return self.minimum

    def Maximum(self):
        return self.maximum

    def SetValue(self, value):
        if not self.minimum <= value <= self.Maximum():
            raise ValueError(f"unsupported {self.name}: {value}")
        self.value = value
        self.events.append((self.name, value))


class FakeCamera:
    def __init__(self, readout_us=0):
        self.events = []
        self.readout_us = readout_us
        self.throughput = FakeNode(
            150_000_000, 100_000_000, 300_000_000, self.events, "throughput"
        )
        self.exposure = FakeNode(80_000, 17.5, 200_000, self.events, "exposure")
        self.fps = FakeNode(7.2, 1.0, 100, self.events, "fps")

    def FindNode(self, name):
        if name == "DeviceLinkThroughputLimit":
            return self.throughput
        if name == "DeviceLinkAcquisitionFrameRateLimit":
            return self.link_limit
        if name == "AcquisitionFrameRate":
            return self.fps
        if name == "ExposureTime":
            return self.exposure
        raise KeyError(name)

    @property
    def link_limit(self):
        camera = self

        class Limit:
            def Value(self):
                transfer_fps = 14.4 if camera.throughput.value == 300_000_000 else 7.2
                return min(
                    transfer_fps,
                    1_000_000 / (camera.exposure.value + camera.readout_us),
                )

        return Limit()


def test_maximize_frame_rate_sets_throughput_before_fps_and_caps_exposure():
    camera = FakeCamera()
    minimum, maximum, fps = maximize_frame_rate(camera)
    assert (minimum, maximum) == (18, 69_444)
    assert fps == pytest.approx(14.4)
    assert camera.exposure.value == maximum
    assert [name for name, _ in camera.events] == [
        "throughput", "exposure", "fps", "exposure", "fps"
    ]


def test_exposure_cap_accounts_for_camera_readout_time_when_fps_drops():
    camera = FakeCamera(readout_us=1_000)
    minimum, maximum, fps = maximize_frame_rate(camera)
    assert minimum == 18
    assert 68_400 <= maximum <= 68_444
    assert fps == pytest.approx(14.4)
    assert camera.link_limit.Value() == pytest.approx(fps)


def test_worker_never_requests_exposure_above_frame_rate_cap():
    worker = CameraWorker()
    worker._exposure_min = 18
    worker._exposure_cap = 69_444
    worker.set_exposure(100_000)
    assert worker._pending_exposure == 69_444


def test_manual_exposure_change_starts_new_temporal_history():
    worker = CameraWorker()
    original = worker._default_processor.process_channels
    observed = []

    def record(channels):
        observed.append(channels["NIR"][0, 0])
        return original(channels)

    worker._default_processor.process_channels = record
    base = np.tile(np.array([[40, 20], [30, 40]], dtype=np.uint16), (2, 2))
    brighter = base.copy()
    brighter[1, 0] = 70
    worker._process(base, 0)
    worker.set_exposure(2400)
    worker._process(brighter, 0)
    worker._process(brighter, 0)
    assert observed == [28, 68, 68]
    worker._close_camera()
    worker._process(base, 0)
    assert observed[-1] == 28


def test_auto_exposure_change_does_not_blend_previous_exposure_into_next_frame():
    worker = CameraWorker()
    observed = []
    original = worker._default_processor.process_channels

    def record(channels):
        observed.append(channels["NIR"][0, 0])
        return original(channels)

    worker._default_processor.process_channels = record
    base = np.tile(np.array([[40, 20], [30, 40]], dtype=np.uint16), (2, 2))
    brighter = base.copy()
    brighter[1, 0] = 70
    worker._process(base, 0)
    worker._auto_exposure = SimpleNamespace(
        done=False, safe=1000, exposure=1000, observe=lambda raw: 2000
    )
    worker._exposure_min = 10
    worker._exposure_cap = 10_000
    worker._camera.set_exposure = lambda requested: requested
    assert worker._advance_auto_exposure(base)
    worker._process(base, 0)
    worker._process(brighter, 0)
    assert observed == [28, 28, 68]


def test_filter_setting_is_applied_by_worker_between_frames(monkeypatch):
    worker = CameraWorker()
    monkeypatch.setattr(worker, "isRunning", lambda: True)
    worker.set_temporal_filter(False, 0.5)
    assert worker._frame_processor.temporal_enabled
    raw = np.tile(np.array([[40, 20], [30, 40]], dtype=np.uint16), (2, 2))
    worker._process(raw, 0)
    assert not worker._frame_processor.temporal_enabled
    assert worker._frame_processor.temporal_weight == 0.5


def test_connection_starts_acquisition_only_after_initial_exposure_and_black_level(monkeypatch):
    camera = FakeCamera()
    camera.black = FakeNode(0, 0, 10, camera.events, "black")
    gains = FakeMap()
    gains.gain.values.update({"Red": 2.0, "Green": 3.0, "Blue": 4.0})
    original_find = camera.FindNode
    start = SimpleNamespace(Execute=lambda: None, WaitUntilDone=lambda: None)
    camera.FindNode = lambda name: (
        camera.black if name == "BlackLevel"
        else SimpleNamespace(Value=lambda: 1024) if name == "PayloadSize"
        else start if name == "AcquisitionStart"
        else gains.FindNode(name) if name in ("GainSelector", "Gain")
        else original_find(name)
    )
    mode = SimpleNamespace(SetCurrentEntry=lambda value: None)
    stream = SimpleNamespace(
        NodeMaps=lambda: [SimpleNamespace(FindNode=lambda name: mode)],
        NumBuffersAnnouncedMinRequired=lambda: 1,
        AllocAndAnnounceBuffer=lambda payload: object(),
        QueueBuffer=lambda buffer: None,
        StartAcquisition=lambda: assert_start_settings(camera, gains),
    )
    device = SimpleNamespace(
        RemoteDevice=lambda: SimpleNamespace(NodeMaps=lambda: [camera]),
        DataStreams=lambda: [SimpleNamespace(OpenDataStream=lambda: stream)],
    )
    manager = SimpleNamespace(
        Update=lambda: None,
        Devices=lambda: [SimpleNamespace(OpenDevice=lambda access: device)],
    )
    monkeypatch.setattr(
        processing.ids_peak, "DeviceManager",
        SimpleNamespace(Instance=lambda: manager),
    )
    monkeypatch.setattr(
        processing.ids_peak, "Library",
        SimpleNamespace(Initialize=lambda: None),
    )
    worker = CameraWorker()
    worker._open_camera()
    assert worker._auto_exposure.exposure == 3500
    assert worker._auto_settle_frames == 2


def test_reconnecting_does_not_restart_autoexposure_or_reset_its_last_exposure(monkeypatch):
    camera = FakeCamera()
    gains = FakeMap()
    original_find = camera.FindNode
    camera.FindNode = lambda name: (
        gains.FindNode(name) if name in ("GainSelector", "Gain", "BlackLevel")
        else SimpleNamespace(Value=lambda: 1024) if name == "PayloadSize"
        else SimpleNamespace(Execute=lambda: None, WaitUntilDone=lambda: None)
        if name in ("AcquisitionStart", "AcquisitionStop")
        else original_find(name)
    )
    mode = SimpleNamespace(SetCurrentEntry=lambda value: None)
    stream = SimpleNamespace(
        NodeMaps=lambda: [SimpleNamespace(FindNode=lambda name: mode)],
        NumBuffersAnnouncedMinRequired=lambda: 1,
        AllocAndAnnounceBuffer=lambda payload: object(),
        QueueBuffer=lambda buffer: None,
        StartAcquisition=lambda: None,
        StopAcquisition=lambda: None,
        Flush=lambda mode: None,
        AnnouncedBuffers=lambda: [],
    )
    device = SimpleNamespace(
        RemoteDevice=lambda: SimpleNamespace(NodeMaps=lambda: [camera]),
        DataStreams=lambda: [SimpleNamespace(OpenDataStream=lambda: stream)],
    )
    manager = SimpleNamespace(
        Update=lambda: None,
        Devices=lambda: [SimpleNamespace(OpenDevice=lambda access: device)],
    )
    monkeypatch.setattr(
        processing.ids_peak, "DeviceManager",
        SimpleNamespace(Instance=lambda: manager),
    )
    monkeypatch.setattr(
        processing.ids_peak, "Library",
        SimpleNamespace(Initialize=lambda: None),
    )

    manual_worker = CameraWorker()
    manual_worker.set_exposure(2100)
    manual_worker._open_camera()
    assert manual_worker._auto_exposure is None
    assert camera.exposure.Value() == 2100
    manual_worker._close_camera()

    worker = CameraWorker()
    worker._open_camera()
    worker._auto_settle_frames = 0
    worker._advance_auto_exposure(np.full((8, 8), 100, dtype=np.uint16))
    exposure_after_search = worker._saved_exposure
    assert exposure_after_search > 3500
    worker._close_camera()
    worker._open_camera()
    assert worker._auto_exposure is None
    assert camera.exposure.Value() == exposure_after_search

    worker.set_exposure(2100)
    worker._close_camera()
    worker._open_camera()
    assert worker._auto_exposure is None
    assert camera.exposure.Value() == 2100


def assert_start_settings(camera, gains):
    assert camera.exposure.Value() == 3500
    assert camera.black.Value() == 2
    assert {key: gains.gain.values[key] for key in ("Red", "Green", "Blue")} == {
        "Red": 1.0, "Green": 1.0, "Blue": 1.0
    }


def test_unity_gain_readback_failure_is_reported():
    from greenview_pro.camera_gains import reset_color_gains

    gains = FakeMap()
    gains.gain.values["Red"] = 2.0
    gains.gain.SetValue = lambda value: None
    with pytest.raises(ValueError, match="Red.*1.0"):
        reset_color_gains(gains)
