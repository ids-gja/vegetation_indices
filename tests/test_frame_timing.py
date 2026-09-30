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


def test_maximize_frame_rate_preserves_exposure_and_uses_its_fps_limit():
    camera = FakeCamera()
    fps = maximize_frame_rate(camera)
    assert fps == pytest.approx(12.5)
    assert camera.exposure.value == 80_000
    assert [name for name, _ in camera.events] == [
        "throughput", "fps"
    ]


def test_maximize_frame_rate_accounts_for_readout_without_changing_exposure():
    camera = FakeCamera(readout_us=1_000)
    fps = maximize_frame_rate(camera)
    assert fps == pytest.approx(1_000_000 / 81_000)
    assert camera.link_limit.Value() == pytest.approx(fps)
    assert camera.exposure.value == 80_000


def test_worker_caps_exposure_at_configured_limit_not_frame_rate():
    worker = CameraWorker()
    worker.set_exposure_limits(10_000, 150_000)
    worker._exposure_min = 10_000
    worker._exposure_cap = 150_000
    worker.set_exposure(180_000)
    assert worker._pending_exposure == 150_000


def test_worker_defers_fullscreen_fps_change_until_between_frames():
    worker = CameraWorker()
    camera = FakeCamera()
    worker._camera.maximize_fps_for_exposure = lambda: maximize_frame_rate(camera)
    worker.request_maximum_frame_rate()
    assert camera.events == []
    worker._apply_pending_frame_rate()
    assert camera.exposure.value == 80_000
    assert camera.fps.value == pytest.approx(12.5)


def test_fullscreen_fps_request_survives_camera_reconnect():
    worker = CameraWorker()
    applied = []
    worker._camera.open = lambda requested: (10_000, 150_000, 80_000)
    worker._camera.maximize_fps_for_exposure = lambda: applied.append(True)
    worker.request_maximum_frame_rate()
    worker._open_camera()
    worker._apply_pending_frame_rate()
    worker._close_camera()
    worker._open_camera()
    worker._apply_pending_frame_rate()
    assert applied == [True, True]


def test_fullscreen_fps_failure_is_reported_without_reconnect_loop():
    worker = CameraWorker()
    errors = []
    worker.recoverable_error.connect(errors.append)

    def reject_change():
        raise ValueError("FPS node unavailable")

    worker._camera.maximize_fps_for_exposure = reject_change
    worker.request_maximum_frame_rate()
    worker._apply_pending_frame_rate()
    assert errors == ["Cannot maximize frame rate: FPS node unavailable"]
    assert not worker._maximize_on_reconnect
    assert not worker._pending_frame_rate


def test_fullscreen_frame_rate_change_pauses_acquisition_not_stream():
    from greenview_pro.camera import CameraSession

    camera = FakeCamera()
    original_find = camera.FindNode
    camera.FindNode = lambda name: (
        SimpleNamespace(
            Execute=lambda: camera.events.append((name, "execute")),
            WaitUntilDone=lambda: None,
        )
        if name in ("AcquisitionStop", "AcquisitionStart")
        else original_find(name)
    )
    session = CameraSession()
    session.remote = camera
    session.stream = object()
    session.maximize_fps_for_exposure()
    assert camera.events == [
        ("AcquisitionStop", "execute"),
        ("throughput", 300_000_000),
        ("fps", 12.5),
        ("AcquisitionStart", "execute"),
    ]
    assert camera.exposure.value == 80_000


def test_fullscreen_restores_exposure_if_camera_alters_it():
    camera = FakeCamera()
    original_set = camera.fps.SetValue

    def changes_exposure(value):
        original_set(value)
        if value != 7.2:
            camera.exposure.value = 70_000

    camera.fps.SetValue = changes_exposure
    with pytest.raises(RuntimeError, match="changed exposure"):
        maximize_frame_rate(camera)
    assert camera.exposure.value == 80_000
    assert camera.fps.value == 7.2
    assert camera.throughput.value == 150_000_000


def test_fullscreen_restores_exposure_after_fps_write_raises():
    camera = FakeCamera()
    original_set = camera.fps.SetValue

    def partly_changes_fps_and_exposure(value):
        original_set(value)
        if value != 7.2:
            camera.exposure.value = 70_000
            raise ValueError("FPS write failed")

    camera.fps.SetValue = partly_changes_fps_and_exposure
    with pytest.raises(ValueError, match="FPS write failed"):
        maximize_frame_rate(camera)
    assert camera.exposure.value == 80_000
    assert camera.fps.value == 7.2
    assert camera.throughput.value == 150_000_000


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
    assert observed == [30, 70, 70]
    worker._close_camera()
    worker._process(base, 0)
    assert observed[-1] == 30


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
    assert observed == [30, 30, 70]


def test_filter_setting_is_applied_by_worker_between_frames(monkeypatch):
    worker = CameraWorker()
    monkeypatch.setattr(worker, "isRunning", lambda: True)
    worker.set_temporal_filter(False, 0.5)
    assert worker._frame_processor.temporal_enabled
    raw = np.tile(np.array([[40, 20], [30, 40]], dtype=np.uint16), (2, 2))
    worker._process(raw, 0)
    assert not worker._frame_processor.temporal_enabled
    assert worker._frame_processor.temporal_weight == 0.5


def test_connection_preserves_cockpit_exposure_and_fps_before_acquisition(monkeypatch):
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
    assert worker._auto_exposure is None
    assert worker._saved_exposure == 80_000
    assert camera.fps.Value() == 7.2
    assert not any(
        name in ("exposure", "fps", "throughput") for name, _ in camera.events
    )


def test_reconnecting_preserves_the_preview_exposure_and_new_manual_selection(monkeypatch):
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
    manual_worker.set_exposure(21_000)
    manual_worker._open_camera()
    assert manual_worker._auto_exposure is None
    assert camera.exposure.Value() == 21_000
    manual_worker._close_camera()

    worker = CameraWorker()
    worker._open_camera()
    assert worker._auto_exposure is None
    assert worker._saved_exposure == 21_000
    worker._close_camera()
    worker._open_camera()
    assert worker._auto_exposure is None
    assert camera.exposure.Value() == 21_000

    worker.set_exposure(30_000)
    worker._close_camera()
    worker._open_camera()
    assert worker._auto_exposure is None
    assert camera.exposure.Value() == 30_000


def assert_start_settings(camera, gains):
    assert camera.exposure.Value() == 80_000
    assert camera.black.Value() == 0
    assert {key: gains.gain.values[key] for key in ("Red", "Green", "Blue")} == {
        "Red": 1.0, "Green": 1.0, "Blue": 1.0
    }


def test_configured_exposure_limits_must_be_increasing():
    from greenview_pro.camera import CameraSession

    session = CameraSession()
    with pytest.raises(ValueError, match="Exposure limits"):
        session.set_exposure_limits(150_000, 10_000)


def test_exposure_slider_range_uses_camera_limit_at_slowest_fps():
    from greenview_pro.camera import CameraSession

    camera = FakeCamera()
    camera.exposure.Maximum = lambda: min(
        200_000, 1_000_000 / camera.fps.Value()
    )
    session = CameraSession()
    session.remote = camera
    assert session._supported_exposure_range() == (10_000, 150_000)
    assert camera.fps.Value() == 7.2
    assert camera.exposure.Value() == 80_000
    assert camera.throughput.Value() == 150_000_000


def test_exposure_range_without_hardware_overlap_is_rejected():
    from greenview_pro.camera import CameraSession

    camera = FakeCamera()
    camera.exposure.minimum = 175_000
    session = CameraSession()
    session.remote = camera
    with pytest.raises(ValueError, match="unsupported"):
        session._supported_exposure_range()


def test_manual_exposure_can_exceed_previous_frame_interval():
    from greenview_pro.camera import CameraSession

    camera = FakeCamera()
    original_find = camera.FindNode
    camera.FindNode = lambda name: (
        SimpleNamespace(Execute=lambda: None, WaitUntilDone=lambda: None)
        if name in ("AcquisitionStop", "AcquisitionStart")
        else original_find(name)
    )
    original_set = camera.exposure.SetValue

    def reject_if_fps_is_too_high(value):
        if value > 1_000_000 / camera.fps.Value():
            raise ValueError("Exposure exceeds current frame interval")
        original_set(value)

    camera.exposure.SetValue = reject_if_fps_is_too_high
    session = CameraSession()
    session.remote = camera
    session.stream = object()
    assert session.set_exposure(150_000) == 150_000
    assert camera.fps.Value() == pytest.approx(1_000_000 / 150_000)
    assert camera.throughput.Value() == 150_000_000


def test_failed_long_exposure_restores_preview_frame_rate():
    from greenview_pro.camera import CameraSession

    camera = FakeCamera()
    events = []
    original_find = camera.FindNode
    camera.FindNode = lambda name: (
        SimpleNamespace(
            Execute=lambda: events.append(name),
            WaitUntilDone=lambda: None,
        )
        if name in ("AcquisitionStop", "AcquisitionStart")
        else original_find(name)
    )
    camera.exposure.SetValue = lambda value: (_ for _ in ()).throw(
        ValueError("Exposure unavailable")
    )
    session = CameraSession()
    session.remote = camera
    session.stream = object()
    with pytest.raises(ValueError, match="Exposure unavailable"):
        session.set_exposure(150_000)
    assert camera.fps.Value() == 7.2
    assert camera.exposure.Value() == 80_000
    assert events == ["AcquisitionStop", "AcquisitionStart"]


def test_manual_exposure_handles_camera_readout_time():
    from greenview_pro.camera import CameraSession

    camera = FakeCamera(readout_us=25_000)
    original_find = camera.FindNode
    camera.FindNode = lambda name: (
        SimpleNamespace(Execute=lambda: None, WaitUntilDone=lambda: None)
        if name in ("AcquisitionStop", "AcquisitionStart")
        else original_find(name)
    )
    original_set = camera.exposure.SetValue

    def reject_if_readout_exceeds_frame_interval(value):
        if value + camera.readout_us > 1_000_000 / camera.fps.Value():
            raise ValueError("Exposure and readout exceed frame interval")
        original_set(value)

    camera.exposure.SetValue = reject_if_readout_exceeds_frame_interval
    session = CameraSession()
    session.remote = camera
    session.stream = object()
    assert session.set_exposure(120_000) == 120_000
    assert camera.fps.Value() == pytest.approx(1_000_000 / 145_000)
    assert camera.exposure.Value() == 120_000


def test_saved_exposure_adjusts_fps_before_acquisition():
    from greenview_pro.camera import configure_camera_exposure

    camera = FakeCamera(readout_us=1_000)
    camera.black = FakeNode(0, 0, 10, camera.events, "black")
    original_find = camera.FindNode
    camera.FindNode = lambda name: (
        camera.black if name == "BlackLevel" else original_find(name)
    )
    camera.fps.value = 14.4
    original_set = camera.exposure.SetValue

    def reject_until_fps_is_low_enough(value):
        if value + camera.readout_us > 1_000_000 / camera.fps.Value():
            raise ValueError("Exposure exceeds frame interval")
        original_set(value)

    camera.exposure.SetValue = reject_until_fps_is_low_enough
    assert configure_camera_exposure(camera, 10_000, 150_000, 150_000) == 150_000
    assert camera.fps.Value() == pytest.approx(1_000_000 / 151_000)
    assert camera.throughput.Value() == 150_000_000


def test_unity_gain_readback_failure_is_reported():
    from greenview_pro.camera_gains import reset_color_gains

    gains = FakeMap()
    gains.gain.values["Red"] = 2.0
    gains.gain.SetValue = lambda value: None
    with pytest.raises(ValueError, match="Red.*1.0"):
        reset_color_gains(gains)
