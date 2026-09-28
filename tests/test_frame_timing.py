import pytest

from greenview_pro.processing import CameraWorker, maximize_frame_rate


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
