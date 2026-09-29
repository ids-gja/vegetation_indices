import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from greenview_pro import app
from greenview_pro.processing import CameraWorker
from test_frame_timing import FakeCamera, FakeNode


def new_search(minimum=18, maximum=10_000):
    from greenview_pro.auto_exposure import ExposureSearch

    return ExposureSearch(minimum, maximum, 3500)


def frame_for(exposure, clip_at):
    raw = np.full((64, 64), 100, dtype=np.uint16)
    if exposure >= clip_at:
        raw[0::2, 1::2] = 4095
    return raw


@pytest.mark.parametrize("clip_at", [1700, 3600, 7800])
def test_binary_search_finds_brightest_unclipped_exposure(clip_at):
    search = new_search()
    assert search.exposure == 3500
    for _ in range(30):
        next_exposure = search.observe(frame_for(search.exposure, clip_at))
        if search.done:
            break
        assert next_exposure is not None
    assert search.done
    assert clip_at - 100 <= search.exposure < clip_at


def test_binary_search_checks_all_bayer_sites_for_clipping():
    search = new_search()
    search.observe(frame_for(3500, 3500))
    assert search.exposure < 3500


def test_binary_search_counts_clipping_across_the_entire_raw_frame():
    search = new_search()
    raw = np.full((64, 64), 100, dtype=np.uint16)
    raw[2::4, 2::4] = 4095
    search.observe(raw)
    assert search.exposure < 3500


def test_autoexposure_limits_clipped_pixels_to_less_than_point_one_percent():
    search = new_search()
    raw = np.full((100, 100), 100, dtype=np.uint8)
    raw.ravel()[:10] = 255
    search.observe(raw)
    assert search.exposure < 3500


def test_autoexposure_uses_selected_sensor_bit_depth():
    from greenview_pro.auto_exposure import ExposureSearch

    raw = np.full((100, 100), 100, dtype=np.uint16)
    raw.ravel()[:10] = 4095
    search = ExposureSearch(18, 10000, 3500, white_level=4095)
    search.observe(raw)
    assert search.exposure < 3500


def test_camera_white_level_uses_pixel_format_when_available():
    from types import SimpleNamespace
    from greenview_pro.camera import sensor_white_level

    for name, expected in (("BayerGR8", 255), ("BayerGR12p", 4095)):
        camera = SimpleNamespace(
            FindNode=lambda _: SimpleNamespace(
                CurrentEntry=lambda: SimpleNamespace(SymbolicValue=lambda: name)
            )
        )
        assert sensor_white_level(camera) == expected


def test_binary_search_selects_upper_cap_if_no_pixels_clip():
    search = new_search(maximum=8000)
    for _ in range(30):
        search.observe(frame_for(0, 9000))
        if search.done:
            break
    assert search.done
    assert search.exposure == 8000


def test_worker_starts_at_three_point_five_ms_and_subtracts_two_dn():
    worker = CameraWorker()
    assert worker._initial_exposure == 3500
    raw = np.array([[4, 12], [22, 8]], dtype=np.uint16)
    indices = worker._default_processor.process_raw(
        raw.astype(np.float32) - worker._black_level
    )
    assert worker._default_processor.sensor.black_level == 0
    np.testing.assert_allclose(worker._raw_indices(raw)["ndvi"], [[1 / 3]])
    np.testing.assert_array_equal(indices.red, [[10]])
    np.testing.assert_array_equal(indices.green, [[4]])
    np.testing.assert_array_equal(indices.nir, [[20]])
    np.testing.assert_allclose(indices.indices["ndvi"], [[1 / 3]])
    np.testing.assert_allclose(indices.indices["cvi"], [[12.5]])


def test_programmatic_exposure_updates_do_not_cancel_autoexposure(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.worker._auto_exposure = new_search(maximum=8000)
        window.configure_exposure(18, 8000, 3500)
        assert window.worker._auto_exposure is not None
        window.exposure.setValue(3400)
        assert window.worker._auto_exposure is None
        assert window.worker._pending_exposure == 3400
    finally:
        window.close()


def test_camera_initialization_sets_black_level_and_starting_exposure():
    from greenview_pro.processing import configure_camera_exposure

    camera = FakeCamera()
    camera.black = FakeNode(0, 0, 10, camera.events, "black")
    original_find = camera.FindNode
    camera.FindNode = lambda name: (
        camera.black if name == "BlackLevel" else original_find(name)
    )
    assert configure_camera_exposure(camera, 18, 69_444) == 3500
    assert camera.exposure.Value() == 3500
    assert camera.black.Value() == 2


def test_autoexposure_updates_the_camera_before_displaying_next_frame():
    camera = FakeCamera()
    worker = CameraWorker()
    worker._remote = camera
    worker._exposure_min = 18
    worker._exposure_cap = 10_000
    worker._auto_exposure = new_search()
    camera.exposure.SetValue(3500)

    assert worker._advance_auto_exposure(frame_for(3500, 3500))
    assert camera.exposure.Value() == worker._auto_exposure.exposure
    assert camera.exposure.Value() < 3500
    assert worker._auto_settle_frames > 0
    worker.set_exposure(2400)
    assert worker._auto_exposure is None


def test_autoexposure_reports_clipping_at_minimum_exposure():
    worker = CameraWorker()
    worker._remote = FakeCamera()
    worker._auto_exposure = new_search(minimum=18, maximum=18)
    messages = []
    worker.recoverable_error.connect(messages.append)
    worker._advance_auto_exposure(frame_for(18, 18))
    assert worker._auto_exposure.done
    assert messages and "minimum" in messages[0]


def test_calibration_offsets_are_not_subtracted_twice():
    from calibration.library import Calibration
    from ndvi_processing import NDVIProcessor, SensorConfig

    worker = CameraWorker()
    sensor = SensorConfig(
        "AR2020", "GRBG", worker._default_processor.sensor.relative_qe_csv,
        black_level=2,
    )
    calibration = Calibration(
        {"R": 1, "G": 1, "NIR": 1},
        offsets={"R": 2, "G": 2, "NIR": 2},
    )
    corrected = NDVIProcessor(sensor, calibration).process_raw(
        np.array([[4, 12], [22, 8]], dtype=np.uint16)
    )
    np.testing.assert_array_equal(corrected.red, [[10]])
    np.testing.assert_array_equal(corrected.green, [[4]])
    np.testing.assert_array_equal(corrected.nir, [[20]])


def test_optional_calibration_uses_read_back_black_level_and_stops_search():
    from calibration.worker import CalibratedCameraWorker

    camera = FakeCamera()
    camera.black = FakeNode(2, 0, 10, camera.events, "black")
    original_find = camera.FindNode
    camera.FindNode = lambda name: (
        camera.black if name == "BlackLevel" else original_find(name)
    )
    worker = CalibratedCameraWorker()
    worker._remote = camera
    worker._auto_exposure = new_search()
    worker.request_calibration("first", 2, 4)
    worker._activate_calibration()

    assert worker._auto_exposure is None
    assert worker._black_level == 4
    np.testing.assert_allclose(
        worker._raw_indices(np.array([[6, 14], [24, 10]], dtype=np.uint16))["ndvi"],
        [[1 / 3]],
    )
    assert worker._processor is worker._default_processor
    worker._camera_closed()
    camera.black.SetValue(2)
    worker._set_default_black_level(2)
    assert worker._black_level == 2
