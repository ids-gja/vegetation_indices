import numpy as np
import pytest
from calibration.library import Calibration

from calibration.flow import make_processor
from calibration import worker as processing
from calibration.worker import CalibratedCameraWorker as CameraWorker
from test_calibration_flow import FakeMap


def test_live_indices_use_shared_processor_when_calibrated():
    worker = CameraWorker()
    worker._processor = make_processor(
        Calibration({"R": 1, "G": 1, "NIR": 1}),
        None,
    )
    values = worker._raw_indices(np.array([[10, 20], [40, 10]], dtype=np.uint16))
    np.testing.assert_allclose(values["ndvi"], 1 / 3)
    np.testing.assert_allclose(values["cvi"], 8)


def test_worker_captures_two_averaged_stages_and_restores_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(processing, "CALIBRATION_FILE", tmp_path / "calibration.json")
    worker = CameraWorker()
    nodes = FakeMap()
    worker._remote = nodes
    worker._saved_exposure = 100
    raw = np.array([[12, 22], [42, 12]], dtype=np.uint16)
    worker.request_calibration("first", 2, 2.0)
    for _ in range(4):
        worker._advance_calibration(raw)
    assert worker._first_gains == {"R": 2.0, "G": 4.0, "NIR": 1.0}
    worker.request_calibration("second", 2, 2.0)
    for _ in range(12):
        worker._advance_calibration(np.full((2, 2), 42, dtype=np.uint16))
    assert processing.CALIBRATION_FILE.is_file()
    assert worker._processor is not None
    worker._processor = None
    worker._load_calibration()
    assert worker._processor is not None


def test_auxiliary_gain_is_applied_saved_restored_and_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(processing, "CALIBRATION_FILE", tmp_path / "calibration.json")
    worker = CameraWorker()
    worker._remote = FakeMap(("DigitalRed", "DigitalGreen", "DigitalBlue", "AnalogAll"))
    white = np.array([[12, 22], [42, 12]], dtype=np.uint16)
    worker.request_calibration("first", 1, 2.0, "AnalogAll", 2.5)
    for _ in range(11):
        worker._advance_calibration(white)
    assert worker._remote.gain.values["AnalogAll"] == 2.5
    worker.request_calibration("second", 1, 2.0, "AnalogAll", 2.5)
    for _ in range(11):
        worker._advance_calibration(np.full((2, 2), 42, dtype=np.uint16))
    calibration = Calibration.load(processing.CALIBRATION_FILE)
    assert calibration.metadata["auxiliary_gain_selector"] == "AnalogAll"
    assert float(calibration.metadata["auxiliary_gain_value"]) == 2.5
    worker._remote.gain.values["AnalogAll"] = 1.0
    with pytest.raises(ValueError, match="AnalogAll"):
        worker._validate_calibration()
    worker._load_calibration()
    assert worker._remote.gain.values["AnalogAll"] == 2.5
    worker._validate_calibration()


def test_auxiliary_gain_change_between_white_captures_is_rejected():
    worker = CameraWorker()
    worker._remote = FakeMap()
    worker.request_calibration("first", 1, 2.0, "AnalogAll", 2.5)
    for _ in range(11):
        worker._advance_calibration(np.array([[12, 22], [42, 12]], dtype=np.uint16))
    worker._remote.gain.values["AnalogAll"] = 3.0
    worker.request_calibration("second", 1, 2.0, "AnalogAll", 2.5)
    with pytest.raises(ValueError, match="AnalogAll"):
        worker._activate_calibration()


def test_exposure_must_remain_fixed_between_white_captures():
    worker = CameraWorker()
    worker._remote = FakeMap()
    worker._saved_exposure = 100
    worker.request_calibration("first", 1, 2.0)
    worker._activate_calibration()
    worker._first_gains = {"R": 1.0, "G": 1.0, "NIR": 1.0}
    worker._capture = None
    worker._remote.exposure.value = 200
    worker.request_calibration("second", 1, 2.0)
    with pytest.raises(ValueError, match="Exposure"):
        worker._activate_calibration()


def test_active_calibration_detects_gain_and_black_level_changes():
    worker = CameraWorker()
    worker._remote = FakeMap()
    worker._calibration = Calibration(
        {"R": 1, "G": 1, "NIR": 1},
        offsets={"R": 2.0, "G": 2.0, "NIR": 2.0},
        hardware_gains={"R": 1.0, "G": 1.0, "NIR": 1.0},
    )
    worker._validate_calibration()
    worker._remote.black.value = 3
    with pytest.raises(ValueError, match="BlackLevel"):
        worker._validate_calibration()
    worker._remote.black.value = 2
    worker._remote.gain.values["Red"] = 2
    with pytest.raises(ValueError, match="hardware gain mismatch"):
        worker._validate_calibration()
