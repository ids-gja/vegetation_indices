import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from greenview_pro import app
from greenview_pro.processing import CameraWorker


def test_snapshots_are_saved_outside_packaged_resources():
    assert app.SNAPSHOT_DIR == Path.cwd() / "snapshots"


def test_default_worker_uses_hardware_balanced_channels():
    worker = CameraWorker()
    raw = np.array([[10, 20], [40, 10]], dtype=np.uint16)
    values = worker._raw_indices(raw)
    np.testing.assert_allclose(values["ndvi"], 5 / 14)
    np.testing.assert_allclose(values["cvi"], 10.6875)
    assert not hasattr(worker, "_calibration")


def test_default_worker_subtracts_black_level_before_calling_core(monkeypatch):
    worker = CameraWorker()
    original = worker._default_processor.process_raw
    core_inputs = []

    def record_input(raw):
        core_inputs.append(raw.copy())
        return original(raw)

    monkeypatch.setattr(worker._default_processor, "process_raw", record_input)
    worker._raw_indices(np.array([[0, 3], [2, 1]], dtype=np.uint16))
    assert len(core_inputs) == 1
    np.testing.assert_array_equal(core_inputs[0], [[-2, 1], [0, -1]])
    assert core_inputs[0].dtype == np.float32


def test_default_window_has_no_calibration_controls(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        assert not hasattr(window, "calibration_button")
        assert not hasattr(window, "calibration_requested")
        assert window.main_cards[2].overlay.text() == "NDVI"
    finally:
        window.close()


def test_cli_exposes_opt_in_calibration_flag():
    result = subprocess.run(
        [sys.executable, "-m", "greenview_pro", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--calibration" in result.stdout


def test_importing_display_does_not_import_optional_calibration_ui():
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import greenview_pro.app, sys; assert 'calibration.ui' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
