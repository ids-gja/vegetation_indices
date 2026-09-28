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
    np.testing.assert_allclose(values["ndvi"], 1 / 3)
    np.testing.assert_allclose(values["cvi"], 8)
    assert not hasattr(worker, "_calibration")


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
