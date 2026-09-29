import os
from pathlib import Path
import subprocess
import sys
import tomllib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QSpinBox

from greenview_pro import app
from greenview_pro.processing import CameraWorker


def test_snapshots_are_saved_outside_packaged_resources():
    assert app.SNAPSHOT_DIR == Path.cwd() / "snapshots"


def test_snapshot_saves_only_live_images_with_ungamma_corrected_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    monkeypatch.setattr(app, "SNAPSHOT_DIR", tmp_path)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    raw = np.full((8, 8, 3), 64, dtype=np.uint8)
    window._last_images = (
        raw,
        np.full_like(raw, 85),
        np.full_like(raw, 125),
    )
    try:
        window.main_cards[1].set_image(raw)
        window.save_snapshot()
        saved = {path.stem.split("_")[-1]: path for path in tmp_path.glob("*.png")}
        assert set(saved) == {"raw", "ndvi", "cvi"}
        np.testing.assert_array_equal(cv2.imread(str(saved["raw"])), raw)
    finally:
        window.close()


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


def test_default_window_has_settings_without_calibration_controls(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        assert not hasattr(window, "calibration_button")
        assert not window.settings_button.icon().isNull()
        assert not hasattr(window, "calibration_requested")
        assert window.main_cards[2].overlay.text() == "NDVI"
    finally:
        window.close()


def test_default_settings_offer_temporal_toggle_and_weight(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    observed = []

    def inspect():
        dialog = next(w for w in application.topLevelWidgets() if isinstance(w, QDialog))
        try:
            enabled = dialog.findChild(QCheckBox)
            weight = dialog.findChild(QSpinBox)
            if enabled is not None and weight is not None:
                initial = (enabled.text(), enabled.isChecked(), weight.value())
                weight.setValue(50)
                enabled.setChecked(False)
                observed.append((initial, weight.value(), weight.isEnabled()))
        finally:
            dialog.accept()

    try:
        QTimer.singleShot(0, inspect)
        window.settings_button.click()
        assert observed
        initial, weight, enabled = observed[0]
        assert initial == ("Temporal filter", True, 25)
        assert weight == 50
        assert window.worker._frame_processor.temporal_weight == 0.5
        assert not window.worker._frame_processor.temporal_enabled
        assert not enabled
        assert not hasattr(window.worker._frame_processor, "median_enabled")
    finally:
        window.close()


def test_save_config_button_persists_live_controls_and_preserves_other_config(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.toml"
    config.write_text(
        '[settings]\nSHOW_PROGRESS_CIRCLE = false\nDISPLAY_TIME = 9.0\n'
        'WATERMARK_TEXT = "keep me"\n\n[custom]\nvalue = "unchanged"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "CONFIG_FILE", config)
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.configure_exposure(100, 3000, 1200)
        window.exposure.setValue(1700)
        window.ndvi_contrast.set_values(20, 60)
        window.cvi_contrast.set_values(10, 90)
        window.worker.set_temporal_filter(False, 0.4)
        window.temporal_enabled = False
        window.temporal_weight = 0.4
        window.save_config_button.click()
        contents = config.read_text(encoding="utf-8")
        assert tomllib.loads(contents)["custom"]["value"] == "unchanged"
        saved = tomllib.loads(contents)["settings"]
        assert saved["WATERMARK_TEXT"] == "keep me"
        assert saved["DISPLAY_TIME"] == 9.0
        assert saved["TEMPORAL_ENABLED"] is False
        assert saved["TEMPORAL_WEIGHT"] == 0.4
        assert (saved["NDVI_LOW_PERCENTILE"], saved["NDVI_HIGH_PERCENTILE"]) == (20, 60)
        assert (saved["CVI_LOW_PERCENTILE"], saved["CVI_HIGH_PERCENTILE"]) == (10, 90)
        assert saved["EXPOSURE_US"] == 1700
    finally:
        window.close()

    restored = app.MainWindow()
    try:
        assert restored.ndvi_contrast.values() == (20, 60)
        assert restored.cvi_contrast.values() == (10, 90)
        assert (restored.worker._frame_processor.temporal_enabled,
                restored.worker._frame_processor.temporal_weight) == (False, 0.4)
        assert restored.worker._saved_exposure == 1700
        assert restored.worker._auto_exposure_available is False
    finally:
        restored.close()


def test_save_config_handles_windows_line_endings(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_bytes(b"[settings]\r\nDISPLAY_TIME = 9.0\r\n")
    monkeypatch.setattr(app, "CONFIG_FILE", config)
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.save_config_button.click()
        assert tomllib.loads(config.read_text(encoding="utf-8"))["settings"]["DISPLAY_TIME"] == 9.0
        assert config.read_bytes().count(b"[settings]") == 1
    finally:
        window.close()


def test_save_config_reports_failure_when_package_is_not_writable(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CONFIG_FILE", tmp_path / "missing" / "config.toml")
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    errors = []
    monkeypatch.setattr(app.QMessageBox, "critical", lambda *args: errors.append(args))
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.save_config_button.click()
        assert errors
        assert not app.CONFIG_FILE.exists()
    finally:
        window.close()


def test_snapshot_button_is_outlined_and_does_not_save_config(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(app, "SNAPSHOT_DIR", tmp_path / "images")
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    window._last_images = tuple(np.full((4, 4, 3), 80, dtype=np.uint8) for _ in range(3))
    try:
        window.snapshot_button.click()
        assert len(list(app.SNAPSHOT_DIR.glob("*.png"))) == 3
        assert not app.CONFIG_FILE.exists()
        assert "border:" in window.snapshot_button.styleSheet()
        assert window.save_config_button.accessibleName() == "Save config"
    finally:
        window.close()


def test_invalid_temporal_config_is_rejected(tmp_path, monkeypatch):
    import pytest

    config = tmp_path / "config.toml"
    config.write_text(
        "[settings]\nTEMPORAL_ENABLED = true\nTEMPORAL_WEIGHT = 0.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "CONFIG_FILE", config)
    application = QApplication.instance() or QApplication([])
    with pytest.raises(ValueError, match="TEMPORAL_WEIGHT"):
        app.MainWindow()


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
