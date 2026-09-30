import os
from pathlib import Path
import subprocess
import sys
import tomllib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
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
        assert set(saved) == {"raw", "ndvi", "tvi"}
        np.testing.assert_array_equal(cv2.imread(str(saved["raw"])), raw)
    finally:
        window.close()


def test_default_worker_uses_hardware_balanced_channels():
    worker = CameraWorker()
    raw = np.array([[10, 20], [40, 10]], dtype=np.uint16)
    values = worker._raw_indices(raw)
    np.testing.assert_allclose(values["ndvi"], 5 / 14)
    np.testing.assert_allclose(values["tvi"], 10.6875)
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
        'WATERMARK_TEXT = "keep me"\n'
        'NDVI_LOW_PERCENTILE = 15\nNDVI_HIGH_PERCENTILE = 74\n'
        'CVI_LOW_PERCENTILE = 16\nCVI_HIGH_PERCENTILE = 77\n'
        '\n[custom]\nvalue = "unchanged"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "CONFIG_FILE", config)
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.configure_exposure(10_000, 30_000, 12_000)
        window.exposure.setValue(17_000)
        assert window.ndvi_contrast.values() == (-100, 100)
        assert window.tvi_contrast.values() == (16, 77)
        window.ndvi_contrast.set_values(-40, 60)
        window.tvi_contrast.set_values(10, 90)
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
        assert (saved["NDVI_MIN"], saved["NDVI_MAX"]) == (-0.4, 0.6)
        assert "NDVI_LOW_PERCENTILE" not in saved
        assert "NDVI_HIGH_PERCENTILE" not in saved
        assert (saved["TVI_LOW_PERCENTILE"], saved["TVI_HIGH_PERCENTILE"]) == (10, 90)
        assert "CVI_LOW_PERCENTILE" not in saved
        assert "CVI_HIGH_PERCENTILE" not in saved
        assert saved["EXPOSURE_US"] == 17_000
        assert (saved["EXPOSURE_MIN_US"], saved["EXPOSURE_MAX_US"]) == (
            10_000, 150_000
        )
    finally:
        window.close()

    restored = app.MainWindow()
    try:
        assert restored.ndvi_contrast.values() == (-40, 60)
        assert restored.tvi_contrast.values() == (10, 90)
        assert (restored.worker._frame_processor.temporal_enabled,
                restored.worker._frame_processor.temporal_weight) == (False, 0.4)
        assert restored.worker._saved_exposure == 17_000
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


def test_exposure_slider_uses_configured_custom_range(
    tmp_path, monkeypatch,
):
    config = tmp_path / "config.toml"
    config.write_text(
        "[settings]\nEXPOSURE_MIN_US = 12000\nEXPOSURE_MAX_US = 110000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "CONFIG_FILE", config)
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        assert window.worker._saved_exposure is None
        assert not window.worker._auto_exposure_available
        assert window.worker._camera.configured_exposure_min == 12_000
        assert window.worker._camera.configured_exposure_max == 110_000
        window.configure_exposure(12_000, 110_000, 80_000)
        assert (window.exposure.minimum(), window.exposure.maximum()) == (
            12_000, 110_000
        )
        window.exposure.setValue(90_000)
        assert window.worker._pending_exposure == 90_000
    finally:
        window.close()


@pytest.mark.parametrize(
    "settings",
    [
        "EXPOSURE_MIN_US = 160000\nEXPOSURE_MAX_US = 150000\n",
        "EXPOSURE_MIN_US = true\nEXPOSURE_MAX_US = 150000\n",
        "EXPOSURE_MIN_US = 10000\nEXPOSURE_MAX_US = 2147483648\n",
        "EXPOSURE_MIN_US = 10000\nEXPOSURE_MAX_US = 150000\nEXPOSURE_US = 9000\n",
    ],
)
def test_invalid_exposure_range_and_saved_exposure_are_rejected(
    tmp_path, monkeypatch, settings,
):
    config = tmp_path / "config.toml"
    config.write_text("[settings]\n" + settings, encoding="utf-8")
    monkeypatch.setattr(app, "CONFIG_FILE", config)
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    with pytest.raises(ValueError, match="(?i)exposure"):
        app.MainWindow()


def test_new_tvi_percentiles_override_legacy_settings(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        "[settings]\n"
        "CVI_LOW_PERCENTILE = 16\nCVI_HIGH_PERCENTILE = 77\n"
        "TVI_LOW_PERCENTILE = 20\nTVI_HIGH_PERCENTILE = 85\n"
        "\n[custom]\nCVI_LOW_PERCENTILE = 4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app, "CONFIG_FILE", config)
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        assert window.tvi_contrast.values() == (20, 85)
        assert "CVI_LOW_PERCENTILE" not in window.marketing_settings
        window.save_config()
        document = tomllib.loads(config.read_text(encoding="utf-8"))
        assert document["settings"]["TVI_LOW_PERCENTILE"] == 20
        assert document["settings"]["TVI_HIGH_PERCENTILE"] == 85
        assert "CVI_LOW_PERCENTILE" not in document["settings"]
        assert "CVI_HIGH_PERCENTILE" not in document["settings"]
        assert document["custom"]["CVI_LOW_PERCENTILE"] == 4
    finally:
        window.close()


@pytest.mark.parametrize(
    "bounds",
    [
        "NDVI_MIN = -1.01\nNDVI_MAX = 1.0\n",
        "NDVI_MIN = 0.0\nNDVI_MAX = 0.01\n",
        "NDVI_MIN = 0.005\nNDVI_MAX = 1.0\n",
        "NDVI_MIN = true\nNDVI_MAX = 1.0\n",
        "NDVI_MIN = 1e308\nNDVI_MAX = 1.0\n",
    ],
)
def test_invalid_ndvi_score_settings_are_rejected(tmp_path, monkeypatch, bounds):
    config = tmp_path / "config.toml"
    config.write_text("[settings]\n" + bounds, encoding="utf-8")
    monkeypatch.setattr(app, "CONFIG_FILE", config)
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    with pytest.raises(ValueError, match="NDVI bounds"):
        app.MainWindow()


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
