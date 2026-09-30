import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFrame, QPushButton, QSpinBox

from greenview_pro import app
from calibration.ui import CalibrationWindow


def test_calibration_gear_opens_frame_count_dialog(monkeypatch):
    monkeypatch.setattr(CalibrationWindow.worker_class, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = CalibrationWindow()
    assert window.calibration_status.text() == "Uncalibrated preview"
    assert not window.calibration_button.icon().isNull()

    def inspect_dialog():
        dialog = next(w for w in application.topLevelWidgets() if isinstance(w, QDialog))
        assert any(spin.value() == 8 for spin in dialog.findChildren(QSpinBox))
        dialog.accept()

    QTimer.singleShot(0, inspect_dialog)
    window.open_calibration()
    window.close()


def test_calibration_settings_offer_white_target_action(monkeypatch):
    monkeypatch.setattr(CalibrationWindow.worker_class, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = CalibrationWindow()
    seen = []

    def inspect():
        dialog = next(w for w in application.topLevelWidgets() if isinstance(w, QDialog))
        seen.append((
            any(box.text() == "Temporal filter" for box in dialog.findChildren(QCheckBox)),
            any(
                button.text() == "White-target calibration"
                for button in dialog.findChildren(QPushButton)
            ),
        ))
        dialog.accept()

    try:
        QTimer.singleShot(0, inspect)
        window.settings_button.click()
        assert seen == [(True, True)]
    finally:
        window.close()


def test_calibration_tvi_label_reflects_uncalibrated_state(monkeypatch):
    monkeypatch.setattr(CalibrationWindow.worker_class, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = CalibrationWindow()
    try:
        window.update_calibration(False, "Uncalibrated preview")
        assert window.main_cards[3].overlay.text() == "TVI (UNCALIBRATED)"
        assert window.view_title("TVI") == "TVI (UNCALIBRATED)"
        window.update_calibration(True, "Calibrated")
        assert window.main_cards[3].overlay.text() == "TVI"
        assert window.view_title("TVI") == "TVI"
    finally:
        window.close()


@pytest.mark.parametrize("increment", [0.25, None])
def test_calibration_dialog_offers_camera_auxiliary_gain(monkeypatch, increment):
    monkeypatch.setattr(CalibrationWindow.worker_class, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = CalibrationWindow()
    window.update_gain_options([("AnalogAll", 1.0, 8.0, increment, 2.0)])
    requests = []
    window.calibration_requested.connect(lambda *args: requests.append(args))

    def inspect_dialog():
        dialog = next(w for w in application.topLevelWidgets() if isinstance(w, QDialog))
        selector = dialog.findChild(QComboBox)
        assert selector.currentData() == "AnalogAll"
        gain = next(
            spin for spin in dialog.findChildren(QDoubleSpinBox)
            if spin.maximum() == 8.0
        )
        assert gain.value() == 2.0
        gain.setValue(3.0)
        next(
            button for button in dialog.findChildren(QPushButton)
            if button.text().startswith("1.")
        ).click()
        dialog.accept()

    QTimer.singleShot(0, inspect_dialog)
    window.open_calibration()
    assert requests[0][-2:] == ("AnalogAll", 3.0)
    window.close()


def test_fullscreen_control_and_exposure_share_one_top_panel(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    panel = window.main_page.layout().itemAt(0).widget()
    assert isinstance(panel, QFrame)
    assert panel.objectName() == "controls"
    assert window.demo_button.parentWidget() is panel
    assert window.exposure.parentWidget() is panel
    assert not window.demo_button.icon().isNull()
    window.show_camera_error("Maximum frame rate unavailable")
    assert window.camera_status.text() == "Maximum frame rate unavailable"
    window.close()


def test_demo_exit_restores_maximized_main_window(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    window.showMaximized()
    application.processEvents()
    window.enter_demo()
    window.exit_demo()
    assert window.isMaximized()
    window.close()
