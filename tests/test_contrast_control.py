import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from greenview_pro import app


def point_for(control, value):
    track = control._track_rect()
    return QPoint(control._value_x(value), track.center().y())


def drag(control, start, end):
    QTest.mousePress(control, Qt.LeftButton, pos=point_for(control, start))
    QTest.mouseMove(control, point_for(control, end))
    QTest.mouseRelease(control, Qt.LeftButton, pos=point_for(control, end))


def test_contrast_grips_never_cross_or_leave_less_than_two_percent():
    from greenview_pro.contrast_control import ContrastRange

    application = QApplication.instance() or QApplication([])
    control = ContrastRange(25, 75)
    control.resize(310, 46)
    control.show()
    application.processEvents()
    changes = []
    control.changed.connect(lambda lo, hi: changes.append((lo, hi)))
    drag(control, 25, 100)
    assert control.values() == (73, 75)
    drag(control, 75, 0)
    assert control.values() == (73, 75)
    assert all(0 <= lo <= hi - 2 <= 98 for lo, hi in changes)
    control.close()


def test_selected_contrast_band_moves_together_and_clamps_to_limits():
    from greenview_pro.contrast_control import ContrastRange

    application = QApplication.instance() or QApplication([])
    control = ContrastRange(20, 40)
    control.resize(310, 46)
    control.show()
    application.processEvents()
    drag(control, 30, 50)
    assert control.values() == (40, 60)
    drag(control, 50, 100)
    assert control.values() == (80, 100)
    drag(control, 90, 0)
    assert control.values() == (0, 20)
    control.close()


def test_click_outside_range_moves_nearest_contrast_grip():
    from greenview_pro.contrast_control import ContrastRange

    application = QApplication.instance() or QApplication([])
    control = ContrastRange(25, 75)
    control.resize(310, 46)
    control.show()
    application.processEvents()
    QTest.mouseClick(control, Qt.LeftButton, pos=point_for(control, 5))
    assert control.values() == (5, 75)
    QTest.mouseClick(control, Qt.LeftButton, pos=point_for(control, 95))
    assert control.values() == (5, 95)
    control.close()


def test_invalid_contrast_bounds_are_rejected_without_changing_selection():
    from greenview_pro.contrast_control import ContrastRange

    application = QApplication.instance() or QApplication([])
    control = ContrastRange(5, 99)
    for low, high in ((0, 1), (99, 100), (-1, 50), (40, 101)):
        with pytest.raises(ValueError, match="gap"):
            control.set_values(low, high)
        assert control.values() == (5, 99)


def test_contrast_range_remains_keyboard_adjustable():
    from greenview_pro.contrast_control import ContrastRange

    application = QApplication.instance() or QApplication([])
    control = ContrastRange(20, 40)
    control.show()
    control.setFocus()
    application.processEvents()
    QTest.keyClick(control, Qt.Key_Right)
    QTest.keyClick(control, Qt.Key_Up)
    assert control.values() == (21, 41)
    QTest.keyClick(control, Qt.Key_Left, Qt.ShiftModifier)
    assert control.values() == (20, 40)
    control.close()


def test_absolute_ndvi_range_uses_hundredth_score_steps():
    from greenview_pro.contrast_control import ContrastRange

    application = QApplication.instance() or QApplication([])
    control = ContrastRange(-50, 50, minimum=-100, maximum=100)
    control.resize(310, 46)
    control.show()
    application.processEvents()
    drag(control, -50, -30)
    assert control.values() == (-30, 50)
    control.setFocus()
    QTest.keyClick(control, Qt.Key_Right)
    assert control.values() == (-29, 50)
    with pytest.raises(ValueError, match="gap"):
        control.set_values(99, 100)
    control.close()


def test_camera_status_changes_do_not_move_top_bar_controls(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.resize(1200, 800)
        window.show()
        application.processEvents()
        positions = tuple(
            widget.mapTo(window, QPoint()).x()
            for widget in (window.exposure, window.ndvi_contrast, window.tvi_contrast, window.snapshot_button)
        )
        window.show_camera_error("Camera disconnected - reconnecting; check USB cable")
        application.processEvents()
        assert positions == tuple(
            widget.mapTo(window, QPoint()).x()
            for widget in (window.exposure, window.ndvi_contrast, window.tvi_contrast, window.snapshot_button)
        )
        assert window.camera_status.text().startswith("Camera disconnected")
        assert window.camera_status.y() > window.tvi_contrast.y()
        window.ndvi_contrast.set_values(-40, 40)
        QTest.mousePress(window.ndvi_contrast, Qt.LeftButton, pos=point_for(window.ndvi_contrast, 0))
        window.update_state("Camera connected", True)
        application.processEvents()
        QTest.mouseMove(window.ndvi_contrast, point_for(window.ndvi_contrast, 20))
        QTest.mouseRelease(
            window.ndvi_contrast, Qt.LeftButton, pos=point_for(window.ndvi_contrast, 20)
        )
        assert window.ndvi_contrast.values() == (-20, 60)
    finally:
        window.close()


def test_exposure_readout_has_space_before_contrast_label(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.show()
        application.processEvents()
        contrast_label = next(
            label for label in window.controls.findChildren(QLabel)
            if label.text() == "NDVI"
        )
        gap = (
            contrast_label.mapTo(window.controls, QPoint()).x()
            - window.exposure_label.mapTo(window.controls, QPoint()).x()
            - window.exposure_label.width()
        )
        assert gap >= 16
    finally:
        window.close()


def test_stacked_contrast_bars_update_independent_ndvi_scores_and_tvi_percentiles(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    monkeypatch.setattr(app, "CONFIG_FILE", tmp_path / "config.toml")
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.show()
        application.processEvents()
        assert window.tvi_contrast.y() > window.ndvi_contrast.y()
        window.ndvi_contrast.set_values(-40, 60)
        assert window.worker._frame_processor.ndvi_bounds == (-0.4, 0.6)
        assert window.worker._frame_processor.percentiles["tvi"] == (30, 80)
        assert window.ndvi_contrast_label.text() == "-0.40 to +0.60"
        window.tvi_contrast.set_values(10, 90)
        assert window.worker._frame_processor.percentiles["tvi"] == (10, 90)
        assert window.worker._frame_processor.ndvi_bounds == (-0.4, 0.6)
        assert window.tvi_contrast_label.text() == "10-90 %"
    finally:
        window.close()


def test_default_ndvi_score_range_and_tvi_percentiles(monkeypatch, tmp_path):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    monkeypatch.setattr(app, "CONFIG_FILE", tmp_path / "config.toml")
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        assert window.ndvi_contrast.values() == (-100, 100)
        assert window.tvi_contrast.values() == (30, 80)
        assert window.ndvi_contrast_label.text() == "-1.00 to +1.00"
        assert window.tvi_contrast_label.text() == "30-80 %"
        assert window.worker._frame_processor.ndvi_bounds == (-1.0, 1.0)
        assert window.worker._frame_processor.percentiles == {"tvi": (30, 80)}
    finally:
        window.close()


def test_tvi_label_uses_tvi_percentile_control(monkeypatch, tmp_path):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    monkeypatch.setattr(app, "CONFIG_FILE", tmp_path / "config.toml")
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        assert any(
            label.text() == "TVI"
            for label in window.controls.findChildren(QLabel)
        )
        assert window.tvi_contrast.accessibleName() == "TVI contrast percentile range"
        assert window.main_cards[3].overlay.text() == "TVI"
        window.tvi_contrast.set_values(20, 85)
        assert window.worker._frame_processor.percentiles["tvi"] == (20, 85)
    finally:
        window.close()
