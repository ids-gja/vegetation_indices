import os
from xml.etree import ElementTree

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QToolButton

from greenview_pro import app
from calibration.ui import ICON_FILE, CalibrationWindow
from PySide6.QtGui import QIcon


def test_gear_has_eight_evenly_spaced_rounded_teeth():
    root = ElementTree.parse(ICON_FILE).getroot()
    rectangles = root.findall("{http://www.w3.org/2000/svg}rect")
    assert len(rectangles) == 8
    assert {rect.get("transform") for rect in rectangles} == {
        f"rotate({angle} 12 12)" for angle in range(0, 360, 45)
    }
    assert all(float(rect.get("rx")) > 0 for rect in rectangles)
    application = QApplication.instance() or QApplication([])
    assert application is not None
    image = QIcon(str(ICON_FILE)).pixmap(QSize(96, 96)).toImage()
    assert image.pixelColor(48, 48).alpha() == 0
    assert image.pixelColor(48, 24).alpha() > 0
    assert image.pixelColor(48, 8).alpha() > 0
    assert image.pixelColor(88, 48).alpha() > 0


def test_save_has_rectangular_slot_and_circular_opening():
    application = QApplication.instance() or QApplication([])
    assert application is not None
    icon = app.load_icon("save.svg")
    image = icon.pixmap(QSize(96, 96)).toImage()
    assert image.pixelColor(48, 26).alpha() == 0  # top slot
    assert image.pixelColor(48, 46).alpha() > 0  # separates the cutouts
    assert image.pixelColor(48, 64).alpha() == 0  # circular opening
    assert image.pixelColor(36, 56).alpha() > 0  # outside circle's diagonal
    assert image.pixelColor(48, 80).alpha() > 0  # beneath the circle


def test_save_and_settings_are_matching_accessible_icon_buttons(monkeypatch):
    monkeypatch.setattr(CalibrationWindow.worker_class, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = CalibrationWindow()
    try:
        for button, label in (
            (window.snapshot_button, "Save snapshot"),
            (window.settings_button, "Settings"),
        ):
            assert isinstance(button, QToolButton)
            assert button.text() == ""
            assert button.toolTip() == label
            assert button.accessibleName() == label
            assert not button.icon().isNull()
        assert window.snapshot_button.iconSize() == window.settings_button.iconSize()
    finally:
        window.close()
