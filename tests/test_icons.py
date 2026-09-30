import os
from pathlib import Path
import tomllib
from xml.etree import ElementTree

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication

from greenview_pro.app import ICON_DIR, load_icon
from calibration.ui import ICON_FILE
from PySide6.QtGui import QIcon


CONTROLS = (
    "arrow_up",
    "arrow_down",
    "camera",
    "camera_off",
    "fullscreen",
    "save",
    "save_config",
    "save_image",
    "gear",
)


def test_control_icons_are_renderable_vectors():
    application = QApplication.instance() or QApplication([])
    assert application is not None
    for name in CONTROLS:
        svg = ICON_DIR / f"{name}.svg"
        root = ElementTree.parse(svg).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert not any(element.tag.endswith("}image") for element in root.iter())
        icon = load_icon(f"{name}.svg")
        assert not icon.isNull()
        for size in (16, 32):
            pixmap = icon.pixmap(QSize(size, size))
            assert not pixmap.isNull()
            image = pixmap.toImage()
            assert image.hasAlphaChannel()
            assert any(
                image.pixelColor(x, y).alpha() > 0
                for y in range(image.height())
                for x in range(image.width())
            )
    assert not QIcon(str(ICON_FILE)).isNull()


def test_wheel_packages_vectors_and_the_original_logo_only():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    icon_files = project["tool"]["setuptools"]["package-data"]["greenview_pro"]
    assert "resources/icons/*" in icon_files
    assert (ICON_DIR / "ids-logo_black_rgb.png").exists()
    assert (ICON_DIR / "gear.svg").exists()
