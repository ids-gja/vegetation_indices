import importlib.util
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

SCRIPT = Path(__file__).resolve().parents[1] / "archive" / "greenview_pro_v4.py"
spec = importlib.util.spec_from_file_location("archive_greenview_v4", SCRIPT)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def test_archive_settings_load_and_reject_invalid_percentiles(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        "exposure_time = 2500\nndvi_percentile_lower = 5\n"
        "ndvi_percentile_upper = 90\ncvi_percentile_lower = 10\n"
        "cvi_percentile_upper = 95\n", encoding="utf-8"
    )
    monkeypatch.setattr(app, "CONFIG_FILE", config)
    settings = app.load_settings()
    assert settings["exposure_time"] == 2500
    assert settings["ndvi_percentile_upper"] == 90
    assert settings["cvi_percentile_lower"] == 10

    config.write_text(config.read_text(encoding="utf-8").replace(
        "cvi_percentile_lower = 10", "cvi_percentile_lower = 99"
    ), encoding="utf-8")
    with pytest.raises(ValueError, match="cvi"):
        app.load_settings()


def test_worker_uses_independent_percentiles_and_only_resets_changed_bounds(monkeypatch):
    settings = {
        "exposure_time": 2500,
        "ndvi_percentile_lower": 5,
        "ndvi_percentile_upper": 90,
        "cvi_percentile_lower": 10,
        "cvi_percentile_upper": 95,
    }
    worker = app.CameraWorker(settings)
    calls = []
    original = app.full_percentile_bounds

    def record(image, low, high):
        calls.append((low, high))
        return original(image, low, high)

    monkeypatch.setattr(app, "full_percentile_bounds", record)
    raw = np.arange(64, dtype=np.uint8).reshape(8, 8)
    worker._process(raw, 30)
    assert calls == [(5, 90), (10, 95)]
    worker.set_percentiles("NDVI", 20, 80)
    assert worker._ndvi_bounds is None
    assert worker._cvi_bounds is not None
    worker._process(raw, 30)
    assert calls == [(5, 90), (10, 95), (20, 80)]
    worker.set_percentiles("CVI", 15, 85)
    assert worker._ndvi_bounds is not None
    assert worker._cvi_bounds is None
    worker._process(raw, 30)
    assert calls[-1] == (15, 85)


def test_window_controls_update_each_index_and_exposure_is_configured(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        settings = app.load_settings()
        assert window.worker._saved_exposure == settings["exposure_time"]
        assert window.ndvi_low.value() == settings["ndvi_percentile_lower"]
        assert window.cvi_high.value() == settings["cvi_percentile_upper"]
        window.cvi_low.increase()
        assert window.worker._cvi_low == settings["cvi_percentile_lower"] + 1
        assert window.worker._ndvi_low == settings["ndvi_percentile_lower"]
        window.ndvi_high.decrease()
        assert window.worker._ndvi_high == settings["ndvi_percentile_upper"] - 1
        assert window.worker._cvi_high == settings["cvi_percentile_upper"]
        assert window.main_page.layout().contentsMargins().left() <= 8
        window._size_demo_layout()
        assert window.demo_large.width() / window.demo_large.height() == pytest.approx(
            16 / 9, abs=0.02
        )
        assert (
            window.demo_large.width()
            + window.demo_reference.width()
            + window.demo_middle.layout().spacing()
            == window.demo_middle.width()
        )
    finally:
        window.close()


def test_demo_stacks_three_cards_and_swaps_index_on_timer_without_new_frame(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    window.marketing_settings["FADE_OUT_TIME"] = "0"
    window.marketing_settings["FADE_IN_TIME"] = "0"
    window.marketing_messages = window.marketing_messages[:1]
    reference = np.full((8, 8, 3), 7, dtype=np.uint8)
    raw = np.full_like(reference, 20)
    ndvi = np.full_like(reference, 30)
    cvi = np.full_like(reference, 40)
    window.color_image = reference
    window.demo_cards[0].set_image(reference)
    monkeypatch.setattr(window.worker, "latest", lambda: ((raw, ndvi, cvi), 1))
    try:
        window.enter_demo()
        application.processEvents()
        window.pull_latest()
        assert len(window.demo_cards) == 3
        assert [card.y() for card in window.demo_cards] == sorted(
            card.y() for card in window.demo_cards
        )
        assert len({card.y() for card in window.demo_cards}) == 3
        assert all(
            card.width() / card.height() == pytest.approx(16 / 9, abs=0.1)
            for card in window.demo_cards
        )
        assert window.demo_cards[0]._array is reference
        assert window.demo_cards[1]._array is raw
        assert window.demo_large._array is ndvi
        assert window.demo_cards[2]._array is cvi
        assert window.demo_large.overlay.text() == "NDVI"
        assert window.demo_cards[2].overlay.text() == "CVI"

        window.next_message()
        QTest.qWait(50)
        assert window.demo_large._array is cvi
        assert window.demo_cards[2]._array is ndvi
        assert window.demo_large.overlay.text() == "CVI"
        assert window.demo_cards[2].overlay.text() == "NDVI"
        assert window.demo_cards[0]._array is reference
        assert window.demo_cards[1]._array is raw
        window.next_message()
        QTest.qWait(50)
        assert window.demo_large._array is ndvi
        assert window.demo_cards[2]._array is cvi
    finally:
        window.close()
