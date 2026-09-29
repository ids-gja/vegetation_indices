import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from greenview_pro import app


def demo_window(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    window.enter_demo()
    application.processEvents()
    return application, window


def wait_for_demo(condition):
    for _ in range(100):
        if condition():
            return
        QTest.qWait(10)
    assert condition(), "Fullscreen transition did not finish"


def test_message_progress_is_a_hollow_ring():
    application = QApplication.instance() or QApplication([])
    ring = app.MessageProgressIndicator("#008A96")
    ring.set_progress(0.5)
    image = ring.grab().toImage()
    center = image.pixelColor(image.width() // 2, image.height() // 2)
    assert center != QColor("#008A96")
    assert image.pixelColor(image.width() // 2, 2) == QColor("#008A96")


def test_normal_preview_uses_source_aspect_and_compact_grid(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.resize(1200, 800)
        window.show()
        application.processEvents()
        window.update_frame_dimensions(1920, 1080)
        application.processEvents()
        cards = window.main_cards
        assert all(
            card.image.width() / card.image.height() == pytest.approx(16 / 9, abs=0.02)
            for card in cards
        )
        assert cards[1].x() - cards[0].geometry().right() - 1 <= 2
        assert cards[2].y() - cards[0].geometry().bottom() - 1 <= 2
        assert all(card.size() == cards[0].size() for card in cards)
        assert window._last_preview_size == tuple(
            (card.image.width(), card.image.height()) for card in cards[1:]
        )
    finally:
        window.close()


def test_fullscreen_uses_display_sized_preview_after_normal_mode(monkeypatch):
    application, window = demo_window(monkeypatch)
    try:
        assert window._last_preview_size[1] == (
            window.demo_large.image.width(), window.demo_large.image.height()
        )
        assert window._last_preview_size[0] == (
            window.demo_cards[1].image.width(), window.demo_cards[1].image.height()
        )
        assert window._last_preview_size[2] == (
            window.demo_cards[3].image.width(), window.demo_cards[3].image.height()
        )
        window._display_view_index = 1
        window._request_preview_size()
        assert window._last_preview_size[2] == (
            window.demo_large.image.width(), window.demo_large.image.height()
        )
        assert window._last_preview_size[1] == (
            window.demo_cards[2].image.width(), window.demo_cards[2].image.height()
        )
    finally:
        window.close()


def test_gamma_brightens_only_the_displayed_raw_cards(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    raw = np.full((8, 8, 3), 64, dtype=np.uint8)
    indices = tuple(np.full_like(raw, 64) for _ in range(2))
    try:
        monkeypatch.setattr(window.worker, "latest", lambda: ((raw, *indices), 1))
        window.pull_latest()
        expected = round(255 * (64 / 255) ** (1 / 2.2))
        assert window.main_cards[1]._array[0, 0, 0] == expected
        assert window.main_cards[2]._array[0, 0, 0] == 64
        np.testing.assert_array_equal(raw, np.full_like(raw, 64))
        window.enter_demo()
        application.processEvents()
        window._refresh_demo_views()
        assert window.demo_cards[1]._array[0, 0, 0] == expected
        assert not window.demo_cards[1]._smooth
        assert window.demo_cards[0]._smooth
        assert window.demo_large._array[0, 0, 0] == 64
    finally:
        window.close()


def test_slideshow_ring_tracks_the_shared_message_and_image_cycle(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(app.time, "monotonic", lambda: now[0])
    original = app.MainWindow.load_marketing

    def enabled(window):
        settings, messages = original(window)
        settings["SHOW_PROGRESS_CIRCLE"] = True
        return settings, messages

    monkeypatch.setattr(app.MainWindow, "load_marketing", enabled)
    application, window = demo_window(monkeypatch)
    try:
        assert not hasattr(window, "view_timer")
        assert not hasattr(window, "view_progress")
        assert window.marketing_timer.isSingleShot()
        assert window.message_progress_timer.interval() <= 16
        assert window.message_progress._progress == 0
        now[0] += float(window.marketing_settings["DISPLAY_TIME"]) / 2
        window.message_progress_timer.timeout.emit()
        assert window.message_progress._progress == pytest.approx(0.5)
        window.exit_demo()
        assert window.message_progress._progress == 0
    finally:
        window.close()


def test_progress_circle_defaults_off_without_stopping_slideshow(monkeypatch):
    application, window = demo_window(monkeypatch)
    try:
        assert window.marketing_settings["SHOW_PROGRESS_CIRCLE"] is False
        assert window.message_progress.isHidden()
        assert not window.message_progress_timer.isActive()
        assert window.marketing_timer.isActive()
    finally:
        window.close()


def test_message_and_large_index_swap_together(monkeypatch):
    application, window = demo_window(monkeypatch)
    try:
        window.marketing_timer.stop()
        window.marketing_settings["FADE_OUT_TIME"] = 0.01
        window.marketing_settings["FADE_IN_TIME"] = 0.01
        window._last_images = tuple(
            np.full((12, 16, 3), i, dtype=np.uint8) for i in range(3)
        )
        window._refresh_demo_views()
        assert window.demo_large.overlay.text() == "NDVI"
        assert [card.overlay.text() for card in window.demo_cards] == [
            "REFERENCE IMAGE", "RAW", "NDVI", "CVI"
        ]
        assert window.demo_large._array is window.demo_cards[2]._array
        initial_message = window.demo_title.text()
        window.next_message()
        assert window.demo_large.overlay.text() == "NDVI"
        wait_for_demo(lambda: window._animation is None)
        assert window.demo_title.text() != initial_message
        assert window.demo_large.overlay.text() == "CVI"
        assert [card.overlay.text() for card in window.demo_cards] == [
            "REFERENCE IMAGE", "RAW", "NDVI", "CVI"
        ]
        assert window.demo_large._array is window.demo_cards[3]._array
        assert window.marketing_timer.isActive()
        window.next_message()
        wait_for_demo(lambda: window._animation is None)
        assert window.demo_large.overlay.text() == "NDVI"
        assert [card.overlay.text() for card in window.demo_cards] == [
            "REFERENCE IMAGE", "RAW", "NDVI", "CVI"
        ]
    finally:
        window.close()


def test_fullscreen_images_fill_height_with_gap_and_hidden_ring(monkeypatch):
    application, window = demo_window(monkeypatch)
    try:
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        window._last_images = (frame, frame, frame)
        window._refresh_demo_views()
        window._size_demo_layout()
        application.processEvents()
        screen = window.screen().size()
        assert window.demo_large.height() == window.demo_reference.height()
        assert window.demo_large.width() / window.demo_large.height() == pytest.approx(
            4 / 3, abs=0.01
        )
        assert window.demo_middle.width() <= screen.width()
        assert window.demo_large.height() <= screen.height()
        assert window.demo_middle.layout().spacing() >= 10
        assert len(window.demo_cards) == 4
        top_left, top_right, bottom_left, bottom_right = window.demo_cards
        assert top_left.y() == top_right.y()
        assert bottom_left.y() == bottom_right.y()
        assert bottom_left.y() > top_left.y()
        assert top_left.x() == bottom_left.x()
        assert top_right.x() == bottom_right.x()
        assert top_right.x() > top_left.x()
        assert (
            top_left.height() + bottom_left.height()
            + window.demo_reference.layout().verticalSpacing()
            == window.demo_reference.height()
        )
        for card in window.demo_cards:
            assert card.image.width() / card.image.height() == pytest.approx(
                4 / 3, abs=0.02
            )
        assert window.demo_large.image.pixmap().size() == window.demo_large.image.size()
        ring = window.message_progress
        assert ring.width() >= 36
        assert ring.isHidden()
        assert not hasattr(window.demo_large, "view_progress")
    finally:
        window.close()


def test_fullscreen_layout_adapts_to_live_sensor_aspect(monkeypatch):
    application, window = demo_window(monkeypatch)
    try:
        live_frame = np.zeros((300, 500, 3), dtype=np.uint8)
        monkeypatch.setattr(
            window.worker,
            "latest",
            lambda: ((live_frame, live_frame, live_frame), 1),
        )
        window.pull_latest()
        application.processEvents()
        assert window.demo_large.width() / window.demo_large.height() == pytest.approx(
            5 / 3, abs=0.01
        )
        assert window.demo_middle.width() <= window.screen().size().width()
        assert window.demo_large.image.width() / window.demo_large.image.height() == pytest.approx(
            5 / 3, abs=0.01
        )
    finally:
        window.close()


def test_fullscreen_layout_stays_fixed_as_bayer_preview_rounds(monkeypatch):
    from greenview_pro.processing import preview_bayer

    application, window = demo_window(monkeypatch)
    try:
        source = np.zeros((800, 1280), dtype=np.uint16)
        window.worker.frame_dimensions_changed.emit(source.shape[1], source.shape[0])
        application.processEvents()
        sizes = []
        for version in range(1, 7):
            preview = preview_bayer(source, window._last_preview_size[1])
            frame = np.zeros((*preview.shape, 3), dtype=np.uint8)
            monkeypatch.setattr(
                window.worker,
                "latest",
                lambda frame=frame, version=version: (
                    (frame, frame, frame), version
                ),
            )
            window.pull_latest()
            application.processEvents()
            sizes.append(window.demo_large.size())
        assert len(set(sizes)) == 1
        assert window.demo_large.width() / window.demo_large.height() == pytest.approx(
            source.shape[1] / source.shape[0], abs=0.01
        )
    finally:
        window.close()


def test_camera_reports_source_size_only_when_it_changes():
    worker = app.CameraWorker()
    dimensions = []
    worker.frame_dimensions_changed.connect(
        lambda width, height: dimensions.append((width, height))
    )
    worker._note_frame_dimensions(np.zeros((800, 1280), dtype=np.uint16))
    worker._note_frame_dimensions(np.zeros((800, 1280), dtype=np.uint16))
    worker._note_frame_dimensions(np.zeros((1080, 1920), dtype=np.uint16))
    assert dimensions == [(1280, 800), (1920, 1080)]


def test_camera_rejects_empty_source_dimensions_before_emitting():
    worker = app.CameraWorker()
    with pytest.raises(RuntimeError, match="expected 2D RAW Bayer image"):
        worker._note_frame_dimensions(np.empty((0, 12), dtype=np.uint16))
