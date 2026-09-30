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


def test_fullscreen_requests_faster_fps_without_changing_preview_exposure(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.worker.set_exposure(80_000)
        assert not window.worker._pending_frame_rate
        window.enter_demo()
        assert window.worker._pending_frame_rate
        assert window.worker._saved_exposure == 80_000
        window.exit_demo()
        assert window.worker._maximize_on_reconnect
    finally:
        window.close()


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
        large = (window.demo_large.image.width(), window.demo_large.image.height())
        assert window._last_preview_size[1] == large
        assert window._last_preview_size[0] == (
            window.demo_cards[1].image.width(), window.demo_cards[1].image.height()
        )
        assert window._last_preview_size[2] == large
        window._display_view_index = 1
        window._request_preview_size()
        assert window._last_preview_size[1:] == (large, large)
    finally:
        window.close()


def test_slideshow_never_enlarges_a_previous_side_card_frame(monkeypatch):
    application, window = demo_window(monkeypatch)
    try:
        raw = np.tile(np.array([[40, 20], [30, 40]], dtype=np.uint16), (400, 640))
        large = (window.demo_large.image.width(), window.demo_large.image.height())
        old_sizes = (
            (window.demo_cards[1].image.width(), window.demo_cards[1].image.height()),
            large,
            (window.demo_cards[3].image.width(), window.demo_cards[3].image.height()),
        )
        old_frame = window.worker._frame_processor.render(raw, old_sizes)
        window._last_images = old_frame
        window._last_image_sizes = old_sizes
        window._refresh_demo_views()
        assert window.demo_large._array is old_frame[1]

        window._display_view_index = 1
        window._request_preview_size()
        window._refresh_demo_views()
        assert window.demo_large._array is not old_frame[2]

        new_frame = window.worker._process(raw, 0)
        window._last_images = new_frame
        window._last_image_sizes = window.worker._render_sizes
        window._refresh_demo_views()
        assert window.demo_large._array is new_frame[2]
        assert new_frame[2].shape == new_frame[1].shape
        assert new_frame[2].shape[1] > old_frame[2].shape[1]
    finally:
        window.close()


def test_worker_publishes_the_sizes_used_to_render_the_latest_frame():
    worker = app.CameraWorker()
    raw = np.tile(np.array([[40, 20], [30, 40]], dtype=np.uint16), (4, 4))
    sizes = ((8, 8), (10, 10), (6, 6))
    worker.set_preview_sizes(sizes)
    images = worker._process(raw, 0)
    with worker._latest_lock:
        worker._latest, worker._latest_version = images, 1
        worker._latest_sizes = worker._render_sizes
    assert worker.latest() == (images, 1)
    assert worker.latest(include_sizes=True) == (images, 1, sizes)


def test_entering_fullscreen_waits_for_large_index_instead_of_scaling_normal_preview(
    monkeypatch,
):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    try:
        window.show()
        application.processEvents()
        normal_sizes = window._last_preview_size
        raw = np.tile(np.array([[40, 20], [30, 40]], dtype=np.uint16), (400, 640))
        window._last_images = window.worker._process(raw, 0)
        window._last_image_sizes = normal_sizes
        window.enter_demo()
        application.processEvents()
        assert window.demo_large._array is None
        window._last_images = window.worker._process(raw, 0)
        window._last_image_sizes = window.worker._render_sizes
        window._refresh_demo_views()
        assert window.demo_large._array is window._last_images[1]
    finally:
        window.close()


def test_gamma_brightens_only_the_displayed_raw_cards(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    raw = np.full((8, 8, 3), 64, dtype=np.uint8)
    indices = tuple(np.full_like(raw, 64) for _ in range(2))
    try:
        monkeypatch.setattr(
            window.worker, "latest",
            lambda include_sizes=False: ((raw, *indices), 1, None),
        )
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
            "REFERENCE IMAGE", "RAW", "NDVI", "TVI"
        ]
        assert window.demo_large._array is window.demo_cards[2]._array
        initial_message = window.demo_title.text()
        window.next_message()
        assert window.demo_large.overlay.text() == "NDVI"
        wait_for_demo(lambda: window._animation is None)
        assert window.demo_title.text() != initial_message
        assert window.demo_large.overlay.text() == "TVI"
        assert [card.overlay.text() for card in window.demo_cards] == [
            "REFERENCE IMAGE", "RAW", "NDVI", "TVI"
        ]
        assert window.demo_large._array is window.demo_cards[3]._array
        assert window.marketing_timer.isActive()
        window.next_message()
        wait_for_demo(lambda: window._animation is None)
        assert window.demo_large.overlay.text() == "NDVI"
        assert [card.overlay.text() for card in window.demo_cards] == [
            "REFERENCE IMAGE", "RAW", "NDVI", "TVI"
        ]
    finally:
        window.close()


def test_packaged_presentation_calls_the_second_index_tvi(monkeypatch):
    application, window = demo_window(monkeypatch)
    try:
        assert any(
            title == "TVI"
            for message in window.marketing_messages
            for title, _ in message["facts"]
        )
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
            lambda include_sizes=False: ((live_frame, live_frame, live_frame), 1, None),
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
                lambda include_sizes=False, frame=frame, version=version: (
                    (frame, frame, frame), version, None
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
