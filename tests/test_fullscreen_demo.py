import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from greenview_pro import app


def demo_window(monkeypatch):
    monkeypatch.setattr(app.CameraWorker, "start", lambda self: None)
    application = QApplication.instance() or QApplication([])
    window = app.MainWindow()
    window.enter_demo()
    application.processEvents()
    return application, window


def test_message_progress_is_a_hollow_ring():
    application = QApplication.instance() or QApplication([])
    ring = app.MessageProgressIndicator("#008A96")
    ring.set_progress(0.5)
    image = ring.grab().toImage()
    center = image.pixelColor(image.width() // 2, image.height() // 2)
    assert center != QColor("#008A96")
    assert image.pixelColor(image.width() // 2, 2) == QColor("#008A96")


def test_slideshow_bar_paints_continuous_fill():
    application = QApplication.instance() or QApplication([])
    bar = app.ViewProgressIndicator("#008A96")
    bar.resize(200, 6)
    bar.set_progress(0.375)
    image = bar.grab().toImage()
    assert image.pixelColor(30, 3) == QColor("#008A96")
    assert image.pixelColor(160, 3) == QColor("#D9E2E3")
    assert bar._progress == 0.375


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
        assert window._last_preview_size == (
            cards[1].image.width() * 2,
            cards[1].image.height() * 2,
        )
    finally:
        window.close()


def test_fullscreen_uses_display_sized_preview_after_normal_mode(monkeypatch):
    application, window = demo_window(monkeypatch)
    try:
        assert window._last_preview_size == (
            window.demo_large.image.width(),
            window.demo_large.image.height(),
        )
    finally:
        window.close()


def test_slideshow_progress_tracks_view_timer_and_resets(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(app.time, "monotonic", lambda: now[0])
    application, window = demo_window(monkeypatch)
    try:
        assert isinstance(window.view_progress, app.ViewProgressIndicator)
        assert window.message_progress_timer.interval() <= 16
        assert window.view_progress._progress == 0
        now[0] += float(window.marketing_settings["VIEW_ROTATION_SECONDS"]) / 2
        window.message_progress_timer.timeout.emit()
        assert window.view_progress._progress == pytest.approx(0.5)
        now[0] += 0.001
        window._update_view_progress()
        assert window.view_progress._progress > 0.5
        window.next_view()
        assert window.view_progress._progress == 0
        now[0] += float(window.marketing_settings["VIEW_ROTATION_SECONDS"]) * 2
        window._update_view_progress()
        assert window.view_progress._progress == 1
        window.exit_demo()
        assert window.view_progress._progress == 0
    finally:
        window.close()


def test_fullscreen_images_fill_height_with_gap_and_aligned_progress(monkeypatch):
    application, window = demo_window(monkeypatch)
    try:
        window._size_demo_layout()
        screen = window.screen().size()
        assert window.demo_large.height() == window.demo_reference.height()
        assert window.demo_large.width() / window.demo_large.height() == pytest.approx(
            4 / 3, abs=0.01
        )
        assert window.demo_middle.width() <= screen.width()
        assert window.demo_large.height() <= screen.height()
        assert window.demo_middle.layout().spacing() >= 10
        assert (
            sum(card.height() for card in window.demo_cards)
            + window.demo_reference.layout().spacing() * 2
            == window.demo_reference.height()
        )
        for card in window.demo_cards:
            assert card.image.width() / card.image.height() == pytest.approx(
                4 / 3, abs=0.02
            )
        assert window.demo_large.image.pixmap().size() == window.demo_large.image.size()
        assert window.view_progress.width() == window.demo_large.width()
        assert (
            window.view_progress.y() + window.view_progress.height()
            == window.demo_large.height()
        )
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
            preview = preview_bayer(source, window._last_preview_size)
            frame = np.zeros((*preview.shape, 3), dtype=np.uint8)
            monkeypatch.setattr(
                window.worker,
                "latest",
                lambda frame=frame, version=version: ((frame, frame, frame), version),
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
