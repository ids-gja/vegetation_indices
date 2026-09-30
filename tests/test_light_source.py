"""Tests for the standalone light-source calibration guide."""

import numpy as np
import pytest

from calibration.light_source import (
    CalibrationGuide, aligned_roi, channel_mean, run_guide, validate_cfa,
)


def test_cfa_accepts_every_two_green_arrangement():
    from itertools import permutations

    for layout in set(permutations("RGGB")):
        assert validate_cfa("".join(layout)) == "".join(layout)
    assert validate_cfa(None) == "GRBG"
    for invalid in ("RBBG", "GGGG", "RGGBG", "RRGG"):
        with pytest.raises(ValueError, match="CFA"):
            validate_cfa(invalid)


@pytest.mark.parametrize("pattern", ["RGGB", "GRBG", "GBRG", "BGGR", "GGRB"])
def test_channel_mean_uses_only_requested_sites(pattern):
    frame = np.zeros((4, 6), dtype=np.uint16)
    for index, channel in enumerate(pattern):
        frame[index // 2 :: 2, index % 2 :: 2] = {"R": 14, "G": 24, "B": 34}[channel]
    assert channel_mean(frame, (0, 0, 6, 4), pattern, "G") == 24
    assert channel_mean(frame, (0, 0, 6, 4), pattern, "R") == 14
    assert channel_mean(frame, (0, 0, 6, 4), pattern, "B") == 34


def test_green_reference_averages_both_green_sites():
    frame = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    assert channel_mean(frame, (0, 0, 2, 2), "RGGB", "G") == 25
    with pytest.raises(ValueError, match="ROI"):
        channel_mean(frame, (1, 0, 2, 2), "RGGB", "G")


def test_roi_snaps_inside_selection_to_even_coordinates():
    assert aligned_roi((3.1, 2.2), (11.9, 9.8), (12, 14)) == (4, 4, 6, 4)
    assert aligned_roi((-8, -4), (18, 18), (11, 13)) == (0, 0, 12, 10)
    with pytest.raises(ValueError, match="2x2"):
        aligned_roi((1, 1), (2, 2), (12, 14))


class FakeCamera:
    def __init__(self, *frames):
        self.frames = list(frames)
        self.reads = 0

    def read(self):
        self.reads += 1
        return self.frames.pop(0)


def test_guide_requires_confirmations_and_reuses_green_roi():
    preview = np.full((4, 2), 8, dtype=np.uint16)
    green = np.array([[60, 2], [3, 60]] * 2, dtype=np.uint16)
    red = np.array([[0, 61], [0, 0]] * 2, dtype=np.uint16)
    ir = np.array([[0, 0], [59, 0]] * 2, dtype=np.uint16)
    camera = FakeCamera(preview, green, red, red, ir)
    guide = CalibrationGuide(camera, "GRBG", settle_frames=0)

    assert guide.stage == "roi"
    with pytest.raises(ValueError, match="preview"):
        guide.confirm_roi((0, 0, 2, 4))
    guide.capture_preview()
    assert guide.stage == "roi"
    with pytest.raises(ValueError, match="Select a slab ROI"):
        guide.confirm_roi(None)
    guide.confirm_roi((0, 0, 2, 4))
    assert guide.stage == "green"
    assert guide.reference is None
    guide.capture_green()
    assert guide.reference == 60
    assert guide.stage == "red"
    assert guide.capture_channel() == 61
    assert guide.capture_channel() == 61
    assert guide.stage == "red"
    guide.confirm_channel()
    assert guide.stage == "ir"
    assert guide.capture_channel() == 59
    guide.confirm_channel()
    assert guide.stage == "done"
    assert camera.reads == 5


def test_guide_rejects_confirmation_without_capture_and_changed_size():
    green = np.full((4, 4), 12, dtype=np.uint8)
    camera = FakeCamera(green, green, green, np.ones((6, 4), dtype=np.uint8))
    guide = CalibrationGuide(camera, "RGGB", settle_frames=0)
    guide.capture_preview()
    guide.confirm_roi((0, 0, 4, 4))
    guide.capture_green()
    with pytest.raises(ValueError, match="Capture"):
        guide.confirm_channel()
    guide.capture_channel()
    with pytest.raises(ValueError, match="shape"):
        guide.capture_channel()
    with pytest.raises(ValueError, match="Capture"):
        guide.confirm_channel()
    assert guide.stage == "red"


def test_window_guides_operator_through_every_confirmation(monkeypatch, capsys):
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets
    from types import SimpleNamespace

    buttons = []
    selectors = []

    class FakeButton:
        def __init__(self, ax, label):
            self.label = SimpleNamespace(set_text=lambda text: None)
            self.click = None
            buttons.append(self)

        def on_clicked(self, callback):
            self.click = callback

    class FakeSelector:
        def __init__(self, ax, callback, **kwargs):
            self.select = callback
            selectors.append(self)

        def set_active(self, active):
            pass

    monkeypatch.setattr(widgets, "Button", FakeButton)
    monkeypatch.setattr(widgets, "RectangleSelector", FakeSelector)
    preview = np.full((8, 8), 10, dtype=np.uint8)
    green = np.full((8, 8), 50, dtype=np.uint8)
    red = np.full((8, 8), 51, dtype=np.uint8)
    ir = np.full((8, 8), 49, dtype=np.uint8)
    camera = FakeCamera(*([preview] * 11 + [green] * 11 + [red] * 11 + [ir] * 11))
    camera.white_level = 255

    def simulate_operator():
        capture, confirm = buttons
        selectors[0].select(SimpleNamespace(xdata=1.1, ydata=1.1),
                            SimpleNamespace(xdata=6.9, ydata=6.9))
        confirm.click(None)
        capture.click(None)
        capture.click(None)
        confirm.click(None)
        capture.click(None)
        confirm.click(None)

    monkeypatch.setattr(plt, "show", simulate_operator)
    try:
        run_guide(camera, "GRBG")
    finally:
        plt.close("all")
    assert "ROI=(2, 2, 4, 4) G=50.00 R=51.00 IR=49.00" in capsys.readouterr().out
