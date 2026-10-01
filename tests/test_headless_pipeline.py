import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from greenview_pro.camera import CameraSession
from greenview_pro.image_processing import FrameProcessor


def test_bayer_cells_form_float_triplets_with_both_green_sites():
    from ndvi_processing import bayer_superpixels

    raw = np.array(
        [[65535, 20, 10, 50], [30, 65535, 40, 20]], dtype=np.uint16
    )
    triplets = bayer_superpixels(raw, "GRBG")
    assert triplets.shape == (1, 2, 3)
    assert triplets.dtype == np.float32
    np.testing.assert_array_equal(triplets, [[[20, 65535, 30], [50, 15, 40]]])


def test_headless_modules_do_not_import_qt():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import greenview_pro.camera, greenview_pro.image_processing, sys; "
            "assert not any(name.startswith('PySide6') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert result.returncode == 0, result.stderr


def test_float_superpixels_are_processed_at_native_resolution_before_display_scaling():
    source = np.tile(np.array([[10, 20], [30, 50]], dtype=np.uint16), (8, 12))
    frame = FrameProcessor()
    seen = []
    original = frame.processor.process_channels

    def record(channels):
        seen.append({name: values.copy() for name, values in channels.items()})
        return original(channels)

    frame.processor.process_channels = record
    rendered = frame.render(source, (8, 6))
    assert len(seen) == 1
    assert seen[0]["R"].shape == (8, 12)
    for name, value in (("R", 18), ("G", 28), ("NIR", 28)):
        assert seen[0][name].dtype == np.float32
        np.testing.assert_array_equal(seen[0][name], value)
    assert rendered[0].shape == (4, 8, 3)
    assert all(image.shape == (5, 8, 3) for image in rendered[1:])


def test_render_processes_once_at_full_resolution_for_different_card_sizes():
    source = np.tile(np.array([[10, 20], [30, 40]], dtype=np.uint16), (20, 30))
    frame = FrameProcessor()
    sizes = ((10, 8), (20, 12), (6, 4))
    seen = []
    original = frame.processor.process_channels

    def record(channels):
        seen.append(channels["R"].shape)
        return original(channels)

    frame.processor.process_channels = record
    images = frame.render(source, sizes)
    assert [image.shape[:2] for image in images] == [
        (6, 10), (12, 18), (4, 6)
    ]
    assert seen == [(20, 30)]


def test_full_resolution_processing_preserves_fractional_channel_values():
    source = np.zeros((6, 6), dtype=np.uint16)
    source[0::2, 0::2] = 10
    source[1::2, 1::2] = 20
    source[1::2, 0::2] = 30
    source[0::2, 1::2] = np.arange(1, 10, dtype=np.uint16).reshape(3, 3)
    frame = FrameProcessor()
    observed = []
    original = frame.processor.process_channels

    def record(channels):
        observed.append(channels["R"].copy())
        return original(channels)

    frame.processor.process_channels = record
    frame.render(source, (2, 2))
    assert observed[0].shape == (3, 3)
    assert observed[0].dtype == np.float32
    np.testing.assert_allclose(observed[0][0, 0], -1, atol=1e-5)


def test_live_render_temporally_averages_each_native_channel_without_blurring_neighbors():
    frame = FrameProcessor()
    raw = np.tile(np.array([[40, 20], [30, 40]], dtype=np.uint16), (4, 4))
    changed = raw.copy()
    changed[1, 0] = 70
    observed = []
    original = frame.processor.process_channels

    def record(channels):
        observed.append({name: values.copy() for name, values in channels.items()})
        return original(channels)

    frame.processor.process_channels = record
    frame.render(raw, (4, 4))
    frame.render(changed, (4, 4))
    np.testing.assert_array_equal(observed[0]["NIR"], np.full((4, 4), 28))
    assert observed[1]["NIR"].dtype == np.float32
    assert observed[1]["NIR"].shape == (4, 4)
    assert observed[1]["NIR"][0, 0] == pytest.approx(38)
    np.testing.assert_array_equal(observed[1]["NIR"][1:, :], 28)
    np.testing.assert_array_equal(observed[1]["R"], 18)
    np.testing.assert_array_equal(observed[1]["G"], 38)
    assert frame.indices(changed)["ndvi"][0, 0] == pytest.approx((68 - 18) / (68 + 18))


def test_live_temporal_history_resets_for_setting_and_sensor_changes():
    frame = FrameProcessor()
    raw = np.tile(np.array([[40, 20], [30, 40]], dtype=np.uint16), (4, 4))
    changed = raw.copy()
    changed[1, 0] = 70
    observed = []
    original = frame.processor.process_channels

    def record(channels):
        observed.append(channels["NIR"].copy())
        return original(channels)

    frame.processor.process_channels = record
    frame.render(raw)
    frame.black_level = 4
    frame.render(changed)
    assert observed[-1][0, 0] == 66

    frame.render(raw)
    frame.reset_temporal()
    frame.render(changed)
    assert observed[-1][0, 0] == 66

    frame.render(raw)
    frame.render(changed[:4, :4])
    assert observed[-1].shape == (2, 2)
    assert observed[-1][0, 0] == 66

    frame.render(raw)
    frame.processor = type(frame.default_processor)(frame.default_processor.sensor)
    fresh_process = frame.processor.process_channels

    def record_fresh(channels):
        observed.append(channels["NIR"].copy())
        return fresh_process(channels)

    frame.processor.process_channels = record_fresh
    frame.render(changed)
    assert observed[-1][0, 0] == 70


def test_temporal_filter_can_be_disabled_and_weight_adjusted():
    frame = FrameProcessor()
    raw = np.tile(np.array([[40, 20], [30, 40]], dtype=np.uint16), (2, 2))
    changed = raw.copy()
    changed[1, 0] = 70
    seen = []
    original = frame.processor.process_channels

    def record(channels):
        seen.append(channels["NIR"][0, 0])
        return original(channels)

    frame.processor.process_channels = record
    frame.render(raw)
    assert frame._ndvi_bounds is not None and frame._tvi_bounds is not None
    frame.set_temporal_filter(False, 0.25)
    assert frame._ndvi_bounds == (-1.0, 1.0) and frame._tvi_bounds is None
    frame.render(changed)
    assert seen[-1] == 68
    assert frame._temporal_superpixels is None
    frame.set_temporal_filter(True, 0.5)
    frame.render(raw)
    frame.render(changed)
    assert seen[-2:] == [28, 48]
    for enabled, weight in ((True, 0), (True, 1.1), (1, 0.5), (False, float("nan"))):
        with pytest.raises(ValueError, match="Temporal"):
            frame.set_temporal_filter(enabled, weight)


def test_tvi_bounds_freeze_in_presentation_and_resume_updating_in_preview():
    frame = FrameProcessor()
    frame.set_temporal_filter(False, 0.25)
    raw = np.tile(
        np.array([[40, 20, 50, 20], [30, 40, 30, 80]], dtype=np.uint16), (4, 4)
    )
    changed = raw.copy()
    changed[1::2, 1::4] = 100
    frame.render(raw, (8, 8))
    preview_bounds = frame._tvi_bounds

    frame.set_tvi_adaptive(False)
    for _ in range(12):
        frame.render(changed, (8, 8))
    assert frame._tvi_bounds == preview_bounds

    frame.set_tvi_adaptive(True)
    for _ in range(10):
        frame.render(changed, (8, 8))
    assert frame._tvi_bounds != preview_bounds


def test_frozen_tvi_bounds_initialize_if_presentation_starts_before_first_frame():
    frame = FrameProcessor()
    frame.set_tvi_adaptive(False)
    raw = np.tile(
        np.array([[40, 20, 50, 20], [30, 40, 30, 80]], dtype=np.uint16), (4, 4)
    )
    frame.render(raw, (8, 8))
    assert frame._tvi_bounds is not None


def test_superpixel_area_scaling_reduces_an_isolated_index_speckle():
    from ndvi_processing import bayer_superpixels, resize_superpixels

    raw = np.full((8, 8), 20, dtype=np.uint16)
    raw[1, 0] = 100
    frame = FrameProcessor()
    native_peak = frame.indices(raw)["ndvi"][0, 0]
    triplets = resize_superpixels(bayer_superpixels(raw, "GRBG"), (2, 2))
    assert triplets.dtype == np.float32
    assert triplets[0, 0, 2] == 40
    channels = {
        name: triplets[..., index] - 2
        for index, name in enumerate(("R", "G", "NIR"))
    }
    reduced_peak = frame.processor.process_channels(channels).indices["ndvi"][0, 0]
    assert 0 < reduced_peak < native_peak


def test_full_resolution_indices_are_reduced_after_calculation(monkeypatch):
    from greenview_pro import image_processing

    raw = np.random.default_rng(12).integers(10, 200, (32, 48), dtype=np.uint16)
    frame = FrameProcessor()
    seen = []
    original = image_processing.cv2.resize

    def record(image, size, **kwargs):
        if image.ndim == 2 and image.dtype == np.float32:
            seen.append((image.shape, size))
        return original(image, size, **kwargs)

    monkeypatch.setattr(image_processing.cv2, "resize", record)
    frame.render(raw, ((8, 6), (16, 12), (8, 6)))
    assert len(seen) == 2
    assert all(shape == (16, 24) for shape, _ in seen)


def test_invalid_native_index_does_not_blank_a_whole_display_pixel(monkeypatch):
    from greenview_pro import image_processing

    frame = FrameProcessor()
    frame._ndvi_bounds = frame._tvi_bounds = (0, 1)
    original = frame.processor.process_channels

    def with_one_invalid(channels):
        result = original(channels)
        result.indices["ndvi"][:] = 0.5
        result.indices["ndvi"][0, 0] = np.nan
        return result

    normalized = []
    original_heatmap = image_processing.heatmap

    def record(image):
        normalized.append(image.copy())
        return original_heatmap(image)

    monkeypatch.setattr(frame.processor, "process_channels", with_one_invalid)
    monkeypatch.setattr(image_processing, "heatmap", record)
    frame.render(np.full((8, 8), 20, dtype=np.uint16), (2, 2))
    assert 0 < normalized[0][0, 0] < 0.5


def test_native_indices_are_not_filtered_or_rescaled():
    source = np.tile(np.array([[10, 20], [30, 40]], dtype=np.uint16), (8, 12))
    source[0::2, 1::4] = 40
    frame = FrameProcessor()
    indices = frame.indices(source)
    assert indices["ndvi"].shape == (8, 12)
    np.testing.assert_allclose(indices["ndvi"][:, 0::2], -5 / 33)
    np.testing.assert_allclose(indices["ndvi"][:, 1::2], 5 / 23)


def test_renderer_returns_only_raw_ndvi_and_tvi():
    source = np.array(
        [
            [5, 10, 10, 10],
            [30, 0, 30, 0],
            [20, 10, 40, 10],
            [30, 0, 30, 0],
        ],
        dtype=np.uint16,
    )
    raw, ndvi, tvi = FrameProcessor().render(source, (4, 4))
    assert tvi.shape == (4, 4, 3)
    assert np.unique(tvi.reshape(-1, 3), axis=0).shape[0] > 1
    assert not np.array_equal(tvi, ndvi)


def test_ndvi_uses_absolute_score_bounds_without_percentile_calculation(monkeypatch):
    from greenview_pro import image_processing

    frame = FrameProcessor()
    source = np.full((2, 6), 20, dtype=np.uint16)
    original_process = frame.processor.process_channels

    def fixed_indices(channels):
        result = original_process(channels)
        result.indices["ndvi"] = np.array([[-0.5, 0, 0.5]], dtype=np.float32)
        return result

    monkeypatch.setattr(frame.processor, "process_channels", fixed_indices)
    seen = []
    original_bounds = image_processing.full_percentile_bounds
    original_heatmap = image_processing.heatmap
    normalized = []

    def record(image, low, high):
        seen.append((low, high))
        return original_bounds(image, low, high)

    def capture(image):
        normalized.append(image.copy())
        return original_heatmap(image)

    monkeypatch.setattr(image_processing, "full_percentile_bounds", record)
    monkeypatch.setattr(image_processing, "heatmap", capture)
    sizes = ((6, 2), (3, 1), (3, 1))
    frame.render(source, sizes)
    np.testing.assert_allclose(normalized[0], [[0.25, 0.5, 0.75]])
    assert seen == [(30, 80)]
    seen.clear()
    normalized.clear()
    frame.set_ndvi_bounds(-0.5, 0.5)
    frame.render(source, sizes)
    np.testing.assert_allclose(normalized[0], [[0, 0.5, 1]])
    assert seen == []
    frame.set_ndvi_bounds(0.27, 0.29)
    assert frame.ndvi_bounds == (0.27, 0.29)
    frame.set_percentiles("tvi", 10, 90)
    frame.render(source, sizes)
    assert seen == [(10, 90)]
    with pytest.raises(ValueError, match="index"):
        frame.set_percentiles("ndvi", 20, 40)
    for low, high in ((-1.1, 0), (0, 1.1), (0, 0.01), (float("nan"), 1)):
        with pytest.raises(ValueError, match="NDVI"):
            frame.set_ndvi_bounds(low, high)


def test_tvi_percentiles_reset_only_the_tvi_display_bounds():
    frame = FrameProcessor()
    frame.render(np.full((4, 4), 20, dtype=np.uint16))
    assert frame.percentiles == {"tvi": (30, 80)}
    assert frame._tvi_bounds is not None
    frame.set_percentiles("tvi", 10, 90)
    assert frame.percentiles == {"tvi": (10, 90)}
    assert frame._tvi_bounds is None
    assert frame.ndvi_bounds == (-1.0, 1.0)


def test_camera_read_copies_buffer_before_requeue(monkeypatch):
    from greenview_pro import camera

    raw = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    queued = []

    def requeue(buffer):
        queued.append(buffer)
        raw[:] = 0

    session = CameraSession()
    session.stream = SimpleNamespace(
        WaitForFinishedBuffer=lambda timeout: object(),
        QueueBuffer=requeue,
    )
    monkeypatch.setattr(
        camera.ids_peak_ipl_extension,
        "BufferToImage",
        lambda buffer: SimpleNamespace(get_numpy_2D=lambda: raw),
    )
    assert np.array_equal(
        session.read(), np.array([[1, 2], [3, 4]], dtype=np.uint16)
    )
    assert len(queued) == 1


def test_failed_headless_open_releases_sdk(monkeypatch):
    from greenview_pro import camera

    calls = []
    monkeypatch.setattr(
        camera.ids_peak,
        "Library",
        SimpleNamespace(
            Initialize=lambda: calls.append("initialize"),
            Close=lambda: calls.append("close"),
        ),
    )
    monkeypatch.setattr(
        camera.ids_peak,
        "DeviceManager",
        SimpleNamespace(
            Instance=lambda: SimpleNamespace(Update=lambda: None, Devices=lambda: [])
        ),
    )
    session = CameraSession()
    with pytest.raises(RuntimeError, match="No IDS camera found"):
        with session:
            pass
    assert calls == ["initialize", "close"]
