"""Qt-independent RAW Bayer processing and live display rendering."""

from importlib.resources import files
import math

import cv2
import matplotlib
import numpy as np

from ndvi_processing import (
    NDVIProcessor,
    SensorConfig,
    bayer_superpixels,
)

LOW_DEFAULT = 30
HIGH_DEFAULT = 80
DEFAULT_PREVIEW_SIZE = (1280, 1280)
TEMPORAL_WEIGHT = np.float32(0.25)

_cmap = matplotlib.colormaps.get_cmap("RdYlGn").resampled(256)
_colors = (_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
COLORMAP = np.ascontiguousarray(_colors.reshape(256, 1, 3)[..., ::-1])


def preview_bayer(raw, maximum_size=DEFAULT_PREVIEW_SIZE):
    """Reduce a Bayer display preview without mixing CFA color sites."""
    if raw.ndim != 2:
        raise ValueError("A Bayer preview requires a two-dimensional image")
    height, width = raw.shape
    maximum_width, maximum_height = maximum_size
    if maximum_width < 2 or maximum_height < 2:
        raise ValueError("Preview dimensions must be at least two pixels")
    scale = min(1.0, maximum_width / width, maximum_height / height)
    target_width = max(2, int(width * scale) // 2 * 2)
    target_height = max(2, int(height * scale) // 2 * 2)
    if target_width == width and target_height == height:
        return raw.copy()
    target_shape = (target_width // 2, target_height // 2)
    preview = np.empty((target_height, target_width), dtype=raw.dtype)
    for y in range(2):
        for x in range(2):
            preview[y::2, x::2] = cv2.resize(
                raw[y::2, x::2], target_shape, interpolation=cv2.INTER_AREA
            )
    return preview


def normalize_with_bounds(image, lo, hi):
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.nan_to_num(
        np.clip((image - np.float32(lo)) / np.float32(hi - lo), 0, 1),
        nan=0,
        posinf=1,
        neginf=0,
    ).astype(np.float32)


def full_percentile_bounds(image, low, high):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, low)), float(np.percentile(finite, high))


def heatmap(image):
    return cv2.applyColorMap(
        np.ascontiguousarray((image * 255).astype(np.uint8)), COLORMAP
    )


def validate_temporal_filter(enabled, weight):
    if (
        not isinstance(enabled, bool)
        or isinstance(weight, bool)
        or not isinstance(weight, (int, float, np.integer, np.floating))
        or not np.isfinite(weight)
        or not 0 < weight <= 1
    ):
        raise ValueError("Temporal filter requires a boolean and weight in (0, 1]")


class FrameProcessor:
    """Compute full-resolution indices and size only the displayed images."""

    def __init__(self, processor=None, black_level=2.0):
        qe = files("ndvi_processing").joinpath("resources", "sensor_AR2020.csv")
        self.default_processor = NDVIProcessor(SensorConfig("AR2020", "GRBG", qe))
        self.processor = processor or self.default_processor
        self.black_level = black_level
        self.percentiles = {"tvi": (LOW_DEFAULT, HIGH_DEFAULT)}
        self._normalization_frame = 0
        self._ndvi_bounds = (-1.0, 1.0)
        self._tvi_bounds = None
        self._bounds_update_interval = 10
        self._bounds_smoothing = np.float32(0.25)
        self.tvi_adaptive = True
        self.temporal_enabled = True
        self.temporal_weight = float(TEMPORAL_WEIGHT)
        self.reset_temporal()

    def reset_temporal(self):
        self._temporal_superpixels = None
        self._temporal_processor = None
        self._temporal_black_level = None

    def set_temporal_filter(self, enabled, weight):
        validate_temporal_filter(enabled, weight)
        if (enabled, weight) != (self.temporal_enabled, self.temporal_weight):
            self.temporal_enabled = enabled
            self.temporal_weight = float(weight)
            self.reset_temporal()
            self._tvi_bounds = None

    @property
    def ndvi_bounds(self):
        return self._ndvi_bounds

    def set_ndvi_bounds(self, low, high):
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in (low, high)
            )
            or not -1 <= low <= high <= 1
            or high - low < 0.02 - 1e-9
        ):
            raise ValueError("NDVI bounds must span at least 0.02 within [-1, 1]")
        self._ndvi_bounds = (float(low), float(high))

    def indices(self, raw):
        if self.processor is self.default_processor:
            raw = np.asarray(raw, dtype=np.float32) - np.float32(self.black_level)
        return self.processor.process_raw(raw).indices

    def set_percentiles(self, index, low, high):
        if index not in self.percentiles:
            raise ValueError(f"Unknown index: {index}")
        if not 0 <= low < high <= 100:
            raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
        self.percentiles[index] = (low, high)
        setattr(self, f"_{index}_bounds", None)

    def set_tvi_adaptive(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("TVI adaptive contrast requires a boolean")
        self.tvi_adaptive = enabled

    def render(self, raw, maximum_size=DEFAULT_PREVIEW_SIZE):
        if raw.ndim != 2 or min(raw.shape) < 2:
            raise RuntimeError(
                "The camera does not provide the expected 2D RAW Bayer image"
            )
        raw = raw[: raw.shape[0] // 2 * 2, : raw.shape[1] // 2 * 2]
        sizes = (
            (maximum_size,) * 3
            if isinstance(maximum_size[0], int)
            else tuple(maximum_size)
        )
        if len(sizes) != 3:
            raise ValueError("Expected one preview size for each live image")
        superpixels = bayer_superpixels(
            raw, self.processor.sensor.cfa_pattern,
            green_mode=self.processor.sensor.green_mode,
        )
        if self.processor is self.default_processor:
            superpixels -= np.float32(self.black_level)
        if not self.temporal_enabled:
            self.reset_temporal()
        elif (
            self._temporal_superpixels is None
            or self._temporal_superpixels.shape != superpixels.shape
            or self._temporal_processor is not self.processor
            or self._temporal_black_level != self.black_level
        ):
            self._temporal_superpixels = superpixels
            self._temporal_processor = self.processor
            self._temporal_black_level = self.black_level
        else:
            np.subtract(superpixels, self._temporal_superpixels, out=superpixels)
            superpixels *= np.float32(self.temporal_weight)
            self._temporal_superpixels += superpixels
        if self.temporal_enabled:
            superpixels = self._temporal_superpixels
        channels = {
            name: superpixels[..., index]
            for index, name in enumerate(("R", "G", "NIR"))
        }
        indices = self.processor.process_channels(channels).indices
        old = self._tvi_bounds
        if self.tvi_adaptive:
            self._normalization_frame += 1
        if old is None or (
            self.tvi_adaptive
            and self._normalization_frame % self._bounds_update_interval == 0
        ):
            new = full_percentile_bounds(indices["tvi"], *self.percentiles["tvi"])
            if old is None:
                self._tvi_bounds = new
            else:
                a = self._bounds_smoothing
                self._tvi_bounds = tuple(
                    (1 - a) * o + a * n for o, n in zip(old, new)
                )
        preview = preview_bayer(raw, sizes[0])
        images = []
        for image, bounds, size in (
            (indices["ndvi"], self._ndvi_bounds, sizes[1]),
            (indices["tvi"], self._tvi_bounds, sizes[2]),
        ):
            image = normalize_with_bounds(image, *bounds)
            height, width = image.shape
            scale = min(size[0] / width, size[1] / height)
            target = (max(1, int(width * scale)), max(1, int(height * scale)))
            if target != (width, height):
                image = cv2.resize(
                    image, target,
                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                )
            colored = heatmap(image)
            images.append(colored)
        maximum = (
            float(np.iinfo(preview.dtype).max)
            if np.issubdtype(preview.dtype, np.integer)
            else max(float(np.max(preview)), 1)
        )
        raw8 = (
            preview
            if preview.dtype == np.uint8
            else np.clip(
                preview.astype(np.float32) * (255 / maximum), 0, 255
            ).astype(np.uint8)
        )
        return cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR), *images
