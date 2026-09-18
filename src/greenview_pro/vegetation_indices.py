"""Hardware-independent Bayer calibration and vegetation-index calculations.

The caller supplies ``CameraImageParameters`` from camera metadata or a persisted
calibration. This module deliberately has no IDS, Qt, OpenCV, or UI dependency.
"""

from dataclasses import dataclass, field

import numpy as np


DEFAULT_COLOR_CORRECTION_MATRIX = np.array(
    [
        [0.116014, -0.017513, -0.016800],
        [-0.000626, 0.077999, -0.013066],
        [-0.092187, -0.045365, 0.129469],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class CameraImageParameters:
    """Calibration and Bayer metadata needed to calculate vegetation indices."""

    pixel_format: str = "BayerGR12"
    bayer_pattern: str = "GRBG"
    red_site: tuple[int, int] = (0, 1)
    green_site: tuple[int, int] = (0, 0)
    blue_site: tuple[int, int] = (1, 0)
    black_level: float = 0.0
    white_level: float | None = None
    red_gain: float = 1.0
    green_gain: float = 1.0
    blue_gain: float = 1.0
    color_correction_matrix: np.ndarray = field(
        default_factory=lambda: DEFAULT_COLOR_CORRECTION_MATRIX.copy()
    )
    denominator_epsilon: float = 0.01


@dataclass(frozen=True)
class VegetationIndices:
    """Calibrated spectral channels and derived vegetation indices."""

    green: np.ndarray
    red: np.ndarray
    nir: np.ndarray
    ndvi: np.ndarray
    cvi: np.ndarray
    tvi: np.ndarray


def _sample_site(raw, site):
    row, column = site
    return raw[row::2, column::2].astype(np.float32)


def calculate_vegetation_indices(raw, parameters=None):
    """Calculate calibrated spectral channels plus NDVI, CVI, and TVI.

    TVI is the transformed vegetation index: ``sqrt(NDVI + 0.5)``. Invalid
    values caused by the transform's negative domain are represented as zero.
    """
    if raw.ndim != 2 or min(raw.shape) < 2:
        raise ValueError("Vegetation indices require a two-dimensional Bayer image")
    if parameters is None:
        parameters = CameraImageParameters()

    raw = raw[: raw.shape[0] // 2 * 2, : raw.shape[1] // 2 * 2]
    red_raw = _sample_site(raw, parameters.red_site)
    green_raw = _sample_site(raw, parameters.green_site)
    blue_raw = _sample_site(raw, parameters.blue_site)

    red_raw = (red_raw - parameters.black_level) * parameters.red_gain
    green_raw = (green_raw - parameters.black_level) * parameters.green_gain
    blue_raw = (blue_raw - parameters.black_level) * parameters.blue_gain

    matrix = np.asarray(parameters.color_correction_matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError("The color correction matrix must have shape (3, 3)")

    green = matrix[0, 0] * red_raw + matrix[0, 1] * green_raw + matrix[0, 2] * blue_raw
    red = matrix[1, 0] * red_raw + matrix[1, 1] * green_raw + matrix[1, 2] * blue_raw
    nir = matrix[2, 0] * red_raw + matrix[2, 1] * green_raw + matrix[2, 2] * blue_raw

    epsilon = np.float32(parameters.denominator_epsilon)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ndvi = (nir - red) / (nir + red + epsilon)
        cvi = (nir * red) / (green * green + epsilon)
        tvi = np.sqrt(ndvi + np.float32(0.5))

    return VegetationIndices(
        green=green,
        red=red,
        nir=nir,
        ndvi=ndvi,
        cvi=cvi,
        tvi=np.nan_to_num(tvi, nan=0.0, posinf=0.0, neginf=0.0),
    )
