"""Reusable raw-image vegetation-index processing."""

from ndvi_processing.processor import IndexResult, NDVIProcessor, SensorConfig
from ndvi_processing.scaling import bayer_superpixels, resize_superpixels
from ndvi_processing.spectral import load_relative_qe, load_filter_curve, plot_spectra

__all__ = [
    "IndexResult",
    "NDVIProcessor",
    "SensorConfig",
    "bayer_superpixels",
    "load_filter_curve",
    "load_relative_qe",
    "plot_spectra",
    "resize_superpixels",
]
