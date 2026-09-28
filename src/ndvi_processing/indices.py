"""Vegetation-index formulas."""

from __future__ import annotations

import numpy as np


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def calculate_indices(
    red: np.ndarray,
    green: np.ndarray,
    nir: np.ndarray,
    *,
    tvi_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Calculate NDVI, CVI, and TVI without Python loops over pixels."""
    red, green, nir = (
        np.asarray(value, dtype=np.float32) for value in (red, green, nir)
    )
    if red.shape != green.shape or red.shape != nir.shape:
        raise ValueError("red, green, and nir must have identical shapes")
    ndvi = _safe_divide(nir - red, nir + red)
    cvi = _safe_divide(nir * red, green * green)
    tvi = 0.5 * (120.0 * (nir - green) - 200.0 * (red - green)) * tvi_scale
    return {"ndvi": ndvi, "cvi": cvi, "tvi": tvi.astype(np.float32, copy=False)}
