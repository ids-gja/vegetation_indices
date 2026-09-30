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
) -> dict[str, np.ndarray]:
    """Calculate NDVI and TVI without Python loops over pixels."""
    red, green, nir = (
        np.asarray(value, dtype=np.float32) for value in (red, green, nir)
    )
    if red.shape != green.shape or red.shape != nir.shape:
        raise ValueError("red, green, and nir must have identical shapes")

    # correct the parasitic NIR contamination in the red and green channels
    red = np.subtract(red, nir)
    green = np.subtract(green, nir)
    np.maximum(red, 0, out=red)
    np.maximum(green, 0, out=green)

    # tvi = 0.5 * (120 * (nir - green) - 200 * (red - green))
    tvi = np.subtract(nir, green)
    np.multiply(tvi, 120, out=tvi)
    np.subtract(red, green, out=green)
    np.multiply(green, 200, out=green)
    np.subtract(tvi, green, out=tvi)
    np.multiply(tvi, 0.5, out=tvi)

    # ndvi = (nir - red) / (nir + red)
    # Reuse the working planes after TVI to avoid NDVI numerator/denominator copies.
    np.add(nir, red, out=green)
    np.subtract(nir, red, out=red)
    ndvi = _safe_divide(red, green)

    return {"ndvi": ndvi, "tvi": tvi}
