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
    """Calculate NDVI and CVI without Python loops over pixels."""
    red, green, nir = (
        np.asarray(value, dtype=np.float32) for value in (red, green, nir)
    )
    if red.shape != green.shape or red.shape != nir.shape:
        raise ValueError("red, green, and nir must have identical shapes")

    # coreect ir parasitic influence on R, B
    red -= nir
    green -= nir

    ndvi = _safe_divide(nir - red, nir + red)
    cvi = _safe_divide(nir * red, green * green)
    return {"ndvi": ndvi, "cvi": cvi}
