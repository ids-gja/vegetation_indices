"""Extract R, G, and NIR planes from a standard two-by-two Bayer CFA."""

from __future__ import annotations

import numpy as np

VALID_CFA_PATTERNS = frozenset({"RGGB", "GRBG", "GBRG", "BGGR"})


def validate_cfa_pattern(cfa_pattern: str) -> None:
    if cfa_pattern not in VALID_CFA_PATTERNS:
        raise ValueError(
            f"cfa_pattern must be one of {', '.join(sorted(VALID_CFA_PATTERNS))}"
        )


def extract_channels(
    raw: np.ndarray,
    cfa_pattern: str,
    green_mode: str = "mean",
) -> dict[str, np.ndarray]:
    """Return R, G, NIR planes from a Bayer mosaic at half resolution.

    The pattern names the four raw sites row by row. The B site provides NIR.
    """
    validate_cfa_pattern(cfa_pattern)
    image = np.asarray(raw)
    if image.ndim != 2:
        raise ValueError("raw image must be a two-dimensional array")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("raw image must contain at least one CFA cell")
    if image.shape[0] % 2 or image.shape[1] % 2:
        raise ValueError("raw image dimensions must be even")
    if green_mode not in {"first", "mean"}:
        raise ValueError("green_mode must be 'first' or 'mean'")

    planes = (
        image[0::2, 0::2],
        image[0::2, 1::2],
        image[1::2, 0::2],
        image[1::2, 1::2],
    )
    green = planes[cfa_pattern.index("G")]
    if green_mode == "mean":
        green = (
            green.astype(np.float32) + planes[cfa_pattern.rindex("G")].astype(np.float32)
        ) * 0.5
    return {
        "R": planes[cfa_pattern.index("R")],
        "G": green,
        "NIR": planes[cfa_pattern.index("B")],
    }
