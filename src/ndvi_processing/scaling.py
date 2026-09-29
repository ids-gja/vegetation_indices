"""Float Bayer superpixels and aspect-preserving channel interpolation."""

import cv2
import numpy as np

from ndvi_processing.bayer import extract_channels


def bayer_superpixels(raw, pattern, green_mode="mean"):
    """Combine each Bayer cell into a float R/G/NIR triplet."""
    channels = extract_channels(raw, pattern, green_mode=green_mode)
    superpixels = np.empty((*channels["R"].shape, 3), dtype=np.float32)
    for index, name in enumerate(("R", "G", "NIR")):
        superpixels[..., index] = channels[name]
    return superpixels


def resize_superpixels(superpixels, maximum_size):
    """Interpolate float triplets to an aspect-preserving pixel budget."""
    height, width = superpixels.shape[:2]
    maximum_width, maximum_height = maximum_size
    if maximum_width < 2 or maximum_height < 2:
        raise ValueError("Preview dimensions must be at least two pixels")
    scale = min(maximum_width / width, maximum_height / height)
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    if target == (width, height):
        return superpixels
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(superpixels, target, interpolation=interpolation)
