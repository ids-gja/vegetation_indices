"""One-time exposure search using the unscaled Bayer sensor values."""

import numpy as np


WHITE_LEVEL = 4095  # AR2020 BayerGR12, right-aligned in uint16 frames.
SATURATED_FRACTION = 0.01
EXPOSURE_PRECISION_US = 100


class ExposureSearch:
    def __init__(self, minimum, maximum, initial):
        if minimum <= 0 or maximum < minimum:
            raise ValueError("Invalid camera exposure range")
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.exposure = max(self.minimum, min(int(initial), self.maximum))
        self.safe = None
        self.clipped = None
        self.done = False
        self.attempts = 0

    def observe(self, raw):
        if self.done:
            return None
        image = np.asarray(raw)
        if image.ndim != 2 or min(image.shape) < 2:
            raise ValueError("Autoexposure requires a two-dimensional Bayer frame")
        saturated = np.count_nonzero(image >= WHITE_LEVEL)
        if saturated < SATURATED_FRACTION * image.size:
            self.safe = self.exposure
        else:
            self.clipped = self.exposure

        self.attempts += 1
        if self.safe is not None and self.clipped is not None:
            if self.clipped - self.safe <= EXPOSURE_PRECISION_US:
                candidate = self.safe
                self.done = True
            else:
                candidate = (self.safe + self.clipped) // 2
        elif self.safe is not None:
            if self.safe == self.maximum:
                self.done = True
                return None
            candidate = min(self.maximum, max(self.exposure + 1, self.exposure * 2))
        else:
            if self.exposure == self.minimum:
                self.done = True
                return None
            candidate = max(self.minimum, (self.minimum + self.exposure) // 2)

        if self.attempts >= 20:
            candidate = self.safe if self.safe is not None else self.minimum
            self.done = True
        if candidate == self.exposure:
            self.done = True
            return None
        self.exposure = candidate
        return candidate
