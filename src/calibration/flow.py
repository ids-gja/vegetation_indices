"""Camera-independent orchestration around the shared NDVI calibration package."""

from importlib.resources import files

import numpy as np
from ids_peak import ids_peak
from .library import (
    Calibration,
    HardwareGainLimits,
    calibrate_raw_white_target,
    suggest_hardware_gains,
)
from ndvi_processing import (
    NDVIProcessor,
    SensorConfig,
)


CHANNELS = {"R": "red", "G": "green", "NIR": "blue"}


class FrameAverage:
    def __init__(self, num_frames):
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        self.num_frames = num_frames
        self.count = 0
        self.total = None

    def add(self, raw):
        image = np.asarray(raw)
        if (
            image.ndim != 2
            or min(image.shape) < 2
            or any(size % 2 for size in image.shape)
        ):
            raise ValueError("Calibration requires an even-sized 2D raw Bayer frame")
        if self.total is not None and image.shape != self.total.shape:
            raise ValueError("Calibration frame shape changed during capture")
        if self.count >= self.num_frames:
            raise ValueError("Calibration capture is already complete")
        if self.total is None:
            self.total = np.zeros(image.shape, dtype=np.float64)
        self.total += image
        self.count += 1
        if self.count == self.num_frames:
            return (self.total / self.num_frames).astype(np.float32)
        return None


class GainControls:
    """Use the camera's linear per-channel GainSelector/Gain nodes."""

    def __init__(self, nodemap):
        self.selector = nodemap.FindNode("GainSelector")
        self.gain = nodemap.FindNode("Gain")
        entries = [entry.StringValue() for entry in self.selector.AvailableEntries()]
        self.channel_entries = {}
        for channel, keyword in CHANNELS.items():
            matches = [name for name in entries if keyword in name.casefold()]
            if not matches:
                raise ValueError(f"missing {keyword} GainSelector entry: {entries}")
            if len(matches) != 1:
                raise ValueError(f"multiple {keyword} GainSelector entries: {matches}")
            self.channel_entries[channel] = matches[0]
        self._auxiliary_entries = tuple(
            name for name in entries if name not in self.channel_entries.values()
        )

    def _select(self, channel):
        self.selector.SetCurrentEntry(self.channel_entries[channel])

    def _select_auxiliary(self, name):
        if name not in self._auxiliary_entries:
            raise ValueError(f"Unavailable auxiliary GainSelector entry: {name}")
        self.selector.SetCurrentEntry(name)

    def _increment(self):
        increment_type = self.gain.IncrementType()
        if increment_type == ids_peak.NodeIncrementType_NoIncrement:
            return None
        if increment_type == ids_peak.NodeIncrementType_FixedIncrement:
            return float(self.gain.Increment())
        raise ValueError(f"Unsupported Gain increment type: ListIncrement ({increment_type})")

    def auxiliary_options(self):
        options = []
        for name in self._auxiliary_entries:
            self._select_auxiliary(name)
            options.append(
                (
                    name,
                    float(self.gain.Minimum()),
                    float(self.gain.Maximum()),
                    self._increment(),
                    float(self.gain.Value()),
                )
            )
        return options

    def read_auxiliary(self, name):
        self._select_auxiliary(name)
        return float(self.gain.Value())

    def apply_auxiliary(self, name, value):
        self._select_auxiliary(name)
        minimum = float(self.gain.Minimum())
        maximum = float(self.gain.Maximum())
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} gain must be between {minimum} and {maximum}")
        self.gain.SetValue(float(value))
        return float(self.gain.Value())

    def read(self):
        result = {}
        for channel in CHANNELS:
            self._select(channel)
            result[channel] = float(self.gain.Value())
        return result

    def limits(self):
        result = {}
        for channel in CHANNELS:
            self._select(channel)
            result[channel] = HardwareGainLimits(
                float(self.gain.Minimum()),
                float(self.gain.Maximum()),
                self._increment(),
            )
        return result

    def apply(self, values):
        if set(values) != set(CHANNELS):
            raise ValueError("Hardware gains require R, G, and NIR")
        for channel in CHANNELS:
            self._select(channel)
            self.gain.SetValue(float(values[channel]))


def recommend_gains(raw, controls, *, black_level):
    return suggest_hardware_gains(
        raw,
        "GRBG",
        controls.limits(),
        controls.read(),
        offsets={channel: black_level for channel in CHANNELS},
    )


def finish_calibration(raw, controls, *, black_level):
    return calibrate_raw_white_target(
        raw,
        "GRBG",
        hardware_gains=controls.read(),
        offsets={channel: black_level for channel in CHANNELS},
    )


def make_processor(calibration: Calibration, hardware_gains):
    qe = files("ndvi_processing").joinpath("resources", "sensor_AR2020.csv")
    sensor = SensorConfig("AR2020", "GRBG", qe, hardware_gains=hardware_gains)
    return NDVIProcessor(sensor, calibration)
