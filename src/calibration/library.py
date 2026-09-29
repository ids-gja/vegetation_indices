"""Calibration data and white-target gain calibration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import numpy as np

from ndvi_processing.bayer import extract_channels


def _validate_hardware_gains(gains: dict[str, float]) -> None:
    if set(gains) != {"R", "G", "NIR"}:
        raise ValueError("hardware gains must contain exactly R, G, and NIR (physical B)")
    if any(not np.isfinite(value) or value <= 0 for value in gains.values()):
        raise ValueError("hardware gains must be finite positive linear multipliers")


@dataclass(frozen=True)
class HardwareGainLimits:
    """Linear gain range; None increment means continuous gain."""

    minimum: float
    maximum: float
    increment: float | None

    def __post_init__(self) -> None:
        if (
            not all(np.isfinite(value) for value in (self.minimum, self.maximum))
            or self.minimum <= 0
            or self.maximum < self.minimum
            or (
                self.increment is not None
                and (not np.isfinite(self.increment) or self.increment <= 0)
            )
        ):
            raise ValueError(
                "gain limits require 0 < minimum <= maximum and a positive "
                "increment or None for continuous gain"
            )

    def quantize(self, gain: float) -> float:
        """Bound continuous gains or round fixed gains to the nearest supported step."""
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("requested gain must be finite and positive")
        if self.increment is None:
            return float(np.clip(gain, self.minimum, self.maximum))
        max_steps = int(np.floor((self.maximum - self.minimum) / self.increment))
        next_gain = self.minimum + (max_steps + 1) * self.increment
        if np.isclose(next_gain, self.maximum, rtol=1e-12, atol=0):
            max_steps += 1
        bounded = np.clip(gain, self.minimum, self.maximum)
        position = (bounded - self.minimum) / self.increment
        lower = int(np.floor(position))
        fraction = position - lower
        round_up = fraction > 0.5 or np.isclose(fraction, 0.5, rtol=0, atol=1e-12)
        steps = min(max_steps, lower + int(round_up))
        return float(min(self.maximum, self.minimum + steps * self.increment))


@dataclass(frozen=True)
class Calibration:
    """Persistent per-channel calibration for one fixed camera setup."""

    gains: dict[str, float]
    offsets: dict[str, float] = field(default_factory=dict)
    matrix: tuple[tuple[float, ...], ...] | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    hardware_gains: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.hardware_gains is not None:
            _validate_hardware_gains(self.hardware_gains)

    def save(self, path: str | Path) -> None:
        payload = {
            "gains": self.gains,
            "offsets": self.offsets,
            "matrix": self.matrix,
            "metadata": self.metadata,
            "hardware_gains": self.hardware_gains,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            gains={str(k): float(v) for k, v in payload["gains"].items()},
            offsets={str(k): float(v) for k, v in payload.get("offsets", {}).items()},
            matrix=(
                tuple(tuple(float(value) for value in row) for row in payload["matrix"])
                if payload.get("matrix") is not None
                else None
            ),
            metadata={str(k): str(v) for k, v in payload.get("metadata", {}).items()},
            hardware_gains=(
                {str(k): float(v) for k, v in payload["hardware_gains"].items()}
                if payload.get("hardware_gains") is not None
                else None
            ),
        )

    def validate_hardware_gains(self, actual: dict[str, float] | None) -> None:
        """Require capture gain readback to match the calibration's hardware state."""
        if self.hardware_gains is None and actual is None:
            return
        if self.hardware_gains is None or actual is None:
            raise ValueError("hardware gains must be supplied for both calibration and capture")
        _validate_hardware_gains(actual)
        _validate_hardware_gains(self.hardware_gains)
        for name, expected in self.hardware_gains.items():
            if not np.isclose(actual[name], expected, rtol=1e-6, atol=0):
                raise ValueError(
                    f"hardware gain mismatch for {name}: calibrated at {expected}, "
                    f"capture uses {actual[name]}; recalibrate after changing gains"
                )


def _white_target_means(
    channels: dict[str, np.ndarray],
    offsets: dict[str, float],
) -> dict[str, float]:
    if not channels:
        raise ValueError("at least one channel is required")
    means: dict[str, float] = {}
    for name, values in channels.items():
        offset = offsets.get(name, 0.0)
        if not np.isfinite(offset):
            raise ValueError(f"channel {name!r} has a non-finite offset")
        data = np.asarray(values, dtype=np.float64) - offset
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            raise ValueError(f"channel {name!r} contains no finite samples")
        mean = float(np.mean(finite))
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError(f"channel {name!r} must have a finite positive calibrated mean")
        means[name] = mean
    return means


def suggest_hardware_gains(
    raw: np.ndarray,
    cfa_pattern: str,
    gain_limits: dict[str, HardwareGainLimits],
    current_gains: dict[str, float],
    *,
    offsets: dict[str, float] | None = None,
    green_mode: str = "mean",
    target: float | None = None,
) -> dict[str, float]:
    """Stage one: recommend absolute quantized gains from a raw white-target frame.

    Gains are linear multipliers for R, G, and NIR (physical B). Current gains
    and offsets must describe this capture. The default target is the largest
    corrected channel mean; an explicit target is in black-subtracted raw units.
    The caller applies the returned gains and obtains a fresh frame for stage two.
    """
    _validate_hardware_gains(current_gains)
    if set(gain_limits) != {"R", "G", "NIR"}:
        raise ValueError("gain limits must contain exactly R, G, and NIR (physical B)")
    for name, limits in gain_limits.items():
        if not np.isclose(limits.quantize(current_gains[name]), current_gains[name], rtol=1e-6, atol=0):
            raise ValueError(f"current gain for {name} is outside its supported range or grid")
    means = _white_target_means(
        extract_channels(raw, cfa_pattern, green_mode), offsets or {}
    )
    target = max(means.values()) if target is None else target
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target must be finite and positive")
    return {
        name: gain_limits[name].quantize(current_gains[name] * (target / mean))
        for name, mean in means.items()
    }


def calibrate_white_target(
    channels: dict[str, np.ndarray],
    reference: float = 1.0,
    offsets: dict[str, float] | None = None,
    *,
    hardware_gains: dict[str, float] | None = None,
) -> Calibration:
    """Calculate residual software gains from a uniform, unsaturated white target.

    Hardware gains, when supplied, are actual readbacks already applied to the
    captured samples. Offsets must be measured at those same settings.
    """
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference must be finite and positive")
    if hardware_gains is not None:
        _validate_hardware_gains(hardware_gains)
        if set(channels) != set(hardware_gains):
            raise ValueError("hardware-aware calibration requires R, G, and NIR channels")
    applied_offsets = offsets or {}
    means = _white_target_means(channels, applied_offsets)
    return Calibration(
        gains={name: reference / mean for name, mean in means.items()},
        offsets=applied_offsets,
        metadata={"method": "uniform_reflectance_white_target"},
        hardware_gains=dict(hardware_gains) if hardware_gains is not None else None,
    )


def calibrate_raw_white_target(
    raw: np.ndarray,
    cfa_pattern: str,
    *,
    hardware_gains: dict[str, float],
    reference: float = 1.0,
    offsets: dict[str, float] | None = None,
    green_mode: str = "mean",
) -> Calibration:
    """Stage two: calibrate a fresh raw frame after setting and reading back gains."""
    result = calibrate_white_target(
        extract_channels(raw, cfa_pattern, green_mode),
        reference=reference,
        offsets=offsets,
        hardware_gains=hardware_gains,
    )
    return replace(result, metadata={**result.metadata, "green_mode": green_mode})
