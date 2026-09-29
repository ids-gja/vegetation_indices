"""Sensor configuration and raw-image processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ndvi_processing.bayer import extract_channels, validate_cfa_pattern
from ndvi_processing.indices import calculate_indices
from ndvi_processing.spectral import load_filter_curve, load_relative_qe, plot_spectra

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class CalibrationLike(Protocol):
    @property
    def gains(self) -> dict[str, float]: ...

    @property
    def offsets(self) -> dict[str, float]: ...

    @property
    def matrix(self) -> tuple[tuple[float, ...], ...] | None: ...

    def validate_hardware_gains(self, actual: dict[str, float] | None) -> None: ...


@dataclass(frozen=True)
class SensorConfig:
    """The immutable optical and CFA description for one camera system."""

    name: str
    cfa_pattern: str
    relative_qe_csv: str | Path
    filter_csv: str | Path | None = None
    green_mode: str = "mean"
    hardware_gains: dict[str, float] | None = None
    black_level: float = 0.0


@dataclass(frozen=True)
class IndexResult:
    red: np.ndarray
    green: np.ndarray
    nir: np.ndarray
    indices: dict[str, np.ndarray]


class NDVIProcessor:
    """Calculate vegetation indices from raw images, optionally applying calibration."""

    def __init__(self, sensor: SensorConfig, calibration: CalibrationLike | None = None):
        self.sensor = sensor
        self.calibration = calibration
        if calibration is not None:
            calibration.validate_hardware_gains(sensor.hardware_gains)
        if not np.isfinite(sensor.black_level) or sensor.black_level < 0:
            raise ValueError("black_level must be finite and nonnegative")
        validate_cfa_pattern(sensor.cfa_pattern)
        if not Path(sensor.relative_qe_csv).is_file():
            raise FileNotFoundError(sensor.relative_qe_csv)
        load_relative_qe(sensor.relative_qe_csv)
        if sensor.filter_csv is not None:
            if not Path(sensor.filter_csv).is_file():
                raise FileNotFoundError(sensor.filter_csv)
            load_filter_curve(sensor.filter_csv)
        if calibration is not None:
            for name in ("R", "G", "NIR"):
                if name not in calibration.gains:
                    raise ValueError(f"calibration is missing gain for {name}")

    def plot_spectra(self) -> Figure:
        """Return the configured sensor/filter comparison without displaying it."""
        if self.sensor.filter_csv is None:
            raise ValueError("Set SensorConfig.filter_csv before plotting spectra")
        return plot_spectra(
            self.sensor.relative_qe_csv,
            self.sensor.filter_csv,
            sensor_name=self.sensor.name,
        )

    def process_raw(self, raw: np.ndarray) -> IndexResult:
        return self.process_channels(
            extract_channels(
                raw, self.sensor.cfa_pattern, green_mode=self.sensor.green_mode
            )
        )

    def process_channels(self, channels: dict[str, np.ndarray]) -> IndexResult:
        """Calculate indices from aligned, optionally resampled float R/G/NIR planes."""
        if set(channels) != {"R", "G", "NIR"}:
            raise ValueError("channels must contain R, G, and NIR")
        corrected = {}
        for name in ("R", "G", "NIR"):
            values = np.asarray(channels[name], dtype=np.float32)
            if self.calibration is not None:
                values = values - np.float32(self.calibration.offsets.get(name, 0.0))
                values = values * np.float32(self.calibration.gains[name])
            else:
                values = values - np.float32(self.sensor.black_level)
            corrected[name] = values

        if self.calibration is not None and self.calibration.matrix is not None:
            vector = np.stack(
                (corrected["R"], corrected["G"], corrected["NIR"]), axis=-1
            )
            transformed = vector @ np.asarray(self.calibration.matrix, dtype=np.float32).T
            corrected["R"], corrected["G"], corrected["NIR"] = (
                transformed[..., 0],
                transformed[..., 1],
                transformed[..., 2],
            )

        return IndexResult(
            red=corrected["R"],
            green=corrected["G"],
            nir=corrected["NIR"],
            indices=calculate_indices(
                corrected["R"], corrected["G"], corrected["NIR"]
            ),
        )
