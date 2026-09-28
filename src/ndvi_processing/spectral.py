"""CSV readers and visual comparison of sensor and filter spectral curves."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _read_curve(path: str | Path, expected_columns: int) -> np.ndarray:
    rows: list[tuple[float, ...]] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                values = tuple(float(value.strip().replace(",", ".")) for value in row[:expected_columns])
            except ValueError:
                continue
            if len(values) == expected_columns:
                rows.append(values)
    if len(rows) < 2:
        raise ValueError(f"{path} does not contain at least two numeric curve rows")
    data = np.asarray(rows, dtype=np.float64)
    order = np.argsort(data[:, 0])
    data = data[order]
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"{path} must contain strictly increasing wavelengths")
    return data


def load_filter_curve(path: str | Path) -> np.ndarray:
    """Load ``wavelength_nm,transmission`` rows."""
    return _read_curve(path, 2)


def load_relative_qe(path: str | Path) -> np.ndarray:
    """Load ``wavelength_nm,qe_r,qe_g,qe_b`` rows."""
    return _read_curve(path, 4)


def plot_spectra(
    relative_qe_csv: str | Path,
    filter_csv: str | Path,
    *,
    sensor_name: str | None = None,
) -> Figure:
    """Return a comparison figure without displaying it; requires the plot extra."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    filter_curve = load_filter_curve(filter_csv)
    sensor_curve = load_relative_qe(relative_qe_csv)
    for name, curve in (("Sensor QE", sensor_curve), ("Filter transmission", filter_curve)):
        if not np.all(np.isfinite(curve)) or np.any(curve[:, 1:] < 0):
            raise ValueError(f"{name} must contain finite values and nonnegative responses")
    if np.any(filter_curve[:, 1] > 1):
        raise ValueError("Filter transmission must be a fraction from 0 to 1")

    colors = ("red", "green", "blue")
    fig, ax = plt.subplots()
    filter_ax = ax.twinx()
    for column, color in enumerate(colors, start=1):
        ax.plot(
            sensor_curve[:, 0],
            sensor_curve[:, column],
            color=color,
            label=f"Sensor {color}",
        )

    fill_wavelengths = np.unique(
        np.concatenate((
            filter_curve[:, 0],
            np.arange(filter_curve[0, 0], filter_curve[-1, 0], 1.0),
            [380.0, 780.0],
        ))
    )
    fill_wavelengths = fill_wavelengths[
        (fill_wavelengths >= filter_curve[0, 0])
        & (fill_wavelengths <= filter_curve[-1, 0])
    ]
    midpoints = (fill_wavelengths[:-1] + fill_wavelengths[1:]) / 2
    # Approximate visible spectral colors; UV and infrared have no visible hue.
    spectral_wavelengths = [380, 440, 490, 510, 580, 645, 780]
    spectral_rgb = np.array([
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
    ])
    fill_colors = np.column_stack(
        [np.interp(midpoints, spectral_wavelengths, channel) for channel in spectral_rgb.T]
    )
    fill_colors[(midpoints < 380) | (midpoints > 780)] = 0.5
    transmission = np.interp(
        fill_wavelengths, filter_curve[:, 0], filter_curve[:, 1] * 100
    )
    vertices = [
        [(left, 0), (left, lower), (right, upper), (right, 0)]
        for left, right, lower, upper in zip(
            fill_wavelengths[:-1], fill_wavelengths[1:], transmission[:-1], transmission[1:]
        )
    ]
    filter_ax.add_collection(PolyCollection(
        vertices,
        facecolors=fill_colors,
        alpha=0.3,
        edgecolors="none",
        antialiaseds=False,
        zorder=1,
    ))

    filter_ax.plot(
        filter_curve[:, 0],
        filter_curve[:, 1] * 100,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Filter: {Path(filter_csv).stem}",
        zorder=3,
    )
    ax.set_title(f"{sensor_name or Path(relative_qe_csv).stem}: RGB Response and Filter")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Sensor QE (CSV units)")
    ax.set_ylim(bottom=0)
    filter_ax.set_ylabel("Filter transmission (%)")
    filter_ax.set_ylim(0, 100)
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    filter_handles, filter_labels = filter_ax.get_legend_handles_labels()
    ax.legend(handles + filter_handles, labels + filter_labels)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    resources = Path(__file__).resolve().parent / "resources"
    plot_spectra(
        resources / "sensor_AR2020.csv",
        resources / "midopt_tb550_660_850.csv",
        sensor_name="AR2020",
    )
    plt.show()