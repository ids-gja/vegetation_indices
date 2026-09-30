import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import ndvi_processing
from calibration.library import Calibration
from ndvi_processing import NDVIProcessor, SensorConfig


QE_CSV = Path(ndvi_processing.__file__).parent / "resources" / "sensor_AR2020.csv"
RAW_GRBG = np.array([[4, 2, 8, 6], [10, 20, 18, 12]], dtype=np.uint16)


def test_default_import_does_not_load_optional_calibration():
    environment = os.environ.copy()
    local_src = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        (local_src, environment.get("PYTHONPATH", ""))
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, ndvi_processing\n"
            "from ndvi_processing import NDVIProcessor, SensorConfig\n"
            "assert not hasattr(ndvi_processing, 'Calibration')\n"
            "assert 'PySide6' not in sys.modules and 'ids_peak' not in sys.modules\n"
            "from calibration.library import Calibration, calibrate_white_target\n"
            "assert callable(calibrate_white_target)\n",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


def sensor(**overrides):
    settings = {
        "name": "test sensor",
        "cfa_pattern": "GRBG",
        "relative_qe_csv": QE_CSV,
    }
    settings.update(overrides)
    return SensorConfig(**settings)


@pytest.mark.parametrize("calibration", [None, "omitted"])
def test_uncalibrated_grbg_averages_both_green_sites(calibration):
    processor = (
        NDVIProcessor(sensor()) if calibration == "omitted"
        else NDVIProcessor(sensor(), calibration=None)
    )
    result = processor.process_raw(RAW_GRBG)

    np.testing.assert_array_equal(result.red, [[2, 6]])
    np.testing.assert_array_equal(result.green, [[12, 10]])
    np.testing.assert_array_equal(result.nir, [[10, 18]])
    np.testing.assert_allclose(result.indices["ndvi"], [[2 / 3, 0.5]])
    np.testing.assert_allclose(result.indices["tvi"], [[20 / 144, 108 / 100]])
    assert set(result.indices) == {"ndvi", "tvi"}


def test_tvi_is_returned_under_its_own_key():
    from ndvi_processing.indices import calculate_indices

    indices = calculate_indices(
        np.array([[50]], dtype=np.uint16),
        np.array([[60]], dtype=np.uint16),
        np.array([[10]], dtype=np.uint16),
    )
    assert set(indices) == {"ndvi", "tvi"}
    np.testing.assert_allclose(indices["tvi"], [[-1400]])


def test_index_calculation_preserves_inputs_and_original_formulas():
    from ndvi_processing.indices import calculate_indices

    red = np.array([[50, 3], [12, 0]], dtype=np.float32)
    green = np.array([[60, 2], [10, 0]], dtype=np.float32)
    nir = np.array([[10, 5], [4, 0]], dtype=np.float32)
    originals = [channel.copy() for channel in (red, green, nir)]
    for channel in (red, green, nir):
        channel.setflags(write=False)

    indices = calculate_indices(red, green, nir)

    for channel, original in zip((red, green, nir), originals):
        np.testing.assert_array_equal(channel, original)
    np.testing.assert_allclose(indices["tvi"], [[-1400, 300], [-320, 0]])
    np.testing.assert_allclose(
        indices["ndvi"], [[-0.6, 1], [-1 / 3, np.nan]], equal_nan=True
    )


def test_index_calculation_matches_formula_on_strided_float32_channels():
    from ndvi_processing.indices import calculate_indices

    rng = np.random.default_rng(17)
    red, green, nir = (
        rng.uniform(-20, 255, (16, 24)).astype(np.float32)[:, ::2]
        for _ in range(3)
    )
    corrected_red = np.maximum(red - nir, 0)
    corrected_green = np.maximum(green - nir, 0)
    expected_tvi = 0.5 * (
        120 * (nir - corrected_green)
        - 200 * (corrected_red - corrected_green)
    )
    denominator = nir + corrected_red
    expected_ndvi = np.full_like(denominator, np.nan)
    np.divide(
        nir - corrected_red, denominator,
        out=expected_ndvi, where=denominator != 0,
    )

    indices = calculate_indices(red, green, nir)

    np.testing.assert_array_equal(indices["tvi"], expected_tvi)
    np.testing.assert_array_equal(indices["ndvi"], expected_ndvi)


def test_float_superpixel_channels_share_raw_calibration_and_index_math():
    raw = np.array([[4, 2], [10, 20]], dtype=np.uint16)
    processor = NDVIProcessor(sensor(black_level=2))
    channels = {
        "R": np.array([[2.0]], dtype=np.float32),
        "G": np.array([[12.0]], dtype=np.float32),
        "NIR": np.array([[10.0]], dtype=np.float32),
    }
    from_raw = processor.process_raw(raw)
    from_channels = processor.process_channels(channels)
    np.testing.assert_array_equal(from_channels.green, [[10]])
    for name in ("ndvi", "tvi"):
        np.testing.assert_allclose(
            from_channels.indices[name], from_raw.indices[name]
        )


def test_explicit_calibration_applies_offsets_gains_and_matrix():
    calibration = Calibration(
        gains={"R": 2, "G": 0.5, "NIR": 3},
        offsets={"R": 1, "G": 2, "NIR": 4},
        matrix=((0, 0, 1), (0, 1, 0), (1, 0, 0)),
    )
    result = NDVIProcessor(sensor(), calibration).process_raw(RAW_GRBG)

    np.testing.assert_array_equal(result.red, [[18, 42]])
    np.testing.assert_array_equal(result.green, [[5, 4]])
    np.testing.assert_array_equal(result.nir, [[2, 10]])
    np.testing.assert_allclose(result.indices["ndvi"], [[-0.8, -32 / 52]])
    np.testing.assert_allclose(result.indices["tvi"], [[36 / 25, 420 / 16]])
    assert set(result.indices) == {"ndvi", "tvi"}


def test_default_worker_sample_uses_uncalibrated_raw_values():
    result = NDVIProcessor(sensor()).process_raw(
        np.array([[10, 20], [40, 10]], dtype=np.uint16)
    )

    np.testing.assert_allclose(result.indices["ndvi"], [[1 / 3]])
    np.testing.assert_allclose(result.indices["tvi"], [[8]])


def test_uncalibrated_still_validates_sensor_and_raw_image():
    with pytest.raises(ValueError, match="cfa_pattern"):
        NDVIProcessor(sensor(cfa_pattern="INVALID"))
    with pytest.raises(FileNotFoundError):
        NDVIProcessor(sensor(relative_qe_csv=QE_CSV.with_name("missing.csv")))
    with pytest.raises(ValueError, match="even"):
        NDVIProcessor(sensor()).process_raw(np.ones((3, 4), dtype=np.uint16))


def test_explicit_calibration_still_validates_gains_and_hardware_readback():
    with pytest.raises(ValueError, match="missing gain for NIR"):
        NDVIProcessor(sensor(), Calibration(gains={"R": 1, "G": 1}))
    with pytest.raises(ValueError, match="hardware gains must be supplied"):
        NDVIProcessor(
            sensor(),
            Calibration(
                gains={"R": 1, "G": 1, "NIR": 1},
                hardware_gains={"R": 1, "G": 1, "NIR": 1},
            ),
        )
    result = NDVIProcessor(
        sensor(hardware_gains={"R": 2, "G": 2, "NIR": 2}),
    ).process_raw(RAW_GRBG)
    np.testing.assert_array_equal(result.red, [[2, 6]])
