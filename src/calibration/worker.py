"""Optional camera calibration, layered on the uncalibrated live worker."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from ids_peak import ids_peak
from PySide6.QtCore import Signal
from calibration.library import Calibration

from greenview_pro.processing import CameraWorker
from calibration.flow import FrameAverage, GainControls, finish_calibration, make_processor, recommend_gains


CALIBRATION_FILE = Path.home() / ".greenview_pro" / "calibration.json"


class CalibratedCameraWorker(CameraWorker):
    calibration_changed = Signal(bool, str)
    calibration_step = Signal(str)
    black_level_changed = Signal(float)
    gain_options_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self._calibration = None
        self._capture_request = None
        self._capture = None
        self._capture_stage = None
        self._first_black_level = None
        self._first_gains = None
        self._first_exposure = None
        self._first_auxiliary_gain = None
        self._gain_controls = None
        self._settle_frames = 0
        self._gain_check_frame = 0

    def request_calibration(
        self, stage, num_frames, black_level, auxiliary_selector="", auxiliary_value=0.0
    ):
        if stage not in ("first", "second") or num_frames < 1:
            raise ValueError("Invalid calibration stage or frame count")
        self._capture_request = (
            stage, num_frames, black_level, auxiliary_selector, auxiliary_value
        )

    def _controls(self):
        if self._gain_controls is None:
            self._gain_controls = GainControls(self._remote)
        return self._gain_controls

    def _activate_calibration(self):
        request = self._capture_request
        if request is None:
            return
        self._capture_request = None
        stage, num_frames, black_level, auxiliary_selector, auxiliary_value = request
        if self._remote is None:
            self.calibration_step.emit("Connect the camera before calibrating")
            return
        if stage == "second" and self._first_gains is None:
            self.calibration_step.emit("Capture the first white target before the second")
            return
        self._auto_exposure = None
        self._auto_settle_frames = 0
        if stage == "first":
            self._first_gains = None
            self._first_auxiliary_gain = None
            self._processor = self._default_processor
            self._ndvi_bounds = self._cvi_bounds = None
            self.calibration_changed.emit(False, "Uncalibrated preview")
            if auxiliary_selector:
                actual = self._controls().apply_auxiliary(auxiliary_selector, auxiliary_value)
                self._first_auxiliary_gain = (auxiliary_selector, actual)
            self._remote.FindNode("BlackLevel").SetValue(float(black_level))
            self._first_black_level = float(self._remote.FindNode("BlackLevel").Value())
            self._set_default_black_level(self._first_black_level)
            self._first_exposure = float(self._remote.FindNode("ExposureTime").Value())
            self.black_level_changed.emit(self._first_black_level)
        else:
            selected = self._first_auxiliary_gain[0] if self._first_auxiliary_gain else ""
            if auxiliary_selector != selected:
                raise ValueError("Auxiliary gain selector changed between white captures")
            self._validate_auxiliary_gain(self._first_auxiliary_gain)
            exposure = float(self._remote.FindNode("ExposureTime").Value())
            if not np.isclose(exposure, self._first_exposure, rtol=0, atol=1e-6):
                raise ValueError("Exposure changed between white captures")
            actual = float(self._remote.FindNode("BlackLevel").Value())
            if not np.isclose(actual, self._first_black_level, rtol=0, atol=1e-6):
                raise ValueError("BlackLevel changed between white captures")
            Calibration(
                {"R": 1, "G": 1, "NIR": 1}, hardware_gains=self._first_gains
            ).validate_hardware_gains(self._controls().read())
        self._capture_stage = stage
        self._capture = FrameAverage(num_frames)
        self._settle_frames = 10 if stage == "second" or auxiliary_selector else 2
        self.calibration_step.emit(f"Capturing {num_frames} fresh white frames ({stage})...")

    def _advance_calibration(self, raw):
        self._activate_calibration()
        if self._capture is None:
            return
        if self._settle_frames:
            self._settle_frames -= 1
            return
        averaged = self._capture.add(raw)
        if averaged is None:
            return
        self._capture = None
        controls = self._controls()
        if self._capture_stage == "first":
            requested = recommend_gains(
                averaged, controls, black_level=self._first_black_level
            )
            controls.apply(requested)
            self._first_gains = controls.read()
            self._validate_auxiliary_gain(self._first_auxiliary_gain)
            self.calibration_step.emit(
                "Gains applied and read back. Keep the target in place and capture "
                "the second white reference."
            )
        else:
            calibration = finish_calibration(
                averaged, controls, black_level=self._first_black_level
            )
            calibration.validate_hardware_gains(self._first_gains)
            if self._first_auxiliary_gain is not None:
                name, value = self._first_auxiliary_gain
                self._validate_auxiliary_gain(self._first_auxiliary_gain)
                calibration = replace(
                    calibration,
                    metadata={
                        **calibration.metadata,
                        "auxiliary_gain_selector": name,
                        "auxiliary_gain_value": str(value),
                    },
                )
            processor = make_processor(calibration, controls.read())
            CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            calibration.save(CALIBRATION_FILE)
            self._calibration = calibration
            self._processor = processor
            self._ndvi_bounds = self._cvi_bounds = None
            self.calibration_changed.emit(True, "Calibrated")
            self.calibration_step.emit(f"Calibration saved to {CALIBRATION_FILE}")
        self._capture_stage = None

    def _load_calibration(self):
        self._processor = self._default_processor
        if not CALIBRATION_FILE.exists():
            self.calibration_changed.emit(False, "Uncalibrated preview")
            return
        calibration = Calibration.load(CALIBRATION_FILE)
        if calibration.metadata.get("green_mode") != "mean":
            raise ValueError("Saved calibration uses the old green-site mode; recalibrate")
        controls = self._controls()
        auxiliary_selector = calibration.metadata.get("auxiliary_gain_selector")
        auxiliary_value = calibration.metadata.get("auxiliary_gain_value")
        if (auxiliary_selector is None) != (auxiliary_value is None):
            raise ValueError("Calibration has incomplete auxiliary gain metadata")
        if auxiliary_selector is not None:
            controls.apply_auxiliary(auxiliary_selector, float(auxiliary_value))
        if calibration.hardware_gains is not None:
            controls.apply(calibration.hardware_gains)
        if calibration.offsets:
            black = next(iter(calibration.offsets.values()))
            if any(
                not np.isclose(value, black, rtol=0, atol=1e-6)
                for value in calibration.offsets.values()
            ):
                raise ValueError("Calibration offsets disagree on camera BlackLevel")
            self._remote.FindNode("BlackLevel").SetValue(float(black))
            self._set_default_black_level(
                float(self._remote.FindNode("BlackLevel").Value())
            )
        self._calibration = calibration
        self._validate_calibration()
        self._processor = make_processor(calibration, controls.read())
        self._ndvi_bounds = self._cvi_bounds = None
        self.calibration_changed.emit(True, "Calibrated")

    def _validate_calibration(self):
        self._calibration.validate_hardware_gains(self._controls().read())
        selector = self._calibration.metadata.get("auxiliary_gain_selector")
        if selector is not None:
            self._validate_auxiliary_gain(
                (selector, float(self._calibration.metadata["auxiliary_gain_value"]))
            )
        black = float(self._remote.FindNode("BlackLevel").Value())
        if any(
            not np.isclose(black, offset, rtol=0, atol=1e-6)
            for offset in self._calibration.offsets.values()
        ):
            raise ValueError("BlackLevel changed; recalibrate")

    def _validate_auxiliary_gain(self, expected):
        if expected is None:
            return
        selector, value = expected
        actual = self._controls().read_auxiliary(selector)
        if not np.isclose(actual, value, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"Auxiliary gain mismatch for {selector}: expected {value}, "
                f"camera reports {actual}; recalibrate"
            )

    def _set_default_black_level(self, value):
        if self._black_level != value:
            self._black_level = value
            if self._processor is self._default_processor:
                self._ndvi_bounds = self._cvi_bounds = None

    def _camera_opened(self):
        self._set_default_black_level(float(self._remote.FindNode("BlackLevel").Value()))
        try:
            self._load_calibration()
        except (ValueError, KeyError, RuntimeError, OSError, ids_peak.Exception) as exc:
            self._processor = self._default_processor
            self.calibration_changed.emit(False, f"Calibration unavailable: {exc}")
        try:
            self.gain_options_changed.emit(self._controls().auxiliary_options())
        except (ValueError, RuntimeError, ids_peak.Exception) as exc:
            self.gain_options_changed.emit([])
            self.calibration_changed.emit(False, f"Gain controls unavailable: {exc}")
        self.black_level_changed.emit(float(self._remote.FindNode("BlackLevel").Value()))

    def _frame_received(self, raw):
        try:
            self._advance_calibration(raw)
        except (ValueError, RuntimeError, OSError, ids_peak.Exception) as exc:
            self._capture = None
            self._capture_stage = None
            self.calibration_step.emit(f"Calibration failed: {exc}")

    def _check_processor(self):
        if self._processor is self._default_processor:
            return
        self._gain_check_frame += 1
        if self._gain_check_frame % 10 == 0:
            try:
                self._validate_calibration()
            except ValueError as exc:
                self._processor = self._default_processor
                self._ndvi_bounds = self._cvi_bounds = None
                self.calibration_changed.emit(False, str(exc))

    def _camera_closed(self):
        self._processor = self._default_processor
        self._capture = None
        self._capture_stage = None
        self._first_gains = None
        self._first_auxiliary_gain = None
        self._gain_controls = None
        self.gain_options_changed.emit([])
        self.calibration_changed.emit(False, "Uncalibrated preview")
