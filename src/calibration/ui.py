"""Opt-in calibration controls for the GreenView Pro display."""

import tomllib
from pathlib import Path

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from calibration.worker import CalibratedCameraWorker
from greenview_pro.app import ICON_DIR, MainWindow

ICON_FILE = ICON_DIR / "gear.svg"
CALIBRATION_CONFIG = Path(__file__).resolve().parent / "config.toml"


class CalibrationWindow(MainWindow):
    worker_class = CalibratedCameraWorker
    calibration_requested = Signal(str, int, float, str, float)

    def __init__(self):
        self._calibrated = False
        self._gain_options = []
        with CALIBRATION_CONFIG.open("rb") as source:
            self.calibration_settings = tomllib.load(source)["settings"]
        super().__init__()
        self.calibration_requested.connect(self.worker.request_calibration)
        self.worker.calibration_changed.connect(self.update_calibration)
        self.worker.calibration_step.connect(self.show_calibration_step)
        self.worker.black_level_changed.connect(self.update_black_level)
        self.worker.gain_options_changed.connect(self.update_gain_options)

    def _build_main(self):
        super()._build_main()
        self.calibration_status = QLabel("Uncalibrated preview")
        self.calibration_status.setWordWrap(True)
        self.status_layout.addWidget(self.calibration_status)
        self.calibration_button = self.settings_button

    def _add_mode_settings(self, layout):
        button = QPushButton("White-target calibration")
        button.clicked.connect(self.open_calibration)
        layout.addWidget(button)

    def open_calibration(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("White-target calibration")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(
            "Fill the view with a uniform, unsaturated white target. "
            "Keep the target and lighting fixed for both captures."
        ))
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Frames per capture"))
        frames = QSpinBox()
        frames.setRange(1, 128)
        frames.setValue(int(self.calibration_settings["CALIBRATION_NUM_FRAMES"]))
        count_row.addWidget(frames)
        layout.addLayout(count_row)
        exposure_row = QHBoxLayout()
        exposure_row.addWidget(QLabel("Exposure (us)"))
        exposure = QSpinBox()
        exposure.setRange(self.exposure.minimum(), max(self.exposure.maximum(), 1))
        exposure.setValue(self.exposure.value())
        exposure.valueChanged.connect(self.exposure.setValue)
        exposure_row.addWidget(exposure)
        layout.addLayout(exposure_row)
        black_row = QHBoxLayout()
        black_row.addWidget(QLabel("BlackLevel (camera DN)"))
        black = QDoubleSpinBox()
        black.setRange(0, 65535)
        black.setDecimals(3)
        black.setValue(
            getattr(self, "_camera_black_level", 2.0)
            if self._calibrated
            else float(self.calibration_settings["CALIBRATION_BLACK_LEVEL"])
        )
        black_row.addWidget(black)
        layout.addLayout(black_row)
        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Analog/global gain"))
        selector = QComboBox()
        selector.addItem("Leave unchanged", "")
        for name, *_ in self._gain_options:
            selector.addItem(name, name)
        if self._gain_options:
            selector.setCurrentIndex(1)
        gain_row.addWidget(selector)
        gain = QDoubleSpinBox()
        gain.setDecimals(6)
        gain_row.addWidget(gain)
        layout.addLayout(gain_row)

        def configure_gain():
            selected = next(
                (option for option in self._gain_options if option[0] == selector.currentData()),
                None,
            )
            gain.setEnabled(selected is not None)
            if selected is not None:
                _, minimum, maximum, increment, current = selected
                gain.setRange(minimum, maximum)
                gain.setSingleStep(
                    increment if increment is not None and increment > 0 else 0.01
                )
                gain.setValue(current)

        selector.currentIndexChanged.connect(configure_gain)
        configure_gain()
        first = QPushButton("1. Set gains from white target")
        second = QPushButton("2. Capture fresh white target and save")
        second.setEnabled(False)
        progress = QLabel("Set exposure and BlackLevel, then capture the white target.")
        progress.setWordWrap(True)
        layout.addWidget(first)
        layout.addWidget(second)
        layout.addWidget(progress)

        def request(stage):
            first.setEnabled(False)
            second.setEnabled(False)
            frames.setEnabled(False)
            black.setEnabled(False)
            exposure.setEnabled(False)
            selector.setEnabled(False)
            gain.setEnabled(False)
            self.calibration_requested.emit(
                stage, frames.value(), black.value(), selector.currentData(), gain.value()
            )

        def report(message):
            progress.setText(message)
            if message.startswith("Gains applied"):
                second.setEnabled(True)
            elif message.startswith(
                ("Calibration failed", "Connect", "Capture the first")
            ):
                first.setEnabled(True)
                second.setEnabled(False)
            elif message.startswith("Calibration saved"):
                first.setEnabled(True)
                second.setEnabled(False)
            else:
                return
            frames.setEnabled(True)
            if not message.startswith("Gains applied"):
                black.setEnabled(True)
                exposure.setEnabled(True)
                selector.setEnabled(True)
                configure_gain()

        first.clicked.connect(lambda: request("first"))
        second.clicked.connect(lambda: request("second"))
        self.worker.calibration_step.connect(report)
        try:
            dialog.exec()
        finally:
            self.worker.calibration_step.disconnect(report)

    @Slot(bool, str)
    def update_calibration(self, calibrated, status):
        self._calibrated = calibrated
        self.calibration_status.setText(status)
        suffix = "" if calibrated else " (UNCALIBRATED)"
        for card, name in ((self.main_cards[2], "NDVI"), (self.main_cards[3], "CVI")):
            card.overlay.setText(name + suffix)
        if self.demo_mode:
            self._refresh_demo_views()

    @Slot(str)
    def show_calibration_step(self, message):
        if message.startswith("Calibration failed") or message.startswith("Connect"):
            self.calibration_status.setText(message)

    @Slot(float)
    def update_black_level(self, value):
        self._camera_black_level = value

    @Slot(object)
    def update_gain_options(self, options):
        self._gain_options = options

    def view_title(self, name):
        if not self._calibrated and name in ("NDVI", "CVI"):
            return name + " (UNCALIBRATED)"
        return name
