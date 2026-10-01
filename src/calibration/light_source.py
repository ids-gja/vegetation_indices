"""Standalone, operator-guided light calibration.

Run before starting the app: python -m calibration.light_source --cfa GRBG
Requires an IDS camera and an interactive Matplotlib backend. The operator
controls the three LED currents; this script never controls the light source.
"""

import argparse
import math

import numpy as np


def validate_cfa(pattern):
    pattern = "GRBG" if pattern is None else pattern.upper()
    if len(pattern) != 4 or sorted(pattern) != ["B", "G", "G", "R"]:
        raise ValueError("CFA must contain exactly one R, one B (IR), and two G sites")
    return pattern


def aligned_roi(start, end, shape):
    """Return an even x, y, width, height inside the dragged sensor area."""
    height, width = shape
    if not all(math.isfinite(value) for value in (*start, *end)):
        raise ValueError("ROI coordinates must be finite")
    x0 = max(0, 2 * math.ceil(min(start[0], end[0]) / 2))
    y0 = max(0, 2 * math.ceil(min(start[1], end[1]) / 2))
    x1 = min(width - width % 2, 2 * math.floor(max(start[0], end[0]) / 2))
    y1 = min(height - height % 2, 2 * math.floor(max(start[1], end[1]) / 2))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("ROI must contain at least one aligned 2x2 Bayer cell")
    return x0, y0, x1 - x0, y1 - y0


def roi_pixels(frame, roi):
    """Return only the aligned RAW pixels inside the selected sensor ROI."""
    image = np.asarray(frame)
    if image.ndim != 2:
        raise ValueError("Expected a 2D RAW Bayer frame")
    x, y, width, height = roi
    if (
            min(x, y) < 0
            or min(width, height) < 2
            or any(value % 2 for value in roi)
            or x + width > image.shape[1]
            or y + height > image.shape[0]
    ):
        raise ValueError("ROI must be even-aligned and inside the RAW frame")
    return image[y:y + height, x:x + width]


def channel_mean(frame, roi, cfa, channel):
    """Average only this channel's Bayer sub-grids inside the selected ROI."""
    cfa = validate_cfa(cfa)
    if channel not in ("R", "G", "B"):
        raise ValueError(f"Unknown CFA channel: {channel}")
    region = roi_pixels(frame, roi)
    site_means = [
        np.mean(region[index // 2::2, index % 2::2], dtype=np.float64)
        for index, site in enumerate(cfa) if site == channel
    ]
    return float(np.mean(site_means))


class CalibrationGuide:
    """Capture each LED-only measurement after operator confirmation."""

    def __init__(self, camera, cfa="GRBG", settle_frames=10):
        self.camera = camera
        self.cfa = validate_cfa(cfa)
        if settle_frames < 0:
            raise ValueError("settle_frames cannot be negative")
        self.settle_frames = settle_frames
        self.stage = "roi"
        self.frame = None
        self.roi = None
        self.reference = None
        self.measured = None
        self.red = None
        self.ir = None

    def _fresh_frame(self):
        for _ in range(self.settle_frames):
            self.camera.read()
        frame = np.asarray(self.camera.read())
        if frame.ndim != 2 or min(frame.shape) < 2:
            raise ValueError("Camera must return a 2D RAW Bayer frame")
        if self.frame is not None and frame.shape != self.frame.shape:
            raise ValueError("Camera frame shape changed during calibration")
        return frame

    def capture_preview(self):
        if self.stage != "roi":
            raise ValueError("Preview is only available before selecting the ROI")
        self.frame = self._fresh_frame()
        return self.frame

    def confirm_roi(self, roi):
        if self.stage != "roi":
            raise ValueError("ROI selection is not the current step")
        if self.frame is None:
            raise ValueError("Capture a preview before confirming the ROI")
        if roi is None:
            raise ValueError("Select a slab ROI before confirming it")
        channel_mean(self.frame, roi, self.cfa, "G")
        self.roi = roi
        self.stage = "green"

    def capture_green(self):
        if self.stage != "green":
            raise ValueError("Confirm the ROI before capturing green")
        frame = self._fresh_frame()
        self.reference = channel_mean(frame, self.roi, self.cfa, "G")
        self.frame = frame
        self.stage = "red"
        return frame

    def capture_channel(self):
        if self.stage not in ("red", "ir"):
            raise ValueError("Select and confirm a green ROI before measuring R or IR")
        self.measured = None
        frame = self._fresh_frame()
        channel = "R" if self.stage == "red" else "B"
        mean = channel_mean(frame, self.roi, self.cfa, channel)
        self.frame = frame
        self.measured = mean
        return mean

    def confirm_channel(self):
        if self.measured is None or self.stage not in ("red", "ir"):
            raise ValueError("Capture the current channel before confirming it")
        if self.stage == "red":
            self.red = self.measured
            self.stage = "ir"
        else:
            self.ir = self.measured
            self.stage = "done"
        self.measured = None


def run_guide(camera, cfa):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.widgets import Button, RectangleSelector
    from ids_peak import ids_peak

    guide = CalibrationGuide(camera, cfa)
    guide.capture_preview()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(bottom=0.25, top=0.88)
    image = ax.imshow(guide.frame, cmap="gray", vmin=0, vmax=camera.white_level)
    ax.set_title("Concrete slab (RAW sensor)")
    status = fig.text(0.08, 0.94, "", va="top")
    capture_ax = fig.add_axes((0.14, 0.06, 0.32, 0.09))
    confirm_ax = fig.add_axes((0.54, 0.06, 0.32, 0.09))
    capture = Button(capture_ax, "Capture green")
    confirm = Button(confirm_ax, "Confirm")
    selection = [None]
    outline = [None]

    def instruction(message=None):
        stage = guide.stage
        prompts = {
            "green": "Set G current to maximum; switch R and IR OFF. Press Capture green.",
            "roi":   "Light the concrete slab to see it; drag a ROI and confirm it first. Refresh preview if needed.",
            "red":   "Switch G and IR OFF. Adjust R current, capture repeatedly, then confirm a match.",
            "ir":    "Switch G and R OFF. Adjust IR current, capture repeatedly, then confirm a match.",
            "done":  "Calibration complete. Turn lights off or set the desired operating currents.",
        }
        capture_ax.set_visible(stage in ("roi", "green", "red", "ir"))
        confirm_ax.set_visible(stage == "roi" and selection[0] is not None or
                               stage in ("red", "ir") and guide.measured is not None)
        capture.label.set_text({"roi": "Refresh preview", "green": "Capture green", "red": "Capture red",
                                "ir":  "Capture IR"}.get(stage, "Capture"))
        confirm.label.set_text("Confirm ROI" if stage == "roi" else f"Confirm {stage.upper()}")
        status.set_text(prompts[stage] + (f"\n{message}" if message else ""))
        fig.canvas.draw_idle()

    def show_selected_roi():
        x, y, width, height = guide.roi
        image.set_data(roi_pixels(guide.frame, guide.roi))
        image.set_extent((x - 0.5, x + width - 0.5,
                          y + height - 0.5, y - 0.5))
        ax.set_xlim(x - 0.5, x + width - 0.5)
        ax.set_ylim(y + height - 0.5, y - 0.5)

    def on_select(eclick, erelease):
        if guide.stage != "roi" or None in (eclick.xdata, eclick.ydata,
                                            erelease.xdata, erelease.ydata):
            return
        try:
            roi = aligned_roi((eclick.xdata, eclick.ydata),
                              (erelease.xdata, erelease.ydata), guide.frame.shape)
        except ValueError as exc:
            selection[0] = None
            if outline[0] is not None:
                outline[0].remove()
                outline[0] = None
            instruction(str(exc))
            return
        selection[0] = roi
        if outline[0] is not None:
            outline[0].remove()
        outline[0] = ax.add_patch(Rectangle(roi[:2], roi[2], roi[3],
                                            fill=False, edgecolor="yellow", linewidth=2))
        instruction(f"Aligned ROI (x, y, w, h): {roi}")

    selector = RectangleSelector(ax, on_select, useblit=True, button=[1],
                                 interactive=False, spancoords="data")

    def on_capture(_event):
        try:
            if guide.stage == "roi":
                image.set_data(guide.capture_preview())
                instruction("Preview refreshed. Select and confirm the slab ROI.")
            elif guide.stage == "green":
                guide.capture_green()
                show_selected_roi()
                instruction(f"Green reference: {guide.reference:.2f}; ROI: {guide.roi}.")
            else:
                mean = guide.capture_channel()
                show_selected_roi()
                name = "Red" if guide.stage == "red" else "IR (blue CFA site)"
                instruction(f"{name}: {mean:.2f}  |  Green reference: {guide.reference:.2f}"
                            f"  |  Difference: {mean - guide.reference:+.2f}")
        except (ValueError, RuntimeError, OSError, ids_peak.Exception) as exc:
            instruction(f"Capture failed: {exc}")

    def on_confirm(_event):
        try:
            if guide.stage == "roi":
                guide.confirm_roi(selection[0])
                selector.set_active(False)
                if outline[0] is not None:
                    outline[0].remove()
                    outline[0] = None
                show_selected_roi()
                instruction(f"ROI confirmed: {guide.roi}. Now set G to maximum; R and IR off.")
            else:
                guide.confirm_channel()
                if guide.stage == "done":
                    instruction(f"G: {guide.reference:.2f}  R: {guide.red:.2f}  "
                                f"IR: {guide.ir:.2f}  ROI: {guide.roi}")
                    print(f"Calibration complete: CFA={guide.cfa} ROI={guide.roi} "
                          f"G={guide.reference:.2f} R={guide.red:.2f} IR={guide.ir:.2f}")
                else:
                    instruction(f"Red confirmed: {guide.red:.2f}.")
        except (ValueError, RuntimeError, OSError, ids_peak.Exception) as exc:
            instruction(f"Confirmation failed: {exc}")

    capture.on_clicked(on_capture)
    confirm.on_clicked(on_confirm)
    instruction()
    plt.show()
    if guide.stage != "done":
        raise RuntimeError("Light calibration was closed before completion")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    def cfa_argument(value):
        try:
            return validate_cfa(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    parser.add_argument("--cfa", type=cfa_argument, default="GRBG",
                        help="2x2 CFA order, row-major (default: GRBG); B is the IR site")
    args = parser.parse_args(argv)
    from greenview_pro.camera import CameraSession

    camera = CameraSession()
    try:
        camera.open()
        run_guide(camera, args.cfa)
    finally:
        camera.shutdown()


if __name__ == "__main__":
    main()
