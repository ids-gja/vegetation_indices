# IDS GreenView Pro

Desktop application for live RAW, NDVI, and CVI visualization with IDS cameras.

## Project layout

```text
src/ndvi_processing/           Reusable Bayer extraction and NDVI/CVI/TVI computation
  resources/                   Sensor and filter spectra
src/greenview_pro/              Live camera worker and Qt demo (no calibration by default)
  resources/                   Demo icons, images, messages, and config
src/calibration/                Optional white-target calibration math, worker, and UI
  resources/                   Calibration settings and gear icon
archive/                       Retained historical application versions
snapshots/                     Generated image captures in the working directory (not tracked)
```

## Setup and run

Install the IDS peak SDK and its Python bindings (`ids_peak`) first. Install
the application; the reusable `ndvi_processing` package is included locally
and does not require a sibling checkout:

```powershell
python -m pip install -e .
greenview-pro
```

Alternatively, run directly from a checkout after installing dependencies:

```powershell
$env:PYTHONPATH = "src"
python -m greenview_pro
```

The demo uses the camera's hardware-balanced R/G/NIR Bayer channels as-is,
without loading saved calibration or applying software white balance. To use
white-target calibration on lighting setups without adjustable channel voltages,
start `greenview-pro --calibration` (or `python -m greenview_pro --calibration`).
Only this mode loads saved calibration, exposes the gear dialog, and applies
calibrated gains. Contrast controls affect display colors, not the underlying
index values.

## Live preview performance

The live processing path is capped at the active square image area's on-screen
dimensions before calculating vegetation indices. It preserves the source aspect ratio,
so the complete camera frame remains visible without distortion.
The app starts maximized, with exposure, contrast, snapshots, and fullscreen
demo controls together in the top panel. Calibration controls appear only
with `--calibration`.

Before starting acquisition, the camera sets `DeviceLinkThroughputLimit` to its
maximum, temporarily minimizes exposure, and sets `AcquisitionFrameRate` to the
highest rate allowed by `DeviceLinkAcquisitionFrameRateLimit` and the camera.
It then sets exposure close to the frame period (`1,000,000 / FPS` microseconds).
If the camera reports a lower attainable FPS at that exposure, the limit is
reduced by the reported timing shortfall and the maximum FPS is reapplied.
The exposure slider cannot exceed this FPS-preserving limit, but remains
adjustable below it.

## Configuration

`src/greenview_pro/resources/config.toml` controls message timing, view rotation,
view spacing, and styling.
`VIEW_SPACING` defaults to 1 px and `VIEW_ROTATION_SECONDS` defaults to 5 seconds.
Add a TOML file under `src/greenview_pro/resources/messages` to add a marketing message; each file defines a
title, subtitle, and three `[[facts]]`.
In fullscreen, a ring beside the message tracks its display time, while a smooth
bar along the main image tracks time until the next image rotates in. The main
image and three stacked side images share an overall height and scale to the
uncropped image aspect ratio, fitting the complete group within the window.

## Reusable vegetation calculations

`src/ndvi_processing` supplies Bayer extraction and NDVI/CVI/TVI calculations.
It can be imported independently by student projects and accepts raw GRBG
frames without software calibration. White-target calibration math and
hardware/UI integration live entirely in `src/calibration`; only the opt-in
mode constructs a processor with a calibration.

```python
from importlib.resources import files
from ndvi_processing import NDVIProcessor, SensorConfig

qe = files("ndvi_processing").joinpath("resources", "sensor_AR2020.csv")
processor = NDVIProcessor(SensorConfig("AR2020", "GRBG", qe))
indices = processor.process_raw(raw_bayer_frame).indices
ndvi, cvi, tvi = (indices[name] for name in ("ndvi", "cvi", "tvi"))
```

## White-target calibration

Run with `--calibration` and open the gear icon while a camera is connected.
Calibration defaults live in `src/calibration/resources/config.toml`. Set the exposure, camera
`BlackLevel`, and frames per capture (default: 8). The dialog also lists the
camera's non-R/G/B `GainSelector` entries (such as global or analog gain) with
their supported ranges and current values. Both continuous (`NoIncrement`) and
fixed-increment Gain controls are supported. Optionally select one and set its
value in camera units to increase brightness when exposure alone is insufficient;
avoid clipping the white target. Fill the full camera frame with an evenly
illuminated, unsaturated neutral target. Capture the first white
reference: the app averages the raw Bayer frames, recommends per-channel linear
hardware gains using the camera's `GainSelector`/`Gain` nodes, applies them and
reads back the actual values. Keep exposure, illumination, BlackLevel, and target
fixed, then capture the second white reference. After buffered frames are
discarded, the app averages fresh raw frames, computes residual software gains,
and stores the calibration in `~\.greenview_pro\calibration.json`.

The AR2020 sensor's Bayer pattern is GRBG. Its physical blue site is an NIR
*proxy*, not a spectrally isolated NIR band. The reusable processor loads the
bundled AR2020 relative-QE file; its values are validated but not used to compute
white-target gains. On restart the app restores saved hardware gains and checks
their readbacks, the selected global/analog gain, and BlackLevel. Channel
`GainSelector` names are discovered from the camera's available entries by
matching red, green, and blue; missing or ambiguous matches cause a clear error
rather than silently using a different gain. If the camera lacks per-channel
linear Gain nodes or saved settings are incompatible, the status explains why the preview is
uncalibrated. Recalibrate after changing optical setup or hardware gains.
Exposure remains adjustable after calibration as requested; absolute TVI values
then vary with exposure, while NDVI/CVI are ratios. Contrast percentiles remain
adjustable independently of calibration.
