# IDS GreenView Pro

Desktop application for live RAW, NDVI, and TVI visualization with IDS cameras.

## Project layout

```text
src/ndvi_processing/           Reusable Bayer extraction and NDVI/TVI computation
  resources/                   Sensor and filter spectra
src/greenview_pro/              Qt-free camera acquisition and frame renderer; Qt demo adapter
  resources/                   Demo icons, images, messages, and config
src/calibration/                Optional white-target calibration math, worker, and UI
  config.toml                   Calibration settings
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

The demo uses the camera's hardware-balanced R/G/NIR Bayer channels without
loading saved calibration or applying software white balance. It sets the
camera BlackLevel to 0 DN, sets the camera's Red, Green, and Blue (NIR-site)
gains to 1.0, and verifies all four readbacks before acquisition. The live
worker subtracts the configured BlackLevel from RAW Bayer values before
passing them to the reusable index processor. To use
white-target calibration on lighting setups without adjustable channel voltages,
start `greenview-pro --calibration` (or `python -m greenview_pro --calibration`).
Only this mode loads saved calibration and applies calibrated gains. The gear
menu is available in both modes; calibration mode also provides the
white-target calibration action. Exposure and contrast remain in the top bar:
separate NDVI and TVI contrast bars are stacked vertically. NDVI uses absolute
index scores from -1.00 to +1.00 in 0.01 steps (defaulting to the full range);
TVI uses image percentiles from 0 to 100 (defaulting to 30% and 80%).
Each bar has two grips; dragging the selected area moves both bounds together.
A gap of at least 0.02 for NDVI or two percentage points for TVI is preserved.
Camera and calibration status messages occupy a separate row so they cannot
shift the controls. Contrast affects display colors, not the underlying index
values. The save icon writes the current filter settings, both contrast ranges,
and exposure to the packaged app `config.toml`; the outlined save icon writes
RAW/NDVI/TVI image snapshots. Exposure is adjustable from 10 to 150 ms by
default (subject to camera limits). A saved exposure is restored on the next
launch; without one, preview starts at the camera's Cockpit exposure.

## Live preview performance

The normal view packs four camera-aspect image cards with a 1 px gap. Each
live card requests its own display-sized image. Fullscreen renders both indices
at the large card's size and scales their side-card copies down; it waits for a
large-sized frame rather than enlarging an older small image after a view change.
Each 2x2 Bayer cell becomes a float R/G/NIR
superpixel, averaging both green sites. Calibration and vegetation indices
are calculated once at native superpixel resolution, with no median filter.
For the live view, an exponential temporal average (25% new frame) smooths
each R/G/NIR superpixel across frames before calculating indices. It preserves
spatial resolution for a still scene but takes several frames to respond to
scene changes. The average is reset on exposure, camera, calibration, sensor
size, and black-level changes; RAW previews and direct `indices()` calls stay
unfiltered. The gear menu in both modes can disable the filter or adjust its
1-100% new-frame weight (100% means no temporal smoothing). Standalone renderers
can call `reset_temporal()` when acquisition settings change.
Only the finished index images are resized for each card. The full-resolution
RAW frame is retained until its final display scaling and for calibration
and camera acquisition. This prioritizes detail over frame rate and memory. Both
views preserve the camera frame's aspect ratio.
The app starts maximized, with exposure, contrast, snapshots, and fullscreen
demo controls together in the top panel. Calibration controls appear only
with `--calibration`.
RAW cards use display-only gamma 2.2 to brighten the preview; snapshots retain
the original, ungamma-corrected 8-bit RAW preview. Snapshots include RAW, NDVI,
and TVI, but not the static reference image.

Preview starts with the frame rate and throughput configured in IDS peak Cockpit,
lowering FPS only when the chosen exposure needs it; there is no automatic
exposure search on startup. A saved `EXPOSURE_US` value
overrides Cockpit exposure, but no saved value leaves the camera exposure
untouched when it lies within the configured range. On entering fullscreen,
acquisition pauses briefly while the app maximizes throughput and frame rate
within the current exposure's limits, then resumes without changing exposure.
Leaving fullscreen keeps the new frame rate. Reconnecting restores the chosen
exposure and, after fullscreen was entered, its frame-rate policy. If a longer
manually selected exposure needs a slower frame rate, preview lowers it just
enough to allow that exposure.

## Configuration

`src/greenview_pro/resources/config.toml` controls message timing, view rotation,
view spacing, and styling.
`VIEW_SPACING` defaults to 1 px and `DISPLAY_TIME` controls the synchronized
fullscreen message and image cycle. `SHOW_PROGRESS_CIRCLE` defaults to `false`;
set it to `true` to display the fullscreen message-cycle ring.
`TEMPORAL_ENABLED` and `TEMPORAL_WEIGHT` default to `true` and `0.25`.
The save icon writes these and current NDVI/TVI contrast bounds and exposure
to this packaged file; a read-only installation will report a save error.
`EXPOSURE_MIN_US` and `EXPOSURE_MAX_US` set the slider limits (defaulting to
10,000 and 150,000 microseconds); the usable range is intersected with camera
capabilities. `EXPOSURE_US` is optional and takes precedence over the exposure
configured in Cockpit.
TVI percentile settings use `TVI_LOW_PERCENTILE` and `TVI_HIGH_PERCENTILE`.
Existing `CVI_*` settings are read on startup and removed on the next save.
Add a TOML file under `src/greenview_pro/resources/messages` to add a marketing message; each file defines a
title, subtitle, and three `[[facts]]`.
When enabled, the ring in the top-right tracks the shared cycle. The
large image alternates NDVI and TVI when the message swaps; the side cards show
the reference image, RAW, NDVI, and TVI in a 2x2 grid, including the selected
index. The main image and the right-hand grid share an overall height and
scale to the original camera frame aspect ratio, fitting the complete group
within the window without resizing as Bayer previews round to even pixel
dimensions.

## Reusable vegetation calculations

`src/ndvi_processing` supplies Bayer extraction and NDVI/TVI calculations.
The TVI result uses the `tvi` dictionary key and the third render output.
It can be imported independently by student projects and accepts raw GRBG
frames without software calibration. The default index and white-target
calibration paths both average the two green sites; saved calibrations made
with the previous single-green-site method require a new white-target capture.
The reusable `bayer_superpixels(raw, "GRBG")` and
`resize_superpixels(triplets, (width, height))` helpers expose reusable float
scaling; the live renderer instead scales finished index images. White-target
calibration math and hardware/UI integration live entirely in
`src/calibration`; only the opt-in mode constructs a processor with a calibration.

Live acquisition and display rendering can also be used without starting Qt:

```python
from greenview_pro.camera import CameraSession
from greenview_pro.image_processing import FrameProcessor

frames = FrameProcessor()
with CameraSession() as camera:
    raw = camera.read()  # Independent, full-resolution NumPy Bayer frame
    indices = frames.indices(raw)  # Native-resolution, unfiltered values
    raw_image, ndvi_image, tvi_image = frames.render(raw, (1280, 960))
```

`CameraSession` requires the IDS peak SDK and its Python bindings, but neither
module imports Qt. To process saved RAW Bayer NumPy arrays, create a
`FrameProcessor` without constructing a camera. The standalone `NDVIProcessor`
below is the sensor-configurable core.

```python
from importlib.resources import files
from ndvi_processing import NDVIProcessor, SensorConfig

qe = files("ndvi_processing").joinpath("resources", "sensor_AR2020.csv")
processor = NDVIProcessor(SensorConfig("AR2020", "GRBG", qe, black_level=0))
indices = processor.process_raw(raw_bayer_frame).indices
ndvi, tvi = (indices[name] for name in ("ndvi", "tvi"))
```

Set `black_level` to the camera's actual DN setting (the standalone processor
defaults to 0); the live worker already subtracts its camera's offset and
passes `black_level=0` to the core. Calibrated processing uses its measured
channel offsets instead.

## White-target calibration

Run with `--calibration`, open the gear menu, and select white-target calibration
while a camera is connected.
Calibration defaults live in `src/calibration/config.toml`. Set the exposure, camera
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
and stores the calibration in `~\.greenview_pro\calibration.json`. On reconnect
the optional mode restores the calibration's recorded camera BlackLevel.

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
Exposure remains adjustable after calibration; NDVI is a ratio and TVI is a
linear combination of the NIR, red, and green channels.
Their display contrast remains adjustable independently of calibration: NDVI
uses fixed score bounds and TVI uses image percentiles.
