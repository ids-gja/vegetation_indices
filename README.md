# IDS GreenView Pro

Desktop application for live RAW, NDVI, and CVI visualization with IDS cameras.

## Project layout

```text
src/greenview_pro/             Application package
  app.py                       Qt application entrypoint
  processing.py                Camera and image-processing pipeline
  vegetation_indices.py        Reusable Bayer calibration and index calculations
src/icons/                     Bundled branding assets
src/images/                    Bundled demonstration images
src/config.toml                Runtime UI and marketing settings
src/messages/                  One TOML file per marketing message
archive/                       Retained historical application versions
snapshots/                     Generated image captures (not tracked)
```

## Setup and run

Install the IDS peak SDK and its Python bindings (`ids_peak`) first, then install the
application and its Python dependencies:

```powershell
python -m pip install -e .
greenview-pro
```

Alternatively, run directly from a checkout:

```powershell
$env:PYTHONPATH = "src"
python -m greenview_pro
```

## Live preview performance

The live processing path is capped at the active square image area's on-screen
dimensions before calculating vegetation indices. It preserves the source aspect ratio,
so the complete camera frame remains visible without distortion.

## Configuration

`src/config.toml` controls message timing, view rotation, view spacing, and styling.
`VIEW_SPACING` defaults to 1 px and `VIEW_ROTATION_SECONDS` defaults to 5 seconds.
Add a TOML file under `src/messages` to add a marketing message; each file defines a
title, subtitle, and three `[[facts]]`.

## Reusable vegetation calculations

`src/greenview_pro/vegetation_indices.py` has no IDS, Qt, OpenCV, or UI dependency.
Pass a Bayer image and `CameraImageParameters` to `calculate_vegetation_indices()` to
obtain calibrated green, red, and NIR channels plus NDVI, CVI, and TVI. The parameter
object holds Bayer site metadata, pixel format, black/white levels, gains, the color
correction matrix, and denominator epsilon, so camera metadata can be supplied when
available.
