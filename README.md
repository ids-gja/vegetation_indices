# IDS GreenView Pro

Desktop application for live RAW, NDVI, CVI, and TVI visualization with IDS cameras.

## Project layout

```text
src/greenview_pro/             Application package
  assets/icons/                Bundled branding assets
  assets/images/               Bundled demonstration images
  config/                      Runtime marketing configuration
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

The live processing path is capped at 1280x720 pixels (`PREVIEW_MAX_PIXELS`) before
calculating vegetation indices. This keeps the preview responsive while the camera
continues acquiring at its configured full resolution.
