# TIFF→LAS Digitizer

Convert well log images (TIFF/PNG/JPG) into QuickSyn-style LAS 1.2 files with AI-assisted defaults that you can always override.

---

## Overview

This app lets you:

- **Upload** a log image (TIFF/PNG/JPG)
- **Define depth** using image pixel positions and real depth values
- **Configure curves** (GR, RHOB, NPHI, DT, etc.) per track
- **Digitize** curves into a LAS 1.2 file, sampled at 0.5 ft when depth unit is feet
- **Review AI help**:
  - Vision OCR (Google Cloud Vision) suggests depth scale and curve labels
  - Color + header text hints for track types and modes
  - Sanity warnings for depth setup and curve ranges

AI is only a helper: you stay in control of every field.

---

## Quick start (local)

### Requirements

- Python 3.9+
- (Optional) Google Cloud Vision credentials for better OCR

### Install

```bash
pip install -r requirements.txt
python web_app.py
```

Then open:

- [http://localhost:5000](http://localhost:5000)

---

## Deployment (Railway)

This repo ships with a `railway.json` that starts the app with Gunicorn.

High-level Railway steps:

1. Create a new Railway project and link this GitHub repo.
2. Let Railway auto-detect the Python app, or keep the existing `railway.json`:
   - Build: Nixpacks
   - Start command:

     ```bash
     gunicorn web_app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2
     ```
3. Set environment variables in Railway:
   - `APP_VERSION` (e.g. `v0.5.0`)
   - Optional Google Vision variables (see below).
4. Deploy. The footer at the bottom of the page shows `Version` and `Build` so you can confirm the running version.

---

## Google Vision OCR setup (optional but recommended)

When configured, Google Cloud Vision is used to read text from the image:

- Depth scale numbers on the left
- Curve header labels (e.g. `GAMMA RAY`, `RHOB`, `SONIC DT`)

The backend then uses these to suggest:

- Depth configuration (top/bottom pixels and depths)
- Curve track positions and type/mnemonic hints

### 1. Create a service account and key

1. In Google Cloud Console, enable the **Vision API** for your project.
2. Create a service account with Vision access.
3. Create a JSON key file for that service account.

### 2. Provide credentials to the app

You have two options:

- **Local:** set `GOOGLE_APPLICATION_CREDENTIALS` to point to your JSON key file.
- **Railway:** place the JSON contents in an environment variable and let the app read it.

See `GOOGLE_VISION_SETUP.md` in this repo for detailed, step-by-step instructions.

If Vision is not configured, the app still works, but with fewer AI suggestions.

---

## How to use the web app

### 1. Upload image (Step 1)

- Drag & drop a TIFF/PNG/JPG or click to browse.
- The app enforces a 100 MB max size.
- On success, you see:
  - A preview image in **Step 2**
  - Basic image dimensions

### 2. Preview & mouse pixel readout (Step 2)

When the image loads:

- **Image dimensions** show width × height.
- A line under the image shows live mouse pixel coordinates:
  - `Mouse pixel: X=…, Y=…`
  - If you are in depth anchor mode it adds `• anchoring TOP` or `• anchoring BOTTOM`.

This lets you see which pixel row you are about to click for depth setup.

### 3. Depth configuration (Step 3)

Depth fields:

- `Top pixel (Y)` / `Bottom pixel (Y)`
- `Top depth value` / `Bottom depth value`
- `Depth unit` (Feet or Meters)

There are two key buttons:

- **Set top from image**
- **Set bottom from image**

Workflow:

1. Click **Set top from image**.
2. Move the mouse over the preview and click on the **depth scale** at the top of your desired interval.
   - The app converts your click to an image Y pixel.
   - If OCR depth labels are available, it **snaps** to the nearest depth label (within a tolerance) and:
     - Fills `Top pixel (Y)` with the snapped pixel.
     - If `Top depth value` is empty, fills it with the snapped depth.
3. Click **Set bottom from image** and repeat for the bottom depth.

You can always edit these numbers manually after anchoring.

> **QuickSyn note:** For QuickSyn workflows, keep `Depth unit = Feet (FT)`. The app will resample the LAS to 0.5 ft depth increments when the unit is feet.

### 4. Depth overlays and session memory

After depth is configured, the app draws **horizontal depth reference lines** on the image:

- Based on your current depth config
- At "nice" intervals (e.g. every 100 ft)
- Labels on the left of the image show the depth value for each reference line

This helps visually confirm the mapping (e.g. that ~1900 ft is near the bottom).

The app also remembers your **last depth configuration** in the browser session:

- Stores top/bottom pixels, depth values, and unit.
- On the next upload (same session) it reuses that setup automatically.

### 5. Curve configuration (Step 3)

- Choose **number of curves** (up to 6) and click **Generate Curve Fields**.
- For each curve you configure:
  - `Curve type` (GR, RHOB, NPHI, DT, CALI, SP, OTHER)
  - `Display name` and `Display unit` (UI labels only)
  - `Left pixel (X)` / `Right pixel (X)`
  - `Left scale value` / `Right scale value`
  - `Curve color / mode`:
    - `Black` for black/dark traces
    - `Red` for red traces

The app uses:

- Auto-detected track boundaries
- OCR curve header labels (e.g. `SONIC DT`)
- Curve strip color (red vs black)

to prefill:

- Pixel ranges
- Suggested `Curve type`
- Suggested LAS mnemonic & unit
- Recommended color mode

You can override any field.

### 6. Digitize & download LAS

Click **Digitize & Download LAS**:

1. The app:
   - Slices the image between your top/bottom pixels.
   - For each curve:
     - Detects trace positions row-by-row (respecting color mode).
     - Maps X to value using the left/right scale values.
   - Builds a depth vector from your depth config.
   - If unit is feet, resamples to 0.5 ft increments for compatibility.
   - Writes LAS 1.2 content.

2. LAS validation:
   - If `lasio` is available, the app attempts to parse the LAS and reports success or error.
   - If `lasio` is missing, it notes that validation is skipped.

3. Sanity warnings:
   - **Depth warnings** (e.g. bottom pixel not below top, extreme depth-per-pixel values, pixels outside image bounds).
   - **Curve warnings** per GR/RHOB/DT curve:
     - Values outside expected ranges.
     - Very flat curves.
     - High percentage of null values.

The final status message shows:

- Validation result
- Any depth warnings
- Any curve warnings

### 7. AI insights (Step 4)

The **AI Insights** panel summarizes what the Vision OCR and heuristics found:

- Depth hints from OCR (if available)
- Curve track hints (pixels and sample values)
- OCR header text and how it maps to types/mnemonics
- Color-based mode hints per track

This is a "glass box" view so you can see *why* the AI suggested certain defaults.

---

## Versioning

The footer shows:

- `Version: <APP_VERSION> • Build: <timestamp>`

Set `APP_VERSION` in your environment (e.g. `v0.5.0`) to track deployed builds.

---

## Notes and limitations

- Designed primarily for **log-style images** with a clear vertical depth scale.
- Works without Google Vision, but with fewer hints and more manual setup.
- Current LAS sampling is fixed at 0.5 ft when depth unit is Feet; meter support does not resample yet.
- Only up to 6 curves per run to keep the UI manageable.

---

## Contributing / issues

If you encounter problems or have feature ideas (new curve types, better OCR heuristics, etc.):

- Open an issue on GitHub: [https://github.com/Ch3fC0d3/tifflas/issues](https://github.com/Ch3fC0d3/tifflas/issues)
- Include a short description and, if possible, a redacted sample image and LAS output.
