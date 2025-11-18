#!/usr/bin/env python3
"""
web_app.py — Flask web app for TIFF→LAS digitizer with Google Vision API

Setup:
1. pip install flask google-cloud-vision opencv-python numpy pandas
2. Get Google Cloud Vision API key: https://console.cloud.google.com
3. Set environment variable: GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json
4. Run: python web_app.py
5. Open: http://localhost:5000

Free hosting: Deploy to Render.com, Railway.app, or Google Cloud Run
"""
from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import pandas as pd
import json
from io import BytesIO, StringIO
import base64
from typing import Dict, List, Tuple
import tempfile
from datetime import datetime
import requests

# Try to import Google Vision API (optional)
VISION_API_AVAILABLE = False
vision_client = None

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    
    # Check for credentials in environment
    if 'GOOGLE_VISION_CREDENTIALS_JSON' in os.environ:
        # Railway/Cloud deployment: JSON in environment variable
        creds_json = json.loads(os.environ['GOOGLE_VISION_CREDENTIALS_JSON'])
        credentials = service_account.Credentials.from_service_account_info(creds_json)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        VISION_API_AVAILABLE = True
        print("✅ Google Vision API: Loaded from environment variable")
    elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        # Local development: JSON file path
        vision_client = vision.ImageAnnotatorClient()
        VISION_API_AVAILABLE = True
        print("✅ Google Vision API: Loaded from file")
    else:
        print("⚠️  Google Vision API: No credentials found")
except ImportError:
    print("⚠️  Google Vision API not available. Install: pip install google-cloud-vision")
except Exception as e:
    print(f"⚠️  Google Vision API error: {e}")

# Optional LAS validator
LASIO_AVAILABLE = False

try:
    import lasio
    LASIO_AVAILABLE = True
    print("✅ lasio imported; LAS validation enabled.")
except Exception as e:
    print(f"ℹ️  lasio unavailable; LAS validation will be skipped: {e}")

# Default LAS curve label mapping by type (kept in sync with frontend curveTypeDefaults)
CURVE_TYPE_DEFAULTS = {
    "GR":   {"mnemonic": "GR",   "unit": "API"},
    "RHOB": {"mnemonic": "RHOB", "unit": "G/CC"},
    "NPHI": {"mnemonic": "NPHI", "unit": "V/V"},
    "DT":   {"mnemonic": "DTC",  "unit": "US/F"},
    "DTC":  {"mnemonic": "DTC",  "unit": "US/F"},
    "CALI": {"mnemonic": "CALI", "unit": "IN"},
    "SP":   {"mnemonic": "SP",   "unit": "MV"},
}

MISSING_MARKERS = [-999.25, -999.0, -9999.0, 999.25]

CURVE_KEYWORDS = {
    "GR":   ["GR", "GAMMA RAY"],
    "RES":  ["RES", "RESISTIVITY", "ILD", "LLD", "LWD RES"],
    "RHOB": ["RHOB", "DENSITY", "RHO B", "BULK DENSITY"],
    "NPHI": ["NPHI", "NEUTRON POROSITY", "NEUT", "PHI N"],
    "PEF":  ["PEF", "PHOTOELECTRIC", "PE"],
}

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")

APP_VERSION = os.environ.get("APP_VERSION", "dev")
APP_BUILD_TIME = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# ----------------------------
# Core Processing Functions
# ----------------------------
def clean_values(values):
    """Replace common missing markers with NaN."""
    arr = np.array(values, dtype=float)
    for m in MISSING_MARKERS:
        arr[arr == m] = np.nan
    return arr


def compute_curve_features(depth, curve_values, curve_name):
    """Compute simple numeric features for a curve to help AI reason about it."""
    values = clean_values(curve_values)
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]

    features = {
        "curve": curve_name,
        "num_samples": int(len(values)),
        "num_valid": int(valid_mask.sum()),
    }

    if len(values) == 0:
        features["pct_missing"] = None
        return features

    features["pct_missing"] = float(100.0 * (1.0 - valid_mask.mean()))

    if len(v) == 0:
        return features

    # Basic stats
    features.update({
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v)),
        "p5": float(np.nanpercentile(v, 5)),
        "p95": float(np.nanpercentile(v, 95)),
    })

    # Gradient stats (change per unit depth)
    try:
        depth_arr = np.asarray(depth, dtype=float)
        depth_valid = depth_arr[valid_mask]
        d_depth = np.diff(depth_valid)
        d_vals = np.diff(v)
        with np.errstate(divide="ignore", invalid="ignore"):
            grad = d_vals / d_depth
        grad = grad[~np.isnan(grad) & ~np.isinf(grad)]
        if grad.size > 0:
            features.update({
                "grad_mean": float(np.mean(grad)),
                "grad_std": float(np.std(grad)),
                "grad_p95": float(np.percentile(grad, 95)),
            })
    except Exception:
        pass

    # Very simple spike detection via z-score
    if v.std() > 0:
        z = (v - v.mean()) / v.std()
        spike_threshold = 4.0
        spikes = np.abs(z) > spike_threshold
        features.update({
            "pct_spikes": float(100.0 * spikes.mean()),
            "num_spikes": int(spikes.sum()),
        })
    else:
        features.update({
            "pct_spikes": 0.0,
            "num_spikes": 0,
        })

    # Longest run of consecutive missing samples
    longest_missing = 0
    current = 0
    for is_valid in valid_mask:
        if not is_valid:
            current += 1
            longest_missing = max(longest_missing, current)
        else:
            current = 0

    features["max_consecutive_missing"] = int(longest_missing)

    return features


def summarize_las_curves_from_str(las_text, depth_mnemonics=("DEPT", "DEPTH")):
    """Read LAS content from a string and compute features for each non-depth curve."""
    if not LASIO_AVAILABLE:
        return None

    try:
        las = lasio.read(StringIO(las_text))
    except Exception as exc:
        print(f"LAS summary: failed to parse LAS content: {exc}")
        return None

    depth_curve = None
    for c in las.curves:
        if c.mnemonic.upper() in [d.upper() for d in depth_mnemonics]:
            depth_curve = c
            break

    if depth_curve is None:
        return None

    depth = depth_curve.data
    all_features = []

    for curve in las.curves:
        if curve is depth_curve:
            continue
        f = compute_curve_features(depth, curve.data, curve.mnemonic)
        all_features.append(f)

    return {
        "well_info": {
            "start_depth": float(depth[0]),
            "end_depth": float(depth[-1]),
            "num_samples": int(len(depth)),
        },
        "curve_features": all_features,
    }


def extract_curve_labels_from_text(full_text: str):
    """Use simple keyword matching over OCR text to find which curves appear on the image."""
    if not full_text:
        return []
    text_upper = full_text.upper()
    found = set()
    for mnemo, keywords in CURVE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_upper:
                found.add(mnemo)
                break
    return sorted(found)


def match_vision_to_las_curves(vision_labels, las_curve_mnemonics):
    """Map OCR-detected labels (GR/RHOB/NPHI/RES, etc.) to LAS mnemonics.

    Uses exact match first, then prefix/contains matching.
    """
    if not vision_labels or not las_curve_mnemonics:
        return {}

    las_upper = {m.upper(): m for m in las_curve_mnemonics}
    mapping = {}

    # 1) exact matches
    for label in vision_labels:
        lu = label.upper()
        if lu in las_upper:
            mapping[label] = las_upper[lu]

    # 2) heuristic prefix/contains for remaining
    for label in vision_labels:
        if label in mapping:
            continue
        lu = label.upper()
        candidates = [
            m for m in las_curve_mnemonics
            if m.upper().startswith(lu) or lu in m.upper()
        ]
        mapping[label] = candidates[0] if candidates else None

    return mapping


def build_ai_analysis_payload(las_text, detected_text):
    """Build a structured payload combining OCR text + LAS summary + simple mapping."""
    if not las_text:
        return None

    # 1) OCR text from Vision
    full_text = ""
    if isinstance(detected_text, str):
        full_text = detected_text
    elif isinstance(detected_text, dict):
        raw_entries = detected_text.get("raw") or []
        texts = [
            (entry.get("text") or "")
            for entry in raw_entries
            if isinstance(entry, dict) and entry.get("text")
        ]
        full_text = "\n".join(texts)

    vision_curve_labels = extract_curve_labels_from_text(full_text)

    # 2) LAS numeric features
    las_summary = summarize_las_curves_from_str(las_text)
    las_curve_mnemonics = []
    if las_summary and las_summary.get("curve_features"):
        las_curve_mnemonics = [cf["curve"] for cf in las_summary["curve_features"]]

    # 3) Map Vision labels to LAS mnemonics
    vision_to_las = match_vision_to_las_curves(vision_curve_labels, las_curve_mnemonics)

    # Optional: basic flags from features
    if las_summary and las_summary.get("curve_features"):
        for cf in las_summary["curve_features"]:
            cf_flags = []
            pct_missing = cf.get("pct_missing") or 0.0
            pct_spikes = cf.get("pct_spikes") or 0.0
            if pct_missing > 30.0:
                cf_flags.append("high_missing")
            if pct_spikes > 5.0:
                cf_flags.append("spiky")
            cf["flags"] = cf_flags

    payload = {
        "ocr_text": full_text,
        "vision_curve_labels": vision_curve_labels,
        "vision_to_las_mapping": vision_to_las,
        "las_summary": las_summary,
    }

    return payload


def call_hf_curve_analysis(ai_payload):
    """Optional: call a Hugging Face text model to get a human-readable curve analysis.

    This is best-effort and will be skipped if credentials are not configured.
    """
    if not HF_API_TOKEN or not HF_MODEL_ID or not ai_payload:
        return None

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    system_msg = (
        "You are a petrophysics assistant. Given OCR text from a well log "
        "image and numeric summaries of each LAS curve, identify which LAS "
        "curves are likely GR, RHOB, NPHI, DT, RES, etc. Be concise and "
        "return a short markdown summary listing each LAS curve with its "
        "most likely identity and any obvious issues (missing data, spikes)."
    )

    prompt = (
        system_msg
        + "\n\nHere is the JSON payload to analyze:\n\n"
        + json.dumps(ai_payload, indent=2)
    )

    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        out = resp.json()
    except Exception as exc:
        print(f"HF API error: {exc}")
        return None

    # The Inference API for text-generation typically returns a list of
    # dicts with a 'generated_text' field.
    try:
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict) and "generated_text" in first:
                return str(first["generated_text"])
        # Fallback: just stringify the response
        return json.dumps(out)
    except Exception:
        return None


def call_hf_curve_chat(ai_payload, question):
    """Optional: chat-style helper to answer user questions about this log.

    Reuses the same HF model but tailors the prompt to the specific question.
    """
    if not HF_API_TOKEN or not HF_MODEL_ID or not ai_payload or not question:
        return None

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    system_msg = (
        "You are a petrophysics assistant. Given OCR text from a well log "
        "image and numeric summaries of each LAS curve, answer the user's "
        "question about which curves are likely GR, RHOB, NPHI, DT, RES, etc. "
        "Comment on whether values and ranges look reasonable. Answer concisely "
        "in markdown, and do not invent curves that are not present."
    )

    prompt = (
        system_msg
        + "\n\nHere is the JSON payload describing this log (OCR + LAS):\n\n"
        + json.dumps(ai_payload, indent=2)
        + "\n\nUser question:\n"
        + question
        + "\n\nAnswer in concise markdown."
    )

    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
        },
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        out = resp.json()
    except Exception as exc:
        print(f"HF API error (chat): {exc}")
        return None

    try:
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict) and "generated_text" in first:
                return str(first["generated_text"])
        return json.dumps(out)
    except Exception:
        return None
def hsv_red_mask(hsv_img):
    lower1, upper1 = np.array([0, 80, 80]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 80, 80]), np.array([180, 255, 255])
    return cv2.bitwise_or(
        cv2.inRange(hsv_img, lower1, upper1),
        cv2.inRange(hsv_img, lower2, upper2),
    )


def hsv_blue_mask(hsv_img):
    lower, upper = np.array([90, 80, 80]), np.array([140, 255, 255])
    return cv2.inRange(hsv_img, lower, upper)


def hsv_green_mask(hsv_img):
    lower, upper = np.array([40, 80, 80]), np.array([90, 255, 255])
    return cv2.inRange(hsv_img, lower, upper)


def black_mask(gray_img):
    return cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        10,
    )

def pick_curve_x_per_row(mask, min_run=2):
    h, w = mask.shape
    xs = np.full(h, np.nan, dtype=np.float32)
    for y in range(h):
        idx = np.flatnonzero(mask[y, :] > 0)
        if idx.size >= min_run:
            xs[y] = float(np.median(idx))
    return xs

def smooth_nanmedian(series, window):
    s = pd.Series(series)
    if window % 2 == 0:
        window += 1
    if window > 1:
        s = s.rolling(window, min_periods=1, center=True).median()
    return s.interpolate(limit_direction="both", limit=50).to_numpy(dtype=np.float32)

def compute_depth_vector(nrows, top_depth, bottom_depth):
    ys = np.arange(nrows, dtype=np.float32)
    return top_depth + (ys / max(1, nrows-1)) * (bottom_depth - top_depth)

def write_las_simple(depth, curve_data, depth_unit="FT"):
    """Generate LAS 1.2-style file compatible with QuickSyn"""
    null_val = -999.25
    unit_token = "F" if depth_unit.upper().startswith("F") else depth_unit.upper()
    eol = "\r\n"

    lines = []

    # Version section (LAS 1.2 style)
    lines.append("~VERSION INFORMATION" + eol)
    lines.append(" VERS.                 1.20:   CWLS LOG ASCII STANDARD -VERSION 1.20" + eol)
    lines.append(" WRAP.                   NO:   ONE LINE PER DEPTH STEP" + eol)

    # Well information section
    lines.append("~WELL INFORMATION BLOCK" + eol)
    lines.append("#MNEM.UNIT       DATA TYPE    INFORMATION" + eol)
    lines.append("#---------     -----------    ---------------------------" + eol)
    lines.append(f" STRT.{unit_token}               {depth[0]:.4f}:" + eol)
    lines.append(f" STOP.{unit_token}               {depth[-1]:.4f}:" + eol)
    step = float(depth[1] - depth[0]) if depth.size > 1 else 0.0
    lines.append(f" STEP.{unit_token}               {step:.4f}:" + eol)
    lines.append(f" NULL.               {null_val:.4f}:" + eol)
    lines.append(" WELL.               WELL NAME:  DIGITIZED_LOG" + eol)

    # Minimal parameter information section (to match legacy LAS 1.2 style)
    lines.append("~PARAMETER INFORMATION BLOCK" + eol)
    lines.append("#MNEM.UNIT       VALUE        DESCRIPTION" + eol)
    lines.append("#---------     -----------    ---------------------------" + eol)
    lines.append(" EKB .               0.0000:  ELEVATION OF KELLY BUSHING" + eol)

    # Curve information section
    lines.append("~CURVE INFORMATION BLOCK" + eol)
    lines.append("#MNEM.UNIT                 API CODE     CURVE DESCRIPTION" + eol)
    lines.append("#---------               -----------    ---------------------------" + eol)
    lines.append(f" DEPT.{unit_token}                 00 000 000 000:  DEPTH" + eol)
    for name, meta in curve_data.items():
        unit = meta.get("unit", "")
        lines.append(f" {name.upper()}.{unit} :  {name.upper()}" + eol)

    # ASCII data section (~A header with column labels)
    names = list(curve_data.keys())
    arrays = [curve_data[n]["values"] for n in names]

    header_cols = ["DEPTH"] + [n.upper() for n in names]
    header = " ".join(f"{c:>10}" for c in header_cols)
    lines.append("~A " + header + eol)

    for i in range(depth.size):
        row_vals = [f"{depth[i]:10.4f}"] + [f"{arrays[j][i]:10.4f}" for j in range(len(arrays))]
        lines.append(" ".join(row_vals) + eol)

    return "".join(lines)

# ----------------------------
# Google Vision API Functions
# ----------------------------
def downsample_for_ocr(image_bytes, max_height=2000):
    """Downsample large images before OCR to reduce memory usage"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return image_bytes
    
    h, w, _ = img.shape
    
    # Only downsample if height exceeds max_height
    if h <= max_height:
        return image_bytes
    
    # Calculate new dimensions maintaining aspect ratio
    scale = max_height / h
    new_w = int(w * scale)
    new_h = max_height
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Re-encode to bytes
    _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()

def detect_text_vision_api(image_bytes):
    """Use Google Vision API to detect text in image"""
    if not VISION_API_AVAILABLE or vision_client is None:
        return {'raw': [], 'numbers': [], 'suggestions': {}}

    try:
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)

        raw_text = []
        numeric_entries = []
        for text in response.text_annotations[1:]:  # Skip first (full text)
            bounding = text.bounding_poly.vertices
            entry = {
                'text': text.description,
                'vertices': [{'x': int(v.x), 'y': int(v.y)} for v in bounding]
            }
            raw_text.append(entry)

            # Extract numeric tokens
            import re
            numbers = re.findall(r'-?\d*\.?\d+', text.description)
            for num in numbers:
                try:
                    value = float(num)
                    x = int(bounding[0].x)
                    y = int(bounding[0].y)
                    numeric_entries.append({
                        'value': value,
                        'text': text.description,
                        'x': x,
                        'y': y
                    })
                except ValueError:
                    continue

        suggestions = build_ocr_suggestions(numeric_entries)
        suggestions = attach_curve_label_hints(suggestions, raw_text)

        return {
            'raw': raw_text,
            'numbers': numeric_entries,
            'suggestions': suggestions
        }
    except Exception as e:
        print(f"Vision API error: {e}")
        return {'raw': [], 'numbers': [], 'suggestions': {}}


def build_ocr_suggestions(numeric_entries):
    """Derive depth and curve hints from numeric OCR entries."""
    if not numeric_entries:
        return {}

    # Sort by y (top to bottom)
    sorted_entries = sorted(numeric_entries, key=lambda n: n['y'])

    depth_candidates = []
    curve_candidates = []

    for entry in sorted_entries:
        value = entry['value']
        y = entry['y']
        x = entry['x']

        # Heuristic: numbers near image left edge likely depth scale
        if x < 0.25 * max(e['x'] for e in sorted_entries + [{'x': x}]):
            depth_candidates.append({'value': value, 'y': y})
        else:
            curve_candidates.append({'value': value, 'x': x, 'y': y})

    depth_hint = None
    if len(depth_candidates) >= 2:
        # Use all detected depth labels to fit a simple linear scale
        ys = np.array([d['y'] for d in depth_candidates], dtype=np.float32)
        vals = np.array([d['value'] for d in depth_candidates], dtype=np.float32)

        try:
            # depth ≈ a * pixel_y + b
            a, b = np.polyfit(ys, vals, 1)
            y_top = float(ys.min())
            y_bottom = float(ys.max())
            top_depth_fit = float(a * y_top + b)
            bottom_depth_fit = float(a * y_bottom + b)

            # Ensure we have a sensible span
            if y_bottom > y_top and bottom_depth_fit != top_depth_fit:
                depth_hint = {
                    'top_depth': top_depth_fit,
                    'bottom_depth': bottom_depth_fit,
                    'top_px': y_top,
                    'bottom_px': y_bottom,
                    'fit_labels': [
                        {'depth': float(v), 'y_px': float(y)}
                        for (v, y) in zip(vals.tolist(), ys.tolist())
                    ]
                }
        except Exception:
            # Fallback to using just the first/last labels if fitting fails
            top = depth_candidates[0]
            bottom = depth_candidates[-1]
            if bottom['y'] > top['y'] and bottom['value'] != top['value']:
                depth_hint = {
                    'top_depth': top['value'],
                    'bottom_depth': bottom['value'],
                    'top_px': top['y'],
                    'bottom_px': bottom['y']
                }

    # Suggest curve bounds by clustering x positions
    curve_hint = None
    if curve_candidates:
        sorted_curves = sorted(curve_candidates, key=lambda c: c['x'])
        clusters = min(3, len(sorted_curves))
        if clusters:
            chunk_size = int(np.ceil(len(sorted_curves) / clusters))
            curve_hint = []
            for idx in range(clusters):
                start = idx * chunk_size
                end = min(len(sorted_curves), (idx + 1) * chunk_size)
                chunk = sorted_curves[start:end]
                if not chunk:
                    continue
                xs = [p['x'] for p in chunk]
                curve_hint.append({
                    'left_px': min(xs),
                    'right_px': max(xs),
                    'sample_value': float(np.mean([p['value'] for p in chunk]))
                })

    suggestions = {}
    if depth_hint:
        suggestions['depth'] = depth_hint
        # Also expose the raw depth label points for snapping in the UI
        labels = depth_hint.get('fit_labels') or [
            {'depth': d['value'], 'y_px': d['y']} for d in depth_candidates
        ]
        suggestions['depth_labels'] = labels
    if curve_hint:
        suggestions['curves'] = curve_hint

    return suggestions


def attach_curve_label_hints(suggestions, raw_text):
    """Attach curve label/type hints to OCR suggestions using nearby text.

    This does NOT auto-apply anything. It only adds optional fields like
    label_type/label_mnemonic/label_text to the curve hint objects so the
    frontend can show them as suggestions.
    """
    if not suggestions or not raw_text:
        return suggestions

    curves = suggestions.get('curves') or []
    if not curves:
        return suggestions

    # Estimate a "header band" (top portion of the image) from text bounding boxes
    y_centers_all = []
    for entry in raw_text:
        verts = entry.get('vertices') or []
        ys_all = [v.get('y') for v in verts if isinstance(v, dict) and 'y' in v]
        if ys_all:
            y_centers_all.append(float(sum(ys_all)) / len(ys_all))

    if not y_centers_all:
        return suggestions

    min_y = min(y_centers_all)
    max_y = max(y_centers_all)
    header_threshold = min_y + 0.3 * (max_y - min_y)  # top ~30% of text as header band

    # Build candidate labels from raw text restricted to the header band
    candidates = []
    for entry in raw_text:
        text = (entry.get('text') or '').strip()
        if not text:
            continue
        label_upper = text.upper()

        label_type = None
        if 'GAMMA' in label_upper or label_upper.startswith('GR'):
            label_type = 'GR'
        elif 'RHOB' in label_upper or 'RHO' in label_upper or 'DENS' in label_upper:
            label_type = 'RHOB'
        elif 'NPHI' in label_upper or 'NEUTRON' in label_upper:
            label_type = 'NPHI'
        elif 'DTC' in label_upper or 'DT ' in label_upper or label_upper.startswith('DT') or 'SONIC' in label_upper:
            label_type = 'DT'
        elif 'CALI' in label_upper or 'CALIPER' in label_upper:
            label_type = 'CALI'
        elif label_upper == 'SP' or 'SPONTANEOUS' in label_upper:
            label_type = 'SP'

        if not label_type:
            continue

        verts = entry.get('vertices') or []
        xs = [v.get('x') for v in verts if isinstance(v, dict) and 'x' in v]
        ys = [v.get('y') for v in verts if isinstance(v, dict) and 'y' in v]
        if not xs or not ys:
            continue

        y_center = float(sum(ys)) / len(ys)
        if y_center > header_threshold:
            # Skip labels that are not in the header band above the tracks
            continue

        x_center = float(sum(xs)) / len(xs)

        candidates.append({
            'type': label_type,
            'text': text,
            'x': x_center,
            'y': y_center,
        })

    if not candidates:
        return suggestions

    # Associate candidate labels with each curve by horizontal proximity
    for curve in curves:
        left_px = curve.get('left_px')
        right_px = curve.get('right_px')
        if left_px is None or right_px is None:
            continue

        track_center = 0.5 * (left_px + right_px)
        best = None
        best_dist = None
        margin = (right_px - left_px) * 0.5 + 30  # allow some slack

        for cand in candidates:
            dx = cand['x'] - track_center
            if abs(dx) > margin:
                continue
            if best is None or abs(dx) < best_dist:
                best = cand
                best_dist = abs(dx)

        if best is not None:
            label_type = best['type']
            defaults = CURVE_TYPE_DEFAULTS.get(label_type, {})
            curve['label_type'] = label_type
            curve['label_mnemonic'] = defaults.get('mnemonic', label_type)
            curve['label_unit'] = defaults.get('unit')
            curve['label_text'] = best['text']
            curve['label_x'] = best.get('x')
            curve['label_y'] = best.get('y')

    return suggestions


def attach_color_hints_to_ocr_curves(image_array, suggestions):
    """Attach simple color-based hints to OCR curve suggestions.

    For each suggested curve track, look at the underlying image region and
    estimate whether it appears predominantly red or dark. This is used only
    to provide hints / default mode suggestions; the user remains in control.
    """
    if not isinstance(suggestions, dict):
        return suggestions

    curves = suggestions.get('curves') or []
    if not curves:
        return suggestions

    h, w = image_array.shape[:2]

    for curve in curves:
        left_px = curve.get('left_px')
        right_px = curve.get('right_px')
        if left_px is None or right_px is None:
            continue

        try:
            left = int(left_px)
            right = int(right_px)
        except Exception:
            continue

        left = max(0, min(w - 1, left))
        right = max(0, min(w, right))
        if right <= left:
            continue

        roi = image_array[:, left:right]
        if roi.size == 0:
            continue

        mean_color = roi.reshape(-1, 3).mean(axis=0)  # B, G, R
        b, g, r = [float(c) for c in mean_color]

        dominant = "mixed"
        recommended_mode = "black"

        if b > r * 1.2 and b > g * 1.2 and b > 60:
            dominant = "blue"
            recommended_mode = "blue"
        elif g > r * 1.2 and g > b * 1.2 and g > 60:
            dominant = "green"
            recommended_mode = "green"
        elif r > g * 1.2 and r > b * 1.2 and r > 60:
            dominant = "red"
            recommended_mode = "red"
        elif max(b, g, r) < 80:
            dominant = "dark"
            recommended_mode = "black"
        elif max(b, g, r) < 150:
            dominant = "gray"
            recommended_mode = "black"

        if dominant == "red":
            hint_text = "Track appears predominantly red; consider using Red mode for detection."
        elif dominant == "green":
            hint_text = "Track appears predominantly green; consider using Green mode for detection."
        elif dominant == "blue":
            hint_text = "Track appears predominantly blue; consider using Blue mode for detection."
        elif dominant in ("dark", "gray"):
            hint_text = "Track appears mostly dark; Black mode is likely appropriate."
        else:
            hint_text = "Track color is mixed; choose Red/Black/Blue/Green mode based on how the curve is drawn."

        curve['color_dominant'] = dominant
        curve['color_recommended_mode'] = recommended_mode
        curve['color_hint_text'] = hint_text

    return suggestions


def compute_curve_outlier_warnings(curves_cfg, las_curve_data, null_val):
    """Simple range-based sanity checks for GR/RHOB/DT curves.

    This does not block LAS generation; it only returns human-readable
    warning strings that the frontend can display alongside status.
    """
    warnings = []
    if not curves_cfg or not las_curve_data:
        return warnings


def compute_depth_warnings(depth_cfg, image_height):
    """Basic sanity checks for depth configuration.

    This checks for monotonicity and a reasonable depth-per-pixel scale.
    Returns a list of human-readable warning strings.
    """
    if not depth_cfg:
        return []

    warnings = []

    try:
        top_px = float(depth_cfg.get('top_px'))
        bottom_px = float(depth_cfg.get('bottom_px'))
        top_depth = float(depth_cfg.get('top_depth'))
        bottom_depth = float(depth_cfg.get('bottom_depth'))
    except Exception:
        return warnings

    if not np.isfinite(top_px) or not np.isfinite(bottom_px) or not np.isfinite(top_depth) or not np.isfinite(bottom_depth):
        return warnings

    if bottom_px <= top_px:
        warnings.append(f"Bottom pixel ({bottom_px:.0f}) is not below top pixel ({top_px:.0f}); check depth window.")

    depth_span = bottom_depth - top_depth
    pix_span = bottom_px - top_px

    if depth_span == 0:
        warnings.append("Top and bottom depths are identical; depth range is zero.")
    else:
        depth_per_pixel = depth_span / max(1.0, pix_span)
        # Heuristic: flag extremely small or large scales
        if abs(depth_per_pixel) < 1e-3:
            warnings.append(f"Depth scale ({depth_per_pixel:.4f} per pixel) is extremely small; check depth values.")
        if abs(depth_per_pixel) > 100.0:
            warnings.append(f"Depth scale ({depth_per_pixel:.2f} per pixel) is extremely large; check depth values.")

    if image_height and (top_px < 0 or bottom_px > image_height):
        warnings.append(f"Depth pixels ({top_px:.0f}–{bottom_px:.0f}) are outside image bounds (0–{image_height - 1}).")

    return warnings

    for c in curves_cfg:
        curve_type = (c.get('type') or '').upper()
        mnemonic = (c.get('las_mnemonic') or c.get('name') or '').upper()
        if not mnemonic or mnemonic not in las_curve_data:
            continue

        meta = las_curve_data.get(mnemonic) or {}
        vals = np.asarray(meta.get("values"), dtype=np.float32)
        if vals.size == 0:
            continue

        valid_mask = vals != null_val
        if not np.any(valid_mask):
            continue

        vals_valid = vals[valid_mask]
        vmin = float(np.nanmin(vals_valid))
        vmax = float(np.nanmax(vals_valid))

        median = float(np.nanmedian(vals_valid))
        std = float(np.nanstd(vals_valid))
        null_pct = 100.0 * (1.0 - float(np.count_nonzero(valid_mask)) / float(vals.size))

        # Decide expected range based on curve type / mnemonic
        low, high = None, None
        if curve_type == 'GR' or mnemonic == 'GR':
            low, high = 0.0, 300.0  # API units
        elif curve_type == 'RHOB' or mnemonic == 'RHOB':
            low, high = 1.7, 3.0    # g/cc
        elif curve_type in ('DT', 'DTC') or mnemonic in ('DT', 'DTC'):
            low, high = 40.0, 200.0 # us/ft

        if low is None or high is None:
            continue

        issues = []
        if vmin < low:
            issues.append(f"min {vmin:.2f} < {low}")
        if vmax > high:
            issues.append(f"max {vmax:.2f} > {high}")

        span = high - low
        dyn_range = vmax - vmin
        if span > 0 and dyn_range < 0.05 * span:
            issues.append(f"curve is very flat (range {dyn_range:.2f})")

        if null_pct > 40.0:
            issues.append(f"{null_pct:.0f}% of samples are null")

        if issues:
            label = c.get('display_name') or mnemonic or curve_type or 'curve'
            summary = (
                f"{label}: {', '.join(issues)} "
                f"(min={vmin:.2f}, max={vmax:.2f}, median={median:.2f}, std={std:.2f}, null≈{null_pct:.0f}%). "
                f"Expected roughly {low}–{high}."
            )
            warnings.append(summary)

    return warnings


def auto_detect_tracks(image_array):
    """Auto-detect track boundaries"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    vertical_sum = np.sum(edges, axis=0)
    
    # Simple peak detection
    threshold = np.max(vertical_sum) * 0.3
    peaks = []
    for i in range(1, len(vertical_sum)-1):
        if vertical_sum[i] > threshold and vertical_sum[i] > vertical_sum[i-1] and vertical_sum[i] > vertical_sum[i+1]:
            peaks.append(i)
    
    # Group into tracks
    if len(peaks) >= 2:
        tracks = [(peaks[i], peaks[i+1]) for i in range(0, len(peaks)-1, 2)]
    else:
        # Fallback: divide into 3 equal sections
        w = image_array.shape[1]
        section_width = w // 3
        tracks = [(i*section_width, (i+1)*section_width) for i in range(3)]
    
    return tracks

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html', app_version=APP_VERSION, build_time=APP_BUILD_TIME)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return image info"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Read image
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Could not read image'}), 400
    
    h, w, _ = img.shape
    
    # Convert to base64 for display
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Auto-detect tracks
    tracks = auto_detect_tracks(img)

    # Try OCR if available
    detected_text = {'raw': [], 'numbers': [], 'suggestions': {}}
    ocr_suggestions = {}
    if VISION_API_AVAILABLE:
        detected_text = detect_text_vision_api(file_bytes)
        ocr_suggestions = detected_text.get('suggestions', {}) or {}
        if ocr_suggestions:
            ocr_suggestions = attach_color_hints_to_ocr_curves(img, ocr_suggestions)
            detected_text['suggestions'] = ocr_suggestions

    return jsonify({
        'success': True,
        'image': f'data:image/png;base64,{img_base64}',
        'width': w,
        'height': h,
        'tracks': tracks,
        'detected_text': detected_text,
        'ocr_suggestions': ocr_suggestions or detected_text.get('suggestions', {}),
        'vision_api_available': VISION_API_AVAILABLE and bool(detected_text.get('raw'))
    })

@app.route('/digitize', methods=['POST'])
def digitize():
    """Process digitization request"""
    data = request.json
    
    # Decode image
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Extract config
    cfg = data['config']
    detected_text = data.get('detected_text') or {}
    depth_cfg = cfg['depth']
    curves = (cfg['curves'] or [])[:6]
    gopt = cfg.get('global_options', {})
    
    null_val = float(gopt.get('null', -999.25))
    downsample = int(gopt.get('downsample', 1))
    blur = int(gopt.get('blur', 3))
    min_run = int(gopt.get('min_run', 2))
    smooth_window = int(gopt.get('smooth_window', 5))
    
    H, W, _ = img.shape
    top = max(0, int(depth_cfg['top_px']))
    bot = min(H, int(depth_cfg['bottom_px']))
    top_depth = float(depth_cfg['top_depth'])
    bottom_depth = float(depth_cfg['bottom_depth'])
    depth_unit = depth_cfg.get('unit', 'FT')
    
    nrows = bot - top
    base_depth = compute_depth_vector(nrows, top_depth, bottom_depth)

    # Depth sanity checks
    depth_warnings = compute_depth_warnings({
        'top_px': top,
        'bottom_px': bot,
        'top_depth': top_depth,
        'bottom_depth': bottom_depth,
    }, H)
    
    curve_data = {}
    curve_traces = {}
    
    for c in curves:
        # LAS-facing name/unit come from las_mnemonic/las_unit (or name/unit as fallback)
        name = c.get('las_mnemonic') or c.get('name')
        unit = c.get('las_unit') or c.get('unit', '')
        left_px = int(c['left_px'])
        right_px = int(c['right_px'])
        left_value = float(c['left_value'])
        right_value = float(c['right_value'])
        mode = c.get('mode', 'black')
        
        roi = img[top:bot, left_px:right_px]
        if blur > 0:
            bb = blur + 1 if blur % 2 == 0 else blur
            roi = cv2.GaussianBlur(roi, (bb, bb), 0)
        
        if mode == 'red':
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = hsv_red_mask(hsv)
        elif mode == 'blue':
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = hsv_blue_mask(hsv)
        elif mode == 'green':
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = hsv_green_mask(hsv)
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mask = black_mask(gray)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

        # Suppress strong vertical grid lines that appear in most rows, so we
        # favor the wiggly curve trace over perfectly vertical grid strokes.
        try:
            h_mask, w_mask = mask.shape
            if h_mask > 0 and w_mask > 0:
                col_hits = (mask > 0).sum(axis=0)
                # Columns that are active in a large fraction of rows are
                # likely grid lines rather than the curve itself.
                threshold = max(10, int(0.7 * h_mask))
                grid_cols = col_hits >= threshold
                if np.any(grid_cols):
                    mask[:, grid_cols] = 0
        except Exception:
            # If anything goes wrong, fall back to the original mask
            pass

        xs = pick_curve_x_per_row(mask, min_run)
        xs = smooth_nanmedian(xs, smooth_window)

        width_px = mask.shape[1]
        vals = np.full(xs.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(xs)
        vals[valid] = left_value + (xs[valid] / max(1, width_px-1)) * (right_value - left_value)

        vals_out = np.where(np.isnan(vals), null_val, vals).astype(np.float32)
        curve_data[name] = {'unit': unit, 'values': vals_out}

        # Build a sparse set of trace points in original image coordinates for UI overlay
        trace_points = []
        if xs.size > 0:
            # Sample at most ~600 points per curve to keep payload small
            step = max(1, int(np.ceil(xs.size / 600)))
            for row_idx in range(0, xs.size, step):
                x_val = xs[row_idx]
                if np.isnan(x_val):
                    continue
                x_img = int(left_px + x_val)
                y_img = int(top + row_idx)
                trace_points.append([x_img, y_img])

        curve_traces[name] = trace_points
    
    # Resample to fixed 0.5 ft step when using feet
    las_depth = base_depth
    las_curve_data = curve_data
    if depth_unit.upper() == "FT" and base_depth.size > 1:
        start = float(base_depth[0])
        stop = float(base_depth[-1])
        step_mag = 0.5

        if stop >= start:
            las_depth = np.arange(start, stop + step_mag * 0.5, step_mag, dtype=np.float32)
        else:
            las_depth = np.arange(start, stop - step_mag * 0.5, -step_mag, dtype=np.float32)

        las_curve_data = {}
        for name, meta in curve_data.items():
            vals = meta["values"].astype(np.float32)
            valid_mask = vals != null_val

            if not np.any(valid_mask):
                new_vals = np.full(las_depth.shape, null_val, dtype=np.float32)
            else:
                depth_valid = base_depth[valid_mask]
                vals_valid = vals[valid_mask]
                order = np.argsort(depth_valid)
                depth_sorted = depth_valid[order]
                vals_sorted = vals_valid[order]
                interp_vals = np.interp(las_depth, depth_sorted, vals_sorted, left=null_val, right=null_val)
                new_vals = interp_vals.astype(np.float32)

            las_curve_data[name] = {"unit": meta.get("unit", ""), "values": new_vals}

    # Run simple curve sanity checks (outlier warnings) on the final LAS depth grid
    outlier_warnings = compute_curve_outlier_warnings(curves, las_curve_data, null_val)

    # Generate LAS file
    las_content = write_las_simple(las_depth, las_curve_data, depth_unit)

    # Validate LAS output if possible
    validation = {
        'passed': True,
        'message': 'LAS validation skipped (lasio not installed).'
    }
    if LASIO_AVAILABLE:
        try:
            lasio.read(StringIO(las_content))
            validation = {
                'passed': True,
                'message': 'LAS parsed successfully with lasio.'
            }
        except Exception as exc:
            validation = {
                'passed': False,
                'message': f'LAS validation failed: {exc}'
            }

    # Build AI analysis payload (OCR + LAS stats) and optionally call Hugging Face
    ai_payload = build_ai_analysis_payload(las_content, detected_text)
    ai_summary = call_hf_curve_analysis(ai_payload) if ai_payload else None

    return jsonify({
        'success': True,
        'las_content': las_content,
        'filename': 'digitized_log.las',
        'validation': validation,
        'outlier_warnings': outlier_warnings,
        'depth_warnings': depth_warnings,
        'curve_traces': curve_traces,
        'ai_payload': ai_payload,
        'ai_summary': ai_summary
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'vision_api': VISION_API_AVAILABLE
    })


@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    """Chat-style endpoint: answer a question about the current log using ai_payload.

    Expects JSON with:
      - ai_payload: the object returned from /digitize
      - question: user's natural language question
    """
    data = request.json or {}
    ai_payload = data.get('ai_payload')
    question = (data.get('question') or '').strip()

    if not ai_payload or not question:
        return jsonify({'success': False, 'error': 'Missing ai_payload or question.'}), 400

    answer = call_hf_curve_chat(ai_payload, question)
    if answer is None:
        return jsonify({'success': False, 'error': 'AI chat is not configured or failed.'}), 500

    return jsonify({'success': True, 'answer': answer})

if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting TIFF→LAS Web App")
    print(f"📊 Google Vision API: {'✅ Available' if VISION_API_AVAILABLE else '⚠️  Not configured'}")
    print("🌐 Open: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
