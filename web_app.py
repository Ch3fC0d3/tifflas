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
import openai
from huggingface_hub import InferenceClient

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID") or os.getenv("OPENAI_MODEL")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID") or "models/gemini-2.0-flash"

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
        f["mnemonic"] = curve.mnemonic
        f["unit"] = getattr(curve, "unit", "") or ""
        f["description"] = getattr(curve, "descr", "") or getattr(curve, "description", "") or ""
        f["rule_type_guess"] = guess_curve_type_from_metadata(
            f["mnemonic"], f["unit"], f["description"]
        )
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


def guess_curve_type_from_metadata(mnemonic, unit, description):
    m = (mnemonic or "").upper()
    u = (unit or "").upper()
    d = (description or "").upper()

    text = " ".join([m, u, d])

    if "GR" in m or "GAMMA" in text:
        return "GR"
    if "RHOB" in m or "DENSITY" in text or "RHO B" in text:
        return "RHOB"
    if "NPHI" in m or "NEUT" in text or "POROSITY" in text:
        return "NPHI"
    if "DT" in m or "DTC" in m or "SONIC" in text:
        return "DT"
    if "CALI" in m or "CALIPER" in text:
        return "CALI"
    if "SP" in m or "SPONTANEOUS" in text:
        return "SP"
    if "RES" in m or "RESISTIVITY" in text or "OHMM" in u or "OHM-M" in u or "OHMM" in text:
        return "RES"
    return None


def build_ai_analysis_payload(las_text, detected_text, user_curves=None):
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

    # 4) User-provided curve config (manual overrides from frontend)
    user_curve_metadata = None
    user_curve_type_by_mnemonic = None
    if user_curves:
        user_curve_metadata = []
        user_curve_type_by_mnemonic = {}
        for idx, c in enumerate(user_curves):
            sel_type = c.get("type")
            las_mnemonic = (c.get("las_mnemonic") or c.get("name") or "").upper()
            entry = {
                "index": idx + 1,
                "selected_type": sel_type,
                "las_mnemonic": las_mnemonic,
                "las_unit": c.get("las_unit") or c.get("unit") or "",
                "display_name": c.get("display_name"),
                "display_unit": c.get("display_unit"),
            }
            user_curve_metadata.append(entry)
            if las_mnemonic:
                user_curve_type_by_mnemonic[las_mnemonic] = sel_type

    payload = {
        "ocr_text": full_text,
        "vision_curve_labels": vision_curve_labels,
        "vision_to_las_mapping": vision_to_las,
        "las_summary": las_summary,
    }

    if user_curve_metadata is not None:
        payload["user_curves"] = user_curve_metadata
        payload["user_curve_type_by_mnemonic"] = user_curve_type_by_mnemonic

    return payload


def call_hf_curve_analysis(ai_payload):
    """Optional: call a Hugging Face text model to get a human-readable curve analysis.

    This is best-effort and will be skipped if credentials are not configured.
    """
    if not ai_payload:
        return None

    system_msg = (
        "You are a petrophysics assistant. Given OCR text from a well log "
        "image and numeric summaries of each LAS curve, identify which LAS "
        "curves are likely GR, RHOB, NPHI, DT, RES, etc. Provide a detailed, "
        "structured markdown report that: (1) explains your methodology for "
        "identifying each curve (OCR labels, value ranges, units, typical scales), "
        "(2) maps each LAS curve to its most likely identity with specific reasoning, "
        "(3) comments on value ranges, units, and typical petrophysical expectations, "
        "(4) highlights data quality issues such as missing data or spikes, and "
        "(5) calls out any unusual depth intervals. Always explain WHY you identified "
        "each curve the way you did. Do not invent curves that are not present."
    )

    prompt = (
        system_msg
        + "\n\nHere is the JSON payload to analyze:\n\n"
        + json.dumps(ai_payload, indent=2)
    )

    # Prefer Gemini if configured
    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            # Use REST API directly to avoid SDK version issues
            # Model ID should include 'models/' prefix (e.g., 'models/gemini-2.0-flash')
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith('models/') else f'models/{GEMINI_MODEL_ID}'
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        text = parts[0].get('text', '')
                        if text:
                            return str(text)
            else:
                print(f"Gemini API error (analysis): {resp.status_code} {resp.text}")
        except Exception as exc:
            print(f"Gemini API error (analysis): {exc}")

    # Fallback to OpenAI if configured
    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(ai_payload, indent=2)},
            ]
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                if content:
                    return str(content)
        except Exception as exc:
            print(f"OpenAI API error (analysis): {exc}")

    # Fallback to Hugging Face Inference if available
    if not HF_API_TOKEN or not HF_MODEL_ID:
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        print(f"HF InferenceClient init error (analysis): {exc}")
        return None

    try:
        out = client.text_generation(
            prompt,
            model=HF_MODEL_ID,
            max_new_tokens=512,
            temperature=0.3,
        )
    except Exception as exc:
        print(f"HF text_generation error (analysis): {exc}")
        return None

    return out if isinstance(out, str) else str(out)


def call_hf_curve_chat(ai_payload, question):
    """Optional: chat-style helper to answer user questions about this log.

    Reuses the same HF model but tailors the prompt to the specific question.
    """
    question = (question or "").strip()
    if not ai_payload or not question:
        return None

    system_msg = (
        "You are a petrophysics assistant. Given OCR text from a well log "
        "image and numeric summaries of each LAS curve, answer the user's "
        "question with a detailed but focused markdown explanation. When "
        "relevant, discuss which curves are likely GR, RHOB, NPHI, DT, RES, "
        "etc., comment on whether values and ranges look reasonable, and refer "
        "to specific depth intervals or data-quality issues. Be precise about "
        "what is supported by the provided payload, and do not invent curves "
        "that are not present."
    )

    payload_text = (
        "Here is the JSON payload describing this log (OCR + LAS):\n\n"
        + json.dumps(ai_payload, indent=2)
        + "\n\nUser question:\n"
        + question
    )

    # Prefer Gemini if configured
    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            # Use REST API directly to avoid SDK version issues
            # Model ID should include 'models/' prefix (e.g., 'models/gemini-2.0-flash')
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith('models/') else f'models/{GEMINI_MODEL_ID}'
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": payload_text}]}]
            }
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        text = parts[0].get('text', '')
                        if text:
                            return str(text)
            else:
                print(f"Gemini API error (chat): {resp.status_code} {resp.text}")
        except Exception as exc:
            print(f"Gemini API error (chat): {exc}")

    # Fallback to OpenAI if configured
    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": payload_text},
            ]
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                if content:
                    return str(content)
        except Exception as exc:
            print(f"OpenAI API error (chat): {exc}")

    # Fallback to Hugging Face Inference if available
    if not HF_API_TOKEN or not HF_MODEL_ID:
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        print(f"HF InferenceClient init error (chat): {exc}")
        return None

    try:
        out = client.text_generation(
            payload_text,
            model=HF_MODEL_ID,
            max_new_tokens=512,
            temperature=0.3,
        )
    except Exception as exc:
        print(f"HF text_generation error (chat): {exc}")
        return f"AI request failed: {str(exc)}"

    return out if isinstance(out, str) else str(out)


def _extract_json_object(text: str):
    """Best-effort helper to parse a single JSON object from a text response.

    Many LLMs sometimes wrap JSON in Markdown or add prose. We first try to
    parse the whole string, then fall back to the first {...} block.
    """
    if not text:
        return None
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def validate_and_fix_calibration(calibration):
    """Validate calibration and fix obvious curve type/scale mismatches."""
    if not isinstance(calibration, dict):
        return calibration
    
    tracks = calibration.get('tracks', [])
    if not tracks:
        return calibration
    
    # Define expected scale ranges for each curve type
    expected_scales = {
        'GR': (0, 150),      # API units
        'RHOB': (1.5, 3.5),  # g/cc
        'NPHI': (-0.2, 0.6), # v/v (porosity)
        'DT': (40, 200),     # us/ft (sonic)
        'CALI': (6, 16),     # inches
        'SP': (-150, 50),    # mV
    }
    
    for track in tracks:
        name = (track.get('name') or '').upper()
        scale_min = track.get('scale_min')
        scale_max = track.get('scale_max')
        
        if not name or not isinstance(scale_min, (int, float)) or not isinstance(scale_max, (int, float)):
            continue
        
        # Check if scale matches the curve type
        if name in expected_scales:
            exp_min, exp_max = expected_scales[name]
            scale_range = scale_max - scale_min
            exp_range = exp_max - exp_min
            
            # If scale is way off (more than 3x different), try to find correct type
            if scale_range > exp_range * 3 or scale_range < exp_range / 3:
                # Try to match scale to correct curve type
                for curve_type, (type_min, type_max) in expected_scales.items():
                    type_range = type_max - type_min
                    # Check if scale is within reasonable bounds for this type
                    if (abs(scale_min - type_min) < type_range * 0.5 and 
                        abs(scale_max - type_max) < type_range * 0.5):
                        print(f"AI calibration: Fixing curve type from {name} to {curve_type} based on scale {scale_min}-{scale_max}")
                        track['name'] = curve_type
                        break
    
    return calibration


def call_ai_calibration(calib_payload):
    """Ask an LLM to propose depth_axis and track calibration JSON.

    calib_payload is a small dict with fields like:
        {
          "image": {"width_px": W, "height_px": H},
          "depth_label_candidates": [{"value": ..., "x_px": ..., "y_px": ...}, ...],
          "header_text_boxes": [{"text": "GR", "x_px": ..., "y_px": ...}, ...],
        }

    Returns a Python dict with optional keys:
        {
          "depth_axis": {
            "top_depth": float,
            "bottom_depth": float,
            "top_pixel": float,
            "bottom_pixel": float,
          },
          "tracks": [
            {
              "name": "GR",
              "left_x": float,
              "right_x": float,
              "scale_min": float,
              "scale_max": float,
              "hot_side": "left" | "right",
              "color_hint": "black" | "red" | "blue" | "green" | null,
            },
            ...
          ]
        }
    """
    if not calib_payload:
        return None

    schema_hint = (
        "You are a petrophysical log calibration assistant. Given OCR-derived "
        "depth label candidates and header text boxes for a single raster log "
        "panel, infer a plausible depth axis and track calibration.\n\n"
        "Always respond with JSON ONLY, no prose, using this schema:\n\n"
        "{\n"
        "  \"depth_axis\": {\n"
        "    \"top_depth\": number,\n"
        "    \"bottom_depth\": number,\n"
        "    \"top_pixel\": number,\n"
        "    \"bottom_pixel\": number\n"
        "  },\n"
        "  \"tracks\": [\n"
        "    {\n"
        "      \"name\": string,\n"
        "      \"left_x\": number,\n"
        "      \"right_x\": number,\n"
        "      \"scale_min\": number,\n"
        "      \"scale_max\": number,\n"
        "      \"hot_side\": \"left\" or \"right\",\n"
        "      \"color_hint\": string or null\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Pixels are in the coordinate system of the provided panel image, where\n"
        "(0,0) is the top-left corner and Y increases downward. Depth should be\n"
        "monotonic with pixel Y (depth increases as Y increases).\n\n"
        "DEPTH AXIS RULES:\n"
        "- Look for depth labels on the LEFT side of the panel (depth_label_candidates)\n"
        "- The SMALLEST depth value should be at the TOP (smallest y_px)\n"
        "- The LARGEST depth value should be at the BOTTOM (largest y_px)\n"
        "- Typical well logs span 10-500 feet per panel\n"
        "- Depth values are usually round numbers (100, 150, 200, etc.)\n"
        "- top_pixel should be the y_px of the topmost depth label\n"
        "- bottom_pixel should be the y_px of the bottommost depth label\n\n"
        "TRACK CALIBRATION RULES:\n"
        "- Match curve names from header_text_boxes to standard petrophysical curves\n"
        "- CRITICAL: The curve NAME must match its SCALE RANGE:\n"
        "  * GR (Gamma Ray): scale_min=0, scale_max=150, units=API\n"
        "  * RHOB (Density): scale_min=1.95, scale_max=2.95, units=g/cc\n"
        "  * NPHI (Neutron Porosity): scale_min=-0.15, scale_max=0.45, units=v/v\n"
        "  * DT (Sonic): scale_min=40, scale_max=140, units=us/ft\n"
        "  * CALI (Caliper): scale_min=6, scale_max=16, units=inches\n"
        "- Common header text variations:\n"
        "  * 'GR', 'GAMMA', 'Gamma Ray' → name='GR'\n"
        "  * 'RHOB', 'DENS', 'Density', 'RHO' → name='RHOB'\n"
        "  * 'NPHI', 'NEUT', 'Neutron', 'PHI' → name='NPHI'\n"
        "  * 'DT', 'SONIC', 'AC', 'Sonic' → name='DT'\n"
        "- Each track should have left_x < right_x\n"
        "- Tracks are ordered left-to-right across the panel\n\n"
        "EXAMPLE: If you see header text 'GR' at x=100, and the track spans x=80-120,\n"
        "then: name='GR', left_x=80, right_x=120, scale_min=0, scale_max=150\n\n"
        "Here is the input JSON you should analyze:\n\n"
    )

    payload_text = schema_hint + json.dumps(calib_payload, indent=2)

    # Prefer Gemini if configured
    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith("models/") else f"models/{GEMINI_MODEL_ID}"
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            body = {"contents": [{"parts": [{"text": payload_text}]}]}
            resp = requests.post(url, json=body, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        calib = _extract_json_object(text)
                        if isinstance(calib, dict):
                            return calib
            else:
                print(f"Gemini API error (calibration): {resp.status_code} {resp.text}")
        except Exception as exc:
            print(f"Gemini API error (calibration): {exc}")

    # Fallback to OpenAI if configured
    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            messages = [
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": payload_text},
            ]
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                calib = _extract_json_object(content)
                if isinstance(calib, dict):
                    return calib
        except Exception as exc:
            print(f"OpenAI API error (calibration): {exc}")

    # Fallback to Hugging Face text-generation if available
    if not HF_API_TOKEN or not HF_MODEL_ID:
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        print(f"HF InferenceClient init error (calibration): {exc}")
        return None

    try:
        out = client.text_generation(
            payload_text,
            model=HF_MODEL_ID,
            max_new_tokens=512,
            temperature=0.1,
        )
        calib = _extract_json_object(out if isinstance(out, str) else str(out))
        if isinstance(calib, dict):
            return calib
    except Exception as exc:
        print(f"HF text_generation error (calibration): {exc}")

    return None


def call_ai_auto_layout(layout_payload):
    """Ask an LLM to infer logging track layout from header text items.

    layout_payload is a small dict with fields like:
        {
          "image": {"width_px": W, "height_px": H},
          "items": [
            {"text": "GR", "x": 650, "y": 120},
            {"text": "0", "x": 620, "y": 140},
            ...
          ]
        }

    The model should return:
        {
          "tracks": [
            {
              "name": "GR",
              "left_x": float,
              "right_x": float,
              "scale_min": number or null,
              "scale_max": number or null,
              "unit": string or null,
              "hot_side": "left" or "right" or null
            },
            ...
          ]
        }
    """
    if not layout_payload:
        return None

    schema_hint = (
        "You are analyzing the HEADER of a raster well log. The user has "
        "cropped the top portion of a single log panel. You see short text "
        "items (curve mnemonics and scale labels) with approximate x/y "
        "centers in pixels.\n\n"
        "Your job is to infer the logging TRACKS present across the width of "
        "the header and return JSON ONLY describing each track.\n\n"
        "Pixels are in the coordinate system of the provided header image, "
        "where x=0 is the left edge and x increases to the right. The overall "
        "image width in pixels is image.width_px.\n\n"
        "OUTPUT FORMAT (JSON ONLY):\n"
        "{\n"
        "  \"tracks\": [\n"
        "    {\n"
        "      \"name\": string,                    // e.g. \"GR\", \"RHOB\", \"NPHI\"\n"
        "      \"left_x\": number,                  // approximate left boundary of this track in pixels\n"
        "      \"right_x\": number,                 // approximate right boundary of this track in pixels\n"
        "      \"scale_min\": number or null,       // inferred scale min if obvious\n"
        "      \"scale_max\": number or null,       // inferred scale max if obvious\n"
        "      \"unit\": string or null,            // e.g. \"API\", \"G/CC\", \"V/V\", \"US/F\"\n"
        "      \"hot_side\": \"left\" | \"right\" | null  // which side is higher / hot values\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "GUIDELINES:\n"
        "- Group header items with similar x positions into the same track.\n"
        "- Track NAME should follow standard petrophysical conventions:\n"
        "  * GR (Gamma Ray)\n"
        "  * RHOB (Density)\n"
        "  * NPHI (Neutron Porosity)\n"
        "  * DT (Sonic)\n"
        "  * CALI (Caliper)\n"
        "  * SP (Spontaneous Potential)\n"
        "- Use typical scale ranges when you see numeric labels near a curve name:\n"
        "  * GR:   ~0–150 API\n"
        "  * RHOB: ~1.95–2.95 g/cc\n"
        "  * NPHI: ~-0.15–0.45 v/v\n"
        "  * DT:   ~40–140 us/ft\n"
        "  * CALI: ~6–16 in\n"
        "- Infer left_x/right_x by placing boundaries midway between adjacent "
        "curve label centers along the x-axis.\n"
        "- Ensure left_x < right_x and tracks are ordered left-to-right.\n"
        "- If you are unsure about scale_min/scale_max or unit, use null.\n\n"
        "Here is the input JSON you should analyze:\n\n"
    )

    payload_text = schema_hint + json.dumps(layout_payload, indent=2)

    # Prefer Gemini if configured
    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith("models/") else f"models/{GEMINI_MODEL_ID}"
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            body = {"contents": [{"parts": [{"text": payload_text}]}]}
            resp = requests.post(url, json=body, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        layout = _extract_json_object(text)
                        if isinstance(layout, dict):
                            return layout
            else:
                print(f"Gemini API error (auto_layout): {resp.status_code} {resp.text}")
        except Exception as exc:
            print(f"Gemini API error (auto_layout): {exc}")

    # Fallback to OpenAI if configured
    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            messages = [
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": payload_text},
            ]
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
                layout = _extract_json_object(content)
                if isinstance(layout, dict):
                    return layout
        except Exception as exc:
            print(f"OpenAI API error (auto_layout): {exc}")

    # Fallback to Hugging Face text-generation if available
    if not HF_API_TOKEN or not HF_MODEL_ID:
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        print(f"HF InferenceClient init error (auto_layout): {exc}")
        return None

    try:
        out = client.text_generation(
            payload_text,
            model=HF_MODEL_ID,
            max_new_tokens=512,
            temperature=0.1,
        )
        layout = _extract_json_object(out if isinstance(out, str) else str(out))
        if isinstance(layout, dict):
            return layout
    except Exception as exc:
        print(f"HF text_generation error (auto_layout): {exc}")

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

def preprocess_curve_track(roi, mode="black"):
    """Clean up a curve track ROI: isolate curve color, remove gridlines, thin.
    
    Args:
        roi: BGR image crop of the track
        mode: "black", "red", "blue", or "green"
    
    Returns:
        Binary mask where curve pixels are 255, background is 0
    """
    if roi is None or roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Step 1: Color isolation
    if mode == "black":
        # Low brightness = curve
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        _, curve_mask = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY_INV)
    elif mode == "red":
        b, g, r = cv2.split(roi)
        curve_mask = ((r > 120) & (r > g + 20) & (r > b + 20)).astype(np.uint8) * 255
    elif mode == "blue":
        b, g, r = cv2.split(roi)
        curve_mask = ((b > 120) & (b > r + 20) & (b > g + 20)).astype(np.uint8) * 255
    elif mode == "green":
        b, g, r = cv2.split(roi)
        curve_mask = ((g > 120) & (g > r + 20) & (g > b + 20)).astype(np.uint8) * 255
    else:
        # Fallback: use existing black mask logic
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        _, curve_mask = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Step 2: Remove vertical gridlines
    if h > 15:
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min(15, h // 3)))
        vert_lines = cv2.morphologyEx(curve_mask, cv2.MORPH_OPEN, vertical_kernel)
        curve_mask = cv2.bitwise_and(curve_mask, cv2.bitwise_not(vert_lines))
    
    # Step 3: Remove horizontal gridlines
    if w > 15:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min(15, w // 3), 1))
        horiz_lines = cv2.morphologyEx(curve_mask, cv2.MORPH_OPEN, horizontal_kernel)
        curve_mask = cv2.bitwise_and(curve_mask, cv2.bitwise_not(horiz_lines))
    
    # Step 4: Slight blur to fill 1-pixel gaps
    blurred = cv2.GaussianBlur(curve_mask, (3, 3), 0)
    _, cleaned = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return cleaned


def trace_curve_with_dp(curve_mask, scale_min, scale_max, curve_type="GR", max_step=3, smooth_lambda=0.5):
    """Trace a curve using dynamic programming for smooth path finding.
    
    Args:
        curve_mask: Binary mask (0 or 255) where curve pixels are bright
        scale_min: Left scale value
        scale_max: Right scale value
        curve_type: Curve type for plausibility checks (GR, RHOB, NPHI, DT, etc.)
        max_step: Max horizontal movement per row (pixels)
        smooth_lambda: Smoothness penalty weight
    
    Returns:
        xs: Array of x-coordinates (one per row), with np.nan for low-confidence rows
        confidence: Array of confidence scores (0-1) per row
    """
    if curve_mask is None or curve_mask.size == 0:
        return np.array([]), np.array([])
    
    h, w = curve_mask.shape
    if h < 2 or w < 2:
        return np.full(h, np.nan), np.zeros(h)
    
    # Define plausible value ranges per curve type
    plausible_ranges = {
        'GR': (0, 200),
        'RHOB': (1.5, 3.5),
        'NPHI': (-0.2, 0.6),
        'DT': (40, 200),
        'CALI': (4, 20),
        'SP': (-200, 100),
    }
    
    # Convert mask to probability (0-1)
    prob = curve_mask.astype(np.float32) / 255.0
    cost = 1.0 - prob  # 0 where curve is strong, 1 where nothing
    
    # Add plausibility penalty (reduced from 10.0 to 2.0 to be less aggressive)
    if curve_type.upper() in plausible_ranges:
        pmin, pmax = plausible_ranges[curve_type.upper()]
        for x in range(w):
            # Map x to value
            value = scale_min + (x / max(1, w - 1)) * (scale_max - scale_min)
            if value < pmin or value > pmax:
                # Penalize implausible values (but not too harshly)
                cost[:, x] += 2.0
    
    # Dynamic programming
    big = 1e6
    dp = np.full((h, w), big, dtype=np.float32)
    prev = np.full((h, w), -1, dtype=np.int16)
    
    # First row: cost as-is
    dp[0, :] = cost[0, :]
    
    for y in range(1, h):
        for x in range(w):
            x0 = max(0, x - max_step)
            x1 = min(w, x + max_step + 1)
            best_val = big
            best_xp = -1
            for xp in range(x0, x1):
                smooth_penalty = smooth_lambda * (x - xp) ** 2
                v = dp[y - 1, xp] + cost[y, x] + smooth_penalty
                if v < best_val:
                    best_val = v
                    best_xp = xp
            dp[y, x] = best_val
            prev[y, x] = best_xp
    
    # Find the best ending point: look for minimum cost in bottom 20% of image
    # This helps avoid starting from a bad position
    bottom_start = max(0, int(h * 0.8))
    best_cost = big
    best_y = h - 1
    best_x = 0
    for y in range(bottom_start, h):
        min_x = int(np.argmin(dp[y, :]))
        if dp[y, min_x] < best_cost:
            best_cost = dp[y, min_x]
            best_y = y
            best_x = min_x
    
    # Backtrack from best position
    path_x = np.full(h, -1, dtype=np.int32)
    path_x[best_y] = best_x
    for y in range(best_y, 0, -1):
        if path_x[y] >= 0:
            path_x[y - 1] = prev[y, path_x[y]]
    
    # Forward fill for any rows after best_y
    if best_y < h - 1:
        last_x = best_x
        for y in range(best_y + 1, h):
            # Find best continuation
            x0 = max(0, last_x - max_step)
            x1 = min(w, last_x + max_step + 1)
            best_local = np.argmax(prob[y, x0:x1])
            path_x[y] = x0 + best_local
            last_x = path_x[y]
    
    # Compute confidence per row
    confidence = np.zeros(h, dtype=np.float32)
    for y in range(h):
        x = path_x[y]
        if x < 0 or x >= w:
            confidence[y] = 0.0
            path_x[y] = -1
            continue
            
        p_best = prob[y, x]
        
        # Find second-best probability within max_step
        x0 = max(0, x - max_step)
        x1 = min(w, x + max_step + 1)
        probs_nearby = prob[y, x0:x1]
        if probs_nearby.size > 1:
            probs_sorted = np.sort(probs_nearby)[::-1]
            p_second = probs_sorted[1] if probs_sorted.size > 1 else 0.0
            confidence[y] = p_best - p_second
        else:
            confidence[y] = p_best
        
        # Mark low-confidence as NaN
        if p_best < 0.15:  # Lowered threshold from 0.2 to 0.15
            path_x[y] = -1
    
    # Convert -1 to np.nan
    xs = path_x.astype(np.float32)
    xs[xs < 0] = np.nan
    
    return xs, confidence


def remove_outliers_and_smooth(xs, window=5, outlier_threshold=3.0):
    """Remove isolated spikes and smooth the curve.
    
    Args:
        xs: Array of x-coordinates with possible NaNs
        window: Window size for median smoothing
        outlier_threshold: Number of std deviations for outlier detection
    
    Returns:
        Smoothed array with outliers removed
    """
    if xs is None or xs.size < 3:
        return xs
    
    # Convert to pandas for easier handling
    s = pd.Series(xs)
    
    # Remove outliers: if a point differs from both neighbors by > threshold * std
    valid = ~s.isna()
    if valid.sum() > 3:
        # Compute rolling std
        rolling_std = s[valid].rolling(window, min_periods=2, center=True).std()
        rolling_mean = s[valid].rolling(window, min_periods=2, center=True).mean()
        
        for i in range(1, len(s) - 1):
            if valid.iloc[i]:
                neighbors = [s.iloc[i - 1], s.iloc[i + 1]]
                neighbors_valid = [x for x in neighbors if not pd.isna(x)]
                if len(neighbors_valid) >= 2:
                    mean_neighbor = np.mean(neighbors_valid)
                    std_val = rolling_std.iloc[i] if i < len(rolling_std) and not pd.isna(rolling_std.iloc[i]) else 1.0
                    if abs(s.iloc[i] - mean_neighbor) > outlier_threshold * max(std_val, 1.0):
                        s.iloc[i] = np.nan
    
    # Smooth with median filter
    if window % 2 == 0:
        window += 1
    if window > 1:
        s = s.rolling(window, min_periods=1, center=True).median()
    
    # Interpolate remaining gaps
    s = s.interpolate(limit_direction="both", limit=50)
    
    return s.to_numpy(dtype=np.float32)


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


@app.route('/reanalyze_panel', methods=['POST'])
def reanalyze_panel():
    """Re-run OCR/AI suggestions on a cropped panel region of the current image.

    Expects JSON with:
      - image: data URL string (same as /digitize)
      - region: { left_px, right_px, top_px, bottom_px } in image pixel coords
    """
    data = request.json or {}
    image_data = data.get('image')
    region = data.get('region') or {}

    if not image_data or ',' not in image_data:
        return jsonify({'success': False, 'error': 'Missing image data'}), 400

    try:
        img_payload = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(img_payload)
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid image data'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'error': 'Could not decode image'}), 400

    H, W, _ = img.shape
    try:
        left = max(0, int(region.get('left_px', 0)))
        right = min(W, int(region.get('right_px', W)))
        top = max(0, int(region.get('top_px', 0)))
        bottom = min(H, int(region.get('bottom_px', H)))
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid region'}), 400

    if right <= left or bottom <= top:
        return jsonify({'success': False, 'error': 'Empty region'}), 400

    # Crop to region
    crop = img[top:bottom, left:right]
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode crop'}), 500

    crop_bytes = buf.tobytes()

    # Run Vision OCR on cropped panel
    detected_text = detect_text_vision_api(crop_bytes)
    ocr_suggestions = detected_text.get('suggestions', {}) or {}

    # Attach color hints so curve suggestions stay consistent with panel
    try:
        ocr_suggestions = attach_color_hints_to_ocr_curves(crop, ocr_suggestions)
        detected_text['suggestions'] = ocr_suggestions
    except Exception:
        # If anything goes wrong here, still return basic OCR suggestions
        pass

    return jsonify({
        'success': True,
        'ocr_suggestions': ocr_suggestions,
        'detected_text': detected_text,
    })


@app.route('/propose_calibration', methods=['POST'])
def propose_calibration():
    """Use Vision + LLM to propose depth_axis and track calibration for a selected panel.

    Expects JSON with:
      - image: data URL string
      - region: { left_px, right_px, top_px, bottom_px } in image pixel coords
    """
    data = request.json or {}
    image_data = data.get('image')
    region = data.get('region') or {}

    if not image_data or ',' not in image_data:
        return jsonify({'success': False, 'error': 'Missing image data'}), 400

    try:
        img_payload = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(img_payload)
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid image data'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'error': 'Could not decode image'}), 400

    H, W, _ = img.shape
    try:
        left = max(0, int(region.get('left_px', 0)))
        right = min(W, int(region.get('right_px', W)))
        top = max(0, int(region.get('top_px', 0)))
        bottom = min(H, int(region.get('bottom_px', H)))
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid region'}), 400

    if right <= left or bottom <= top:
        return jsonify({'success': False, 'error': 'Empty region'}), 400

    # Crop to region
    crop = img[top:bottom, left:right]
    crop_h, crop_w, _ = crop.shape
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode crop'}), 500

    crop_bytes = buf.tobytes()

    # Run Vision OCR on cropped panel
    detected_text = detect_text_vision_api(crop_bytes)
    raw_text = detected_text.get('raw', [])
    numeric_entries = detected_text.get('numbers', [])

    if not numeric_entries:
        return jsonify({
            'success': False,
            'error': 'No numeric OCR entries found in selected panel; cannot propose calibration.'
        }), 400

    # Build calibration payload for the LLM
    # 1) Depth label candidates: left-side numeric entries
    xs_all = [float(e['x']) for e in numeric_entries if 'x' in e]
    if xs_all:
        min_x = min(xs_all)
        max_x = max(xs_all)
        span_x = max(max_x - min_x, 1.0)
        depth_x_threshold = min_x + 0.35 * span_x
    else:
        depth_x_threshold = None

    depth_label_candidates = []
    for e in numeric_entries:
        if depth_x_threshold is not None and float(e['x']) <= depth_x_threshold:
            depth_label_candidates.append({
                'value': float(e['value']),
                'x_px': float(e['x']),
                'y_px': float(e['y']),
            })

    # 2) Header text boxes: text in top ~30% of crop
    y_vals_all = [float(e['y']) for e in numeric_entries if 'y' in e]
    if y_vals_all:
        y_min = min(y_vals_all)
        y_max = max(y_vals_all)
        header_threshold = y_min + 0.3 * (y_max - y_min)
    else:
        header_threshold = crop_h * 0.3

    header_text_boxes = []
    for entry in raw_text:
        text = (entry.get('text') or '').strip()
        if not text:
            continue
        verts = entry.get('vertices') or []
        ys = [v.get('y') for v in verts if isinstance(v, dict) and 'y' in v]
        xs = [v.get('x') for v in verts if isinstance(v, dict) and 'x' in v]
        if not ys or not xs:
            continue
        y_center = float(sum(ys)) / len(ys)
        x_center = float(sum(xs)) / len(xs)
        if y_center <= header_threshold:
            header_text_boxes.append({
                'text': text,
                'x_px': x_center,
                'y_px': y_center,
            })

    calib_payload = {
        'image': {
            'width_px': crop_w,
            'height_px': crop_h,
        },
        'depth_label_candidates': depth_label_candidates,
        'header_text_boxes': header_text_boxes,
    }

    # Call LLM to propose calibration
    calibration = call_ai_calibration(calib_payload)
    if not calibration:
        return jsonify({
            'success': False,
            'error': 'AI calibration failed or returned no result. Check server logs.'
        }), 500

    # Validate and fix obvious mismatches
    calibration = validate_and_fix_calibration(calibration)

    return jsonify({
        'success': True,
        'calibration': calibration,
    })


@app.route('/api/auto_layout', methods=['POST'])
def auto_layout_tracks():
    data = request.json or {}
    image_data = data.get('image')
    region = data.get('region') or {}

    if not image_data or ',' not in image_data:
        return jsonify({'success': False, 'error': 'Missing image data'}), 400

    try:
        img_payload = image_data.split(',', 1)[1]
        img_bytes = base64.b64decode(img_payload)
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid image data'}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'error': 'Could not decode image'}), 400

    H, W, _ = img.shape
    try:
        left = max(0, int(region.get('left_px', 0)))
        right = min(W, int(region.get('right_px', W)))
        top = max(0, int(region.get('top_px', 0)))
        bottom = min(H, int(region.get('bottom_px', H)))
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid region'}), 400

    if right <= left or bottom <= top:
        return jsonify({'success': False, 'error': 'Empty region'}), 400

    panel = img[top:bottom, left:right]
    panel_h, panel_w, _ = panel.shape
    if panel_h < 2 or panel_w < 2:
        return jsonify({'success': False, 'error': 'Panel too small for layout detection'}), 400

    header_h = max(10, int(panel_h * 0.15))
    header = panel[0:header_h, :]

    ok, buf = cv2.imencode('.jpg', header, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode header crop'}), 500

    header_bytes = buf.tobytes()

    detected_text = detect_text_vision_api(header_bytes)
    raw_text = detected_text.get('raw', []) or []

    items = []
    for entry in raw_text:
        text = (entry.get('text') or '').strip()
        if not text:
            continue
        verts = entry.get('vertices') or []
        xs = [v.get('x') for v in verts if isinstance(v, dict) and 'x' in v]
        ys = [v.get('y') for v in verts if isinstance(v, dict) and 'y' in v]
        if not xs or not ys:
            continue
        x_center = float(sum(xs)) / len(xs)
        y_center = float(sum(ys)) / len(ys)
        items.append({
            'text': text,
            'x': x_center,
            'y': y_center,
        })

    if not items:
        return jsonify({'success': False, 'error': 'No header text items found for layout detection.'}), 400

    layout_payload = {
        'image': {
            'width_px': panel_w,
            'height_px': header_h,
        },
        'items': items,
    }

    layout = call_ai_auto_layout(layout_payload)
    if not layout:
        return jsonify({
            'success': False,
            'error': 'AI layout detection failed or returned no result. Check server logs.'
        }), 500

    raw_tracks = layout.get('tracks') or []
    tracks_out = []
    for t in raw_tracks:
        try:
            lx = float(t.get('left_x'))
            rx = float(t.get('right_x'))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lx) or not np.isfinite(rx):
            continue
        lx = max(0.0, min(float(panel_w), lx))
        rx = max(0.0, min(float(panel_w), rx))
        if rx <= lx:
            continue

        track_out = {
            'name': t.get('name'),
            'left_px': float(left) + lx,
            'right_px': float(left) + rx,
            'scale_min': t.get('scale_min'),
            'scale_max': t.get('scale_max'),
            'unit': t.get('unit'),
            'hot_side': t.get('hot_side'),
        }
        tracks_out.append(track_out)

    if not tracks_out:
        return jsonify({'success': False, 'error': 'AI layout returned no usable tracks.'}), 400

    return jsonify({
        'success': True,
        'tracks': tracks_out,
        'raw_layout': layout,
    })


def build_ocr_suggestions(numeric_entries):
    """Derive depth and curve hints from numeric OCR entries."""
    if not numeric_entries:
        return {}

    # Sort by y (top to bottom)
    sorted_entries = sorted(numeric_entries, key=lambda n: n['y'])

    depth_candidates = []
    curve_candidates = []

    # Estimate horizontal extent of all numeric text so we can treat roughly
    # the left ~35% as potential depth-scale labels and the rest as curve
    # values. This mirrors the reference approach that only uses left-side
    # numbers for depth.
    xs_all = [float(e['x']) for e in sorted_entries if 'x' in e]
    if xs_all:
        min_x = min(xs_all)
        max_x = max(xs_all)
        span_x = max(max_x - min_x, 1.0)
        depth_x_threshold = min_x + 0.35 * span_x
    else:
        depth_x_threshold = None

    for entry in sorted_entries:
        value = entry['value']
        y = entry['y']
        x = entry['x']

        use_as_depth = False
        if depth_x_threshold is not None:
            use_as_depth = float(x) <= depth_x_threshold

        if use_as_depth:
            depth_candidates.append({'value': value, 'y': y})
        else:
            curve_candidates.append({'value': value, 'x': x, 'y': y})

    depth_hint = None
    if len(depth_candidates) >= 2:
        # Filter out obviously non-depth values (e.g., huge magnitudes that would
        # produce spans like 700000 to -200000). If filtering removes everything,
        # fall back to the original set.
        filtered = [d for d in depth_candidates if abs(d['value']) <= 100000]
        use_entries = filtered if len(filtered) >= 2 else depth_candidates

        # Use detected depth labels to fit a simple linear scale
        ys = np.array([d['y'] for d in use_entries], dtype=np.float32)
        vals = np.array([d['value'] for d in use_entries], dtype=np.float32)

        try:
            # depth  a * pixel_y + b
            a, b = np.polyfit(ys, vals, 1)
            y_top = float(ys.min())
            y_bottom = float(ys.max())
            top_depth_fit = float(a * y_top + b)
            bottom_depth_fit = float(a * y_bottom + b)

            # Ensure we have a sensible span and that depth is reasonably
            # monotonic with pixel position.
            span_val = abs(bottom_depth_fit - top_depth_fit)
            max_span = 100000.0  # reject clearly insane ranges
            min_span = 5.0       # avoid noise from tiny spans
            ok_span = (span_val >= min_span and span_val <= max_span)

            corr = np.corrcoef(ys, vals)[0, 1] if ys.size >= 2 else 1.0
            ok_corr = np.isfinite(corr) and abs(corr) >= 0.9

            ok_magnitude = all(abs(v) <= 1e6 for v in (top_depth_fit, bottom_depth_fit))

            if y_bottom > y_top and top_depth_fit != bottom_depth_fit and ok_span and ok_corr and ok_magnitude:
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
            # Fallback to using just the first/last labels if fitting fails,
            # but still reject clearly unreasonable spans.
            top = depth_candidates[0]
            bottom = depth_candidates[-1]
            span_val = abs(bottom['value'] - top['value'])
            if (
                bottom['y'] > top['y']
                and bottom['value'] != top['value']
                and span_val >= 5.0
                and span_val <= 100000.0
                and all(abs(v) <= 1e6 for v in (top['value'], bottom['value']))
            ):
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

    # Try to refine depths using header/table information. Many logs print a
    # small table with "Top" / "Bottom" / "Total depth" values in ft near the
    # top of the page. We look only in that header band and only at numbers that
    # look like depths in feet (e.g. 4449.90 ft, 10026.53 ft), ignoring other
    # units like us/ft or lbf.
    header_top_val = None
    header_bottom_val = None
    if sorted_entries:
        y_vals_all = [e['y'] for e in sorted_entries]
        if y_vals_all:
            y_min = min(y_vals_all)
            y_max = max(y_vals_all)
            # Top 25% of text as a rough "header" band
            band_cut = y_min + 0.25 * (y_max - y_min)

            header_depth_vals_strict = []  # require explicit "ft" (not us/ft)
            header_depth_vals_loose = []   # any plausible depth magnitude
            for e in sorted_entries:
                if e['y'] > band_cut:
                    continue
                text_l = str(e.get('text', '')).lower()
                val = e.get('value')
                if not np.isfinite(val):
                    continue
                # Depths in feet are typically hundreds to tens of thousands of
                # units, not tiny (0.0) and not enormous.
                if abs(val) < 100 or abs(val) > 50000:
                    continue

                header_depth_vals_loose.append(float(val))

                # Strict: same token mentions "ft" but not sonic units.
                if 'ft' in text_l and 'us/ft' not in text_l and 'ms/ft' not in text_l:
                    header_depth_vals_strict.append(float(val))

            use_vals = header_depth_vals_strict
            if not use_vals and len(header_depth_vals_loose) >= 2:
                # Fallback: no explicit "ft" nearby, but we do see multiple
                # plausible depth values in the header band. Use the min/max
                # only if they span a reasonable interval so we do not mistake
                # small curve scales (e.g. 1.95–2.95) for depths.
                header_depth_vals_loose.sort()
                span = header_depth_vals_loose[-1] - header_depth_vals_loose[0]
                if span >= 200.0:  # require at least a few hundred feet span
                    use_vals = header_depth_vals_loose

            if use_vals:
                use_vals.sort()
                if len(use_vals) >= 2:
                    header_top_val = use_vals[0]
                    header_bottom_val = use_vals[-1]
                else:
                    # Only a single header depth (e.g. "Total Depth @ 10015 ft")
                    header_top_val = header_bottom_val = use_vals[0]

    if depth_hint and header_top_val is not None and header_bottom_val is not None:
        # Override the fitted depths so they match the header values, while
        # preserving the original orientation (increasing or decreasing depth).
        d_top = float(depth_hint['top_depth'])
        d_bottom = float(depth_hint['bottom_depth'])
        if d_top <= d_bottom:
            depth_hint['top_depth'] = header_top_val
            depth_hint['bottom_depth'] = header_bottom_val
        else:
            depth_hint['top_depth'] = header_bottom_val
            depth_hint['bottom_depth'] = header_top_val

    # If we could not infer a depth scale from the physical log but we do have
    # plausible header depths, still provide a depth_hint so the UI can
    # auto-fill the top/bottom depth values. Pixel positions (top_px,
    # bottom_px) will remain unchanged in that case.
    if depth_hint is None and header_top_val is not None and header_bottom_val is not None:
        depth_hint = {
            'top_depth': header_top_val,
            'bottom_depth': header_bottom_val,
        }

    suggestions = {}
    if depth_hint:
        suggestions['depth'] = depth_hint
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
    elif pix_span > 0:
        depth_per_pixel = depth_span / pix_span
        # Heuristic similar to the reference compute_depth_scale usage:
        # flag unusual but not impossible scales outside ~0.1–10 depth
        # units per pixel so the user can double-check anchors.
        if abs(depth_per_pixel) < 0.1 or abs(depth_per_pixel) > 10.0:
            warnings.append(
                f"Unusual depth scale (~{depth_per_pixel:.2f} depth units per pixel). Check anchors."
            )

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


def select_primary_track_region(tracks, image_width):
    """Cluster tracks into horizontal panels and select the best one.

    This is used when the TIFF contains multiple side-by-side copies of the
    same log. We group tracks by gaps between their horizontal centers and
    then pick the widest, densest panel as the default region to use.
    """
    if not tracks:
        return None

    centers = []
    widths = []
    for left, right in tracks:
        try:
            l = int(left)
            r = int(right)
        except Exception:
            continue
        width = max(1, r - l)
        widths.append(width)
        centers.append(0.5 * (l + r))

    if not widths or not centers:
        return None

    # Sort tracks by horizontal center and compute gaps between neighbors.
    sorted_indices = sorted(range(len(tracks)), key=lambda i: centers[i])
    gaps = []
    for idx in range(len(sorted_indices) - 1):
        c0 = centers[sorted_indices[idx]]
        c1 = centers[sorted_indices[idx + 1]]
        gaps.append(c1 - c0)

    if gaps:
        median_gap = float(np.median(gaps))
        gap_threshold = max(median_gap * 2.5, 40.0)
    else:
        gap_threshold = max(int(image_width * 0.25), 40)

    panels = []
    current = []
    last_center = None
    for idx in sorted_indices:
        center = centers[idx]
        if last_center is not None and (center - last_center) > gap_threshold and current:
            panels.append(current)
            current = []
        current.append(idx)
        last_center = center
    if current:
        panels.append(current)

    if not panels:
        return None

    best_panel = None
    best_score = None
    best_left = None
    best_right = None
    for panel in panels:
        left_vals = []
        right_vals = []
        for i in panel:
            try:
                l = int(tracks[i][0])
                r = int(tracks[i][1])
            except Exception:
                continue
            left_vals.append(l)
            right_vals.append(r)
        if not left_vals or not right_vals:
            continue
        left = min(left_vals)
        right = max(right_vals)
        total_width = max(1, right - left)
        score = total_width * len(panel)
        if best_panel is None or score > best_score:
            best_panel = panel
            best_score = score
            best_left = left
            best_right = right

    if best_panel is None:
        return None

    return {
        "left_px": int(best_left),
        "right_px": int(best_right),
        "track_indices": best_panel,
    }

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html', app_version=APP_VERSION, build_time=APP_BUILD_TIME)


@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon to prevent 404 errors."""
    return '', 204

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

    # If multiple panels are present, pick the "best" region of tracks as a hint,
    # but still return all tracks so the user can manually choose.
    primary_region = select_primary_track_region(tracks, w)

    # OCR is now deferred until after the user selects a specific panel.
    # We return empty placeholders here and let /reanalyze_panel do the actual
    # Vision OCR work on the cropped region the user cares about.
    detected_text = {'raw': [], 'numbers': [], 'suggestions': {}}
    ocr_suggestions = {}

    return jsonify({
        'success': True,
        'image': f'data:image/png;base64,{img_base64}',
        'width': w,
        'height': h,
        'tracks': tracks,
        'all_tracks': tracks,
        'primary_region': {
            'left_px': primary_region.get('left_px'),
            'right_px': primary_region.get('right_px'),
            'track_indices': primary_region.get('track_indices'),
        } if primary_region else None,
        'detected_text': detected_text,
        'ocr_suggestions': ocr_suggestions or detected_text.get('suggestions', {}),
        'vision_api_available': bool(VISION_API_AVAILABLE)
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
        
        # NEW: Use advanced preprocessing to clean the curve track
        mask = preprocess_curve_track(roi, mode=mode)
        
        # NEW: Use DP-based smooth path tracing with plausibility checks
        curve_type = c.get('type', 'GR')  # Get curve type for plausibility
        xs, confidence = trace_curve_with_dp(
            mask, 
            scale_min=left_value, 
            scale_max=right_value,
            curve_type=curve_type,
            max_step=3,
            smooth_lambda=0.5
        )
        
        # NEW: Remove outliers and smooth
        xs = remove_outliers_and_smooth(xs, window=smooth_window, outlier_threshold=3.0)

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

    # Build AI analysis payload (OCR + LAS stats + user curve config)
    ai_payload = build_ai_analysis_payload(las_content, detected_text, curves)
    ai_summary = call_hf_curve_analysis(ai_payload) if ai_payload else None

    # Prepare digitized vectors for frontend cursor readout
    try:
        digitized_depth = las_depth.tolist()
        digitized_curves = {
            name: {
                "unit": meta.get("unit", ""),
                "values": (meta.get("values") or []).tolist(),
            }
            for name, meta in las_curve_data.items()
        }
    except Exception:
        digitized_depth = None
        digitized_curves = None

    return jsonify({
        'success': True,
        'las_content': las_content,
        'filename': 'digitized_log.las',
        'validation': validation,
        'outlier_warnings': outlier_warnings,
        'depth_warnings': depth_warnings,
        'curve_traces': curve_traces,
        'ai_payload': ai_payload,
        'ai_summary': ai_summary,
        'digitized_depth': digitized_depth,
        'digitized_curves': digitized_curves,
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'vision_api': VISION_API_AVAILABLE
    })


@app.route('/debug-env')
def debug_env():
    """Debug endpoint to check environment variable configuration."""
    return jsonify({
        'HF_API_TOKEN': 'set' if HF_API_TOKEN else 'missing',
        'HF_MODEL_ID': HF_MODEL_ID or 'missing',
        'OPENAI_API_KEY': 'set' if OPENAI_API_KEY else 'missing',
        'OPENAI_MODEL_ID': OPENAI_MODEL_ID or 'missing',
        'GEMINI_API_KEY': 'set' if GEMINI_API_KEY else 'missing',
        'GEMINI_MODEL_ID': GEMINI_MODEL_ID or 'missing',
        'VISION_API_AVAILABLE': VISION_API_AVAILABLE,
        'GOOGLE_VISION_CREDENTIALS_JSON': 'set' if os.getenv('GOOGLE_VISION_CREDENTIALS_JSON') else 'missing',
        'GOOGLE_APPLICATION_CREDENTIALS': 'set' if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') else 'missing'
    })


@app.route('/test-ai')
def test_ai():
    """Test endpoint to verify Hugging Face API is working."""
    # Prefer Gemini if configured
    if GEMINI_API_KEY and GEMINI_MODEL_ID:
        try:
            # Use REST API directly to avoid SDK version issues
            # Model ID should include 'models/' prefix (e.g., 'models/gemini-2.0-flash')
            model_name = GEMINI_MODEL_ID if GEMINI_MODEL_ID.startswith('models/') else f'models/{GEMINI_MODEL_ID}'
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": "What is 2+2?"}]}]
            }
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        text = parts[0].get('text', '')
                        return jsonify({
                            'success': True,
                            'status_code': 200,
                            'provider': 'gemini',
                            'model': model_name,
                            'response': text,
                        })
            return jsonify({
                'success': False,
                'provider': 'gemini',
                'model': model_name,
                'error': f"{resp.status_code} {resp.text}",
            })
        except Exception as exc:
            return jsonify({
                'success': False,
                'provider': 'gemini',
                'model': GEMINI_MODEL_ID,
                'error': str(exc),
            })

    # Fallback to OpenAI if configured
    if OPENAI_API_KEY and OPENAI_MODEL_ID:
        try:
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL_ID,
                messages=[
                    {"role": "user", "content": "What is 2+2?"},
                ],
                max_tokens=50,
                temperature=0.3,
            )
            choices = resp.get("choices") or []
            content = ""
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content") or ""
            return jsonify({
                'success': True,
                'status_code': 200,
                'provider': 'openai',
                'model': OPENAI_MODEL_ID,
                'response': content
            })
        except Exception as exc:
            return jsonify({
                'success': False,
                'provider': 'openai',
                'model': OPENAI_MODEL_ID,
                'error': str(exc),
            })

    if not HF_API_TOKEN or not HF_MODEL_ID:
        return jsonify({
            'success': False,
            'error': 'No AI provider configured (missing Gemini/OpenAI/HF credentials).',
            'HF_API_TOKEN': 'set' if HF_API_TOKEN else 'missing',
            'HF_MODEL_ID': HF_MODEL_ID or 'missing'
        })

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        return jsonify({
            'success': False,
            'error': f'InferenceClient init error: {str(exc)}',
            'model': HF_MODEL_ID
        })

    try:
        out = client.text_generation(
            "What is 2+2?",
            model=HF_MODEL_ID,
            max_new_tokens=50,
            temperature=0.3,
        )
        return jsonify({
            'success': True,
            'status_code': 200,
            'provider': 'hf-inference',
            'model': HF_MODEL_ID,
            'response': out if isinstance(out, str) else str(out)
        })
    except Exception as exc:
        return jsonify({
            'success': False,
            'provider': 'hf-inference',
            'error': str(exc),
            'model': HF_MODEL_ID
        })


@app.route('/list-gemini-models')
def list_gemini_models():
    """List available Gemini models from the API."""
    if not GEMINI_API_KEY:
        return jsonify({'error': 'GEMINI_API_KEY not set'}), 400
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models?key={GEMINI_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get('models', [])
            # Filter for models that support generateContent
            generate_models = [
                {
                    'name': m.get('name', ''),
                    'displayName': m.get('displayName', ''),
                    'supportedMethods': m.get('supportedGenerationMethods', [])
                }
                for m in models
                if 'generateContent' in m.get('supportedGenerationMethods', [])
            ]
            return jsonify({
                'success': True,
                'models': generate_models,
                'total': len(generate_models)
            })
        else:
            return jsonify({
                'success': False,
                'error': f"{resp.status_code} {resp.text}"
            })
    except Exception as exc:
        return jsonify({
            'success': False,
            'error': str(exc)
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
        return jsonify({'success': False, 'error': 'AI chat is not configured. Please set GEMINI_API_KEY, OPENAI_API_KEY, or HF_API_TOKEN in your environment.'}), 500
    
    # If answer contains error message from AI API, still return success but show the error
    return jsonify({'success': True, 'answer': answer})

if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting TIFF→LAS Web App")
    print(f"📊 Google Vision API: {'✅ Available' if VISION_API_AVAILABLE else '⚠️  Not configured'}")
    print("🌐 Open: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
