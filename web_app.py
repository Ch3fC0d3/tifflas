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
# Vision API is optional; initialize later after env vars are loaded.
VISION_API_AVAILABLE = False
vision_client = None

# Load environment variables from .env and .env.local
from dotenv import load_dotenv
load_dotenv()  # Load .env
load_dotenv('.env.local', override=True)  # Load .env.local (overrides .env)

from flask import Flask, render_template, request, jsonify, send_file, Response, session, redirect, url_for, flash
import math
import os
import random
import re
import shutil
import string
import tempfile
import textwrap
import time
import heapq
from collections import defaultdict
import cv2
import numpy as np
import fast_tracer
import pandas as pd
import json
from io import BytesIO, StringIO
import base64
import zipfile
from typing import Dict, List, Tuple, Optional
import tempfile
from datetime import datetime
import uuid
from pathlib import Path
import requests
import openai
from huggingface_hub import InferenceClient

TORCH_AVAILABLE = False
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None

# Phase 1 & 2: Learning system imports
from user_tracker import tracker
from parameter_learner import ParameterLearner
from ai_tracer import AITracer

# Initialize learning system after all imports
learner = ParameterLearner(tracker)
import hashlib

# Initialize AI tracer
ai_tracer = AITracer("curve_trace_model.pt")

# Try to import Google Vision API (optional)
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
        # Local development: JSON file path in env var
        vision_client = vision.ImageAnnotatorClient()
        VISION_API_AVAILABLE = True
        print("✅ Google Vision API: Loaded from file")
    else:
        # Auto-detect key file in project directory
        _local_key = Path(__file__).parent / 'GOOGLE_APPLICATION_CREDENTIALS.json'
        if _local_key.exists():
            credentials = service_account.Credentials.from_service_account_file(str(_local_key))
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            VISION_API_AVAILABLE = True
            print(f"✅ Google Vision API: Auto-loaded from {_local_key.name}")
        else:
            print("⚠️  Google Vision API: No credentials found")
            vision_client = None
            VISION_API_AVAILABLE = False
except ImportError:
    print("⚠️  Google Vision API not available. Install: pip install google-cloud-vision")
    vision_client = None
    VISION_API_AVAILABLE = False
except Exception as e:
    print(f"⚠️  Google Vision API error: {e}")
    vision_client = None
    VISION_API_AVAILABLE = False

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

try:
    CURVE_TRACE_UPSCALE = float(os.environ.get("CURVE_TRACE_UPSCALE", "2.0"))
except Exception:
    CURVE_TRACE_UPSCALE = 2.0
CURVE_TRACE_UPSCALE = max(1.0, min(4.0, CURVE_TRACE_UPSCALE))

APP_VERSION = os.environ.get("APP_VERSION", "dev")
APP_BUILD_TIME = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max request size
app.secret_key = os.environ.get("SECRET_KEY", "tiflas-dev-secret-key-change-in-prod")

# ----------------------------
# Auth Decorator
# ----------------------------
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function



@app.errorhandler(500)
def _handle_internal_server_error(exc):
    if request.path == '/digitize':
        import traceback
        original = getattr(exc, 'original_exception', None)
        if original is not None:
            print(f"/digitize 500 error: {original}")
            print(traceback.format_exc())
            err_msg = str(original)
        else:
            print(f"/digitize 500 error: {exc}")
            err_msg = str(exc)

        tb = traceback.format_exc()
        tb_lines = tb.splitlines()[-25:]
        tb_short = "\n".join(tb_lines)

        return jsonify({
            'success': False,
            'error': err_msg,
            'traceback': tb_short,
        }), 500

    return jsonify({
        'success': False,
        'error': 'Internal server error',
    }), 500

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
        "centers in pixels, and a 'full_text' block containing all recognized text.\n\n"
        "Your job is to:\n"
        "1. Infer the logging TRACKS present across the width of the header.\n"
        "2. Extract generic HEADER METADATA (Well, Company, API, etc.) from the 'full_text'.\n\n"
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
        "  ],\n"
        "  \"header_metadata\": {\n"
        "    \"well\": string or null,\n"
        "    \"company\": string or null,\n"
        "    \"api\": string or null,\n"
        "    \"date\": string or null,\n"
        "    \"field\": string or null,\n"
        "    \"location\": string or null,\n"
        "    \"county\": string or null,\n"
        "    \"state\": string or null,\n"
        "    \"province\": string or null,\n"
        "    \"service_company\": string or null\n"
        "  }\n"
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
        "- For header_metadata, look for text items like 'WELL:', 'COMPANY:', 'API:', etc.\n"
        "  and try to associate the value next to them. If not found, use null.\n"
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

    # Step 5: Remove near-solid vertical "spines" that look like grid/border lines.
    # These are typically track borders or grid lines, not the actual log curve,
    # and they can cause the DP tracer to hug the wrong column.
    if h >= 40 and w >= 4:
        col_fraction = np.mean(cleaned > 0, axis=0)  # fraction of rows that are "on" per column
        edge_margin = max(1, int(0.15 * w))
        edge_mask = np.zeros_like(col_fraction, dtype=bool)
        edge_mask[:edge_margin] = True
        edge_mask[-edge_margin:] = True

        # Strong spines near the left/right edges (likely track borders)
        # Use 0.90 threshold: true borders span ~100% of height; slow curves (RHOB/DTC)
        # can occupy 40-80% of one column and must not be removed.
        edge_spines = (col_fraction > 0.90) & edge_mask

        # Very strong vertical spines anywhere inside the band.
        interior_spines = (col_fraction > 0.90) & ~edge_mask

        spine_cols = edge_spines | interior_spines
        if np.any(spine_cols):
            cleaned[:, spine_cols] = 0

    return cleaned


def detect_dominant_curve_hue(roi_bgr, sample_fraction=0.3):
    """Detect the dominant hue of the curve in a sample region.
    
    Samples the middle portion of the image (where the curve is likely to be)
    and finds the most common saturated hue, excluding near-white/black pixels.
    
    Returns:
        (hue_center, hue_range) or None if no dominant hue found
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    
    h, w = roi_bgr.shape[:2]
    if h < 10 or w < 10:
        return None
    
    # Sample the middle portion of the image (where curve likely is)
    x_start = int(w * (0.5 - sample_fraction / 2))
    x_end = int(w * (0.5 + sample_fraction / 2))
    sample_region = roi_bgr[:, x_start:x_end]
    
    hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
    
    # Filter for saturated, non-white, non-black pixels (likely curve pixels)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    
    # Curve pixels: reasonably saturated and not too dark or bright
    curve_mask = (saturation > 50) & (value > 40) & (value < 240)
    
    if np.sum(curve_mask) < 100:
        return None
    
    # Get hues of curve pixels
    curve_hues = hue[curve_mask]
    
    # Build histogram of hues (0-180 in OpenCV)
    hist, bins = np.histogram(curve_hues, bins=36, range=(0, 180))
    
    # Find the dominant hue bin (excluding very low counts)
    threshold = np.max(hist) * 0.3
    dominant_bins = np.where(hist > threshold)[0]
    
    if len(dominant_bins) == 0:
        return None
    
    # Use the bin with highest count
    peak_bin = np.argmax(hist)
    hue_center = (bins[peak_bin] + bins[peak_bin + 1]) / 2
    
    # Adaptive hue range based on how spread the hues are
    hue_std = np.std(curve_hues)
    hue_range = max(10, min(25, hue_std * 2))
    
    return (float(hue_center), float(hue_range))


def apply_local_contrast_normalization(img_bgr):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    faded sections while preserving color information.
    
    This normalizes brightness in small windows so faded curve sections
    don't get lost during detection.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    
    # Convert to LAB color space (L = lightness, A/B = color)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to the L (lightness) channel only
    # clipLimit controls contrast amplification (lower = less noise amplification)
    # tileGridSize controls the window size for local normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    # Merge back and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # SATURATION BOOST: "Up the contrast" for color
    # Make faint ink look vibrant
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.5) # Boost saturation by 50%
    s = np.clip(s, 0, 255).astype(np.uint8)
    enhanced_bgr = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    return enhanced_bgr


def remove_grid_lines_aggressive(gray_img, aggressive=True):
    """
    Aggressively detect and remove grid lines from black and white log images.
    Returns a mask with grid lines removed.
    
    Args:
        gray_img: Grayscale image
        aggressive: If True, use very aggressive grid detection
    """
    if gray_img is None or gray_img.size == 0:
        return gray_img
    
    h, w = gray_img.shape[:2]
    if h < 20 or w < 20:
        return gray_img
    
    # Create a copy to work with
    result = gray_img.copy()
    
    # Detect vertical lines (most common in grid)
    if aggressive:
        # Very aggressive vertical line detection
        v_kernel_size = max(15, min(80, h // 3))  # Larger kernel for aggressive detection
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        v_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, v_kernel)
        
        # Detect horizontal lines
        h_kernel_size = max(15, min(80, w // 3))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        h_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, h_kernel)
        
        # Combine detected lines
        grid_lines = cv2.bitwise_or(v_lines, h_lines)
        
        # Dilate slightly to ensure complete removal
        dilate_kernel = np.ones((2, 2), np.uint8)
        grid_lines = cv2.dilate(grid_lines, dilate_kernel, iterations=1)
        
        # Remove grid lines from original
        result = cv2.subtract(result, grid_lines)
    else:
        # Standard grid removal
        v_kernel_size = max(10, min(60, h // 2))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        v_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, v_kernel)
        
        h_kernel_size = max(10, min(60, w // 2))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        h_lines = cv2.morphologyEx(result, cv2.MORPH_OPEN, h_kernel)
        
        grid_lines = cv2.bitwise_or(v_lines, h_lines)
        result = cv2.subtract(result, grid_lines)
    
    return result


def detect_if_black_and_white_log(roi_bgr):
    """
    Auto-detect if an image is a black and white log (vs colored).
    Returns True if the image appears to be black and white.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False
    
    try:
        # Convert to HSV and check saturation
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        
        # Calculate percentage of low-saturation pixels
        low_sat_pixels = np.sum(saturation < 30)
        total_pixels = saturation.size
        low_sat_ratio = low_sat_pixels / max(1, total_pixels)
        
        # If >90% of pixels have low saturation, it's likely black and white
        return low_sat_ratio > 0.90
    except Exception:
        return False


def enhance_curve_roi(roi_bgr):
    """
    Apply lightweight denoise + horizontal super-resolution to a curve ROI.
    Returns (processed_roi, horizontal_scale_factor).
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr, 1.0
    proc = roi_bgr
    scale = 1.0
    try:
        proc = cv2.bilateralFilter(proc, d=5, sigmaColor=25, sigmaSpace=12)
    except Exception:
        pass
    if CURVE_TRACE_UPSCALE > 1.01 and proc.shape[1] >= 2:
        h, w = proc.shape[:2]
        new_w = max(2, int(round(w * CURVE_TRACE_UPSCALE)))
        try:
            proc = cv2.resize(proc, (new_w, h), interpolation=cv2.INTER_CUBIC)
            scale = new_w / max(1, w)
        except Exception:
            scale = 1.0
    try:
        # Mild sharpening to boost ink contrast after upscaling
        blur = cv2.GaussianBlur(proc, (0, 0), sigmaX=0.8)
        proc = cv2.addWeighted(proc, 1.2, blur, -0.2, 0)
    except Exception:
        pass
    return proc, scale


def suppress_grid_hough(gray, h_thresh_ratio=0.25, v_thresh_ratio=0.25):
    """
    Use Probabilistic Hough Transform to detect and remove long straight grid lines
    while preserving jagged curve data.
    """
    if gray is None:
        return gray
        
    h, w = gray.shape
    # Pre-process for edge detection
    # Use adaptive threshold to get binary edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    # 1. Horizontal Lines
    min_len_h = int(w * h_thresh_ratio)
    lines_h = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=min_len_h, maxLineGap=10)
    
    # 2. Vertical Lines
    min_len_v = int(h * v_thresh_ratio)
    lines_v = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=min_len_v, maxLineGap=10)
    
    # Mask to draw the lines to be removed
    grid_mask = np.zeros_like(gray)
    
    if lines_h is not None:
        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2-y1, x2-x1) * 180 / np.pi)
            # Strict horizontal check (+/- 2 degrees)
            if angle < 2 or angle > 178:
                cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 2)
                
    if lines_v is not None:
        for line in lines_v:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2-y1, x2-x1) * 180 / np.pi)
            # Strict vertical check (+/- 2 degrees from 90)
            if 88 < angle < 92:
                cv2.line(grid_mask, (x1, y1), (x2, y2), 255, 2)
                
    # Dilate mask slightly to clean up edge artifacts
    grid_mask = cv2.dilate(grid_mask, np.ones((3,3), np.uint8), iterations=1)
    
    # Inpaint/Erase grid lines (set to white)
    cleaned = gray.copy()
    cleaned[grid_mask > 0] = 255
    
    return cleaned


def compute_prob_map(roi_bgr, mode="black", ui_filters=None, _dual_polarity_allowed=True):
    """Build a soft probability map for the curve in a track ROI.

    Returns an 8-bit image (0–255) where higher values mean higher likelihood
    of belonging to the curve. This can be fed directly to the existing
    DP tracer, which internally rescales to [0, 1].
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    h, w = roi_bgr.shape[:2]
    if h < 2 or w < 2:
        return np.zeros((h, w), dtype=np.uint8)

    ui_filters = ui_filters or {}
    try:
        use_contrast = bool(ui_filters.get('contrast'))
        use_invert = bool(ui_filters.get('invert'))
    except Exception:
        use_contrast = False
        use_invert = False

    if _dual_polarity_allowed and mode == "auto" and use_invert:
        try:
            ui_filters_no_invert = dict(ui_filters)
        except Exception:
            ui_filters_no_invert = {}
        ui_filters_no_invert['invert'] = False
        try:
            inv_roi = cv2.bitwise_not(roi_bgr)
        except Exception:
            inv_roi = np.clip(255 - roi_bgr, 0, 255).astype(np.uint8)
        prob_a = compute_prob_map(
            roi_bgr,
            mode=mode,
            ui_filters=ui_filters_no_invert,
            _dual_polarity_allowed=False,
        )
        prob_b = compute_prob_map(
            inv_roi,
            mode=mode,
            ui_filters=ui_filters_no_invert,
            _dual_polarity_allowed=False,
        )
        try:
            if prob_a is None:
                return prob_b
            if prob_b is None:
                return prob_a
            if prob_a.shape != prob_b.shape:
                return prob_a
            return np.maximum(prob_a, prob_b)
        except Exception:
            return prob_a

    roi_pre = roi_bgr
    if use_invert and mode == "auto":
        try:
            roi_pre = cv2.bitwise_not(roi_pre)
        except Exception:
            roi_pre = np.clip(255 - roi_pre, 0, 255).astype(np.uint8)
    if use_contrast:
        try:
            alpha = 2.0
            roi_f = roi_pre.astype(np.float32)
            roi_f = (roi_f - 128.0) * alpha + 128.0
            roi_pre = np.clip(roi_f, 0, 255).astype(np.uint8)
        except Exception:
            pass

    roi_enhanced = apply_local_contrast_normalization(roi_pre)
    
    # Use enhanced image for HSV and grayscale conversion
    hsv = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2GRAY)
    
    # Also apply CLAHE directly to grayscale for edge detection
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 1) Base color/intensity mask
    # For colored modes, try to detect the actual curve hue for better tracking
    # Use enhanced image for better hue detection in faded areas
    colored_modes = {"green", "red", "blue", "auto", "cyan", "magenta", "yellow", "orange", "purple"}
    detected_hue = None
    if mode in colored_modes:
        detected_hue = detect_dominant_curve_hue(roi_enhanced)
    
    if mode == "green":
        # PERMISSIVE green detection again - we need to catch the faint tips
        # We'll rely on the tracer logic to ignore grid noise
        # Increased saturation min from 30 to 50 to reject gray vertical lines
        lower = np.array([25, 50, 30], dtype=np.uint8)
        upper = np.array([95, 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower, upper)
        
        # Suppress red/orange pixels
        b, g, r = cv2.split(roi_enhanced)
        r16 = r.astype(np.int16)
        g16 = g.astype(np.int16)
        b16 = b.astype(np.int16)
        
        # Only suppress clearly red pixels
        clearly_red = (r16 > g16 + 30) & (r16 > b16 + 30)
        color_mask[clearly_red] = 0
        
        # Weak G-dominance check (allow if G is just slightly higher or equal)
        g_dominant = (g16 >= r16 - 5) & (g16 >= b16 - 5)
        color_mask[~g_dominant] = 0

    elif mode == "auto":
        # Auto-detect the curve hue and track it
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            h_lo = max(0, int(hue_center - hue_range))
            h_hi = min(180, int(hue_center + hue_range))
            lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, lower, upper)
            
            # Refine with median hue from detected pixels
            nonzero = np.nonzero(color_mask)
            if len(nonzero[0]) > 50:
                h_channel = hsv[:, :, 0]
                valid_h = h_channel[nonzero]
                med_h = float(np.median(valid_h))
                band = 12.0
                h_lo = max(0, int(med_h - band))
                h_hi = min(180, int(med_h + band))
                dyn_lower = np.array([h_lo, 45, 45], dtype=np.uint8)
                dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
                color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)
        else:
            # Fallback to detecting any saturated colored pixels
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            color_mask = ((saturation > 50) & (value > 40) & (value < 240)).astype(np.uint8) * 255

    elif mode == "red":
        # Red wraps around hue 0/180, so handle both ends
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            # Only use detected hue if it's in the red-ish range (0-15 or 160-180)
            if hue_center <= 20 or hue_center >= 155:
                # Use detected hue with adaptive range
                if hue_center <= 20:
                    h_lo = max(0, int(hue_center - hue_range))
                    h_hi = min(25, int(hue_center + hue_range))
                    lower1 = np.array([h_lo, 60, 40], dtype=np.uint8)
                    upper1 = np.array([h_hi, 255, 255], dtype=np.uint8)
                    color_mask = cv2.inRange(hsv, lower1, upper1)
                else:
                    h_lo = max(150, int(hue_center - hue_range))
                    h_hi = min(180, int(hue_center + hue_range))
                    lower2 = np.array([h_lo, 60, 40], dtype=np.uint8)
                    upper2 = np.array([h_hi, 255, 255], dtype=np.uint8)
                    color_mask = cv2.inRange(hsv, lower2, upper2)
            else:
                # Detected hue outside red range, use default
                lower1 = np.array([0, 70, 40], dtype=np.uint8)
                upper1 = np.array([15, 255, 255], dtype=np.uint8)
                lower2 = np.array([160, 70, 40], dtype=np.uint8)
                upper2 = np.array([180, 255, 255], dtype=np.uint8)
                m1 = cv2.inRange(hsv, lower1, upper1)
                m2 = cv2.inRange(hsv, lower2, upper2)
                color_mask = cv2.bitwise_or(m1, m2)
        else:
            lower1 = np.array([0, 70, 40], dtype=np.uint8)
            upper1 = np.array([15, 255, 255], dtype=np.uint8)
            lower2 = np.array([160, 70, 40], dtype=np.uint8)
            upper2 = np.array([180, 255, 255], dtype=np.uint8)
            m1 = cv2.inRange(hsv, lower1, upper1)
            m2 = cv2.inRange(hsv, lower2, upper2)
            color_mask = cv2.bitwise_or(m1, m2)
        
        # Refine with median hue from detected pixels
        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 12.0
            # Handle red's wrap-around at 0/180
            if med_h <= 20:
                h_lo = max(0, int(med_h - band))
                h_hi = min(30, int(med_h - band))
                dyn_lower = np.array([h_lo, 60, 40], dtype=np.uint8)
                dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
                color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)
            elif med_h >= 160:
                h_lo = max(150, int(med_h - band))
                h_hi = min(180, int(med_h + band))
                dyn_lower = np.array([h_lo, 60, 40], dtype=np.uint8)
                dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
                color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "blue":
        # Use detected hue if available, otherwise fall back to fixed blue range
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            # Only use detected hue if it's in the blue-ish range (90-140)
            if 85 <= hue_center <= 145:
                h_lo = max(80, int(hue_center - hue_range))
                h_hi = min(150, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([90, 40, 40], dtype=np.uint8)
                upper = np.array([140, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([90, 40, 40], dtype=np.uint8)
            upper = np.array([140, 255, 255], dtype=np.uint8)
        
        color_mask = cv2.inRange(hsv, lower, upper)
        
        # Refine with median hue from detected pixels
        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 12.0
            h_lo = max(80, int(med_h - band))
            h_hi = min(150, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "cyan":
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            if 70 <= hue_center <= 110:
                h_lo = max(60, int(hue_center - hue_range))
                h_hi = min(120, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([70, 40, 40], dtype=np.uint8)
                upper = np.array([110, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([70, 40, 40], dtype=np.uint8)
            upper = np.array([110, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 12.0
            h_lo = max(60, int(med_h - band))
            h_hi = min(120, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "yellow":
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            if 10 <= hue_center <= 55:
                h_lo = max(0, int(hue_center - hue_range))
                h_hi = min(70, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([15, 40, 40], dtype=np.uint8)
                upper = np.array([45, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([15, 40, 40], dtype=np.uint8)
            upper = np.array([45, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 10.0
            h_lo = max(0, int(med_h - band))
            h_hi = min(70, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "orange":
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            if 0 <= hue_center <= 50:
                h_lo = max(0, int(hue_center - hue_range))
                h_hi = min(60, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([5, 40, 40], dtype=np.uint8)
                upper = np.array([35, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([5, 40, 40], dtype=np.uint8)
            upper = np.array([35, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 10.0
            h_lo = max(0, int(med_h - band))
            h_hi = min(60, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "magenta":
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            if hue_center >= 110:
                h_lo = max(100, int(hue_center - hue_range))
                h_hi = min(180, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([140, 40, 40], dtype=np.uint8)
                upper = np.array([180, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([140, 40, 40], dtype=np.uint8)
            upper = np.array([180, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 12.0
            h_lo = max(100, int(med_h - band))
            h_hi = min(180, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)

    elif mode == "purple":
        if detected_hue is not None:
            hue_center, hue_range = detected_hue
            if 105 <= hue_center <= 175:
                h_lo = max(90, int(hue_center - hue_range))
                h_hi = min(180, int(hue_center + hue_range))
                lower = np.array([h_lo, 40, 40], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            else:
                lower = np.array([115, 40, 40], dtype=np.uint8)
                upper = np.array([170, 255, 255], dtype=np.uint8)
        else:
            lower = np.array([115, 40, 40], dtype=np.uint8)
            upper = np.array([170, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, lower, upper)

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 12.0
            h_lo = max(90, int(med_h - band))
            h_hi = min(180, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)
    else:
        # "black" or fallback: dark pixels relative to local background
        # Auto-detect if this is a black and white log for aggressive grid removal
        is_bw_log = detect_if_black_and_white_log(roi_bgr)
        
        # Apply aggressive grid removal to grayscale before thresholding if B&W detected
        gray_processed = gray
        if is_bw_log:
            # Enhanced with Hough Transform for better preservation of jagged curves
            gray_processed = suppress_grid_hough(gray_processed)
            # Use original morphology as backup but less aggressive
            gray_processed = remove_grid_lines_aggressive(gray_processed, aggressive=False)
        
        color_mask = cv2.adaptiveThreshold(
            gray_processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 21, 4
        )

        # Suppress colored pixels (grid/track lines are often red/green/blue).
        # In black mode we want low-saturation dark ink.
        try:
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            colored = (sat > 55) & (val > 40)
            color_mask[colored] = 0
        except Exception:
            pass

        # Additional grid removal on the mask itself (less aggressive if already processed)
        if h >= 20 and w >= 20:
            if is_bw_log:
                # Lighter grid removal since we already did aggressive removal
                k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, min(40, h // 3))))
                # AGGRESSIVE: Remove horizontal lines > 10px to kill grid shelves
                k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
            else:
                # Standard grid removal for non-B&W images
                k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, min(60, h // 2))))
                # AGGRESSIVE: Remove horizontal lines > 15px to kill grid shelves even if not detected as B&W
                k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            
            v_lines = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k_v)
            h_lines = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k_h)
            lines = cv2.bitwise_or(v_lines, h_lines)
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(lines))

    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, 1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, 1)
    color_score = color_mask.astype(np.float32) / 255.0

    # Compute 1-pixel skeleton for black mode using ximgproc thinning.
    # Applied AFTER grid removal so only curve pixels are thinned, not grid lines.
    skel_thin = None
    if mode not in colored_modes:
        try:
            if hasattr(cv2, 'ximgproc'):
                skel_thin = cv2.ximgproc.thinning(
                    color_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
                )
        except Exception:
            skel_thin = None

    # 2) Enhanced edge detection using both Canny and Sobel.
    #    Canny finds strong edges; Sobel emphasizes horizontal gradients
    #    which helps track curves that move left/right.
    edges_canny = cv2.Canny(gray, 40, 120)
    
    # Sobel for horizontal gradient (curve moving left/right)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    sobel_x = (sobel_x / (sobel_x.max() + 1e-6) * 255).astype(np.uint8)
    
    # Combine Canny and Sobel - Canny for sharp edges, Sobel for gradients
    edges_combined = cv2.addWeighted(edges_canny, 0.6, sobel_x, 0.4, 0)
    edges_blur = cv2.GaussianBlur(edges_combined, (5, 5), 0)
    edge_score = edges_blur.astype(np.float32) / 255.0
    
    # For color modes, gate edges by color mask to suppress non-colored edges
    if mode in colored_modes:
        edge_score *= color_score

    # 3) Suppress vertical "rails" (grid / borders) and track edges
    # REMOVED morphological erasure because it was deleting steep curve segments.
    # We will rely on the column-statistics penalty (rail_penalty) instead.

    if h >= 4 and w >= 2:
        col_on_frac = (color_score > 0).mean(axis=0)
        
        # For black mode, slow-varying curves (DTC, RHOB) can occupy one column for
        # 40-70% of the image height — use a high threshold to only kill near-solid
        # gridline/border columns (~90%+ occupancy). Colored modes can stay at 0.35
        # since their color_score is already hue-gated and gridlines are unsaturated.
        col_rail_threshold = 0.80 if mode not in colored_modes else 0.35
        rail_cols = col_on_frac > col_rail_threshold
        if np.any(rail_cols):
            color_score[:, rail_cols] *= 0.005  # Almost eliminate vertical rails
            edge_score[:, rail_cols] *= 0.005

        if mode not in colored_modes:
            row_on_frac = (color_score > 0).mean(axis=1)
            rail_rows = row_on_frac > 0.80
            if np.any(rail_rows):
                color_score[rail_rows, :] *= 0.02
                edge_score[rail_rows, :] *= 0.02
        
        # REMOVED edge suppression. Gamma Ray curves often hit the track edges.
        # We should rely on the specific 'preprocess_curve_track' logic for borders,
        # not a blind gradient suppression.

    # 4) Centerline boost via distance transform.
    # For colored modes, compute this from the filled stroke area (color mask),
    # not an edge-gated mask. Edge-only masks pull the DT peak toward one side
    # of a thick stroke.
    if mode in colored_modes:
        bin_for_dt = (color_score > 0.12).astype(np.uint8)
        try:
            # Fill small holes so DT peaks at true stroke center.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            bin_for_dt = cv2.morphologyEx(bin_for_dt, cv2.MORPH_CLOSE, k, iterations=1)
        except Exception:
            pass
    else:
        # For black mode, keep the conservative ink definition (helps avoid grid).
        ink_mask = (color_score > 0.25) & (edge_score > 0.10)
        if not np.any(ink_mask):
            ink_mask = (color_score > 0.15) & (edge_score > 0.05)
        bin_for_dt = ink_mask.astype(np.uint8)

    if np.any(bin_for_dt):
        dist = cv2.distanceTransform(bin_for_dt, cv2.DIST_L2, 5)
        center_score = dist.astype(np.float32)
        maxd = float(center_score.max())
        if maxd > 0:
            center_score /= maxd
            # Boost toward the center of thick strokes.
            center_score = np.power(center_score, 0.65)
    else:
        center_score = np.zeros_like(color_score, dtype=np.float32)

    # 5) Enhanced probability map with edge-aware filtering
    # For color modes, use adaptive weighting based on local characteristics
    if mode in colored_modes:
        # Edge-enhanced weighting for GR logs
        # Boost edge detection specifically for jagged features
        edge_enhanced = edge_score.copy()
        
        # Apply directional filtering to enhance horizontal edges (curve movement)
        kernel_horizontal = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
        horizontal_edges = cv2.filter2D(edge_score, -1, kernel_horizontal)
        horizontal_edges = np.maximum(0, horizontal_edges)
        
        # Combine with original edges
        edge_enhanced = np.maximum(edge_score, horizontal_edges * 0.5)
        
        # Adaptive weighting based on local edge density.
        # Keep edges helpful but avoid consistent outer-edge bias on thick ink.
        local_edge_density = cv2.GaussianBlur(edge_enhanced, (5, 5), 0)
        edge_weight = 0.18 + 0.16 * local_edge_density

         # Centerline boost; for thick strokes DT already peaks at the true center.
        center_boost = center_score * (1.0 + 0.2 * edge_enhanced)

        # Final probability map: slightly lower color weight, stronger center.
        prob = 0.08 * color_score + edge_weight * edge_enhanced + (0.92 - edge_weight) * center_boost
        
        # Apply gamma correction to boost faint signals
        prob = np.power(prob, 0.8)  # Gamma < 1 boosts low values
        
    else:
        # Enhanced black mode detection
        edge_enhanced = edge_score.copy()
        
        # Boost weak edges in black mode
        weak_edges = (edge_score > 0.05) & (edge_score < 0.3)
        edge_enhanced[weak_edges] = edge_score[weak_edges] * 1.5
        
        # --- Vertical Derivative Boost ---
        # Calculate Sobel Y to detect horizontal changes (edges of horizontal lines/spikes)
        # Vertical grid rails have dy ~ 0. Wiggly curves (even steep ones) have higher dy components.
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)
        # Normalize
        max_sy = sobel_y.max()
        if max_sy > 0:
            sobel_y_score = (sobel_y / max_sy).astype(np.float32)
        else:
            sobel_y_score = np.zeros_like(edge_score)

        # --- Harris Corner Boost ---
        # Harris response is high at "corners" (jagged peaks) and low on straight edges (grid lines).
        # This is perfect for highlighting the high-frequency nature of GR curves.
        # blockSize=2 (local), ksize=3 (gradients), k=0.04 (sensitivity)
        harris = cv2.cornerHarris(gray, 2, 3, 0.04)
        # Normalize strictly to 0-1
        harris = np.maximum(0, harris) # Clip negatives (flat regions)
        max_h = harris.max()
        if max_h > 0:
            harris_score = (harris / max_h).astype(np.float32)
            # Dilate slightly to make the corner "dots" connect
            harris_score = cv2.dilate(harris_score, np.ones((3,3), np.uint8))
        else:
            harris_score = np.zeros_like(edge_score)
            
        # --- Diagonal Derivative Boost ---
        # Grid lines are 0 or 90 degrees. Curves have diagonal segments.
        # Calculate diagonal gradients: |dx| + |dy| is a simple approx, but we can be more specific.
        # Actually, just using the magnitude of the gradient vector (already largely covered by Canny)
        # isn't enough. We want specifically 45/135 degree energy.
        # Rotate 45 degrees? Easier: |dx| * |dy| is high only when BOTH are present (diagonal).
        diag_score = (sobel_x.astype(np.float32)/255.0) * (sobel_y.astype(np.float32)/max_sy if max_sy > 0 else 0)
        # Normalize
        if diag_score.max() > 0:
            diag_score /= diag_score.max()
            
        # Combine:
        # - color_score (15%): Base intensity
        # - edge_enhanced (30%): Canny + SobelX (strong edges) - reduced slightly
        # - center_score (20%): Distance transform (center of strokes)
        # - sobel_y_score (15%): Boost for wiggles/spikes (dy)
        # - harris_score (10%): Boost for jagged peaks/corners
        # - diag_score (10%): Boost for diagonal segments (non-grid orientations)
        # Build skeleton score from ximgproc thinning (1-pixel centerline)
        skel_score = None
        if skel_thin is not None and skel_thin.any():
            skel_f = cv2.GaussianBlur(skel_thin.astype(np.float32), (3, 3), 0)
            skel_max = float(skel_f.max())
            if skel_max > 0:
                skel_score = skel_f / skel_max

        if skel_score is not None:
            # Skeleton gets 35% — massive bias to true centerline to prevent corner cutting
            prob = 0.10 * color_score + 0.15 * edge_enhanced + 0.15 * center_score + 0.10 * sobel_y_score + 0.10 * harris_score + 0.05 * diag_score + 0.35 * skel_score
        else:
            prob = 0.15 * color_score + 0.30 * edge_enhanced + 0.20 * center_score + 0.15 * sobel_y_score + 0.10 * harris_score + 0.10 * diag_score

    # 6) Reuse the stronger grid-removal heuristics from preprocess_curve_track
    #    as a gating mask. This aggressively down-weights columns/rows that
    #    look like grid or track borders, while preserving the wiggly curve
    #    strokes.
    if mode not in colored_modes:
        try:
            cleaned_binary = preprocess_curve_track(roi_bgr, mode=mode)
            if cleaned_binary is not None and cleaned_binary.size == prob.size:
                cleaned_score = cleaned_binary.astype(np.float32) / 255.0
                # Where cleaned_score == 0 (likely grid/border), push probability
                # almost to zero; where == 1, keep prob as-is.
                gate = 0.05 + 0.95 * cleaned_score
                prob *= gate
        except Exception:
            # If preprocessing fails for any reason, fall back to the ungated map.
            pass
    maxp = float(prob.max())
    if maxp > 0:
        prob = prob / maxp
    prob = np.clip(prob, 1e-4, 1.0).astype(np.float32)

    return (prob * 255.0).astype(np.uint8)


def trace_curve_with_dp(
    curve_mask,
    scale_min,
    scale_max,
    curve_type="GR",
    max_step=3,
    smooth_lambda=0.5,
    curv_lambda=0.0,
    hot_side=None,
):
    """Trace a curve using dynamic programming for smooth path finding.
    
    Args:
        curve_mask: Binary mask (0 or 255) where curve pixels are bright
        scale_min: Left scale value
        scale_max: Right scale value
        curve_type: Curve type for plausibility checks (GR, RHOB, NPHI, DT, etc.)
        max_step: Max horizontal movement per row (pixels)
        smooth_lambda: First-derivative smoothness penalty weight (penalizes jumps)
        curv_lambda: Second-derivative curvature penalty weight (penalizes kinks)
    
    Returns:
        xs: Array of x-coordinates (one per row), with np.nan for low-confidence rows
        confidence: Array of confidence scores (0-1) per row
    """
    if curve_mask is None or curve_mask.size == 0:
        return np.array([]), np.array([])
    
    h, w = curve_mask.shape
    if h < 2 or w < 2:
        return np.full(h, np.nan), np.zeros(h)

    if not getattr(fast_tracer, "NUMBA_AVAILABLE", False) and w > 420:
        w_small = int(max(64, min(420, w)))
        if w_small < w:
            mask_small = cv2.resize(curve_mask, (w_small, h), interpolation=cv2.INTER_AREA)
            max_step_small = max(1, int(round(max_step * (w_small - 1) / max(1, w - 1))))
            xs_small, conf_small = trace_curve_with_dp(
                mask_small,
                scale_min=scale_min,
                scale_max=scale_max,
                curve_type=curve_type,
                max_step=max_step_small,
                smooth_lambda=smooth_lambda,
                curv_lambda=curv_lambda,
                hot_side=hot_side,
            )
            if xs_small is None or xs_small.size == 0:
                return xs_small, conf_small
            scale_back = (w - 1) / max(1, (w_small - 1))
            xs = xs_small.astype(np.float32, copy=True)
            finite = np.isfinite(xs)
            xs[finite] = xs[finite] * scale_back
            return xs, conf_small

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

    def _morphological_skeleton(bin_img):
        """Simple morphological skeletonization (Zhang-Suen style via erode-open)."""
        size = np.size(bin_img)
        skel = np.zeros_like(bin_img, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        prev = None
        iteration = 0
        max_iter = 512  # safety
        while True:
            eroded = cv2.erode(bin_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(bin_img, temp)
            skel = cv2.bitwise_or(skel, temp)
            bin_img = eroded.copy()
            iteration += 1
            if cv2.countNonZero(bin_img) == 0 or iteration >= max_iter:
                break
            if prev is not None and np.array_equal(bin_img, prev):
                break
            prev = bin_img
        return skel

    # Live-wire style node score combining probability and centerline distance
    bin_mask = prob > 0.10
    skeleton_score = np.zeros_like(prob, dtype=np.float32)
    if np.any(bin_mask):
        try:
            if hasattr(cv2, 'ximgproc'):
                skel = cv2.ximgproc.thinning(
                    bin_mask.astype(np.uint8) * 255,
                    thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
                )
            else:
                skel = _morphological_skeleton((bin_mask.astype(np.uint8) * 255))
            if skel is not None and skel.size == prob.size:
                skel_f = skel.astype(np.float32) / 255.0
                # Feather skeleton to nearby pixels so DP can stay on the ridge
                skel_f = cv2.GaussianBlur(skel_f, (3, 3), 0)
                skel_max = float(skel_f.max())
                if skel_max > 0:
                    skeleton_score = skel_f / skel_max
        except Exception:
            skeleton_score = np.zeros_like(prob, dtype=np.float32)
    if np.any(bin_mask):
        dist = cv2.distanceTransform(bin_mask.astype(np.uint8), cv2.DIST_L2, 5)
        dist_norm = dist.astype(np.float32)
        maxd = float(dist_norm.max())
        if maxd > 0:
            dist_norm /= maxd
        center_score = np.power(dist_norm, 0.9)
    else:
        center_score = np.zeros_like(prob, dtype=np.float32)

    eps = 1e-6
    live_score = np.power(prob, 0.7) * (0.15 + 0.85 * center_score)
    # Boost with skeleton ridge to keep micro-bumps
    if skeleton_score is not None:
        live_score = np.maximum(live_score, 0.55 * skeleton_score + 0.05 * bin_mask.astype(np.float32))
    live_score = np.clip(live_score, eps, 1.0)

    # Blend in distance transform so the DP prefers the *center* of thick
    # ink strokes over their edges.  A pixel at the stroke center gets a
    # bonus proportional to its distance from the nearest background pixel.
    # 70% original score + 30% centredness keeps the colour/intensity
    # signal dominant while removing the edge-of-stroke bias.
    if bin_mask.any():
        _dist = cv2.distanceTransform(bin_mask.astype(np.uint8), cv2.DIST_L2, 3)
        _d_max = _dist.max()
        if _d_max > 0:
            _dist_norm = (_dist / _d_max).astype(np.float32)
            live_score = live_score * (0.7 + 0.3 * _dist_norm)
            live_score = np.clip(live_score, eps, 1.0)

    cost = -np.log(live_score)

    # Soft rail penalty: down-weight columns that stay on for many rows, without banning them
    if h >= 4 and w >= 2:
        col_frac = bin_mask.mean(axis=0)
        rail_mask = col_frac > 0.60
        # Expand to runs of length ≥3 using a 3-wide moving window
        rail_run = np.convolve(rail_mask.astype(np.float32), np.ones(3, dtype=np.float32), mode='same') >= 2.5
        rail_weight = 15.0  # keep modest per guidance (5–30)
        if np.any(rail_run):
            cost += (rail_weight * rail_run.astype(np.float32))[np.newaxis, :]

    # Use live_score for Viterbi likelihoods
    prob = live_score
    
    if hot_side in ("left", "right") and w >= 2:
        frac = np.linspace(0.0, 1.0, w, dtype=np.float32)
        if hot_side == "left":
            dist = frac
        else:
            dist = 1.0 - frac
        side_lambda = 1.0
        side_penalty = side_lambda * dist
        cost += side_penalty[np.newaxis, :]
    
    # Add plausibility penalty only when the display scale is in physical units.
    # If scale_min/scale_max don't overlap the known physical range at all (e.g.
    # display range 0-150 vs RHOB physical 1.5-3.5 g/cc), skip the penalty to
    # avoid penalising every column and forcing the DP toward one edge.
    if curve_type.upper() in plausible_ranges:
        pmin, pmax = plausible_ranges[curve_type.upper()]
        scale_lo = min(scale_min, scale_max)
        scale_hi = max(scale_min, scale_max)
        # Only apply if the display range meaningfully overlaps the physical range
        if scale_lo <= pmax and scale_hi >= pmin:
            for x in range(w):
                value = scale_min + (x / max(1, w - 1)) * (scale_max - scale_min)
                if value < pmin or value > pmax:
                    cost[:, x] += 1.0
    
    # Horizontal grid line suppression using morphological opening.
    # A true grid line has many *consecutive* pixels across the row; the actual
    # curve only spans ~1-5 pixels per row and won't survive a wide horizontal
    # opening. This is more discriminative than a raw row-fraction threshold.
    if h >= 4 and w >= 8:
        horiz_kernel_w = max(5, w // 4)
        horiz_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_w, 1))
        horiz_detected = cv2.morphologyEx(bin_mask.astype(np.uint8), cv2.MORPH_OPEN, horiz_kern)
        horiz_row_frac = horiz_detected.mean(axis=1)
        horiz_mask = horiz_row_frac > 0.30  # >30% of row survived wide-kernel opening → true grid line
        if np.any(horiz_mask):
            uniform_cost = float(-np.log(1e-3))
            cost[horiz_mask, :] = uniform_cost
            prob[horiz_mask, :] = 1e-3

    # Run optimized DP (Forward Pass)
    xs_fwd, conf_fwd = fast_tracer.run_viterbi(
        cost.astype(np.float32), 
        prob.astype(np.float32), 
        int(max_step), 
        float(smooth_lambda), 
        float(curv_lambda)
    )
    
    # Run optimized DP (Backward Pass)
    # Flip cost and prob arrays upside down
    xs_bwd_flipped, conf_bwd_flipped = fast_tracer.run_viterbi(
        cost[::-1].astype(np.float32), 
        prob[::-1].astype(np.float32), 
        int(max_step), 
        float(smooth_lambda), 
        float(curv_lambda)
    )
    # Flip results back to match original orientation
    xs_bwd = xs_bwd_flipped[::-1]
    conf_bwd = conf_bwd_flipped[::-1]
    
    # Merge Forward and Backward results
    # Averaging the positions cancels out directional lag/hysteresis
    xs = np.full_like(xs_fwd, np.nan)
    confidence = np.zeros_like(conf_fwd)
    
    for y in range(h):
        v1 = xs_fwd[y]
        v2 = xs_bwd[y]
        valid1 = np.isfinite(v1)
        valid2 = np.isfinite(v2)
        
        if valid1 and valid2:
            # If the two passes disagree significantly (e.g. took different paths around an obstacle),
            # pick the one with higher local probability instead of averaging (which would land in the middle of nowhere).
            if abs(v1 - v2) > 5:
                p1 = prob[y, int(min(w-1, max(0, v1)))]
                p2 = prob[y, int(min(w-1, max(0, v2)))]
                xs[y] = v1 if p1 > p2 else v2
            else:
                # Otherwise, average them to center the trace on peaks
                xs[y] = (v1 + v2) * 0.5
            confidence[y] = (conf_fwd[y] + conf_bwd[y]) * 0.5
        elif valid1:
            xs[y] = v1
            confidence[y] = conf_fwd[y]
        elif valid2:
            xs[y] = v2
            confidence[y] = conf_bwd[y]
    
    return xs, confidence


def trace_curve_skeleton_path(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Photocopy-style tracer: binarize, skeletonize, take top-to-bottom seam along skeleton.
    """
    if mask is None or mask.size == 0:
        return np.array([]), np.array([])
    h, w = mask.shape
    bin_mask = (mask > 25).astype(np.uint8)
    if cv2.countNonZero(bin_mask) == 0:
        return np.full(h, np.nan, dtype=np.float32), np.zeros(h, dtype=np.float32)

    # Light rail removal
    dark = (mask < 40).astype(np.uint8)
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 24)))
    k_horz = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 24), 1))
    rail_v = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_vert)
    rail_h = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_horz)
    rail_mask = cv2.bitwise_or(rail_v, rail_h)
    bin_mask = cv2.subtract(bin_mask, rail_mask)
    bin_mask = np.clip(bin_mask, 0, 1).astype(np.uint8)
    if cv2.countNonZero(bin_mask) == 0:
        return np.full(h, np.nan, dtype=np.float32), np.zeros(h, dtype=np.float32)

    skel = _skeletonize_binary((bin_mask * 255).astype(np.uint8))
    if skel is None or skel.size != mask.size or cv2.countNonZero(skel) == 0:
        return np.full(h, np.nan, dtype=np.float32), np.zeros(h, dtype=np.float32)

    num_labels, labels = cv2.connectedComponents(skel, connectivity=8)
    if num_labels > 1:
        areas = [(labels == i).sum() for i in range(1, num_labels)]
        keep_label = 1 + int(np.argmax(areas))
        skel = np.where(labels == keep_label, skel, 0).astype(np.uint8)
        if cv2.countNonZero(skel) == 0:
            return np.full(h, np.nan, dtype=np.float32), np.zeros(h, dtype=np.float32)

    prob = skel.astype(np.float32) / 255.0
    cost = 1.0 - prob
    dp = cost.copy()
    prev = np.full_like(labels, -1, dtype=np.int16)
    for y in range(1, h):
        for x in range(w):
            best = dp[y - 1, x]
            px = x
            if x > 0 and dp[y - 1, x - 1] < best:
                best = dp[y - 1, x - 1]; px = x - 1
            if x + 1 < w and dp[y - 1, x + 1] < best:
                best = dp[y - 1, x + 1]; px = x + 1
            dp[y, x] += best
            prev[y, x] = px
    end_x = int(np.argmin(dp[-1]))
    xs_path = np.full(h, np.nan, dtype=np.float32)
    x = end_x
    for y in range(h - 1, -1, -1):
        xs_path[y] = float(x)
        x_prev = prev[y, x]
        if x_prev < 0:
            break
        x = int(x_prev)
    conf = np.where(np.isfinite(xs_path), 1.0, 0.0).astype(np.float32)
    return xs_path, conf


def _skeletonize_binary(bin_img: np.ndarray) -> np.ndarray:
    """Simple morphological skeletonization (Zhang-Suen style via erode-open)."""
    skel = np.zeros_like(bin_img, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    prev = None
    iteration = 0
    max_iter = 1024  # safety
    work = bin_img.copy()
    while True:
        eroded = cv2.erode(work, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(work, temp)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded.copy()
        iteration += 1
        if cv2.countNonZero(work) == 0 or iteration >= max_iter:
            break
        if prev is not None and np.array_equal(work, prev):
            break
        prev = work
    return skel


def _parabola_subpixel_x(row_slice: np.ndarray, start: int):
    """Fit a parabola to the log-intensity of a row slice and return subpixel x."""
    if row_slice is None or row_slice.size < 3:
        return float(start + np.argmax(row_slice))
    y = row_slice.astype(np.float32)
    idx = int(np.argmax(y))
    if idx == 0 or idx == y.size - 1:
        return float(start + idx)
    y1, y2, y3 = y[idx - 1:idx + 2]
    denom = (y1 - 2 * y2 + y3)
    if abs(denom) < 1e-9:
        return float(start + idx)
    offset = 0.5 * (y1 - y3) / denom
    return float(start + idx + offset)


def _postprocess_missed_peaks(mask: np.ndarray, prob: np.ndarray, xs: np.ndarray, search_radius: int = 12, min_prob: float = 0.03):
    """Hybrid post-processor: force missed ink peaks onto the curve."""
    h, w = mask.shape
    if xs.size != h:
        return xs
    for y in range(h):
        if not np.isfinite(xs[y]):
            continue
        xi = int(round(xs[y]))
        start = max(0, xi - search_radius)
        end = min(w, xi + search_radius + 1)
        row = prob[y, start:end]
        if row.size == 0:
            continue
        # Find all peaks above min_prob (robust to plateaus)
        peaks = []
        plateau_start = -1
        for i in range(1, row.size - 1):
            val = row[i]
            if val < min_prob:
                plateau_start = -1
                continue
            
            prev = row[i - 1]
            next_val = row[i + 1]
            
            # Rising edge
            if val > prev:
                if val > next_val:
                    peaks.append(i) # Sharp peak
                    plateau_start = -1
                elif val == next_val:
                    plateau_start = i # Start plateau
                else:
                    plateau_start = -1
            # Flat
            elif val == prev:
                if val > next_val:
                    if plateau_start != -1:
                        peaks.append((plateau_start + i) // 2) # End plateau
                    plateau_start = -1
                elif val < next_val:
                    plateau_start = -1
            # Falling
            else:
                plateau_start = -1

        if not peaks:
            continue
        # If the current position is not already on a peak, move to the nearest peak
        cur_rel = xi - start
        on_peak = any(abs(cur_rel - p) <= 2 for p in peaks)
        if not on_peak:
            # Choose the peak closest to the current x
            best = min(peaks, key=lambda p: abs(p - cur_rel))
            # Weighted centroid around the chosen peak
            peak_val = row[best]
            thresh = max(peak_val * 0.6, min_prob)
            left = best
            right = best
            while left > 0 and row[left - 1] >= thresh:
                left -= 1
            while right + 1 < row.size and row[right + 1] >= thresh:
                right += 1
            seg = row[left:right + 1].astype(np.float32)
            coords = np.arange(start + left, start + right + 1, dtype=np.float32)
            wsum = seg.sum()
            if wsum > 1e-6:
                xs[y] = float((coords * seg).sum() / wsum)
            else:
                xs[y] = float(start + best)


def align_rgb_channels(bgr: np.ndarray) -> np.ndarray:
    """Align RGB channels via phase correlation to reduce color fringing."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    try:
        b, g, r = cv2.split(bgr)

        def _shift_to_ref(src, ref):
            src_f = src.astype(np.float32)
            ref_f = ref.astype(np.float32)
            shift, _ = cv2.phaseCorrelate(ref_f, src_f)
            dx, dy = shift
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        r_aligned = _shift_to_ref(r, g)
        b_aligned = _shift_to_ref(b, g)
        return cv2.merge([b_aligned, g, r_aligned])
    except Exception:
        return bgr


def align_rgb_channels(bgr: np.ndarray) -> np.ndarray:
    """Align RGB channels via phase correlation to reduce color fringing."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    try:
        b, g, r = cv2.split(bgr)

        def _shift_to_ref(src, ref):
            src_f = src.astype(np.float32)
            ref_f = ref.astype(np.float32)
            shift, _ = cv2.phaseCorrelate(ref_f, src_f)
            dx, dy = shift
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        r_aligned = _shift_to_ref(r, g)
        b_aligned = _shift_to_ref(b, g)
        return cv2.merge([b_aligned, g, r_aligned])
    except Exception:
        return bgr
    for y in range(h):
        if not np.isfinite(xs[y]):
            continue
        xi = int(round(xs[y]))
        start = max(0, xi - search_radius)
        end = min(w, xi + search_radius + 1)
        row = prob[y, start:end]
        if row.size == 0:
            continue
        # Find all peaks above min_prob
        peaks = []
        for i in range(1, row.size - 1):
            if row[i] > row[i - 1] and row[i] > row[i + 1] and row[i] >= min_prob:
                peaks.append(i)
        if not peaks:
            continue
        # If the current position is not already on a peak, move to the nearest peak
        cur_rel = xi - start
        on_peak = any(abs(cur_rel - p) <= 2 for p in peaks)
        if not on_peak:
            # Choose the peak closest to the current x
            best = min(peaks, key=lambda p: abs(p - cur_rel))
            # Weighted centroid around the chosen peak
            peak_val = row[best]
            thresh = max(peak_val * 0.6, min_prob)
            left = best
            right = best
            while left > 0 and row[left - 1] >= thresh:
                left -= 1
            while right + 1 < row.size and row[right + 1] >= thresh:
                right += 1
            seg = row[left:right + 1].astype(np.float32)
            coords = np.arange(start + left, start + right + 1, dtype=np.float32)
            wsum = seg.sum()
            if wsum > 1e-6:
                xs[y] = float((coords * seg).sum() / wsum)
            else:
                xs[y] = float(start + best)
        else:
            # Even if on a peak, refine to weighted centroid for subpixel accuracy
            cur_idx = int(cur_rel)
            if 0 < cur_idx < row.size - 1:
                peak_val = row[cur_idx]
                thresh = max(peak_val * 0.6, min_prob)
                left = cur_idx
                right = cur_idx
                while left > 0 and row[left - 1] >= thresh:
                    left -= 1
                while right + 1 < row.size and row[right + 1] >= thresh:
                    right += 1
                seg = row[left:right + 1].astype(np.float32)
                coords = np.arange(start + left, start + right + 1, dtype=np.float32)
                wsum = seg.sum()
                if wsum > 1e-6:
                    xs[y] = float((coords * seg).sum() / wsum)
    return xs


def trace_curve_pixel_perfect(mask: np.ndarray, grayscale: np.ndarray = None, bgr: np.ndarray = None, hot_side=None, preserve_wiggles: bool = False, crest_boost: bool = False):
    """Pixel-perfect tracing optimized for dot-matrix style prints: row-by-row peak following."""
    if mask is None or mask.size == 0:
        return np.array([]), np.array([])
    h, w = mask.shape
    if h < 4 or w < 2:
        return np.full(h, np.nan, dtype=np.float32), np.zeros(h, dtype=np.float32)

    # Build probability map with hue weighting if available
    hue_weight = None
    if bgr is not None and bgr.size == mask.size * 3:
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            ink_mask = (mask > np.percentile(mask, 70)).astype(np.uint8)
            if ink_mask.sum() > 20:
                ink_hues = hsv[..., 0][ink_mask.astype(bool)]
                ang = ink_hues.astype(np.float32) * (2.0 * np.pi / 180.0)
                mean_hue = math.atan2(np.sin(ang).mean(), np.cos(ang).mean())
                if mean_hue < 0:
                    mean_hue += 2 * np.pi
                mean_deg = mean_hue * 180.0 / np.pi
                hue = hsv[..., 0].astype(np.float32)
                dh = np.abs(((hue - mean_deg + 90) % 180) - 90)
                hue_weight = np.exp(-(dh ** 2) / (2 * (12.0 ** 2))).astype(np.float32)
                hue_weight = np.clip(hue_weight, 0.15, 1.0)
        except Exception:
            hue_weight = None

    prob_base = mask.astype(np.float32) / 255.0
    if hue_weight is not None:
        prob_base = prob_base * hue_weight
    # Morphological close to connect dot-matrix dots (slightly larger to bridge gaps)
    # Taller kernel for crest_boost to bridge vertical dot gaps
    k_size = (3, 11) if crest_boost else (4, 6)
    prob_closed = cv2.morphologyEx(prob_base, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size))
    prob = np.maximum(prob_base, prob_closed)
    
    # Only blur if NOT crest_boost to preserve sharp dot edges
    if not crest_boost:
        prob = cv2.GaussianBlur(prob, (3, 3), 0)
    
    prob = np.clip(prob * 1.25, 0.0, 1.0)

    # Per-row global maxima (photocopy-style fallback)
    row_max_xs = np.full(h, np.nan, dtype=np.float32)
    for y in range(h):
        row = prob[y]
        peak = row.max()
        if peak < 0.01:
            continue
        idx = int(np.argmax(row))
        thresh = max(peak * 0.6, 0.01)
        left = idx
        right = idx
        while left > 0 and row[left - 1] >= thresh:
            left -= 1
        while right + 1 < row.size and row[right + 1] >= thresh:
            right += 1
        seg = row[left:right + 1].astype(np.float32)
        coords = np.arange(left, right + 1, dtype=np.float32)
        wsum = seg.sum()
        if wsum > 1e-6:
            row_max_xs[y] = float((coords * seg).sum() / wsum)
        else:
            row_max_xs[y] = float(idx)

    def _seam_path(prob_map: np.ndarray):
        """Min-cost vertical seam on (1-prob)."""
        h_s, w_s = prob_map.shape
        if h_s < 2 or w_s < 1:
            return np.full(h_s, np.nan, dtype=np.float32)
        cost = 1.0 - prob_map
        dp = cost.copy()
        prev = np.full((h_s, w_s), -1, dtype=np.int16)
        for y in range(1, h_s):
            for x in range(w_s):
                best = dp[y - 1, x]
                px = x
                if x > 0 and dp[y - 1, x - 1] < best:
                    best = dp[y - 1, x - 1]; px = x - 1
                if x + 1 < w_s and dp[y - 1, x + 1] < best:
                    best = dp[y - 1, x + 1]; px = x + 1
                dp[y, x] += best
                prev[y, x] = px
        end_x = int(np.argmin(dp[-1]))
        xs_path = np.full(h_s, np.nan, dtype=np.float32)
        x = end_x
        for y in range(h_s - 1, -1, -1):
            xs_path[y] = float(x)
            x_prev = prev[y, x]
            if x_prev < 0:
                break
            x = int(x_prev)
        return xs_path

    # ---- Simple row-by-row peak following (dot-matrix style) ----
    # Optional crest boost: use stronger ridge-enhanced prob to stay on tops
    ridge_prob = prob
    if crest_boost:
        # Taller vertical blur to bridge larger dot gaps
        # Also slightly wider to help horizontal connectivity
        k_size = (5, 11)
        ridge_prob = np.maximum(prob, cv2.blur(prob, (1, 15)))
        sobel_y = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)
        if sobel_y.max() > 1e-6:
            sobel_y = sobel_y / (sobel_y.max() + 1e-6)
            ridge_prob = np.maximum(ridge_prob, ridge_prob * (1.0 + 0.6 * sobel_y))
        ridge_prob = cv2.dilate(ridge_prob, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    xs = np.full(h, np.nan, dtype=np.float32)
    confidence = np.zeros(h, dtype=np.float32)
    base_search_rad = 80  # widened to catch larger jumps

    # Find initial x from the row with strongest total ink
    row_sums = ridge_prob.sum(axis=1)
    start_row = int(np.argmax(row_sums))
    best_x = int(np.argmax(ridge_prob[start_row]))
    xs[start_row] = float(best_x)
    confidence[start_row] = ridge_prob[start_row, best_x]

    def find_peak_in_row(y, prev_x, wide_search=False):
        """Find strongest peak near prev_x in row y with adaptive continuity penalty."""
        xi = int(round(prev_x))
        row_strength = float(ridge_prob[y].max()) if crest_boost else 0.0
        
        # Determine search window
        s_rad = base_search_rad + int(6 * row_strength) if crest_boost else base_search_rad
        if wide_search:
            s_rad = s_rad * 3  # Emergency wide search
            
        start = max(0, xi - s_rad)
        end = min(w, xi + s_rad + 1)
        row = ridge_prob[y, start:end]
        
        # Lower threshold to catch faint dots
        min_peak_thresh = 0.005 if crest_boost else 0.012
        if row.size == 0 or row.max() < min_peak_thresh:
            return np.nan, 0.0
        
        # Find all significant peaks (robust to plateaus)
        peaks = []
        plateau_start = -1
        for i in range(1, row.size - 1):
            val = row[i]
            if val < min_peak_thresh:
                plateau_start = -1
                continue
                
            prev = row[i - 1]
            next_val = row[i + 1]
            
            # Rising edge
            if val > prev:
                if val > next_val:
                    peaks.append(i) # Sharp peak
                    plateau_start = -1
                elif val == next_val:
                    plateau_start = i # Start plateau
                else:
                    plateau_start = -1
            # Flat
            elif val == prev:
                if val > next_val:
                    if plateau_start != -1:
                        peaks.append((plateau_start + i) // 2) # End plateau
                    plateau_start = -1
                elif val < next_val:
                    plateau_start = -1
            # Falling
            else:
                plateau_start = -1

        if crest_boost and peaks:
            # Use scoring with quadratic penalty to prefer closer peaks
            # but allow jumping to strong peaks if they are reasonably close.
            best_score = -1e9
            idx_rel = peaks[0]
            for pk in peaks:
                pk_val = row[pk]
                pk_x = start + pk
                dist = abs(pk_x - prev_x)
                
                # Quadratic penalty: penalize large jumps much more than small ones
                # at dist=0, penalty=0
                # at dist=10, penalty is small
                # at dist=80, penalty is large
                norm_dist = dist / float(s_rad) # 0 to 1
                
                # Base penalty factor
                penalty_weight = 0.15 # Stronger weight to prevent jumping to grid lines
                
                continuity_penalty = penalty_weight * (norm_dist ** 2) * (1.0 - pk_val * 0.5)
                
                score = pk_val - continuity_penalty
                if score > best_score:
                    best_score = score
                    idx_rel = pk
            peak_val = row[idx_rel]
        elif crest_boost and row.max() >= 0.008:
             # No peaks found but signal is strong (likely edge peak), use argmax
             idx_rel = int(np.argmax(row))
             peak_val = row[idx_rel]
        elif not peaks:
            idx_rel = int(np.argmax(row))
            peak_val = row[idx_rel]
        else:
            # Choose peak with best score: peak_val - adaptive continuity_penalty
            best_score = -1e9
            idx_rel = peaks[0]
            for pk in peaks:
                pk_val = row[pk]
                pk_x = start + pk
                # Adaptive penalty: lighter to allow close switch-backs; allow bigger excursions on strong peaks
                dist = abs(pk_x - prev_x)
                if crest_boost:
                     continuity_penalty = 0.005 * (dist / float(s_rad)) * (1.0 - pk_val)
                else:
                     continuity_penalty = 0.020 * (dist / float(s_rad)) * (1.0 - pk_val)
                score = pk_val - continuity_penalty
                if score > best_score:
                    best_score = score
                    idx_rel = pk
            peak_val = row[idx_rel]
        
        # Weighted centroid for subpixel
        # Use cleaner 'prob' map for localization if available to avoid blur bias
        if crest_boost:
            raw_row = prob[y, start:end]
            if raw_row.size == row.size:
                # Re-check value on raw map
                pv_raw = raw_row[idx_rel]
                # If raw map has signal, use it. Otherwise fall back to ridge_row
                if pv_raw > 0.001:
                    row_for_centroid = raw_row
                    peak_val = pv_raw
                    # Tighter threshold for crest_boost to stay on peak
                    thresh = max(peak_val * 0.70, 0.0025)
                else:
                    row_for_centroid = row
                    thresh = max(peak_val * 0.70, 0.0025)
            else:
                row_for_centroid = row
                thresh = max(peak_val * 0.70, 0.0025)
        else:
            row_for_centroid = row
            thresh = max(peak_val * 0.40, 0.003)

        left = idx_rel
        right = idx_rel
        while left > 0 and row_for_centroid[left - 1] >= thresh:
            left -= 1
        while right + 1 < row_for_centroid.size and row_for_centroid[right + 1] >= thresh:
            right += 1
        seg = row_for_centroid[left:right + 1].astype(np.float32)
        coords = np.arange(start + left, start + right + 1, dtype=np.float32)
        wsum = seg.sum()
        if wsum > 1e-6:
            x_out = float((coords * seg).sum() / wsum)
        else:
            x_out = float(start + idx_rel)
        return x_out, peak_val

    # Trace downward from start_row
    prev_x = xs[start_row]
    for y in range(start_row + 1, h):
        x_new, conf = find_peak_in_row(y, prev_x)
        if not np.isfinite(x_new) and crest_boost:
            # Fallback: try wide search
            x_new, conf = find_peak_in_row(y, prev_x, wide_search=True)
            
        xs[y] = x_new
        confidence[y] = conf
        if np.isfinite(x_new):
            prev_x = x_new

    # Trace upward from start_row
    prev_x = xs[start_row]
    for y in range(start_row - 1, -1, -1):
        x_new, conf = find_peak_in_row(y, prev_x)
        if not np.isfinite(x_new) and crest_boost:
            # Fallback: try wide search
            x_new, conf = find_peak_in_row(y, prev_x, wide_search=True)
            
        xs[y] = x_new
        confidence[y] = conf
        if np.isfinite(x_new):
            prev_x = x_new

    # Fill small gaps with linear interpolation (more permissive to bridge gaps)
    s = pd.Series(xs)
    s = s.interpolate(method='linear', limit_direction='both', limit=25)
    xs = s.to_numpy(dtype=np.float32)

    # No inertia smoothing to keep every wiggle

    # Photocopy-style fusion: row tracer vs row maxima vs seam
    # 1) Fuse with row maxima when stronger (no margin to keep detail)
    for y in range(h):
        if not np.isfinite(row_max_xs[y]):
            continue
        x_row = xs[y]
        x_max = row_max_xs[y]
        p_max = ridge_prob[y, int(round(np.clip(x_max, 0, w - 1)))]
        p_row = ridge_prob[y, int(round(np.clip(x_row, 0, w - 1)))] if np.isfinite(x_row) else -1.0
        if p_max > p_row:
            xs[y] = x_max

    # 2) Optional seam fusion; skip when crest_boost is enabled
    if not crest_boost:
        xs_seam = _seam_path(prob)
        if xs_seam.size == h:
            xs_fused = xs.copy()
            for y in range(h):
                x_row = xs[y]
                x_seam = xs_seam[y]
                if not np.isfinite(x_seam):
                    continue
                p_seam = prob[y, int(round(np.clip(x_seam, 0, w - 1)))]
                p_row = prob[y, int(round(np.clip(x_row, 0, w - 1)))] if np.isfinite(x_row) else -1.0
                if p_seam > p_row:  # no margin, prefer stronger seam locally
                    xs_fused[y] = x_seam
            xs = xs_fused

    return xs, confidence

def trace_curve_multiscale(curve_mask, scale_min, scale_max, curve_type="GR", max_step=3, smooth_lambda=0.1, hot_side=None):
    """
    Enhanced multi-scale curve tracing with 5 scales and weighted fusion.
    
    This improves detection by:
    - More granular scales for better feature capture
    - Weighted fusion based on confidence and scale
    - Scale-adaptive parameters for GR logs
    - Better handling of jagged Gamma Ray spikes
    
    Scales used: 1.0, 0.75, 0.5, 0.33, 0.25, 0.125 (6 scales)
    """
    if curve_mask is None or curve_mask.size == 0:
        return np.array([]), np.array([])
    
    # For GR curves, we want to allow very sharp peaks, so we lower the smoothing significantly.
    # For other curves (like Res), we keep it higher to avoid noise.
    # Note: We respect the passed smooth_lambda, but ensure it's not too high for GR.
    if curve_type.upper() == "GR" and smooth_lambda > 0.01:
        smooth_lambda = 0.001
    
    h, w = curve_mask.shape
    if h < 4 or w < 4:
        return trace_curve_with_dp(curve_mask, scale_min, scale_max, curve_type, max_step, smooth_lambda, hot_side)
    
    # Adaptive scale selection based on image content
    def adaptive_scale_selection(curve_mask, curve_type):
        """Choose optimal scales based on image characteristics"""
        h, w = curve_mask.shape
        prob = curve_mask.astype(np.float32) / 255.0
        
        if curve_type.upper() == "GR":
            # Analyze jaggedness for GR logs
            gray = (prob * 255).astype(np.uint8)
            
            # Edge detection to measure jaggedness
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Measure intensity variance (indicates sharp transitions)
            intensity_var = np.var(prob)
            
            # Determine scale set based on jaggedness
            if edge_density > 0.15 or intensity_var > 0.1:  # Very jagged
                scales = [1.0, 0.8, 0.6, 0.4, 0.2]  # More fine scales
            elif edge_density > 0.08 or intensity_var > 0.05:  # Moderately jagged
                scales = [1.0, 0.75, 0.5, 0.33, 0.25]  # Balanced
            else:  # Smooth
                scales = [1.0, 0.67, 0.33, 0.17]  # More coarse scales
        else:
            # Default for other curve types
            scales = [1.0, 0.67, 0.33, 0.17]
        
        # Filter scales that are too small
        valid_scales = []
        for scale in scales:
            h_s = max(4, int(h * scale))
            w_s = max(4, int(w * scale))
            if h_s >= 4 and w_s >= 4:
                valid_scales.append(scale)
        
        return valid_scales if len(valid_scales) >= 2 else [1.0, 0.5]
    
    # Use adaptive scale selection
    valid_scales = adaptive_scale_selection(curve_mask, curve_type)
    
    all_xs = []
    all_confs = []
    
    # Enhanced scale-adaptive parameters
    def get_scale_params(scale, curve_type, jaggedness_factor=1.0):
        """Get parameters optimized for each scale and curve type"""
        if curve_type.upper() == "GR":
            # Adjust based on jaggedness
            return {
                "smooth_lambda": max(0.000001, smooth_lambda * scale * jaggedness_factor),
                "max_step": max(1, int(max_step * scale * 1.2)),  # Moderate movement to prevent teleportation
                "rail_threshold": max(0.01, 0.1 * scale * jaggedness_factor),
                "curv_lambda": max(0.000001, 0.001 * scale * jaggedness_factor)
            }
        else:
            return {
                "smooth_lambda": smooth_lambda * scale,
                "max_step": max(1, int(max_step * scale)),
                "rail_threshold": 0.1 * scale,
                "curv_lambda": 0.05 * scale
            }
    
    # Calculate jaggedness factor for parameter tuning
    prob = curve_mask.astype(np.float32) / 255.0
    gray = (prob * 255).astype(np.uint8)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    jaggedness_factor = max(0.5, min(2.0, 1.0 + edge_density * 5))
    
    if len(valid_scales) < 2:
        return trace_curve_with_dp(curve_mask, scale_min, scale_max, curve_type, max_step, smooth_lambda, hot_side)
    
    prob = curve_mask.astype(np.float32) / 255.0
    
    for scale in valid_scales:
        if scale == 1.0:
            mask_scaled = curve_mask
            h_s, w_s = h, w
        else:
            h_s = max(4, int(h * scale))
            w_s = max(4, int(w * scale))
            
            # Use appropriate interpolation
            if scale > 0.5:
                mask_scaled = cv2.resize(curve_mask, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
            else:
                mask_scaled = cv2.resize(curve_mask, (w_s, h_s), interpolation=cv2.INTER_AREA)
        
        # Get scale-adaptive parameters
        params = get_scale_params(scale, curve_type)
        
        # Run DP at this scale
        xs_scaled, conf_scaled = trace_curve_with_dp(
            mask_scaled,
            scale_min=scale_min,
            scale_max=scale_max,
            curve_type=curve_type,
            max_step=params["max_step"],
            smooth_lambda=params["smooth_lambda"],
            curv_lambda=params["curv_lambda"],
            hot_side=hot_side,
        )
        
        # Upsample back to full resolution with sub-pixel precision
        if scale != 1.0 and xs_scaled.size > 0:
            xs_full = np.full(h, np.nan, dtype=np.float32)
            conf_full = np.zeros(h, dtype=np.float32)
            
            for y_full in range(h):
                y_scaled = y_full * scale
                y_int = int(y_scaled)
                y_frac = y_scaled - y_int
                
                if y_int < xs_scaled.size and np.isfinite(xs_scaled[y_int]):
                    # Linear interpolation for sub-pixel accuracy
                    x_scaled = xs_scaled[y_int] / scale
                    conf_scaled_val = conf_scaled[y_int]
                    
                    # Handle fractional positions
                    if y_int + 1 < xs_scaled.size and np.isfinite(xs_scaled[y_int + 1]):
                        x_next = xs_scaled[y_int + 1] / scale
                        conf_next = conf_scaled[y_int + 1]
                        
                        # Interpolate both position and confidence
                        x_full = x_scaled * (1 - y_frac) + x_next * y_frac
                        conf_full[y_full] = conf_scaled_val * (1 - y_frac) + conf_next * y_frac
                    else:
                        x_full = x_scaled
                        conf_full[y_full] = conf_scaled_val
                    
                    xs_full[y_full] = x_full
            
            xs_scaled = xs_full
            conf_scaled = conf_full
        
        all_xs.append(xs_scaled)
        all_confs.append(conf_scaled)
    
    # Nuclear option - ultra-aggressive peak detection for missed GR spikes
    def detect_local_peaks(prob_map, min_prominence=0.005, sensitivity_boost=True):
        """Nuclear option: ultra-aggressive peak detection for GR spike detection"""
        h, w = prob_map.shape
        peaks = []
        
        # Use raw probability map for maximum sensitivity (no blur)
        raw = prob_map
        
        search_window = 2       # Very small neighborhood
        
        # Multi-scale detection with different sensitivities
        for y in range(1, h-1):
            row = raw[y]
            
            # Find ALL local maxima, even tiny ones
            for x in range(1, w-1):
                # Check 3x3 neighborhood for local max
                local_max = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        if row[x] < raw[y+dy, x+dx]:
                            local_max = False
                            break
                    if not local_max:
                        break
                
                # Accept even tiny peaks for GR logs
                if local_max and row[x] > min_prominence:
                    # Minimal prominence requirement
                    window_size = min(8, w//8)
                    left_start = max(0, x-window_size)
                    right_end = min(w, x+window_size+1)
                    
                    left_region = row[left_start:x]
                    right_region = row[x+1:right_end]
                    
                    if len(left_region) > 0 and len(right_region) > 0:
                        # Use 20th percentile for valleys (very relaxed)
                        left_valley = np.percentile(left_region, 20)
                        right_valley = np.percentile(right_region, 20)
                        prominence = row[x] - max(left_valley, right_valley)
                        
                        # Use provided min_prominence
                        if prominence > min_prominence:
                            # Accept even flat peaks
                            peaks.append((y, x, prominence))
        
        # Ultra-aggressive edge peak detection
        if curve_type.upper() == "GR":
            for y in range(h):
                # Very low threshold for edge detection
                if prob_map[y, 0] > 0.1:  # Very low edge threshold
                    peaks.append((y, 0, prob_map[y, 0]))
                if prob_map[y, w-1] > 0.1:  # Very low edge threshold
                    peaks.append((y, w-1, prob_map[y, w-1]))
        
        # Accept all peaks, no filtering for maximum sensitivity
        return peaks  # Return all detected peaks, no filtering
    
    def is_near_peak(y, peaks, window=4):
        """Check if y coordinate is near a detected peak with expanded window"""
        return any(abs(py - y) <= window for py, px, prom in peaks)
    
    # AI-powered peak detection fallback
    def ai_detect_peaks(image_roi, curve_type="GR"):
        """Use Google Vision API to detect curve peaks as fallback"""
        if not VISION_API_AVAILABLE or vision_client is None:
            print("AI detection: Vision API not available")
            return []
        
        print(f"AI detection: Starting for {curve_type} curve...")
        try:
            # Ensure ROI is proper format for Vision API
            if len(image_roi.shape) == 3:
                # Convert BGR to RGB for Vision API
                roi_rgb = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
            else:
                roi_rgb = cv2.cvtColor(image_roi, cv2.COLOR_GRAY2RGB)
            
            # Convert to bytes
            _, buffer = cv2.imencode('.jpg', roi_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_bytes = buffer.tobytes()
            
            print("AI detection: Sending to Vision API...")
            
            # Use Vision API for object detection
            image = vision.Image(content=image_bytes)
            response = vision_client.object_localization(image=image)
            
            peaks = []
            objects = response.localized_object_annotations
            print(f"AI detection: Found {len(objects)} objects")
            
            h, w = image_roi.shape[:2]
            
            for obj in objects:
                print(f"AI detection: Object '{obj.name}' with confidence {obj.score:.2f}")
                if obj.name.lower() in ['line', 'curve', 'graph', 'drawing'] and obj.score > 0.3:
                    # Convert normalized vertices to pixel coordinates
                    vertices = obj.bounding_poly.normalized_vertices
                    if len(vertices) >= 4:
                        # Calculate center points for peaks
                        y_coords = [v.y * h for v in vertices]
                        x_coords = [v.x * w for v in vertices]
                        
                        # Create peaks along detected curve
                        for i in range(len(y_coords)):
                            peaks.append((int(y_coords[i]), int(x_coords[i]), obj.score))
            
            # Also try document text detection for numeric patterns
            try:
                doc_response = vision_client.document_text_detection(image=image)
                texts = doc_response.text_annotations
                if texts:
                    print(f"AI detection: Found text - {texts[0].description[:50]}...")
            except Exception as e:
                print(f"AI text detection failed: {e}")
            
            print(f"AI detection: Returning {len(peaks)} AI-detected peaks")
            return peaks
            
        except Exception as e:
            print(f"AI detection error: {e}")
            return []
        
        try:
            # Convert ROI to bytes for Vision API
            _, buffer = cv2.imencode('.jpg', image_roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_bytes = buffer.tobytes()
            
            # Use Vision API to detect curve features
            image = vision.Image(content=image_bytes)
            response = vision_client.text_detection(image=image)
            
            # Alternative: use document text detection for better curve detection
            response = vision_client.document_text_detection(image=image)
            
            # Extract text and analyze for curve patterns
            texts = response.text_annotations
            if texts:
                full_text = texts[0].description
                # Look for numeric patterns that might indicate curve values
                numbers = re.findall(r'\d+\.?\d*', full_text)
                
            # Use object detection for curve features (if available)
            response = vision_client.object_localization(image=image)
            objects = response.localized_object_annotations
            
            peaks = []
            for obj in objects:
                if obj.name.lower() in ['line', 'curve', 'graph']:
                    # Convert bounding box to peak coordinates
                    vertices = obj.bounding_poly.normalized_vertices
                    if len(vertices) >= 4:
                        # Calculate center as peak
                        center_x = (vertices[0].x + vertices[2].x) * w / 2
                        center_y = (vertices[0].y + vertices[2].y) * h / 2
                        peaks.append((int(center_y), int(center_x), 0.8))  # High confidence
            
            return peaks
        except Exception as e:
            print(f"AI peak detection failed: {e}")
            return []
    
    # Hybrid AI + Traditional peak detection
    peaks_traditional = detect_local_peaks(prob, min_prominence=0.005) if curve_type.upper() == "GR" else []
    peaks_ai = ai_detect_peaks(roi, curve_type) if curve_type.upper() == "GR" and VISION_API_AVAILABLE else []
    
    # Merge traditional and AI peaks
    all_peaks = peaks_traditional + peaks_ai
    
    # Final refinement with AI-enhanced sub-pixel accuracy
    def refine_subpixel_parabola(mask, xs, prob_map):
        """Refine positions using parabolic sub-pixel interpolation
        Given peak pixel x, fits parabola to (x-1), x, (x+1) to find true max.
        
        Formula: offset = 0.5 * (left - right) / (left - 2*center + right)
        """
        if mask is None or xs is None:
            return xs, np.zeros_like(xs)
        
        h, w = prob_map.shape
        xs_refined = xs.copy()
        subpixel_conf = np.ones(h, dtype=np.float32)
        
        for y in range(h):
            if not np.isfinite(xs[y]):
                continue
                
            x = int(round(xs[y]))
            if x < 1 or x >= w-1:
                continue
            
            # Get intensity values at peak and neighbors
            center = prob_map[y, x]
            left = prob_map[y, x-1] if x-1 >= 0 else center
            right = prob_map[y, x+1] if x+1 < w else center
            
            # Check if we have a valid parabola (concave down)
            denom = left - 2*center + right
            if abs(denom) > 1e-6:  # Avoid division by zero
                # Calculate sub-pixel offset
                offset = 0.5 * (left - right) / denom
                
                # Only apply if offset is reasonable (< 1 pixel)
                if abs(offset) < 1.0:
                    xs_refined[y] = x + offset
                else:
                    xs_refined[y] = xs[y]
            else:
                xs_refined[y] = xs[y]
        
        return xs_refined, subpixel_conf
    
    # Curvature-based refinement for missed peaks
    def curvature_based_refinement(xs, prob_map, curve_type="GR"):
        """Refine based on local curvature analysis to catch missed peaks"""
        h, w = prob_map.shape
        xs_refined = xs.copy()
        curv_conf = np.ones(h, dtype=np.float32)
        # Only meaningful for jagged GR logs; for smooth curves (RHOB, DTC, etc.)
        # the ±5px snap with a 5% threshold causes hundreds of random jumps on a
        # noisy prob_map, making the trace more erratic than it started.
        if curve_type.upper() != "GR":
            return xs_refined, curv_conf
        # Calculate curvature using second derivative
        valid_mask = np.isfinite(xs)
        if np.sum(valid_mask) < 5:
            return xs_refined, curv_conf
        
        # Fill NaNs for curvature calculation
        xs_smooth = xs.copy()
        xs_smooth[~valid_mask] = np.interp(np.where(~valid_mask)[0], 
                                          np.where(valid_mask)[0], 
                                          xs[valid_mask])
        
        # Calculate curvature (second derivative)
        curvature = np.gradient(np.gradient(xs_smooth))
        
        # Identify high curvature regions (likely missed peaks)
        high_curvature = np.abs(curvature) > np.percentile(np.abs(curvature[valid_mask]), 80)
        
        # Refine high curvature regions
        for y in range(h):
            if not valid_mask[y] or not high_curvature[y]:
                continue
                
            # Search locally around current position
            x_current = int(round(xs[y]))
            search_radius = 5
            
            # Define search window
            start = max(0, x_current - search_radius)
            end = min(w, x_current + search_radius + 1)
            
            # Find local maximum in probability map
            local_prob = prob_map[y, start:end]
            if local_prob.size > 0:
                local_max_idx = np.argmax(local_prob)
                local_max_val = local_prob[local_max_idx]
                
                # Only refine if local max is significantly better
                current_val = prob_map[y, x_current] if 0 <= x_current < w else 0
                if local_max_val > current_val * 1.05:  # 5% improvement threshold (Medium Greedy)
                    xs_refined[y] = start + local_max_idx
        
        return xs_refined, curv_conf

    def ensure_peak_crests(xs, conf, prob_map, peaks, hot_side=None, y_merge_window=3):
        """Ensure each vertical GR peak cluster has at least one crest sample.

        For each cluster of peaks in Y, we pick a single row and move that
        row's sample to the crest (tip) on the hot side, but only if the
        crest candidate is at least as strong in the probability map and
        farther toward the hot side than the current position. This avoids
        reintroducing zig-zag artifacts while still guaranteeing a tip dot.
        """
        if xs is None or peaks is None:
            return xs, conf

        if len(peaks) == 0:
            return xs, conf

        h, w = prob_map.shape
        xs_out = xs.copy()
        conf_out = conf.copy()

        # Filter to reasonably strong peaks to avoid pure noise
        strong_peaks = []
        for py, px, prom in peaks:
            if not np.isfinite(py) or not np.isfinite(px):
                continue
            py_i = int(py)
            px_i = int(px)
            if py_i < 0 or py_i >= h or px_i < 0 or px_i >= w:
                continue
            # Use either provided prominence or local prob as strength
            strength = float(prom) if np.isfinite(prom) else float(prob_map[py_i, px_i])
            if strength <= 0.005:
                continue
            strong_peaks.append((py_i, px_i, strength))

        if not strong_peaks:
            return xs_out, conf_out

        # Group peaks into vertical clusters by Y
        strong_peaks.sort(key=lambda p: p[0])
        clusters = []
        current = [strong_peaks[0]]
        for py, px, prom in strong_peaks[1:]:
            if abs(py - current[-1][0]) <= y_merge_window:
                current.append((py, px, prom))
            else:
                clusters.append(current)
                current = [(py, px, prom)]
        clusters.append(current)

        for cluster in clusters:
            # Pick a crest candidate within this vertical group.
            # Strategy: Find the strongest peak, but prefer "tips" (hot side)
            # if they are reasonably strong (at least 50% of the max score in the cluster).
            
            # 1. Find max score in cluster
            max_s = -1.0
            for py, px, prom in cluster:
                s = float(prob_map[py, px])
                if s > max_s:
                    max_s = s
            
            # 2. Select best candidate among those with reasonable score
            threshold = max(0.01, max_s * 0.5)
            best_key = None
            best_py, best_px, best_score = None, None, -1.0
            
            for py, px, prom in cluster:
                score = float(prob_map[py, px])
                if score < threshold:
                    continue
                    
                # Bias toward the hot side for the "key" (tie-breaking or primary sort)
                if hot_side == "right":
                    # Prioritize right-most X, then score
                    key = (px, score)
                elif hot_side == "left":
                    # Prioritize left-most X (largest -px), then score
                    key = (-px, score)
                else:
                    # Just score
                    key = (score, 0)

                # Select best candidate based on direction-aware key
                if best_key is None or key > best_key:
                    best_key = key
                    best_score = score
                    best_py, best_px = py, px

            if best_py is None:
                continue

            y = int(best_py)
            x_peak = int(best_px)
            if y < 0 or y >= h or x_peak < 0 or x_peak >= w:
                continue

            x_curr = xs_out[y] if np.isfinite(xs_out[y]) else None
            
            # TELEPORTATION GUARD:
            # Prevent snapping to peaks far from the existing trace.
            # If the current row has no trace, check neighbors to establish context.
            ref_x = x_curr
            if ref_x is None:
                # Search neighbors for a valid reference point
                for offset in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                    ny = y + offset
                    if 0 <= ny < h and np.isfinite(xs_out[ny]):
                        ref_x = xs_out[ny]
                        break
            
            # If we have a reference, enforce distance limit (30px)
            if ref_x is not None:
                if abs(x_peak - ref_x) > 30:
                    continue
            else:
                # If no reference nearby (isolated point), unsafe to snap -> skip
                continue

            x_curr_int = int(round(x_curr)) if x_curr is not None else None

            p_peak = float(prob_map[y, x_peak])
            p_curr = float(prob_map[y, x_curr_int]) if x_curr_int is not None and 0 <= x_curr_int < w else 0.0

            move = False
            if x_curr is None:
                move = True
            else:
                if hot_side == "right":
                    if x_peak > x_curr_int and p_peak >= max(0.1, p_curr * 0.9):
                        move = True
                elif hot_side == "left":
                    if x_peak < x_curr_int and p_peak >= max(0.1, p_curr * 0.9):
                        move = True
                else:
                    if p_peak >= p_curr * 1.05:
                        move = True

            if not move:
                continue

            xs_out[y] = float(x_peak)
            conf_out[y] = max(conf_out[y], p_peak)

        return xs_out, conf_out

    # Apply final refinements
    xs_fused = np.full(h, np.nan, dtype=np.float32)
    confidence = np.zeros(h, dtype=np.float32)
    
    # Peak-aware fusion with curvature refinement
    peaks = detect_local_peaks(prob, min_prominence=0.005)
    
    for y in range(h):
        valid_indices = []
        valid_xs = []
        valid_confs = []
        cand_scales = []  # accumulator for per-candidate scale values (renamed to avoid shadowing outer valid_scales)
        
        for i, (xs_s, conf_s, scale) in enumerate(zip(all_xs, all_confs, valid_scales)):
            if xs_s is not None and y < xs_s.size and np.isfinite(xs_s[y]):
                x_int = int(round(xs_s[y]))
                if 0 <= x_int < w:
                    valid_indices.append(i)
                    valid_xs.append(xs_s[y])
                    valid_confs.append(conf_s[y])
                    cand_scales.append(scale)

        # Confidence-weighted fusion across all scales
        if valid_xs:
            total_conf = sum(valid_confs)
            if total_conf > 0:
                xs_fused[y] = sum(x * c for x, c in zip(valid_xs, valid_confs)) / total_conf
                confidence[y] = max(valid_confs)
            else:
                xs_fused[y] = float(np.mean(valid_xs))
                confidence[y] = 0.0

    # Push sampled points to hot-side crest tips
    if peaks:
        xs_fused, confidence = ensure_peak_crests(
            xs_fused, confidence, prob, peaks, hot_side=hot_side
        )

    return xs_fused, confidence


def trace_curve_greedy_peaks(mask, max_jump=30, min_prob=0.02):
    """Trace curve by greedily following the strongest peak in each row.

    This tracer finds the brightest pixel in each row within a search window,
    with NO smoothness penalty - it follows every peak and valley exactly.

    Args:
        mask: Probability map (0-255)
        max_jump: Maximum horizontal jump between rows
        min_prob: Minimum probability to consider

    Returns:
        Array of x-coordinates, one per row
    """
    if mask is None:
        return None

    h, w = mask.shape[:2]
    prob = mask.astype(np.float32) / 255.0
    xs = np.full(h, np.nan, dtype=np.float32)

    row_max_vals = prob.max(axis=1)
    row_max_xs = prob.argmax(axis=1).astype(np.float32)

    valid_rows = row_max_vals >= min_prob

    if not np.any(valid_rows):
        return xs

    start_row = int(np.argmax(row_max_vals))
    xs[start_row] = row_max_xs[start_row]

    current_x = xs[start_row]
    for y in range(start_row + 1, h):
        row = prob[y]
        x0 = max(0, int(current_x) - max_jump)
        x1 = min(w, int(current_x) + max_jump + 1)
        window = row[x0:x1]
        if window.max() >= min_prob:
            best_local = np.argmax(window)
            current_x = float(x0 + best_local)
            xs[y] = current_x

    current_x = xs[start_row]
    for y in range(start_row - 1, -1, -1):
        row = prob[y]
        x0 = max(0, int(current_x) - max_jump)
        x1 = min(w, int(current_x) + max_jump + 1)
        window = row[x0:x1]
        if window.max() >= min_prob:
            best_local = np.argmax(window)
            current_x = float(x0 + best_local)
            xs[y] = current_x

    return xs


def refine_to_smart_edges(mask, xs, min_prob=0.005):
    """
    Smart refinement based on local geometry (SR Tuned):
    - Vertical runs (width < 10px): Center on ink (CoM).
    - Spikes/Bumps: Snap to EDGE if deviation > 3.0px.
    """
    if mask is None or xs is None:
        return xs
    
    h, w = mask.shape
    prob = mask.astype(np.float32)
    xs_smart = xs.copy()
    
    # 1. Compute local median
    try:
        s = pd.Series(xs)
        local_med = s.rolling(window=15, center=True, min_periods=1).median().to_numpy() # Increased window for SR
    except:
        local_med = xs.copy()

    for y in range(h):
        x = xs_smart[y]
        if not np.isfinite(x):
            continue
            
        ix = int(round(x))
        if ix < 0 or ix >= w:
            continue
            
        # 2. Find connected ink chunk
        # TELEPORTATION GUARD: Reduced from 200 to 30 to prevent snapping to distant noise
        w_start = max(0, ix - 30)
        w_end = min(w, ix + 31)
        row_slice = prob[y, w_start:w_end]
        
        # Simple threshold
        ink_indices = np.where(row_slice > min_prob * 255)[0] if prob.max() > 1.0 else np.where(row_slice > min_prob)[0]
        if ink_indices.size == 0:
            continue
            
        ink_indices_global = ink_indices + w_start
        min_ink = ink_indices_global[0]
        max_ink = ink_indices_global[-1]
        ink_width = max_ink - min_ink
        
        # 3. Geometric Decision
        med_val = local_med[y]
        if not np.isfinite(med_val):
            med_val = x
            
        diff = x - med_val
        
        # LOGIC FOR 2x SUPER-RES:
        # Width < 14px (was 10) -> Center
        # Deviation > 5.0px (was 3.0) -> Snap
        
        if ink_width < 14:
            # Narrow Line -> Center of Mass
            weights = row_slice[ink_indices]
            coords = ink_indices_global
            total_w = weights.sum()
            if total_w > 0:
                xs_smart[y] = (coords * weights).sum() / total_w
            else:
                xs_smart[y] = (min_ink + max_ink) / 2.0
        else:
            # Wide/Feature
            if diff > 5.0:
                xs_smart[y] = float(max_ink) # Snap Right
            elif diff < -5.0:
                xs_smart[y] = float(min_ink) # Snap Left
            elif abs(diff) < 2.5:
                # Stable Zone -> Center of Mass
                weights = row_slice[ink_indices]
                coords = ink_indices_global
                total_w = weights.sum()
                if total_w > 0:
                    xs_smart[y] = (coords * weights).sum() / total_w
                else:
                    xs_smart[y] = (min_ink + max_ink) / 2.0
            else:
                # Transition Zone -> Keep current
                xs_smart[y] = x

    return xs_smart

def refine_peaks_and_valleys(mask, xs, search_radius=25, min_prob=0.1):
    """Specifically refine peaks and valleys where the curve changes direction.
    
    This function detects where the traced curve has local extrema (peaks/valleys)
    and searches more aggressively in those regions to find the true curve position.
    
    Args:
        mask: Probability map (0-255)
        xs: Array of x-coordinates from initial trace
        search_radius: How far to search horizontally at peaks/valleys
        min_prob: Minimum probability to consider a pixel as curve
    
    Returns:
        Refined x-coordinates
    """
    if mask is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size < 5:
        return xs
    
    h, w = mask.shape[:2]
    if h < 5 or w < 5:
        return xs
    
    prob = mask.astype(np.float32) / 255.0
    xs_ref = xs.copy()
    
    # Detect peaks and valleys by looking at the derivative of x positions
    # A peak is where x goes from increasing to decreasing (or vice versa for valley)
    valid_mask = np.isfinite(xs_ref)
    if np.sum(valid_mask) < 5:
        return xs_ref
    
    # Fill gaps for derivative calculation
    xs_filled = xs_ref.copy()
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0:
        xs_filled[:valid_indices[0]] = xs_ref[valid_indices[0]]
        xs_filled[valid_indices[-1]:] = xs_ref[valid_indices[-1]]
        # Linear interpolation for gaps
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            if end_idx - start_idx > 1:
                xs_filled[start_idx:end_idx] = np.linspace(
                    xs_ref[start_idx], xs_ref[end_idx], end_idx - start_idx
                )
    
    # Calculate first derivative (velocity)
    dx = np.diff(xs_filled)
    
    # Find zero crossings in derivative (peaks and valleys) - vectorized
    # Sign change indicates peak or valley
    sign_change = (dx[:-1] * dx[1:]) < 0
    # Large magnitude change also indicates sharp turn; use a slightly lower
    # threshold so more subtle bends get refined
    large_change = np.abs(np.diff(dx)) > 2
    extrema_mask = sign_change | large_change
    extrema_rows = np.where(extrema_mask)[0] + 1  # +1 because diff reduces length
    
    # Refine at each extremum with a wider search
    for y in extrema_rows:
        if y < 0 or y >= h:
            continue
        x = xs_ref[y]
        if not np.isfinite(x):
            continue
        
        x_c = int(round(float(x)))
        if x_c < 0 or x_c >= w:
            continue
        
        row = prob[y]
        x0 = max(0, x_c - search_radius)
        x1 = min(w, x_c + search_radius + 1)
        window = row[x0:x1]
        if window.size == 0:
            continue
        
        max_p = float(window.max())
        if max_p < min_prob:
            continue
        
        # Find the peak position
        local_peak_idx = int(np.argmax(window))
        x_peak = x0 + local_peak_idx
        
        # Use weighted centroid of strong pixels around the peak
        # Use HIGH power (4) to stay very close to the peak
        xs_local = np.arange(x0, x1, dtype=np.float32)
        weights = window.astype(np.float32)
        peak_mask = weights >= max_p * 0.3  # Include pixels at 30% of peak
        
        if np.any(peak_mask):
            weights_masked = (weights * peak_mask.astype(np.float32)) ** 4
            wsum = float(weights_masked.sum())
            if wsum > 0:
                x_centroid = float((xs_local * weights_masked).sum() / wsum)
                
                # EXTREMUM PUSH: If this is a sharp peak/valley, push towards the extremity
                # Check 3 rows above and below to see curvature
                if y > 2 and y < h - 3:
                    prev_x = np.nanmedian(xs_ref[y-3:y])
                    next_x = np.nanmedian(xs_ref[y+1:y+4])
                    if np.isfinite(prev_x) and np.isfinite(next_x):
                        # Detected a peak (point is > neighbors) or valley (point is < neighbors)
                        # Use 0.2 buffer to catch almost all real features while ignoring microscopic noise.
                        # LOGIC UPDATE: Allow "Plateaus" (flat tops) to be detected as peaks.
                        # Strict peak: > prev AND > next
                        # Plateau start: > prev AND >= next
                        # Plateau end: >= prev AND > next
                        buf = 0.2
                        
                        # Right Peak (High value)
                        # Check if strictly higher than at least one side, and at least equal (within tolerance) to the other
                        is_right_peak = (x_centroid > prev_x + buf and x_centroid >= next_x - buf) or \
                                        (x_centroid >= prev_x - buf and x_centroid > next_x + buf)

                        # Left Peak (Low value) - Valley
                        is_left_peak = (x_centroid < prev_x - buf and x_centroid <= next_x + buf) or \
                                       (x_centroid <= prev_x + buf and x_centroid < next_x - buf)
                        
                        # Helper to find connected ink span around the centroid
                        def get_connected_range(weights, center_idx):
                            # Ensure center_idx is within bounds
                            center_idx = max(0, min(len(weights)-1, center_idx))
                            
                            # If we landed on empty space, find nearest ink
                            if weights[center_idx] == 0:
                                valid = np.where(weights > 0)[0]
                                if valid.size == 0:
                                    return center_idx, center_idx
                                # Closest valid index
                                center_idx = valid[np.abs(valid - center_idx).argmin()]
                                
                            # Expand Left
                            l_idx = center_idx
                            while l_idx > 0 and weights[l_idx-1] > 0:
                                l_idx -= 1
                                
                            # Expand Right
                            r_idx = center_idx
                            while r_idx < len(weights)-1 and weights[r_idx+1] > 0:
                                r_idx += 1
                                
                            return l_idx, r_idx

                        center_idx_local = int(round(x_centroid - x0))
                        
                        if is_right_peak:
                            # Find rightmost CONNECTED pixel
                            l_idx, r_idx = get_connected_range(weights, center_idx_local)
                            rightmost_x = x0 + r_idx
                            xs_ref[y] = float(rightmost_x)
                            
                        elif is_left_peak:
                            # Find leftmost CONNECTED pixel
                            l_idx, r_idx = get_connected_range(weights, center_idx_local)
                            leftmost_x = x0 + l_idx
                            xs_ref[y] = float(leftmost_x)
                        else:
                            xs_ref[y] = x_centroid
                else:
                    xs_ref[y] = x_centroid
                
                # Also refine a couple rows above and below (reduced from ±5 to ±2 for speed)
                half_radius = search_radius // 3
                for dy in [-2, -1, 1, 2]:
                    y2 = y + dy
                    if y2 < 0 or y2 >= h:
                        continue
                    x2 = xs_ref[y2]
                    if not np.isfinite(x2):
                        continue
                    
                    x2_c = int(round(float(x2)))
                    row2 = prob[y2]
                    x0_2 = max(0, x2_c - half_radius)
                    x1_2 = min(w, x2_c + half_radius + 1)
                    window2 = row2[x0_2:x1_2]
                    if window2.size == 0:
                        continue
                    
                    # Simplified: just snap to max in window
                    best_idx = np.argmax(window2)
                    if window2[best_idx] >= min_prob:
                        xs_ref[y2] = float(x0_2 + best_idx)
    
    return xs_ref


def ensure_peaks_have_points(
    mask,
    xs,
    min_prob=0.08,
    min_peak_prominence=0.03,
    max_shift=40,
):
    """Ensure every significant peak in the probability map has a traced point.
    
    Optimized version using vectorized operations.
    
    Args:
        mask: Probability map (0-255)
        xs: Array of x-coordinates from trace
        min_prob: Minimum probability to consider a peak
        min_peak_prominence: Minimum prominence (height above neighbors) for a peak
    
    Returns:
        Refined x-coordinates with peaks properly captured
    """
    if mask is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size < 3:
        return xs
    
    h, w = mask.shape[:2]
    if h < 3 or w < 3:
        return xs
    
    prob = mask.astype(np.float32) / 255.0
    xs_ref = xs.copy()
    
    # Vectorized peak detection: find where each pixel is greater than both neighbors
    # This is much faster than row-by-row Python loops
    left_shift = np.roll(prob, 1, axis=1)
    right_shift = np.roll(prob, -1, axis=1)
    is_peak = (prob > left_shift) & (prob > right_shift) & (prob >= min_prob)
    # Zero out edges
    is_peak[:, 0] = False
    is_peak[:, -1] = False
    
    # Process only rows that have peaks
    rows_with_peaks = np.where(np.any(is_peak, axis=1))[0]
    
    for y in rows_with_peaks:
        row = prob[y]
        current_x = xs_ref[y]
        
        # Get candidate peak positions in this row
        peak_xs = np.where(is_peak[y])[0]
        if peak_xs.size == 0:
            continue
        
        peak_probs = row[peak_xs]
        # Estimate local prominence vs immediate neighbors
        left_vals = row[peak_xs - 1]
        right_vals = row[peak_xs + 1]
        prominences = peak_probs - np.maximum(left_vals, right_vals)
        valid_mask = prominences >= min_peak_prominence
        if not np.any(valid_mask):
            continue
        peak_xs = peak_xs[valid_mask]
        peak_probs = peak_probs[valid_mask]
        prominences = prominences[valid_mask]
        
        # Sort peaks by strength descending
        sort_idx = np.argsort(-peak_probs)
        peak_xs = peak_xs[sort_idx]
        peak_probs = peak_probs[sort_idx]
        
        # If current trace is far from the best peak, snap to it
        if np.isfinite(current_x):
            current_x_int = int(round(current_x))
            current_prob = row[current_x_int] if 0 <= current_x_int < w else 0
            
            # Prefer a peak near the current trace when possible
            target_peak_x = None
            for px in peak_xs:
                if abs(px - current_x) <= max_shift:
                    target_peak_x = px
                    break
            
            # If no peak is close enough, DO NOT jump to a far-away peak.
            # Stick to the current trace.
            if target_peak_x is None:
                continue

            target_prob = row[target_peak_x]
            
            # Only move if the nearby target is clearly stronger than current position
            if target_prob > current_prob * 1.05:
                x0 = max(0, target_peak_x - 6)
                x1 = min(w, target_peak_x + 7)
                window = row[x0:x1]
                if window.size == 0:
                    continue
                xs_local = np.arange(x0, x1, dtype=np.float32)
                weights = window ** 3
                wsum = weights.sum()
                if wsum > 0:
                    xs_ref[y] = float((xs_local * weights).sum() / wsum)
        else:
            xs_ref[y] = float(peak_xs[0])
    
    return xs_ref


def _push_crest_hot_side(mask, xs, hot_side, curve_type=None, min_prob=0.01, max_shift=30):
    # Only apply this aggressive crest push for GR-type curves
    if curve_type is not None and str(curve_type).upper() != "GR":
        return xs
    if mask is None or xs is None:
        return xs
    if hot_side not in ("left", "right"):
        return xs
    if not hasattr(xs, "size") or xs.size == 0:
        return xs
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return xs
    prob = mask.astype(np.float32) / 255.0
    xs_out = xs.copy()
    moved = 0
    for y in range(h):
        x = xs_out[y]
        if not np.isfinite(x):
            continue
        x_int = int(round(x))
        if x_int < 0 or x_int >= w:
            continue
        row = prob[y]
        if hot_side == "right":
            start = x_int
            end = min(w, x_int + max_shift + 1)
        else:
            start = max(0, x_int - max_shift)
            end = x_int + 1
        if end <= start:
            continue
        seg = row[start:end]
        if seg.size == 0:
            continue
        # Find all pixels above a very low probability floor and move to the
        # **furthest** such pixel toward the hot side. This guarantees that,
        # wherever the GR ink extends horizontally, at least some rows will
        # put their sample at the outer tip.
        if hot_side == "right":
            cand = np.where(seg >= min_prob)[0]
            if cand.size == 0:
                continue
            peak_idx = int(cand[-1])  # furthest right
        else:
            cand = np.where(seg >= min_prob)[0]
            if cand.size == 0:
                continue
            peak_idx = int(cand[0])  # furthest left
        x_peak = start + peak_idx
        if x_peak != x_int:
            xs_out[y] = float(x_peak)
            moved += 1
    try:
        if moved > 0:
            print(f"[crest] moved {moved} rows toward {hot_side} (curve_type={curve_type})")
        else:
            print(f"[crest] no moves (hot_side={hot_side}, curve_type={curve_type})")
    except Exception:
        pass
    return xs_out


def ensure_gr_peak_crests(xs, prob_map, hot_side=None, min_prob=0.002, y_merge_window=5, max_shift_frac=0.6):
    """For GR colored-mode traces, guarantee at least one crest sample per spike.

    This is a conservative helper that, for each vertical cluster of rows, moves
    at most ONE row further toward the hot-side tip, based on the probability
    map. It avoids reshaping the whole trace while ensuring every spike has a
    dot at its outermost visible tip.
    """
    if xs is None or prob_map is None:
        return xs
    if hot_side not in ("left", "right"):
        return xs
    if not hasattr(xs, "size") or xs.size == 0:
        return xs

    h, w = prob_map.shape
    n = xs.size
    m = min(h, n)
    if m == 0:
        return xs

    prob = prob_map[:m].astype(np.float32)
    xs_out = xs.copy()

    # Allow large moves, but not across the entire track; cap at a fraction
    # of the track width.
    # TELEPORTATION GUARD: Cap at 30px to prevent shooting off to distant noise
    max_dx_allowed = min(30, max(1, int(max_shift_frac * w)))

    # 1) Build crest candidates per row
    candidates = []  # (y, crest_x, dx)
    for y in range(m):
        x_curr = xs_out[y]
        if not np.isfinite(x_curr):
            continue
        row = prob[y]
        x_curr_int = int(round(x_curr))
        if x_curr_int < 0 or x_curr_int >= w:
            continue

        on = row >= float(min_prob)
        if not np.any(on):
            continue

        if bool(on[x_curr_int]):
            l_idx = int(x_curr_int)
            r_idx = int(x_curr_int)
        else:
            idxs = np.where(on)[0]
            if idxs.size == 0:
                continue
            nearest = int(idxs[int(np.argmin(np.abs(idxs - x_curr_int)))])
            if abs(nearest - x_curr_int) > max_dx_allowed:
                continue
            l_idx = int(nearest)
            r_idx = int(nearest)

        while l_idx > 0 and bool(on[l_idx - 1]):
            l_idx -= 1
        while r_idx + 1 < w and bool(on[r_idx + 1]):
            r_idx += 1

        crest_x = int(r_idx if hot_side == "right" else l_idx)
        dx = crest_x - x_curr_int if hot_side == "right" else x_curr_int - crest_x
        if dx <= 0 or dx > max_dx_allowed:
            continue
        candidates.append((y, crest_x, dx))

    if not candidates:
        return xs_out

    # 2) Group into vertical clusters by Y (each cluster ≈ one spike)
    candidates.sort(key=lambda c: c[0])
    clusters = []
    current = [candidates[0]]
    for y, crest_x, dx in candidates[1:]:
        if abs(y - current[-1][0]) <= y_merge_window:
            current.append((y, crest_x, dx))
        else:
            clusters.append(current)
            current = [(y, crest_x, dx)]
    clusters.append(current)

    # 3) For each cluster, move exactly one row: the one that can move
    #    furthest toward the hot side.
    moved_clusters = 0
    for cluster in clusters:
        y_best, x_best, dx_best = max(cluster, key=lambda t: t[2])
        dx_keep = max(1, int(round(float(dx_best) * 0.8)))
        keep = [t for t in cluster if int(t[2]) >= dx_keep]
        keep.sort(key=lambda t: abs(int(t[0]) - int(y_best)))
        for (yy, xx, _dd) in keep[:3]:
            xs_out[int(yy)] = float(xx)
        moved_clusters += 1

    try:
        if moved_clusters:
            print(f"[gr-crest] moved {moved_clusters} crest rows (clusters={len(clusters)})")
    except Exception:
        pass

    return xs_out


def refine_to_stroke_centerline(mask, xs, threshold_ratio=0.5, window_size=None):
    """Refine trace to the centerline of the curve stroke width.
    
    Uses a half-maximum window (FWHM) to estimate stroke width and blends
    geometric and weighted centers for stability.
    """
    if mask is None or xs is None:
        return xs
    if not hasattr(xs, "size") or xs.size < 3:
        return xs
    
    h, w = mask.shape[:2]
    if h < 3 or w < 3:
        return xs
    
    prob = mask.astype(np.float32) / 255.0
    xs_ref = xs.copy()
    
    valid_rows = np.where(np.isfinite(xs_ref))[0]
    try:
        if window_size is not None:
            search_radius = max(1, int(window_size))
        else:
            search_radius = 15
    except Exception:
        search_radius = 15
    
    for y in valid_rows:
        x_prev = float(xs_ref[y])
        x_c = int(round(x_prev))
        if x_c < 0 or x_c >= w:
            continue
        
        row = prob[y]
        x0 = max(0, x_c - search_radius)
        x1 = min(w, x_c + search_radius + 1)
        window = row[x0:x1]
        if window.size == 0:
            continue
        
        max_val = window.max()
        if max_val < 0.05:
            continue
        
        peak_idx = int(np.argmax(window))
        thr = float(max_val) * float(threshold_ratio)
        if thr <= 0:
            continue

        above = window >= thr
        if not np.any(above):
            continue

        # Choose the contiguous above-threshold band that overlaps the current
        # x (or is nearest). Using argmax alone can lock onto a bright edge on
        # thick strokes.
        segs = []
        in_seg = False
        seg_start = 0
        for i in range(int(above.size)):
            if bool(above[i]) and not in_seg:
                in_seg = True
                seg_start = i
            elif (not bool(above[i])) and in_seg:
                segs.append((int(seg_start), int(i - 1)))
                in_seg = False
        if in_seg:
            segs.append((int(seg_start), int(above.size) - 1))

        x_rel = float(x_prev) - float(x0)
        chosen = None
        best_dist = None
        for (l, r) in segs:
            if l <= x_rel <= r:
                chosen = (l, r)
                best_dist = 0.0
                break
            if x_rel < l:
                dist = float(l) - x_rel
            else:
                dist = x_rel - float(r)
            if best_dist is None or dist < best_dist:
                chosen = (l, r)
                best_dist = dist
            elif best_dist is not None and abs(dist - best_dist) < 1e-6:
                # Tie-breaker: prefer the segment with higher local intensity
                try:
                    if chosen is not None:
                        l0, r0 = chosen
                        if float(window[l:r + 1].max()) > float(window[l0:r0 + 1].max()):
                            chosen = (l, r)
                except Exception:
                    pass

        if chosen is None:
            continue

        left_idx, right_idx = chosen

        stroke_center = (left_idx + right_idx) / 2.0

        # Optional weighted correction within the band. Keep it mild to prevent
        # a consistent pull toward stronger edges.
        final_center = float(stroke_center)
        try:
            stroke_slice = window[left_idx:right_idx + 1]
            if stroke_slice.size > 0:
                xs_local = np.arange(left_idx, right_idx + 1, dtype=np.float32)
                weights = np.power(stroke_slice.astype(np.float32), 1.5)
                wsum = float(weights.sum())
                if wsum > 1e-8:
                    weighted_center = float((xs_local * weights).sum() / wsum)
                    final_center = 0.75 * float(stroke_center) + 0.25 * weighted_center
        except Exception:
            final_center = float(stroke_center)

        x_new = float(x0 + final_center)

        # Clamp and blend for stability
        max_shift = max(1.5, float(search_radius) * 0.6)
        dx = x_new - x_prev
        if dx > max_shift:
            x_new = x_prev + max_shift
        elif dx < -max_shift:
            x_new = x_prev - max_shift

        alpha = 0.85
        xs_ref[y] = float((1.0 - alpha) * x_prev + alpha * x_new)
    
    return xs_ref


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
        # Compute rolling std/mean for outlier detection. Pandas requires
        # min_periods <= window, so ensure the rolling window is at least 2
        # even when the smoothing window is 1 (our "almost no smoothing" case).
        win_outlier = max(2, int(window))
        rolling_std = s[valid].rolling(win_outlier, min_periods=2, center=True).std()
        rolling_mean = s[valid].rolling(win_outlier, min_periods=2, center=True).mean()
        
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

def trace_curve_direct_centerline(mask, threshold=10):
    """
    Find the exact centerline by finding the peak intensity in each row.
    Uses a weighted centroid around the peak for sub-pixel accuracy.
    
    Args:
        mask: Grayscale probability map (0-255)
        threshold: Minimum intensity to consider as ink
    
    Returns:
        Array of x-coordinates, one per row
    """
    h, w = mask.shape[:2]
    xs = np.full(h, np.nan, dtype=np.float32)
    
    # For each row, find the peak intensity
    for y in range(h):
        row = mask[y, :].astype(np.float32)
        
        # Find pixels above threshold
        valid_mask = row >= threshold
        
        if np.any(valid_mask):
            # Find the maximum intensity in this row
            max_intensity = row[valid_mask].max()
            
            if max_intensity > threshold:
                # Find all pixels near the peak (within 5% of max for good balance)
                peak_threshold = max_intensity * 0.95
                peak_mask = row >= peak_threshold
                peak_indices = np.where(peak_mask)[0]
                
                if len(peak_indices) > 0:
                    # Use intensity-weighted centroid with cubic weighting
                    # This gives maximum emphasis to the absolute peak
                    weights = row[peak_indices]
                    # Apply cubic emphasis on the peak for ultra-precise centering
                    gaussian_weights = weights ** 3  # Cubic emphasizes peak even more
                    xs[y] = float(np.sum(peak_indices * gaussian_weights) / np.sum(gaussian_weights))
                else:
                    # Fallback: weighted centroid of all valid pixels
                    valid_indices = np.where(valid_mask)[0]
                    weights = row[valid_indices]
                    xs[y] = float(np.sum(valid_indices * weights) / np.sum(weights))
    
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

def write_las_simple(depth, curve_data, depth_unit="FT", header_metadata=None):
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

    md = header_metadata if isinstance(header_metadata, dict) else {}
    comp = (md.get('comp') or md.get('company') or '').strip() if isinstance(md.get('comp') or md.get('company') or '', str) else ''
    well = (md.get('well') or '').strip() if isinstance(md.get('well') or '', str) else ''
    fld = (md.get('fld') or md.get('field') or '').strip() if isinstance(md.get('fld') or md.get('field') or '', str) else ''
    loc = (md.get('loc') or md.get('location') or '').strip() if isinstance(md.get('loc') or md.get('location') or '', str) else ''
    county = (md.get('county') or '').strip() if isinstance(md.get('county') or '', str) else ''
    state = (md.get('state') or '').strip() if isinstance(md.get('state') or '', str) else ''
    prov = (md.get('prov') or md.get('province') or '').strip() if isinstance(md.get('prov') or md.get('province') or '', str) else ''
    srvc = (md.get('srvc') or md.get('service') or md.get('service_company') or '').strip() if isinstance(md.get('srvc') or md.get('service') or md.get('service_company') or '', str) else ''
    date = (md.get('date') or '').strip() if isinstance(md.get('date') or '', str) else ''
    api = (md.get('api') or '').strip() if isinstance(md.get('api') or '', str) else ''
    uwi = (md.get('uwi') or '').strip() if isinstance(md.get('uwi') or '', str) else ''

    if comp:
        lines.append(f" COMP.       {comp}:" + eol)
    lines.append(f" WELL.       {(well or 'DIGITIZED_LOG')}:" + eol)
    if fld:
        lines.append(f" FLD .       {fld}:" + eol)
    if loc:
        lines.append(f" LOC .       {loc}:" + eol)
    if county:
        lines.append(f" CNTY.       {county}:" + eol)
    if state:
        lines.append(f" STAT.       {state}:" + eol)
    if prov:
        lines.append(f" PROV.       {prov}:" + eol)
    if srvc:
        lines.append(f" SRVC.       {srvc}:" + eol)
    if date:
        lines.append(f" DATE.       {date}:" + eol)
    if api:
        lines.append(f" API .       {api}:" + eol)
    if uwi:
        lines.append(f" UWI .       {uwi}:" + eol)

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


def build_las_filename_from_metadata(header_metadata, default_name="digitized_log.las"):
    if not isinstance(header_metadata, dict):
        return default_name
    well = header_metadata.get('well')
    comp = header_metadata.get('comp') or header_metadata.get('company')

    def _clean(s):
        if not isinstance(s, str):
            return ''
        s = s.strip()
        if not s:
            return ''
        import re
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
        s = s.strip("_-")
        return s[:80]

    well_s = _clean(well)
    comp_s = _clean(comp)
    if well_s and comp_s:
        return f"{comp_s}__{well_s}.las"
    if well_s:
        return f"{well_s}.las"
    return default_name

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

        full_text = ""
        if response.text_annotations:
            full_text = response.text_annotations[0].description

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
            'suggestions': suggestions,
            'full_text': full_text
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


@app.route('/crop_to_panel', methods=['POST'])
def crop_to_panel():
    """Crop the uploaded image to a working panel/depth window.

    Expects JSON with:
      - image: data URL string (same as /digitize)
      - region: { left_px, right_px, top_px, bottom_px } in image pixel coords

    Returns a new data URL plus the cropped width/height so the frontend can
    treat it as a shorter working image.
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

    crop = img[top:bottom, left:right]
    ch, cw, _ = crop.shape

    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode crop'}), 500

    b64 = base64.b64encode(buf).decode('ascii')
    data_url = f"data:image/jpeg;base64,{b64}"

    return jsonify({
        'success': True,
        'image': data_url,
        'width': int(cw),
        'height': int(ch),
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
    # 1) Depth label candidates: include left-side AND center-column numeric entries
    # Many logs print depth in a center column, not just on the left edge.
    xs_all = [float(e['x']) for e in numeric_entries if 'x' in e]
    if xs_all:
        min_x = min(xs_all)
        max_x = max(xs_all)
        span_x = max(max_x - min_x, 1.0)
        # Expand threshold to 60% to catch center-column depth labels
        depth_x_threshold = min_x + 0.60 * span_x
    else:
        depth_x_threshold = None

    depth_label_candidates = []
    for e in numeric_entries:
        val = float(e['value'])
        # Filter: depth values are typically in range 0-50000 ft, not tiny curve scales
        if val < 0 or val > 50000:
            continue
        if depth_x_threshold is not None and float(e['x']) <= depth_x_threshold:
            depth_label_candidates.append({
                'value': val,
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


@app.route('/propose_curves', methods=['POST'])
def propose_curves():
    """Use Vision + LLM to propose curve tracks for a selected panel.

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

    panel = img[top:bottom, left:right]
    panel_h, panel_w, _ = panel.shape
    if panel_h < 2 or panel_w < 2:
        return jsonify({'success': False, 'error': 'Panel too small for curve suggestion'}), 400

    # Detect tracks within the panel using edge-based detector
    local_tracks = auto_detect_tracks(panel) or []
    tracks_out = []
    for idx, (lx, rx) in enumerate(local_tracks):
        try:
            lx_f = float(lx)
            rx_f = float(rx)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lx_f) or not np.isfinite(rx_f) or rx_f <= lx_f:
            continue
        tracks_out.append({
            'index': idx,
            'left_px': lx_f,
            'right_px': rx_f,
        })

    # Fallback: if we only found 0 or 1 track on a reasonably wide panel,
    # synthesize several equal-width tracks so curves can cover the full width.
    if len(tracks_out) <= 1 and panel_w >= 400:
        synth_tracks = []
        n_segments = 4
        seg_w = float(panel_w) / float(n_segments)
        for i in range(n_segments):
            lx_f = i * seg_w
            rx_f = (i + 1) * seg_w
            synth_tracks.append({
                'index': i,
                'left_px': lx_f,
                'right_px': rx_f,
            })
        if synth_tracks:
            print(f"⚠️  auto_detect_tracks found {len(tracks_out)} track(s); using {len(synth_tracks)} synthetic tracks instead.")
            tracks_out = synth_tracks

    if not tracks_out:
        return jsonify({'success': False, 'error': 'No tracks detected in panel for curve suggestion.'}), 400

    # Run Vision OCR on the same panel to get numeric + label hints
    ok, buf = cv2.imencode('.jpg', panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode panel for OCR'}), 500

    crop_bytes = buf.tobytes()
    detected_text = detect_text_vision_api(crop_bytes)
    ocr_suggestions = detected_text.get('suggestions', {}) or {}

    # Attach color hints based on panel image content
    try:
        ocr_suggestions = attach_color_hints_to_ocr_curves(panel, ocr_suggestions)
        detected_text['suggestions'] = ocr_suggestions
    except Exception:
        pass

    curve_payload = build_curve_suggestion_payload(panel, tracks_out, ocr_suggestions, detected_text)
    ai_result = call_ai_curve_suggestions(curve_payload)
    if not ai_result or not isinstance(ai_result, dict):
        return jsonify({'success': False, 'error': 'AI curve suggestion failed or returned no result.'}), 500

    ai_curves = ai_result.get('curves') or []
    print(f"🤖 AI returned {len(ai_curves)} curve suggestions for {len(tracks_out)} detected tracks")

    curves_cfg = []
    rejected_reasons = []
    for idx, c in enumerate(ai_curves):
        try:
            track_index = int(c.get('track_index'))
        except (TypeError, ValueError):
            rejected_reasons.append(f"Curve {idx}: invalid track_index type")
            continue
        if track_index < 0 or track_index >= len(tracks_out):
            rejected_reasons.append(f"Curve {idx}: track_index {track_index} out of range (0-{len(tracks_out)-1})")
            continue

        track = tracks_out[track_index]
        abs_left = float(left) + float(track.get('left_px', 0.0))
        abs_right = float(left) + float(track.get('right_px', 0.0))
        if not np.isfinite(abs_left) or not np.isfinite(abs_right) or abs_right <= abs_left:
            rejected_reasons.append(f"Curve {idx}: invalid pixel range")
            continue

        mnemonic = (c.get('mnemonic') or track.get('name') or '').strip()
        if not mnemonic:
            mnemonic = f"CURVE{len(curves_cfg) + 1}"

        mode = (c.get('mode') or 'black').strip().lower()
        if mode not in ('black', 'red', 'blue', 'green'):
            mode = 'black'

        preferred = bool(c.get('preferred', False))

        curves_cfg.append({
            'mnemonic': mnemonic,
            'track_index': track_index,
            'preferred': preferred,
            'mode': mode,
            'left_px': abs_left,
            'right_px': abs_right,
        })

    if not curves_cfg:
        print(f"❌ All {len(ai_curves)} AI curve suggestions rejected:")
        for reason in rejected_reasons:
            print(f"   - {reason}")

        # Fallback: if we have detected tracks, synthesize simple curves so the UI can proceed
        if tracks_out:
            print("⚠️  Falling back to heuristic curves from detected tracks.")
            for idx, t in enumerate(tracks_out[:6]):
                try:
                    abs_left = float(left) + float(t.get('left_px', 0.0))
                    abs_right = float(left) + float(t.get('right_px', 0.0))
                except Exception:
                    continue
                if not np.isfinite(abs_left) or not np.isfinite(abs_right) or abs_right <= abs_left:
                    continue

                mnemonic = (t.get('name') or f'CURVE{len(curves_cfg) + 1}').strip() or f'CURVE{len(curves_cfg) + 1}'

                curves_cfg.append({
                    'mnemonic': mnemonic,
                    'track_index': idx,
                    'preferred': idx == 0,
                    'mode': 'black',
                    'left_px': abs_left,
                    'right_px': abs_right,
                })

        if not curves_cfg:
            return jsonify({'success': False, 'error': 'AI returned no usable curve suggestions.'}), 400

    print(f"✅ Accepted {len(curves_cfg)} curves, rejected {len(rejected_reasons)}")

    return jsonify({
        'success': True,
        'curves': curves_cfg,
        'raw_ai': ai_result,
        'payload': curve_payload,
    })


@app.route('/api/auto_layout', methods=['POST'])
def auto_layout_tracks():
    data = request.json or {}
    image_data = data.get('image')
    region = data.get('region') or {}
    treat_region_as_header = bool(data.get('treat_region_as_header'))

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

    # For normal panel-based layout, only the top band is treated as the
    # header. For explicit header/key capture, the entire region is the
    # header strip, so skip the extra crop.
    if treat_region_as_header:
        header = panel
        header_h = panel_h
    else:
        header_h = max(10, int(panel_h * 0.15))
        header = panel[0:header_h, :]

    ok, buf = cv2.imencode('.jpg', header, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({'success': False, 'error': 'Failed to encode header crop'}), 500

    header_bytes = buf.tobytes()

    detected_text = detect_text_vision_api(header_bytes)
    raw_text = detected_text.get('raw', []) or []

    def _extract_header_metadata(raw_entries):
        if not isinstance(raw_entries, list) or not raw_entries:
            return None
        try:
            items_local = []
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue
                text = (entry.get('text') or '').strip()
                if not text:
                    continue
                verts = entry.get('vertices') or []
                ys_local = [v.get('y') for v in verts if isinstance(v, dict) and 'y' in v]
                xs_local = [v.get('x') for v in verts if isinstance(v, dict) and 'x' in v]
                if not ys_local or not xs_local:
                    continue
                y = float(sum(ys_local)) / len(ys_local)
                x = float(sum(xs_local)) / len(xs_local)
                items_local.append((y, x, text))
            if not items_local:
                return None
            items_local.sort(key=lambda t: (t[0], t[1]))

            lines = []
            y_tol = 8.0
            current_y = None
            current_tokens = []
            for y, x, text in items_local:
                if current_y is None or abs(y - current_y) <= y_tol:
                    if current_y is None:
                        current_y = y
                    current_tokens.append((x, text))
                else:
                    current_tokens.sort(key=lambda t: t[0])
                    lines.append(' '.join(t[1] for t in current_tokens if t[1]).strip())
                    current_y = y
                    current_tokens = [(x, text)]
            if current_tokens:
                current_tokens.sort(key=lambda t: t[0])
                lines.append(' '.join(t[1] for t in current_tokens if t[1]).strip())

            import re

            def pick_after(label_re, s):
                m = re.search(label_re, s, flags=re.IGNORECASE)
                if not m:
                    return None
                tail = s[m.end():].strip(" :-\t")
                return tail.strip() if tail else None

            md = {}
            for s in lines:
                if not s:
                    continue
                for key, pat in (
                    ('comp', r"\bCOMPANY\b"),
                    ('well', r"\bWELL\b"),
                    ('fld', r"\bFIELD\b"),
                    ('loc', r"\bLOCATION\b"),
                    ('county', r"\bCOUNTY\b"),
                    ('state', r"\bSTATE\b"),
                    ('prov', r"\bPROV(?:INCE)?\b"),
                    ('srvc', r"\bSERVICE\s+COMPANY\b"),
                    ('date', r"\bDATE\b"),
                    ('api', r"\bAPI\b"),
                    ('uwi', r"\bUWI\b"),
                ):
                    if key in md:
                        continue
                    val = pick_after(pat, s)
                    if val:
                        md[key] = val

                if 'api' not in md:
                    m = re.search(r"\b(\d{2}[- ]?\d{3}[- ]?\d{5})\b", s)
                    if m:
                        md['api'] = m.group(1).replace(' ', '-')
                if 'date' not in md:
                    m = re.search(r"\b(\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", s)
                    if m:
                        md['date'] = m.group(1)

            return md if md else None
        except Exception:
            return None

    header_metadata = _extract_header_metadata(raw_text) if treat_region_as_header else None

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

    full_text_blob = detected_text.get('full_text', '')

    # If no header text found, fall back to edge-based track detection
    if not items and not full_text_blob:
        print("⚠️  No header text found; falling back to edge-based track detection")
        try:
            local_tracks = auto_detect_tracks(panel)
            tracks_out = []
            for idx, (lx, rx) in enumerate(local_tracks or []):
                try:
                    lx_f = float(lx)
                    rx_f = float(rx)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(lx_f) or not np.isfinite(rx_f) or rx_f <= lx_f:
                    continue
                tracks_out.append({
                    'name': f'Track{idx+1}',
                    'left_px': float(left) + lx_f,
                    'right_px': float(left) + rx_f,
                    'scale_min': None,
                    'scale_max': None,
                    'unit': None,
                    'hot_side': None,
                })
        except Exception as exc:
            import traceback
            return jsonify({
                'success': False,
                'error': f'Edge fallback failed: {str(exc)}',
                'traceback': traceback.format_exc()[-1500:]
            }), 500
        
        if not tracks_out:
            return jsonify({'success': False, 'error': 'No tracks detected (neither header text nor edge detection found tracks).'}), 400
        
        return jsonify({
            'success': True,
            'tracks': tracks_out,
            'raw_layout': {'tracks': [], 'fallback': 'edge_detection'},
        })

    layout_payload = {
        'image': {
            'width_px': panel_w,
            'height_px': header_h,
        },
        'items': items,
        'full_text': full_text_blob,
    }

    layout = call_ai_auto_layout(layout_payload)
    if not layout:
        # If no AI providers are configured, give an actionable error.
        has_provider = bool((GEMINI_API_KEY and GEMINI_MODEL_ID) or (OPENAI_API_KEY and OPENAI_MODEL_ID) or (HF_API_TOKEN and HF_MODEL_ID))
        if not has_provider:
            return jsonify({
                'success': False,
                'error': 'AI layout detection is not configured. Set GEMINI_API_KEY (or OPENAI_API_KEY / HF_API_TOKEN) in the server environment.'
            }), 500

        # Otherwise fall back to edge-based track detection on the panel.
        print("⚠️  AI layout inference returned no result; falling back to edge-based track detection")
        try:
            local_tracks = auto_detect_tracks(panel)
            tracks_out = []
            for idx, (lx, rx) in enumerate(local_tracks or []):
                try:
                    lx_f = float(lx)
                    rx_f = float(rx)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(lx_f) or not np.isfinite(rx_f) or rx_f <= lx_f:
                    continue
                tracks_out.append({
                    'name': f'Track{idx+1}',
                    'left_px': float(left) + lx_f,
                    'right_px': float(left) + rx_f,
                    'scale_min': None,
                    'scale_max': None,
                    'unit': None,
                    'hot_side': None,
                    'color_hint': None,
                })
        except Exception as exc:
            import traceback
            return jsonify({
                'success': False,
                'error': f'AI layout returned no result, and edge fallback failed: {str(exc)}',
                'traceback': traceback.format_exc()[-1500:]
            }), 500

        if tracks_out:
            return jsonify({
                'success': True,
                'tracks': tracks_out,
                'raw_layout': {
                    'tracks': [],
                    'fallback': 'edge_detection_after_ai_failure',
                    'ocr_items': len(items),
                },
            })

        return jsonify({
            'success': False,
            'error': f"AI layout detection failed and edge fallback found no tracks. OCR items={len(items)}. Try selecting a larger/clearer header region."
        }), 500

    raw_tracks = layout.get('tracks') or []
    
    # Merge AI-extracted metadata (often better than regex)
    ai_meta = layout.get('header_metadata')
    if ai_meta and isinstance(ai_meta, dict):
        if header_metadata is None:
            header_metadata = {}
        for k, v in ai_meta.items():
            if v and isinstance(v, str) and v.strip():
                val = v.strip()
                # Map AI keys to internal keys where they differ
                if k == 'company': header_metadata['comp'] = val
                elif k == 'field': header_metadata['fld'] = val
                elif k == 'location': header_metadata['loc'] = val
                elif k == 'province': header_metadata['prov'] = val
                elif k == 'service_company': header_metadata['srvc'] = val
                else:
                    # well, api, date, county, state, etc. match or are new
                    header_metadata[k] = val

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
            'color_hint': t.get('color_hint'),
        }
        tracks_out.append(track_out)

    if not tracks_out:
        return jsonify({'success': False, 'error': 'AI layout returned no usable tracks.'}), 400

    return jsonify({
        'success': True,
        'tracks': tracks_out,
        'raw_layout': layout,
        'header_metadata': header_metadata,
    })


def build_curve_suggestion_payload(panel_image, tracks_out, ocr_suggestions, detected_text):
    h, w = panel_image.shape[:2]

    tracks = []
    for idx, t in enumerate(tracks_out or []):
        try:
            left_px = float(t.get('left_px'))
            right_px = float(t.get('right_px'))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(left_px) or not np.isfinite(right_px) or right_px <= left_px:
            continue
        track = {
            'index': idx,
            'left_px': left_px,
            'right_px': right_px,
        }
        name = t.get('name')
        if isinstance(name, str) and name:
            track['name'] = name
        unit = t.get('unit')
        if isinstance(unit, str) and unit:
            track['unit'] = unit
        smin = t.get('scale_min')
        smax = t.get('scale_max')
        if isinstance(smin, (int, float)) and np.isfinite(smin):
            track['scale_min'] = float(smin)
        if isinstance(smax, (int, float)) and np.isfinite(smax):
            track['scale_max'] = float(smax)
        tracks.append(track)

    curves_hint = (ocr_suggestions or {}).get('curves') or []
    header_labels = []
    for c in curves_hint:
        try:
            lx = float(c.get('left_px'))
            rx = float(c.get('right_px'))
        except (TypeError, ValueError):
            lx = None
            rx = None
        x_center = None
        if lx is not None and rx is not None and np.isfinite(lx) and np.isfinite(rx):
            x_center = 0.5 * (lx + rx)
        label_text = c.get('label_text') or c.get('label_mnemonic') or c.get('type')
        if not label_text:
            continue
        label = {'text': str(label_text)}
        if x_center is not None:
            label['x_px'] = x_center
        label_type = c.get('label_type') or c.get('type')
        if label_type:
            label['curve_type'] = str(label_type)
        header_labels.append(label)

    full_text = ''
    if isinstance(detected_text, dict):
        raw_entries = detected_text.get('raw') or []
        texts = []
        for entry in raw_entries:
            if isinstance(entry, dict):
                t = entry.get('text')
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        full_text = '\n'.join(texts)
    elif isinstance(detected_text, str):
        full_text = detected_text

    return {
        'image': {
            'width_px': int(w),
            'height_px': int(h),
        },
        'tracks': tracks,
        'header_labels': header_labels,
        'raw_text': full_text,
    }


def call_ai_curve_suggestions(curve_payload):
    if not curve_payload:
        return None

    schema_hint = (
        "You are helping digitize paper well logs into LAS. You receive a list of "
        "tracks with x positions, rough names, and header labels. "
        "Your job is to decide which tracks correspond to which curves and which "
        "2-3 curves should be digitized by default. Always respond with JSON ONLY "
        "using this schema:\n\n"
        "{\n"
        "  \"curves\": [\n"
        "    {\n"
        "      \"mnemonic\": string,\n"
        "      \"track_index\": integer,\n"
        "      \"preferred\": boolean,\n"
        "      \"mode\": string\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Guidelines:\n"
        "- Match header_labels to nearest track by x position when possible.\n"
        "- Prefer GR, DT, RHOB, NPHI, RES when choosing preferred curves.\n"
        "- Never invent track indices; only use those present in tracks[].\n"
    )

    payload_text = schema_hint + json.dumps(curve_payload, indent=2)

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
                        curves = _extract_json_object(text)
                        if isinstance(curves, dict):
                            return curves
        except Exception as exc:
            print(f"Gemini API error (curve_suggestions): {exc}")

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
                curves = _extract_json_object(content)
                if isinstance(curves, dict):
                    return curves
        except Exception as exc:
            print(f"OpenAI API error (curve_suggestions): {exc}")

    if not HF_API_TOKEN or not HF_MODEL_ID:
        return None

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_API_TOKEN)
    except Exception as exc:
        print(f"HF InferenceClient init error (curve_suggestions): {exc}")
        return None

    try:
        out = client.text_generation(
            payload_text,
            model=HF_MODEL_ID,
            max_new_tokens=512,
            temperature=0.1,
        )
        curves = _extract_json_object(out if isinstance(out, str) else str(out))
        if isinstance(curves, dict):
            return curves
    except Exception as exc:
        print(f"HF text_generation error (curve_suggestions): {exc}")

    return None


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
    """Auto-detect track boundaries by finding vertical edges, filtering out narrow depth columns"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    vertical_sum = np.sum(edges, axis=0)
    
    # Peak detection for vertical edges
    threshold = np.max(vertical_sum) * 0.3
    peaks = []
    for i in range(1, len(vertical_sum)-1):
        if vertical_sum[i] > threshold and vertical_sum[i] > vertical_sum[i-1] and vertical_sum[i] > vertical_sum[i+1]:
            peaks.append(i)
    
    # Group consecutive peaks into tracks
    if len(peaks) >= 2:
        tracks = []
        for i in range(len(peaks) - 1):
            left = peaks[i]
            right = peaks[i + 1]
            width = right - left
            # Filter out narrow regions (likely depth columns, not data tracks)
            # Typical depth columns are 30-80px wide; data tracks are usually 80-300px
            if width >= 30:
                tracks.append((left, right))
    else:
        # Fallback: divide into equal sections
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
    # If already logged in, go to dashboard
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html', app_version=APP_VERSION, build_time=APP_BUILD_TIME)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Simple hardcoded auth as requested
        if email == 'admin@tiflas.com' and password == 'password':
            session['user'] = email
            # Handle "next" redirect if present
            next_url = request.args.get('next')
            return redirect(next_url or url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid email or password')
            
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', app_version=APP_VERSION, build_time=APP_BUILD_TIME)


@app.route('/las_viewer')
@login_required
def las_viewer():
    return render_template('las_viewer.html', app_version=APP_VERSION)


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

    # Lightweight header-only OCR: try to infer global top/bottom depth values
    # (e.g. from a Pass Summary table) without running full-panel OCR yet.
    detected_text = {'raw': [], 'numbers': [], 'suggestions': {}}
    ocr_suggestions = {}
    if VISION_API_AVAILABLE and vision_client is not None:
        try:
            header_h = max(100, int(h * 0.3))
            header_crop = img[0:header_h, :]
            ok_header, header_buf = cv2.imencode('.jpg', header_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok_header:
                header_bytes = header_buf.tobytes()
                detected_text = detect_text_vision_api(header_bytes)
                ocr_suggestions = detected_text.get('suggestions', {}) or {}
        except Exception as exc:
            print(f"Header OCR error on upload: {exc}")
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
    preview_filters = data.get('preview_filters') or {}
    detected_text = data.get('detected_text') or {}
    depth_cfg = cfg['depth']
    curves = (cfg['curves'] or [])[:6]
    gopt = cfg.get('global_options', {})

    header_metadata = data.get('header_metadata') if isinstance(data, dict) else None

    null_val = float(gopt.get('null', -999.25))
    downsample = int(gopt.get('downsample', 1))
    blur = int(gopt.get('blur', 3))
    min_run = int(gopt.get('min_run', 2))
    smooth_window = int(gopt.get('smooth_window', 5))
    snap_threshold = float(gopt.get('snap_threshold', 1.20)) # Default to 1.20 (20% brighter) as requested/observed

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
    curve_warnings = []

    for c in curves:
        # LAS-facing name/unit come from las_mnemonic/las_unit (or name/unit as fallback)
        name = c.get('las_mnemonic') or c.get('name')
        unit = c.get('las_unit') or c.get('unit', '')
        left_px = int(c['left_px'])
        right_px = int(c['right_px'])
        left_value = float(c['left_value'])
        right_value = float(c['right_value'])
        mode = c.get('mode', 'black')
        hot_side = c.get('hot_side')
        pixel_perfect = bool(c.get('pixel_perfect'))
        trace_mode = c.get('trace_mode')
        align_channels = bool(c.get('align_channels'))
        preserve_wiggles = bool(c.get('preserve_wiggles'))
        crest_boost = bool(c.get('crest_boost'))
        if not hot_side and np.isfinite(left_value) and np.isfinite(right_value):
            hot_side = 'right' if right_value >= left_value else 'left'

        # Defensive ROI bounds check: avoid empty slices that crash OpenCV ops.
        # (This can happen if the UI sends left/right reversed, or values are out of range.)
        img_w = int(img.shape[1])
        img_h = int(img.shape[0])
        left_px = max(0, min(img_w - 1, left_px))
        right_px = max(0, min(img_w, right_px))
        if right_px <= left_px:
            curve_warnings.append({
                'curve': name,
                'error': 'Invalid curve bounds (right_px must be > left_px).',
                'left_px': left_px,
                'right_px': right_px,
                'image_width': img_w,
            })
            continue

        top_clamped = max(0, min(img_h - 1, int(top)))
        bot_clamped = max(0, min(img_h, int(bot)))
        if bot_clamped <= top_clamped:
            curve_warnings.append({
                'curve': name,
                'error': 'Invalid depth bounds (bottom_px must be > top_px).',
                'top_px': top_clamped,
                'bottom_px': bot_clamped,
                'image_height': img_h,
            })
            continue

        roi = img[top_clamped:bot_clamped, left_px:right_px]
        if roi is None or roi.size == 0:
            curve_warnings.append({
                'curve': name,
                'error': 'Empty ROI for curve (check left/right and top/bottom).',
                'top_px': top_clamped,
                'bottom_px': bot_clamped,
                'left_px': left_px,
                'right_px': right_px,
            })
            continue

        if align_channels:
            roi = align_rgb_channels(roi)
        if blur > 0:
            bb = blur + 1 if blur % 2 == 0 else blur
            roi = cv2.GaussianBlur(roi, (bb, bb), 0)

        # Define colored modes set (including auto which detects hue automatically)
        colored_modes = {"green", "red", "blue", "auto", "cyan", "magenta", "yellow", "orange", "purple"}

        # NEW: Build a soft probability mask for the curve using color/edges
        # plus vertical-rail suppression. This returns an 8-bit image where
        # higher values mean higher likelihood of curve pixels.
        # Use compute_prob_map for all modes - it has sophisticated edge detection
        # and centerline boost that works well
        mask = compute_prob_map(roi, mode=mode, ui_filters=preview_filters)

        if mode not in {"green", "red", "blue", "auto", "cyan", "magenta", "yellow", "orange", "purple"}:
            _pm = mask.astype(np.float32) / 255.0
            _pct_nonzero = float(np.mean(_pm > 0.01) * 100)
            _pm_max = float(_pm.max())
            _pm_mean = float(_pm.mean())
            # Debug info removed

        # NEW: Use DP-based smooth path tracing with plausibility checks
        curve_type = c.get('type', 'GR')  # Get curve type for plausibility

        # For explicit color modes, allow more left-right wiggle (lower
        # smoothness penalty) and rely mostly on the DP + local maxima
        # refinement rather than heavy 1D smoothing so the traced path can
        # hug the colored curve as tightly as possible.
        curve_smooth_window = smooth_window
        refine_kwargs = {}
        outlier_threshold = 3.0
        if mode in colored_modes:
            # NO smoothing: window = 1 means no median filter applied
            curve_smooth_window = 1
            # MAXIMUM local window and absolute minimum threshold to snap to any ink
            refine_kwargs = {"dominance_ratio": snap_threshold, "max_shift": 25, "min_prob": 0.005}
            # Disable outlier removal - keep every point for maximum accuracy
            outlier_threshold = 100.0  # Effectively disabled
        else:
            # Use user threshold for non-colored modes too (default was 1.1)
            refine_kwargs = {"dominance_ratio": snap_threshold}
        # Effectively zero smoothness penalty for colored modes to prefer jagged ink over smooth artifacts
        dp_smooth_lambda = 0.001 if mode in colored_modes else (0.001 if curve_type == 'GR' else 0.02)
        # ALSO zero out curvature penalty to allow high-frequency wiggles/jitter
        dp_curv_lambda = 0.001 if mode in colored_modes else (0.001 if curve_type == 'GR' else 0.005)
        max_step_dp = 200 if mode in colored_modes else (30 if curve_type == 'GR' else 50)  # Restrict movement to prevent teleportation

        # Optional pixel-perfect skeleton tracer (preserve every bump)
        if ai_tracer.is_available() and trace_mode == "ai_tracer":
            # Use the AI model for tracing
            try:
                # The AI model predicts coordinates relative to the ROI's left edge
                # and already handles scaling to the ROI width.
                xs = ai_tracer.trace(roi)
                confidence = np.ones_like(xs) * 0.95 # Mock high confidence for AI
            except Exception as e:
                print(f"⚠️ AI Tracer failed for {name}: {e}")
                # Fallback to empty if AI fails
                xs = np.full(roi.shape[0], np.nan)
                confidence = np.zeros(roi.shape[0])
        elif pixel_perfect and mode in colored_modes:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if trace_mode == "skeleton_path":
                xs, confidence = trace_curve_skeleton_path(mask)
            else:
                xs, confidence = trace_curve_pixel_perfect(
                    mask,
                    grayscale=gray_roi,
                    bgr=roi,
                    hot_side=hot_side,
                    preserve_wiggles=preserve_wiggles,
                    crest_boost=crest_boost,
                )
            width_px = mask.shape[1]
            # Fill gaps gently to avoid dropping rows
            if xs.size:
                s = pd.Series(xs)
                s = s.interpolate(method='linear', limit_direction='both', limit=max(10, int(xs.size * 0.02)))
                xs = s.to_numpy(dtype=np.float32)
            # Hybrid post-processing: force missed ink peaks onto the curve
            prob = mask.astype(np.float32) / 255.0
            if crest_boost:
                xs = _postprocess_missed_peaks(mask, prob, xs, search_radius=40, min_prob=0.004)
            else:
                xs = _postprocess_missed_peaks(mask, prob, xs, search_radius=30, min_prob=0.008)
        # For colored modes, use the "Fusion" strategy from successful memories:
        # Run both DP and Direct Centerline tracers, then merge per-row based on probability.
        # AND DISABLE EXTRA REFINEMENTS which cause the zig-zag snapping.
        elif mode in colored_modes:
            # SUPER-RESOLUTION: Upscale mask by 2x to allow sub-pixel precision
            # Use LINEAR interpolation to create smooth gradients between pixels
            mask_orig = mask
            h_orig, w_orig = mask.shape
            mask = cv2.resize(mask, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LINEAR)
            
            # Adjust parameters for 2x scale
            max_step_dp_sr = max_step_dp * 2
            
            # 1. Run DP Tracer (provides continuity)
            xs_dp, conf_dp = trace_curve_with_dp(
                mask,
                scale_min=left_value,
                scale_max=right_value,
                curve_type=curve_type,
                max_step=max_step_dp_sr,
                smooth_lambda=dp_smooth_lambda,
                curv_lambda=dp_curv_lambda,
                hot_side=hot_side,
            )

            # 2. Local Peak Search Fusion
            # Instead of a global direct tracer (which gets distracted by far-away curves),
            # search locally around the DP path for the true tip of the spike.
            h_mask, w_mask = mask.shape
            prob_map = mask.astype(np.float32) / 255.0
            xs = np.full(h_mask, np.nan, dtype=np.float32)
            
            search_window = 100
            
            for y in range(h_mask):
                x_dp = xs_dp[y]
                
                if not np.isfinite(x_dp):
                    xs[y] = x_dp
                    continue
                    
                ix_dp = int(round(x_dp))
                if not (0 <= ix_dp < w_mask):
                    xs[y] = x_dp
                    continue
                    
                p_dp = prob_map[y, ix_dp]
                
                # Search for a better peak in the window
                start = max(0, ix_dp - search_window)
                end = min(w_mask, ix_dp + search_window + 1)
                
                # Extract local slice
                local_prob = prob_map[y, start:end]
                if local_prob.size > 0:
                    # Find ALL peaks, not just the first one
                    max_p = local_prob.max()
                    if max_p > 0:
                        # Get all indices that are essentially the max
                        candidates = np.where(local_prob >= max_p * 0.99)[0]

                        # Prefer the strongest local peak that stays close to the DP path.
                        best_cand = candidates[0]
                        best_score = -1e9
                        for c in candidates:
                            x_cand = start + c
                            d = abs(x_cand - x_dp)
                            score = local_prob[c] - 0.15 * d  # strongest penalty to stay on DP path center
                            if score > best_score:
                                best_score = score
                                best_cand = c
                                
                        p_local = float(local_prob[best_cand])

                        # Ridge-centroid snap: if the peak is a short plateau (common on thick ink),
                        # take the weighted centroid of the contiguous plateau region around best_cand.
                        # This avoids consistent 1-2px edge bias from argmax selection.
                        x_local = float(start + best_cand)
                        try:
                            peak_thr = float(max_p) * 0.99
                            left_i = int(best_cand)
                            right_i = int(best_cand)
                            while left_i > 0 and float(local_prob[left_i - 1]) >= peak_thr:
                                left_i -= 1
                            while right_i + 1 < int(local_prob.size) and float(local_prob[right_i + 1]) >= peak_thr:
                                right_i += 1

                            seg = local_prob[left_i:right_i + 1].astype(np.float32)
                            s = float(seg.sum())
                            if s > 1e-8:
                                coords = np.arange(start + left_i, start + right_i + 1, dtype=np.float32)
                                x_local = float((coords * seg).sum() / s)
                        except Exception:
                            x_local = float(start + best_cand)
                        
                        # Fusion Logic:
                        # If local peak is found and is at least 6% as bright as the DP point,
                        # AND it is substantially far away (indicating a missed spike), take it.
                        if p_local > p_dp * 0.06:
                            xs[y] = float(x_local)
                        else:
                            xs[y] = x_dp
                    else:
                        xs[y] = x_dp
                else:
                    xs[y] = x_dp

            # 3. Pure Center-of-Mass refinement (no edge snapping)
            # Use weighted COM per row to center on ink
            h_sr, w_sr = mask.shape
            prob_sr = mask.astype(np.float32) / 255.0
            for y in range(h_sr):
                x_cur = xs[y]
                if not np.isfinite(x_cur):
                    continue
                ix = int(round(x_cur))
                # Search in a local window around current position
                win = 12
                start = max(0, ix - win)
                end = min(w_sr, ix + win + 1)
                row_slice = prob_sr[y, start:end]
                if row_slice.sum() > 1e-6:
                    coords = np.arange(start, end, dtype=np.float32)
                    xs[y] = (coords * row_slice).sum() / row_slice.sum()

            # 4. Refine Peaks (MOVED to after downsampling)
            # We don't run it here to avoid the downsampling smoothing out the sharp tips.
            
            # 8. Minimal cleanup only - NO aggressive snapping to far-away peaks
            s = pd.Series(xs)
            xs = s.interpolate(method='linear', limit_direction='both', limit=max(25, int(xs.size * 0.02))).to_numpy(dtype=np.float32)
            
            # DOWNSAMPLE: Map back to original resolution
            # Take every 2nd point and divide coordinate by 2
            # Use averaging to reduce noise: (y*2 + y*2+1) / 2
            xs_down = np.full(h_orig, np.nan, dtype=np.float32)
            for y_orig in range(h_orig):
                y_sr = y_orig * 2
                val1 = xs[y_sr]
                val2 = xs[y_sr + 1] if y_sr + 1 < h_mask else val1
                
                if np.isfinite(val1) and np.isfinite(val2):
                    xs_down[y_orig] = (val1 + val2) / 4.0 # Divide by 2 (avg) then divide by 2 (scale) -> /4
                elif np.isfinite(val1):
                    xs_down[y_orig] = val1 / 2.0
                elif np.isfinite(val2):
                    xs_down[y_orig] = val2 / 2.0
            
            xs = xs_down
            # Restore original mask for downstream
            mask = mask_orig

            # 8b. Clean up artifacts (single-pixel horizontal glitches)
            # The high-sensitivity plateau logic can sometimes trigger on noise.
            # A median filter of size 3 removes single-pixel spikes but keeps real features (width >= 2).
            try:
                 from scipy.signal import medfilt
                 xs_valid_mask = np.isfinite(xs)
                 if np.sum(xs_valid_mask) > 3:
                     # Fill NaNs temporarily
                     xs_filled = xs.copy()
                     xs_filled[~xs_valid_mask] = np.nanmedian(xs)
                     # Apply median filter
                     xs_smooth = medfilt(xs_filled, kernel_size=3)
                     # Restore valid pixels
                     xs[xs_valid_mask] = xs_smooth[xs_valid_mask]
            except ImportError:
                 pass

            # 9. FINAL TIP REFINEMENT (Post-Downsample)
            # Run the peak pusher on the original resolution to catch the absolute edges
            # that might have been smoothed by downsampling.
            # Use small buffer (0.2) to be very sticky to tips.
            # Reduced search radius to 30px to prevent teleporting to distant noise.
            xs = refine_peaks_and_valleys(mask, xs, search_radius=30, min_prob=0.005)

            # Gentle centerline refinement to re-center on ink after outer-edge bias and fusion
            try:
                xs = refine_to_stroke_centerline(mask, xs, threshold_ratio=0.5, window_size=8)
            except Exception:
                pass

            # Global centering correction: subtract median residual vs. ink center-of-mass
            try:
                h_mask, w_mask = mask.shape
                xs_valid = xs[~np.isnan(xs)]
                if xs_valid.size > 0:
                    # Compute per-row center-of-mass of probability
                    probs = mask.astype(np.float32) / 255.0
                    weight_sums = probs.sum(axis=1)
                    com = np.full(h_mask, np.nan, dtype=np.float32)
                    nonzero_rows = weight_sums > 1e-6
                    com[nonzero_rows] = (probs[nonzero_rows] * np.arange(w_mask, dtype=np.float32)).sum(axis=1) / weight_sums[nonzero_rows]
                    deltas = []
                    for y in range(h_mask):
                        if np.isnan(xs[y]) or np.isnan(com[y]):
                            continue
                        deltas.append(xs[y] - com[y])
                    if deltas:
                        median_delta = float(np.median(deltas))
                        xs = xs - median_delta
                    
                    # Do not apply a fixed pixel shift; it prevents true centerline alignment.
            except Exception:
                pass
            
        else:
            # For black/other modes, use enhanced multi-scale tracer
            # This tracer now includes "Grid-Safe Snapping" to handle black grids.
            # It fuses 5 different scales to find the most consistent path and rejects vertical rails.
            xs, confidence = trace_curve_multiscale(
                mask,
                scale_min=left_value,
                scale_max=right_value,
                curve_type=curve_type,
                max_step=max_step_dp,
                smooth_lambda=dp_smooth_lambda,
                hot_side=hot_side,
            )

            # Push trace to hot-side ink edge (tip/crest of each spike)
            prob_map_bm = mask.astype(np.float32) / 255.0
            xs = ensure_gr_peak_crests(xs, prob_map_bm, hot_side=hot_side)

            # Optional final smoothing for non-GR curves (GR needs to stay jagged)
            if curve_type.upper() != "GR":
                 xs = remove_outliers_and_smooth(xs, window=curve_smooth_window, outlier_threshold=outlier_threshold)

        width_px = mask.shape[1]

        # UNIVERSAL GAP FILLING:
        # Aggressive grid removal can leave small gaps where the curve crossed a grid line.
        # We linearly interpolate these gaps for all modes to ensure continuity.
        if xs.size > 0:
            s = pd.Series(xs)
            h_mask, w_mask = mask.shape
            # Allow filling gaps up to 2% of image height or 25px (to cover 15px grid cuts), whichever is larger
            max_gap = max(25, int(h_mask * 0.02))
            s = s.interpolate(method='linear', limit_direction='both', limit=max_gap, limit_area=None)
            # Handle edge cases
            if s.isna().any():
                s = s.fillna(method='ffill', limit=max_gap).fillna(method='bfill', limit=max_gap)
            xs = s.to_numpy(dtype=np.float32)

        # For colored modes, apply specific enhancements (peaks, centerline refinement)
        if mode in colored_modes:
            if curve_type.upper() == "GR":
                prob_map = mask.astype(np.float32) / 255.0
                xs = ensure_gr_peak_crests(xs, prob_map, hot_side=hot_side, min_prob=0.01)

            # Final centerline snap for ALL colored modes
            try:
                xs = refine_to_stroke_centerline(mask, xs, threshold_ratio=0.5, window_size=10)
            except Exception:
                pass

            try:
                prob = mask.astype(np.float32) / 255.0
                h_mask, w_mask = prob.shape
                radius = 4
                xs2 = xs.copy()
                for y in range(h_mask):
                    x0 = xs2[y]
                    if not np.isfinite(x0):
                        continue
                    ix = int(round(x0))
                    x_min = max(0, ix - radius)
                    x_max = min(w_mask, ix + radius + 1)
                    row = prob[y, x_min:x_max]
                    if row.size == 0:
                        continue
                    s = float(row.sum())
                    if s <= 1e-8:
                        continue
                    coords = np.arange(x_min, x_max, dtype=np.float32)
                    xs2[y] = float((coords * row).sum() / s)
                xs = xs2
            except Exception:
                pass

            if curve_type.upper() == "GR":
                prob_map = mask.astype(np.float32) / 255.0
                xs = ensure_gr_peak_crests(xs, prob_map, hot_side=hot_side)

            # Optional final local peak snap; kept disabled because it
            # quantizes to integer columns and can reintroduce zig-zags.
            do_final_peak_snap = False
            if do_final_peak_snap:
                # FINAL STEP: refine each point to local probability maximum
                # Tight window to avoid sideways wander
                h_mask, w_mask = mask.shape
                xs_refined_final = np.copy(xs)
                
                local_search_radius = 2  # Very tight window to avoid sideways wander
                
                for y in range(h_mask):
                    if not np.isnan(xs[y]):
                        x_current = int(round(xs[y]))
                        
                        # Define tight search window around current position
                        x_min = max(0, x_current - local_search_radius)
                        x_max = min(w_mask, x_current + local_search_radius + 1)
                        
                        # Find local maximum within this small window
                        row_segment = mask[y, x_min:x_max].astype(np.float32)
                        
                        if len(row_segment) > 0 and row_segment.max() > 0:
                            # Find peak position within window
                            local_peak_idx = np.argmax(row_segment)
                            # Convert back to full image coordinates
                            xs_refined_final[y] = x_min + local_peak_idx
                
                xs = xs_refined_final
        else:
            # For non-colored modes, keep the original vertical-rail rejection logic
            xs_valid = xs[~np.isnan(xs)]
            if xs_valid.size > 0:
                dyn_range = float(np.nanmax(xs_valid) - np.nanmin(xs_valid))
                min_dyn = max(4.0, 0.02 * float(width_px))
                if dyn_range < min_dyn:
                    xs_fallback = pick_curve_x_per_row(mask, min_run=min_run)
                    xs_fallback = smooth_nanmedian(xs_fallback, window=curve_smooth_window)
                    xs = xs_fallback
                    xs_valid = xs[~np.isnan(xs)]

            if xs_valid.size > 0:
                std_x = float(np.nanstd(xs_valid))
                std_threshold = max(1.0, 0.005 * float(width_px))
                # Only reject near-perfectly-vertical traces (rail lock-on).
                # Use a very tight threshold: 0.5% of track width or 1.0px minimum.
                # Slow curves like DTC/RHOB can legitimately have low std.
                if std_x < std_threshold:
                    xs[:] = np.nan

        vals = np.full(xs.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(xs)
        vals[valid] = left_value + (xs[valid] / max(1, width_px-1)) * (right_value - left_value)

        vals_out = np.where(np.isnan(vals), null_val, vals).astype(np.float32)
        curve_data[name] = {'unit': unit, 'values': vals_out}

        # Build a sparse set of trace points in original image coordinates for UI overlay
        trace_points = []
        if xs.size > 0:
            # Only sample from rows where the DP tracer produced a valid X.
            # This avoids the corner-case where all sampled indices land on
            # NaNs even though some rows are valid, which would yield an
            # empty trace and no cyan dots in the UI.
            valid_rows = np.where(~np.isnan(xs))[0]
            if valid_rows.size > 0:
                # Send EVERY single traced point - no sampling at all.
                # This creates a completely solid line that shows the exact trace.
                for row_idx in valid_rows:
                    x_val = xs[row_idx]
                    x_img = round(left_px + x_val)
                    y_img = int(top + row_idx)
                    trace_points.append([x_img, y_img])

        curve_traces[name] = trace_points
    
    # Resample to fixed 0.5 ft step when using feet
    las_depth = base_depth
    las_curve_data = curve_data
    ai_payload = None
    ai_summary = None
    digitized_depth = None
    digitized_curves = None
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

    if not curve_data:
        return jsonify({
            'error': 'No valid curves to digitize. Please check curve bounds and depth settings.',
            'curve_warnings': curve_warnings,
            'depth_warnings': depth_warnings,
        }), 400

    # Run simple curve sanity checks (outlier warnings) on the final LAS depth grid
    outlier_warnings = compute_curve_outlier_warnings(curves, las_curve_data, null_val)

    # Prepare digitized vectors for frontend cursor readout (always, even without lasio)
    try:
        digitized_depth = las_depth.tolist()
        digitized_curves = {
            name: {
                "unit": meta.get("unit", ""),
                "values": (meta.get("values").tolist() if meta.get("values") is not None else None),
            }
            for name, meta in las_curve_data.items()
        }
    except Exception:
        digitized_depth = None
        digitized_curves = None

    # Generate LAS file
    las_content = write_las_simple(las_depth, las_curve_data, depth_unit, header_metadata=header_metadata)

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

    return jsonify({
        'success': True,
        'las_content': las_content,
        'filename': build_las_filename_from_metadata(header_metadata, default_name='digitized_log.las'),
        'validation': validation,
        'outlier_warnings': outlier_warnings,
        'depth_warnings': depth_warnings,
        'curve_warnings': curve_warnings,
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


@app.route('/refine_edit', methods=['POST'])
def refine_edit():
    """
    Refine a curve edit using multi-scale tracing on a local segment.
    
    Takes a small vertical window around the edited point and runs
    the improved line detection to find the best x-position.
    
    Request JSON:
        image: base64 encoded image
        track: {leftX, rightX, leftValue, rightValue}
        editY: pixel Y coordinate of the edit
        editX: current pixel X coordinate (user's drag position)
        windowSize: vertical window size in pixels (default 50)
        curveType: curve type (GR, RHOB, etc.)
        mode: detection mode (green, black, etc.)
    
    Returns:
        refinedX: the best x-position from multi-scale detection
        confidence: detection confidence
    """
    try:
        data = request.json
        
        # Decode image
        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'})
        
        h_img, w_img = img.shape[:2]
        
        # Get parameters
        track = data.get('track', {})
        # Note: if isCrop is True, leftX/rightX are used mainly for scale_min/max mapping,
        # but the image provided is ALREADY the cropped track.
        left_value = float(track.get('leftValue', 0))
        right_value = float(track.get('rightValue', 100))
        
        edit_y = int(data.get('editY', h_img // 2))
        edit_x = float(data.get('editX', w_img // 2))
        window_size = int(data.get('windowSize', 50))
        curve_type = data.get('curveType', 'GR').upper()
        mode = data.get('mode', 'green').lower()
        
        is_crop = data.get('isCrop', False)
        crop_origin_x = int(data.get('cropOriginX', 0))
        edit_relative_y = int(data.get('editRelativeY', -1))

        ui_filters = data.get('uiFilters') or data.get('ui_filters') or {}
        try:
            ui_filters = {
                'invert': bool(ui_filters.get('invert')),
                'contrast': bool(ui_filters.get('contrast')),
            }
        except Exception:
            ui_filters = {}

        request_max_snap_dist = data.get('maxSnapDist')
        try:
            snap_threshold = float(data.get('snapThreshold', 1.05))
        except Exception:
            snap_threshold = 1.05

        if is_crop:
        # Image is already cropped to the track/ROI
            track_crop = img
            # If editRelativeY is provided, use it; otherwise assume center
            if edit_relative_y >= 0:
                edit_row_in_window = edit_relative_y
            else:
                edit_row_in_window = h_img // 2
            
            # For multiscale tracer, the "track" is the whole image provided
            # effectively left_x=0, right_x=w_img in local coords
            # BUT we need to be careful if the user provided leftX/rightX in the request
            # they might be absolute. For the tracer, we just need scale mapping.
            pass
        else:
            # Standard mode: image is the full page, we crop it
            left_x = int(track.get('leftX', 0))
            right_x = int(track.get('rightX', w_img))
            
            left_x = max(0, min(left_x, w_img - 1))
            right_x = max(left_x + 1, min(right_x, w_img))
            
            # Extract vertical window around edit point
            y_start = max(0, edit_y - window_size // 2)
            y_end = min(h_img, edit_y + window_size // 2)
            
            track_crop = img[y_start:y_end, left_x:right_x]
            
            edit_row_in_window = edit_y - y_start
            
            # We want to add this offset back to the result
            crop_origin_x = left_x
            
        track_proc = track_crop
        x_scale_factor = 1.0
        if track_crop.size == 0:
            return jsonify({'success': False, 'error': 'Empty track region'})
        try:
            track_proc, x_scale_factor = enhance_curve_roi(track_crop)
        except Exception:
            track_proc = track_crop
            x_scale_factor = 1.0
        if x_scale_factor <= 0:
            x_scale_factor = 1.0
        
        # Build probability map for this segment
        colored_modes = {'green', 'red', 'blue', 'auto', 'cyan', 'magenta', 'yellow', 'orange', 'purple'}
        
        try:
            # ALWAYS use compute_prob_map so we get grid removal, UI filters, and enhanced detection
            # for both colored and black modes.
            mask = compute_prob_map(track_proc, mode, ui_filters=ui_filters)
        except Exception as _prob_err:
            print(f'refine_edit: compute_prob_map failed ({_prob_err}), returning edit position')
            return jsonify({'success': True, 'refinedX': float(edit_x), 'confidence': 0.0,
                            'originalX': float(edit_x), 'refinedPath': []})

        weight_map = mask.astype(np.float32)
        # Calculate distance transform for weighting if needed (compute_prob_map already does this internally,
        # but we do it here for the refined_centroid helper)
        try:
            bin_mask = (mask > 0).astype(np.uint8)
            dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 3)
            if dist is not None and dist.size:
                maxv = float(np.max(dist))
                if maxv > 1e-6:
                    weight_map = (dist / maxv) * 255.0
        except Exception:
            weight_map = mask.astype(np.float32)
        
        # Run multi-scale tracing on this segment
        # Use parameters consistent with the main digitization loop
        smooth_l = 0.001 if (mode in colored_modes or curve_type == 'GR') else 0.02
        curv_l = 0.001 if (mode in colored_modes or curve_type == 'GR') else 0.005
        max_s = 200 if mode in colored_modes else (30 if curve_type == 'GR' else 50)

        try:
            xs_refined, confidence = trace_curve_multiscale(
                mask,
                scale_min=left_value,
                scale_max=right_value,
                curve_type=curve_type,
                max_step=max_s,
                smooth_lambda=smooth_l,
                curv_lambda=curv_l,
                hot_side=None,
            )
        except Exception as _trace_err:
            print(f'refine_edit: trace_curve_multiscale failed ({_trace_err}), returning edit position')
            return jsonify({'success': True, 'refinedX': float(edit_x), 'confidence': 0.0,
                            'originalX': float(edit_x), 'refinedPath': []})
        
        # UNIVERSAL GAP FILLING (Same as main loop):
        # Fill gaps from grid removal so the refined path is continuous
        if xs_refined is not None and xs_refined.size > 0:
            try:
                s = pd.Series(xs_refined)
                # Use at least 25px gap fill to bridge grid cuts
                max_gap = 25
                s = s.interpolate(method='linear', limit_direction='both', limit=max_gap, limit_area=None)
                if s.isna().any():
                    s = s.fillna(method='ffill', limit=max_gap).fillna(method='bfill', limit=max_gap)
                xs_refined = s.to_numpy(dtype=np.float32)
            except Exception:
                pass

        # Helper for centroid refinement
        def get_refined_centroid(x_viterbi, row_idx):
            try:
                search_r = 10
                center_int = int(round(x_viterbi))
                r_start = max(0, center_int - search_r)
                r_end = min(mask.shape[1], center_int + search_r + 1)
                
                if r_end > r_start:
                    row_vals = weight_map[row_idx, r_start:r_end].astype(float)
                    row_vals = cv2.GaussianBlur(row_vals.reshape(1, -1), (3, 1), 0).flatten()
                    
                    total_mass = np.sum(row_vals)
                    if total_mass > 1e-3:
                        indices = np.arange(len(row_vals))
                        com_local = np.sum(indices * row_vals) / total_mass
                        refined_pos = r_start + com_local
                        
                        # Only accept if close to Viterbi
                        if abs(refined_pos - x_viterbi) < search_r:
                            return refined_pos
            except Exception:
                pass
            return x_viterbi

        if 0 <= edit_row_in_window < len(xs_refined) and np.isfinite(xs_refined[edit_row_in_window]):
            # Refine the specific click point
            refined_x_local = get_refined_centroid(xs_refined[edit_row_in_window], edit_row_in_window)

            try:
                w_local = mask.shape[1]
                edit_x_local = float(edit_x) - float(crop_origin_x)
                edit_x_local_proc = edit_x_local * x_scale_factor
                if w_local > 2 and np.isfinite(edit_x_local_proc):
                    try:
                        max_snap = float(request_max_snap_dist) if request_max_snap_dist is not None else 15.0
                        max_snap = float(np.clip(max_snap, 4.0, 60.0))
                    except Exception:
                        max_snap = 15.0
                    max_snap_proc = max_snap * x_scale_factor

                    search_r = int(max(4, min(80, round(max_snap_proc + 6))))
                    center_int = int(round(edit_x_local_proc))
                    r_start = max(0, center_int - search_r)
                    r_end = min(w_local, center_int + search_r + 1)
                    if r_end > r_start:
                        row_vals = weight_map[edit_row_in_window, r_start:r_end].astype(np.float32)
                        if row_vals.size > 0:
                            local_best = int(np.argmax(row_vals))
                            local_best_x = float(r_start + local_best)

                            dp_x = float(refined_x_local)
                            local_score = float(row_vals[local_best])
                            dp_score = float(weight_map[edit_row_in_window, int(np.clip(round(dp_x), 0, w_local - 1))])

                            if abs(local_best_x - edit_x_local_proc) <= max_snap_proc and (
                                abs(dp_x - edit_x_local_proc) > max_snap_proc or local_score > dp_score * 1.10
                            ):
                                refined_x_local = get_refined_centroid(local_best_x, edit_row_in_window)
            except Exception:
                pass

            refined_x_orig = refined_x_local / x_scale_factor
            refined_x = crop_origin_x + refined_x_orig  # Convert back to full image coordinates
            conf = float(confidence[edit_row_in_window]) if edit_row_in_window < len(confidence) else 0.5
        else:
            # Fallback: return the user's edit position
            refined_x = edit_x
            conf = 0.0
        
        # Prepare the full path segment relative to the crop, with centroid refinement applied to ALL points
        refined_path_segment = []
        if xs_refined is not None and len(xs_refined) > 0:
            for i, val in enumerate(xs_refined):
                if np.isfinite(val):
                    # Refine every point for sub-pixel accuracy and smoother crests
                    final_val = get_refined_centroid(val, i)
                    refined_path_segment.append({
                        'offsetY': i,
                        'x': float(crop_origin_x + (final_val / x_scale_factor)),
                        'confidence': float(confidence[i]) if confidence is not None and i < len(confidence) else 0.0
                    })

        return jsonify({
            'success': True,
            'refinedX': float(refined_x),
            'confidence': conf,
            'originalX': float(edit_x),
            'refinedPath': refined_path_segment
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


@app.route('/api/log_correction', methods=['POST'])
def log_correction():
    try:
        data = request.json or {}
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

        now = datetime.utcnow()
        date_str = now.strftime('%Y-%m-%d')
        ts = now.strftime('%Y%m%dT%H%M%S.%fZ')
        event_id = data.get('event_id') or str(uuid.uuid4())

        base_dir = Path(__file__).resolve().parent
        out_dir = base_dir / 'corrections' / date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        image_path = None
        image_data = data.get('image')
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            try:
                header, b64 = image_data.split(',', 1)
                ext = 'jpg'
                if 'image/png' in header:
                    ext = 'png'
                raw = base64.b64decode(b64)
                image_path = out_dir / f'{ts}_{event_id}.{ext}'
                image_path.write_bytes(raw)
            except Exception:
                image_path = None

        record = {
            'event_id': event_id,
            'ts_utc': now.isoformat() + 'Z',
            'type': data.get('type'),
            'curve_id': data.get('curve_id'),
            'curve_index': data.get('curve_index'),
            'mode': data.get('mode'),
            'track': data.get('track'),
            'depth_index': data.get('depth_index'),
            'depth': data.get('depth'),
            'before': data.get('before'),
            'after': data.get('after'),
            'refine': data.get('refine'),
            'fallback': data.get('fallback'),
            'notes': data.get('notes'),
            'image_path': str(image_path) if image_path else None,
        }

        jsonl_path = out_dir / 'corrections.jsonl'
        with jsonl_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        return jsonify({'success': True, 'event_id': event_id})
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/learn_from_user', methods=['POST'])
def learn_from_user():
    """Record user curve adjustments for learning (Phase 1)"""
    data = request.json or {}
    
    required_fields = ['curve_type', 'original_params', 'user_params']
    for field in required_fields:
        if field not in data:
            return jsonify({
                'success': False, 
                'error': f'Missing required field: {field}'
            }), 400
    
    try:
        curve_type = data['curve_type']
        original_params = data['original_params']
        user_params = data['user_params']
        quality_score = data.get('quality_score', 1.0)
        image_context = data.get('image_context')
        
        # Validate curve type
        valid_types = ['GR', 'RHOB', 'NPHI', 'DT', 'CALI', 'SP', 'OTHER']
        if curve_type not in valid_types:
            curve_type = 'OTHER'
        
        # Record the adjustment
        tracker.record_adjustment(
            curve_type=curve_type,
            original_params=original_params,
            user_params=user_params,
            quality_score=quality_score,
            image_context=image_context
        )
        
        # Return stats for feedback
        stats = tracker.get_stats(curve_type)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'message': f'Adjustment recorded for {curve_type}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/user_preferences', methods=['GET'])
def get_user_preferences():
    """Get user preference statistics"""
    curve_type = request.args.get('curve_type')
    
    if curve_type:
        adjustments = tracker.get_adjustments(curve_type)
        stats = tracker.get_stats(curve_type)
        return jsonify({
            'curve_type': curve_type,
            'adjustments': adjustments,
            'stats': stats
        })
    else:
        all_adjustments = tracker.get_all_adjustments()
        all_stats = {ct: tracker.get_stats(ct) for ct in all_adjustments.keys()}
        return jsonify({
            'all_adjustments': all_adjustments,
            'all_stats': all_stats
        })


@app.route('/api/clear_preferences', methods=['POST'])
def clear_preferences():
    """Clear user preferences (for testing/reset)"""
    curve_type = request.json.get('curve_type')
    
    if curve_type:
        tracker.adjustments[curve_type] = []
    else:
        tracker.adjustments.clear()
    
    tracker.save_preferences()
    
    return jsonify({
        'success': True,
        'message': f'Preferences cleared for {curve_type or "all curves"}'
    })


@app.route('/api/batch_digitize', methods=['POST'])
def batch_digitize():
    """Process multiple TIFF images for ML training dataset generation.

    Expects JSON with:
      - jobs: list of { image, config, preview_filters, detected_text, header_metadata }
      - export_format: 'json' (default) or 'las'
      - include_images: bool (include cropped panel images in output)

    Returns:
      - results: list of digitization results with metadata
      - summary: { total, success, failed }
    """
    data = request.json or {}
    jobs = data.get('jobs', [])
    export_format = data.get('export_format', 'json')
    include_images = data.get('include_images', True)

    if not jobs:
        return jsonify({'success': False, 'error': 'No jobs provided'}), 400

    results = []
    success_count = 0
    failed_count = 0

    for idx, job in enumerate(jobs):
        try:
            image_data = job.get('image')
            image_path = job.get('image_path')
            config = job.get('config')
            preview_filters = job.get('preview_filters', {})
            detected_text = job.get('detected_text', {})
            header_metadata = job.get('header_metadata')

            if not image_data and not image_path:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': 'Missing image or image_path'
                })
                failed_count += 1
                continue

            if not config:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': 'Missing config'
                })
                failed_count += 1
                continue

            # Load image (path or base64)
            img = None
            if image_path:
                if os.path.exists(image_path):
                    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                else:
                    results.append({
                        'index': idx,
                        'success': False,
                        'error': f'Image path not found: {image_path}'
                    })
                    failed_count += 1
                    continue
            elif image_data:
                try:
                    # Decode image
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    img_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception as e:
                    results.append({
                        'index': idx,
                        'success': False,
                        'error': f'Failed to decode image: {e}'
                    })
                    failed_count += 1
                    continue

            if img is None:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': 'Failed to decode image'
                })
                failed_count += 1
                continue

            # Extract config
            depth_cfg = config['depth']
            curves = (config['curves'] or [])[:6]
            gopt = config.get('global_options', {})

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

            curve_data = {}
            curve_traces = {}

            for c in curves:
                name = c.get('las_mnemonic') or c.get('name')
                unit = c.get('las_unit') or c.get('unit', '')
                left_px = int(c['left_px'])
                right_px = int(c['right_px'])
                left_value = float(c['left_value'])
                right_value = float(c['right_value'])
                mode = c.get('mode', 'black')
                hot_side = c.get('hot_side')
                pixel_perfect = bool(c.get('pixel_perfect'))
                trace_mode = c.get('trace_mode')
                align_channels = bool(c.get('align_channels'))
                preserve_wiggles = bool(c.get('preserve_wiggles'))
                crest_boost = bool(c.get('crest_boost'))

                if not hot_side and np.isfinite(left_value) and np.isfinite(right_value):
                    hot_side = 'right' if right_value >= left_value else 'left'

                left_px = max(0, min(W - 1, left_px))
                right_px = max(0, min(W, right_px))

                if right_px <= left_px:
                    continue

                top_clamped = max(0, min(H - 1, int(top)))
                bot_clamped = max(0, min(H, int(bot)))

                if bot_clamped <= top_clamped:
                    continue

                roi = img[top_clamped:bot_clamped, left_px:right_px]
                if roi is None or roi.size == 0:
                    continue

                if align_channels:
                    roi = align_rgb_channels(roi)
                if blur > 0:
                    bb = blur + 1 if blur % 2 == 0 else blur
                    roi = cv2.GaussianBlur(roi, (bb, bb), 0)

                mask = compute_prob_map(roi, mode=mode, ui_filters=preview_filters)
                curve_type = c.get('type', 'GR')

                colored_modes = {"green", "red", "blue", "auto", "cyan", "magenta", "yellow", "orange", "purple"}

                if pixel_perfect and mode in colored_modes:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    if trace_mode == "skeleton_path":
                        xs, confidence = trace_curve_skeleton_path(mask)
                    else:
                        xs, confidence = trace_curve_pixel_perfect(
                            mask, grayscale=gray_roi, bgr=roi, hot_side=hot_side,
                            preserve_wiggles=preserve_wiggles, crest_boost=crest_boost,
                        )
                    width_px = mask.shape[1]
                    if xs.size:
                        s = pd.Series(xs)
                        s = s.interpolate(method='linear', limit_direction='both', limit=max(10, int(xs.size * 0.02)))
                        xs = s.to_numpy(dtype=np.float32)
                    prob = mask.astype(np.float32) / 255.0
                    if crest_boost:
                        xs = _postprocess_missed_peaks(mask, prob, xs, search_radius=40, min_prob=0.004)
                    else:
                        xs = _postprocess_missed_peaks(mask, prob, xs, search_radius=30, min_prob=0.008)
                elif mode in colored_modes:
                    mask_orig = mask
                    h_orig, w_orig = mask.shape
                    mask = cv2.resize(mask, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LINEAR)
                    max_step_dp = 200 * 2
                    dp_smooth_lambda = 0.001
                    dp_curv_lambda = 0.001

                    xs_dp, conf_dp = trace_curve_with_dp(
                        mask, scale_min=left_value, scale_max=right_value,
                        curve_type=curve_type, max_step=max_step_dp,
                        smooth_lambda=dp_smooth_lambda, curv_lambda=dp_curv_lambda,
                        hot_side=hot_side,
                    )

                    h_mask, w_mask = mask.shape
                    prob_map = mask.astype(np.float32) / 255.0
                    xs = np.full(h_mask, np.nan, dtype=np.float32)

                    for row in range(h_mask):
                        dp_x = xs_dp[row] if row < len(xs_dp) else None
                        if np.isnan(dp_x):
                            continue

                        search_radius = 30
                        x_start = max(0, int(dp_x) - search_radius)
                        x_end = min(w_mask, int(dp_x) + search_radius + 1)
                        row_probs = prob_map[row, x_start:x_end]

                        if row_probs.size == 0:
                            continue

                        local_max_idx = np.argmax(row_probs)
                        xs[row] = x_start + local_max_idx

                    xs = xs / 2.0

                    mask = mask_orig
                else:
                    xs, confidence = trace_curve_with_dp(
                        mask, scale_min=left_value, scale_max=right_value,
                        curve_type=curve_type, max_step=3, smooth_lambda=0.5, curv_lambda=0.05,
                        hot_side=hot_side,
                    )

                if xs.size != nrows:
                    if xs.size > nrows:
                        xs = xs[:nrows]
                    else:
                        xs = np.pad(xs, (0, nrows - xs.size), mode='edge')

                xs = pd.Series(xs).interpolate(method='linear', limit_direction='both', limit=10).to_numpy()

                scale_range = right_value - left_value
                if scale_range == 0:
                    scale_range = 1.0

                values = left_value + (xs / (right_px - left_px)) * scale_range
                values = np.where(np.isnan(values), null_val, values)

                # Clean NaN/inf from xs and values before converting to list
                xs_clean = np.where(np.isnan(xs) | np.isinf(xs), null_val, xs)
                values_clean = np.where(np.isnan(values) | np.isinf(values), null_val, values)

                if downsample > 1:
                    values_clean = values_clean[::downsample]
                    base_depth = base_depth[::downsample]

                curve_data[name] = values_clean.tolist()
                curve_traces[name] = xs_clean.tolist()

            # Clean NaN/inf from depth values before converting to list
            base_depth_clean = np.where(np.isnan(base_depth) | np.isinf(base_depth), null_val, base_depth)

            result = {
                'index': idx,
                'success': True,
                'depth': {
                    'top_px': top,
                    'bottom_px': bot,
                    'top_depth': top_depth,
                    'bottom_depth': bottom_depth,
                    'unit': depth_unit,
                    'values': base_depth_clean.tolist(),
                },
                'curves': curve_data,
                'curve_traces': curve_traces,
                'metadata': {
                    'image_width': W,
                    'image_height': H,
                    'curve_count': len(curve_data),
                    'null_value': null_val,
                }
            }

            if include_images:
                ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    result['image'] = base64.b64encode(buf.tobytes()).decode('utf-8')

            if header_metadata:
                result['header_metadata'] = header_metadata

            results.append(result)
            success_count += 1

        except Exception as e:
            results.append({
                'index': idx,
                'success': False,
                'error': str(e)
            })
            failed_count += 1
            continue

    return jsonify({
        'success': True,
        'results': results,
        'summary': {
            'total': len(jobs),
            'success': success_count,
            'failed': failed_count
        }
    })


@app.route('/api/export_training_data', methods=['POST'])
def export_training_data():
    """Export digitized data as ML-ready training dataset.

    Expects JSON with:
      - data: list of digitization results (from batch_digitize or digitize)
      - format: 'json' (default) or 'csv'
      - include_metadata: bool (include image metadata)

    Returns:
      - JSON or CSV formatted training data with:
        - image_id
        - depth_values
        - curve_data (pixel traces and value mappings)
        - curve_metadata (type, scale, parameters)
    """
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    data = request.json or {}
    dataset = data.get('data', [])
    export_format = data.get('format', 'json')
    include_metadata = data.get('include_metadata', True)

    if not dataset:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    training_data = []

    for idx, item in enumerate(dataset):
        if not item.get('success'):
            continue

        depth_info = item.get('depth', {})
        curves = item.get('curves', {})
        curve_traces = item.get('curve_traces', {})
        metadata = item.get('metadata', {})
        header_metadata = item.get('header_metadata', {})

        # Clean depth values of NaN/inf
        depth_values_raw = depth_info.get('values', [])
        depth_values_clean = [None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for v in depth_values_raw]

        training_item = {
            'image_id': f"img_{idx:04d}",
            'depth': {
                'top_px': depth_info.get('top_px'),
                'bottom_px': depth_info.get('bottom_px'),
                'top_depth': depth_info.get('top_depth'),
                'bottom_depth': depth_info.get('bottom_depth'),
                'unit': depth_info.get('unit', 'FT'),
                'values': depth_values_clean,
            },
            'curves': []
        }

        for curve_name, values in curves.items():
            trace = curve_traces.get(curve_name, [])
            # Filter out NaN and inf values for JSON serialization
            trace_clean = [None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for v in trace]
            values_clean = [None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for v in values]

            training_item['curves'].append({
                'name': curve_name,
                'pixel_trace': trace_clean,
                'depth_values': values_clean,
                'sample_count': len(values),
            })

        if include_metadata:
            # Clean metadata values
            training_item['metadata'] = {
                'image_width': metadata.get('image_width'),
                'image_height': metadata.get('image_height'),
                'curve_count': metadata.get('curve_count'),
                'null_value': metadata.get('null_value'),
            }
            if header_metadata:
                training_item['header_metadata'] = header_metadata

        training_data.append(training_item)

    if export_format == 'csv':
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        writer.writerow(['image_id', 'depth_top_px', 'depth_bottom_px', 'depth_top_depth',
                        'depth_bottom_depth', 'depth_unit', 'curve_name', 'pixel_trace',
                        'depth_values'])

        for item in training_data:
            depth = item['depth']
            for curve in item['curves']:
                # Filter out NaN and inf values for JSON serialization
                pixel_trace = curve['pixel_trace']
                depth_values = curve['depth_values']

                # Replace NaN with null for JSON compatibility
                pixel_trace_clean = [None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for v in pixel_trace]
                depth_values_clean = [None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v for v in depth_values]

                writer.writerow([
                    item['image_id'],
                    depth['top_px'],
                    depth['bottom_px'],
                    depth['top_depth'],
                    depth['bottom_depth'],
                    depth['unit'],
                    curve['name'],
                    json.dumps(pixel_trace_clean),
                    json.dumps(depth_values_clean),
                ])

        csv_output = output.getvalue()
        return jsonify({
            'success': True,
            'format': 'csv',
            'data': csv_output,
            'count': len(training_data)
        })

    # Use custom encoder to handle numpy types and NaN/inf
    response_data = {
        'success': True,
        'format': 'json',
        'data': training_data,
        'count': len(training_data)
    }

    json_str = json.dumps(response_data, cls=NpEncoder)
    return Response(json_str, mimetype='application/json')


_ML_CURVE_TRACE_MODEL_CACHE = {
     'model_path': None,
     'model': None,
     'meta': None,
 }


if TORCH_AVAILABLE:
     class _CurveTraceNet(nn.Module):
         def __init__(self, in_ch: int = 1, base: int = 16):
             super().__init__()
             self.enc = nn.Sequential(
                 nn.Conv2d(in_ch, base, 3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(base, base, 3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2),
                 nn.Conv2d(base, base * 2, 3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(base * 2, base * 2, 3, padding=1),
                 nn.ReLU(inplace=True),
             )
             self.dec = nn.Sequential(
                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                 nn.Conv2d(base * 2, base, 3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(base, base, 3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(base, 1, 1),
             )

         def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
             feat = self.enc(x)
             logits = self.dec(feat).squeeze(1)
             prob = torch.softmax(logits, dim=-1)
             xs = torch.linspace(0.0, 1.0, logits.shape[-1], device=logits.device)
             pred = (prob * xs).sum(dim=-1)
             return pred


def _ml_decode_image_data_url(image_data: str) -> np.ndarray:
     img_data = image_data.split(',', 1)[1]
     img_bytes = base64.b64decode(img_data)
     nparr = np.frombuffer(img_bytes, np.uint8)
     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
     if img is None:
         raise ValueError('Failed to decode image')
     return img


def _ml_load_curve_trace_model(model_path: str, device: str = 'cpu') -> Tuple['torch.nn.Module', Dict]:
     cache = _ML_CURVE_TRACE_MODEL_CACHE
     if cache.get('model') is not None and cache.get('model_path') == model_path:
         return cache['model'], (cache.get('meta') or {})

     payload = torch.load(model_path, map_location=device)
     state_dict = payload.get('state_dict')
     if not state_dict:
         raise ValueError('Model file missing state_dict')

     model = _CurveTraceNet()
     model.load_state_dict(state_dict)
     model.eval()
     model.to(device)

     meta = {
         'input_h': int(payload.get('input_h', 256)),
         'input_w': int(payload.get('input_w', 128)),
         'curve': payload.get('curve'),
     }

     cache['model_path'] = model_path
     cache['model'] = model
     cache['meta'] = meta

     return model, meta


def _ml_resolve_curve_trace_model_path(requested_path: Optional[str]) -> str:
     if requested_path:
         return requested_path

     env_path = os.environ.get('CURVE_TRACE_MODEL_PATH')
     if env_path:
         return env_path

     candidates = [
         Path(__file__).with_name('curve_trace_model.pt'),
         Path.cwd() / 'curve_trace_model.pt',
     ]

     try:
         desktop_dir = Path(__file__).resolve().parent.parent
         candidates.append(desktop_dir / 'TestTiflas' / 'curve_trace_model.pt')
     except Exception:
         pass

     for p in candidates:
         try:
             if p.exists():
                 return str(p)
         except Exception:
             continue

     return str(candidates[0])


@app.route('/api/download_las_zip', methods=['POST'])
def download_las_zip():
    """Generate a ZIP file containing individual LAS files for each curve."""
    data = request.json or {}
    
    depths = data.get('depths')
    curves = data.get('curves')
    header_metadata = data.get('header_metadata') or {}
    depth_unit = data.get('depth_unit', 'FT')
    
    if not depths or not curves:
        return jsonify({'error': 'Missing depth or curve data'}), 400
        
    try:
        depth_arr = np.array(depths, dtype=np.float32)
        
        # Create in-memory ZIP
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            base_name = build_las_filename_from_metadata(header_metadata, default_name='digitized_log').replace('.las', '')
            
            for curve_name, curve_info in curves.items():
                # Prepare single-curve data dict
                single_curve_data = {
                    curve_name: curve_info
                }
                
                # Generate LAS content
                las_content = write_las_simple(depth_arr, single_curve_data, depth_unit, header_metadata)
                
                # Add to ZIP
                # Filename: WellName_CurveName.las
                # Sanitize curve name for filename
                safe_curve_name = "".join([c for c in curve_name if c.isalnum() or c in ('_', '-')])
                filename = f"{base_name}_{safe_curve_name}.las"
                zf.writestr(filename, las_content)
                
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{base_name}_curves.zip"
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml_predict_curve_trace', methods=['POST'])
def ml_predict_curve_trace():
     if not TORCH_AVAILABLE:
         return jsonify({'success': False, 'error': 'torch is not available in this environment'}), 400

     data = request.json or {}
     image_data = data.get('image')
     roi = data.get('roi') or {}

     if not image_data:
         return jsonify({'success': False, 'error': 'Missing image'}), 400

     model_path = _ml_resolve_curve_trace_model_path(data.get('model_path'))

     device = data.get('device') or 'cpu'

     try:
         img = _ml_decode_image_data_url(image_data)
         H, W = img.shape[:2]

         top_px = int(roi.get('top_px', 0))
         bottom_px = int(roi.get('bottom_px', H))
         left_px = int(roi.get('left_px', 0))
         right_px = int(roi.get('right_px', W))

         top_px = max(0, min(H - 1, top_px))
         bottom_px = max(0, min(H, bottom_px))
         left_px = max(0, min(W - 1, left_px))
         right_px = max(0, min(W, right_px))

         if bottom_px <= top_px or right_px <= left_px:
             return jsonify({'success': False, 'error': 'Invalid ROI'}), 400

         roi_img = img[top_px:bottom_px, left_px:right_px]
         if roi_img is None or roi_img.size == 0:
             return jsonify({'success': False, 'error': 'Empty ROI'}), 400

         roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

         if not Path(model_path).exists():
             return jsonify({
                 'success': False,
                 'error': f'Model file not found: {model_path}',
             }), 400

         model, meta = _ml_load_curve_trace_model(model_path=model_path, device=device)
         in_h = int(meta.get('input_h', 256))
         in_w = int(meta.get('input_w', 128))

         roi_resized = cv2.resize(roi_gray, (in_w, in_h), interpolation=cv2.INTER_AREA)
         x = (roi_resized.astype(np.float32) / 255.0)[None, None, :, :]
         x_t = torch.from_numpy(x).to(device)

         with torch.no_grad():
             pred_norm = model(x_t)[0].detach().cpu().numpy().astype(np.float32)

         roi_w = int(right_px - left_px)
         roi_h = int(bottom_px - top_px)
         if roi_w <= 1 or roi_h <= 1:
             return jsonify({'success': False, 'error': 'ROI too small'}), 400

         pred_px = pred_norm * float(roi_w - 1)

         src_y = np.linspace(0.0, float(in_h - 1), num=in_h, dtype=np.float32)
         dst_y = np.linspace(0.0, float(in_h - 1), num=roi_h, dtype=np.float32)
         pred_px_full = np.interp(dst_y, src_y, pred_px).astype(np.float32)

         return jsonify({
             'success': True,
             'model_path': model_path,
             'model_meta': meta,
             'roi': {
                 'top_px': top_px,
                 'bottom_px': bottom_px,
                 'left_px': left_px,
                 'right_px': right_px,
             },
             'pixel_trace': pred_px_full.tolist(),
         })
     except Exception as e:
         return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting TIFF→LAS Web App")
    print(f"📊 Google Vision API: {'✅ Available' if VISION_API_AVAILABLE else '⚠️  Not configured'}")
    print("🌐 Open: http://localhost:5000")
    
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
