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
except ImportError:
    print("ℹ️  lasio not installed; LAS validation will be skipped.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# ----------------------------
# Core Processing Functions
# ----------------------------
def hsv_red_mask(hsv_img):
    lower1, upper1 = np.array([0, 80, 80]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 80, 80]), np.array([180, 255, 255])
    return cv2.bitwise_or(cv2.inRange(hsv_img, lower1, upper1), 
                         cv2.inRange(hsv_img, lower2, upper2))

def black_mask(gray_img):
    return cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 51, 10)

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
    if curve_hint:
        suggestions['curves'] = curve_hint

    return suggestions

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
    return render_template('index.html')

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
    if VISION_API_AVAILABLE:
        detected_text = detect_text_vision_api(file_bytes)

    return jsonify({
        'success': True,
        'image': f'data:image/png;base64,{img_base64}',
        'width': w,
        'height': h,
        'tracks': tracks,
        'detected_text': detected_text,
        'ocr_suggestions': detected_text.get('suggestions', {}),
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
    
    curve_data = {}
    
    for c in curves:
        name = c['name']
        unit = c.get('unit', '')
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
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mask = black_mask(gray)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
        
        xs = pick_curve_x_per_row(mask, min_run)
        xs = smooth_nanmedian(xs, smooth_window)
        
        width_px = mask.shape[1]
        vals = np.full(xs.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(xs)
        vals[valid] = left_value + (xs[valid] / max(1, width_px-1)) * (right_value - left_value)
        
        vals_out = np.where(np.isnan(vals), null_val, vals).astype(np.float32)
        curve_data[name] = {'unit': unit, 'values': vals_out}
    
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
    
    return jsonify({
        'success': True,
        'las_content': las_content,
        'filename': 'digitized_log.las',
        'validation': validation
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'vision_api': VISION_API_AVAILABLE
    })

if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting TIFF→LAS Web App")
    print(f"📊 Google Vision API: {'✅ Available' if VISION_API_AVAILABLE else '⚠️  Not configured'}")
    print("🌐 Open: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
