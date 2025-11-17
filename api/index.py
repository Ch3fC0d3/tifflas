"""Vercel entry point for the TIFF→LAS Flask application."""
import sys
import os

# Add parent directory to path so we can import web_app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import pandas as pd
import json
from io import BytesIO
import base64
from typing import Dict, List, Tuple

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
    """Generate LAS file as string"""
    null_val = -999.25
    lines = []
    lines.append("~Version\n")
    lines.append(" VERS. 2.0 : CWLS LOG ASCII STANDARD\n")
    lines.append(" WRAP. NO  : One line per depth step\n")
    lines.append("~Well\n")
    lines.append(f" STRT.{depth_unit} {depth[0]:.4f} : START DEPTH\n")
    lines.append(f" STOP.{depth_unit} {depth[-1]:.4f} : STOP DEPTH\n")
    step = float(depth[1] - depth[0]) if depth.size > 1 else 0.0
    lines.append(f" STEP.{depth_unit} {step:.4f} : STEP\n")
    lines.append(f" NULL. {null_val} : NULL VALUE\n")
    lines.append("~Curve\n")
    lines.append(f" DEPT.{depth_unit} : Depth\n")
    for name, meta in curve_data.items():
        unit = meta.get("unit", "")
        lines.append(f" {name}.{unit} : {name}\n")
    lines.append("~ASCII\n")

    names = list(curve_data.keys())
    arrays = [curve_data[n]["values"] for n in names]
    for i in range(depth.size):
        row = [f"{depth[i]:.4f}"] + [f"{arrays[j][i]:.4f}" for j in range(len(arrays))]
        lines.append(" ".join(row) + "\n")

    return "".join(lines)

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
    try:
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
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}',
            'width': w,
            'height': h,
            'tracks': tracks,
            'detected_text': [],
            'vision_api_available': False
        })
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/digitize', methods=['POST'])
def digitize():
    """Process digitization request"""
    try:
        data = request.json
        
        # Decode image
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract config
        cfg = data['config']
        depth_cfg = cfg['depth']
        curves = cfg['curves']
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
        
        # Generate LAS file
        las_content = write_las_simple(base_depth, curve_data, depth_unit)
        
        return jsonify({
            'success': True,
            'las_content': las_content,
            'filename': 'digitized_log.las'
        })
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'vision_api': False
    })
