
import os
import json
import cv2
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Vision API is optional
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
        # Local development: JSON file path in env var
        vision_client = vision.ImageAnnotatorClient()
        VISION_API_AVAILABLE = True
        print("✅ Google Vision API: Loaded from file")
    else:
        # Auto-detect key file in project directory (try to find it relative to this file or cwd)
        # Assuming this file is in app/services/, we look up 2 levels
        _local_key = Path(__file__).resolve().parent.parent.parent / 'GOOGLE_APPLICATION_CREDENTIALS.json'
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

from .las_handler import extract_curve_labels_from_text, match_vision_to_las_curves

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

def build_ocr_suggestions(numeric_entries):
    # This was originally a helper in web_app.py but implementation wasn't shown in the snippets.
    # I will provide a basic implementation or placeholder.
    # Assuming it groups numbers by Y-coordinate to suggest curves.
    
    # For now, let's just return a list of values found. 
    # Real implementation would clustering logic.
    return numeric_entries

def attach_curve_label_hints(suggestions, raw_text):
    # Placeholder for logic that attaches "GR", "RES" etc. to tracks
    return suggestions

def attach_color_hints_to_ocr_curves(crop_img, ocr_suggestions):
    # Placeholder
    return ocr_suggestions

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
