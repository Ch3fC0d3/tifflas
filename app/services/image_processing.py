
import cv2
import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Optional, Dict

# ----------------------------
# Core Image Processing Functions
# ----------------------------

def hsv_red_mask(hsv_img):
    # H=0-15: reds/oranges; H=140-180: magentas/pinks — both map to "red" mode
    lower1, upper1 = np.array([0, 40, 40]), np.array([15, 255, 255])
    lower2, upper2 = np.array([140, 40, 40]), np.array([180, 255, 255])
    return cv2.bitwise_or(
        cv2.inRange(hsv_img, lower1, upper1),
        cv2.inRange(hsv_img, lower2, upper2),
    )

def hsv_blue_mask(hsv_img):
    lower, upper = np.array([90, 40, 40]), np.array([140, 255, 255])
    return cv2.inRange(hsv_img, lower, upper)

def hsv_green_mask(hsv_img):
    lower, upper = np.array([35, 40, 40]), np.array([90, 255, 255])
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
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if mode == "black":
        # For black-on-black grids, we need to balance between keeping faded ink
        # and dropping gridlines. A pure adaptive threshold often drops thin/faded log sections.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Blur slightly to smooth out dot-matrix printer artifacts
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use a generous global threshold to ensure we don't drop the curve ink.
        # (Grid lines will also be included, but we'll remove them morphologically later)
        _, curve_mask = cv2.threshold(blur, 190, 255, cv2.THRESH_BINARY_INV)
        
        # Use an adaptive threshold just as a "boost" for very dark local regions
        # that might be slightly above the global 190 threshold due to uneven lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 51, 10
        )
        curve_mask = cv2.bitwise_or(curve_mask, thresh)
    elif mode == "red":
        # HSV-based: catches red (H=0-15) AND magenta/pink (H=140-180)
        curve_mask = hsv_red_mask(hsv)
    elif mode == "blue":
        curve_mask = hsv_blue_mask(hsv)
    elif mode == "green":
        curve_mask = hsv_green_mask(hsv)
    else:
        # Fallback: low-value (dark) pixels
        _, _, v = cv2.split(hsv)
        _, curve_mask = cv2.threshold(v, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Step 2: Remove THIN vertical gridlines
    if h > 15:
        line_len = min(15, h // 3)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
        vert_structures = cv2.morphologyEx(curve_mask, cv2.MORPH_OPEN, vert_kernel)
        
        # Thick structures (>=3px wide) are likely the curve
        thick_vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, line_len))
        thick_vert_structures = cv2.morphologyEx(curve_mask, cv2.MORPH_OPEN, thick_vert_kernel)
        
        # Only subtract thin vertical lines
        thin_vert_lines = cv2.subtract(vert_structures, thick_vert_structures)
        curve_mask = cv2.bitwise_and(curve_mask, cv2.bitwise_not(thin_vert_lines))
    
    # Step 3: Remove THIN horizontal gridlines
    if w > 15:
        line_len = min(15, w // 3)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
        horiz_structures = cv2.morphologyEx(curve_mask, cv2.MORPH_OPEN, horiz_kernel)
        
        # Thick structures (>=3px tall) are likely the curve
        thick_horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 3))
        thick_horiz_structures = cv2.morphologyEx(curve_mask, cv2.MORPH_OPEN, thick_horiz_kernel)
        
        # Only subtract thin horizontal lines
        thin_horiz_lines = cv2.subtract(horiz_structures, thick_horiz_structures)
        curve_mask = cv2.bitwise_and(curve_mask, cv2.bitwise_not(thin_horiz_lines))
    
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

def compute_prob_map(roi_bgr, mode="black"):
    """Build a soft probability map for curve tracing using HSV color isolation,
    edge detection, and distance-transform centerline boosting.

    Returns an 8-bit image (0-255) where higher values = higher curve likelihood.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    h, w = roi_bgr.shape[:2]
    if h < 2 or w < 2:
        return np.zeros((h, w), dtype=np.uint8)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Color mask
    if mode == "green":
        lower = np.array([35, 40, 40], dtype=np.uint8)
        upper = np.array([90, 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower, upper)

        b, g, r = cv2.split(roi_bgr)
        r16 = r.astype(np.int16)
        g16 = g.astype(np.int16)
        b16 = b.astype(np.int16)
        red_dominant = (r16 > g16 + 10) & (r16 > b16 + 10)
        color_mask[red_dominant] = 0

        nonzero = np.nonzero(color_mask)
        if len(nonzero[0]) > 50:
            h_channel = hsv[:, :, 0]
            valid_h = h_channel[nonzero]
            med_h = float(np.median(valid_h))
            band = 15.0
            h_lo = max(0, int(med_h - band))
            h_hi = min(180, int(med_h + band))
            dyn_lower = np.array([h_lo, 40, 40], dtype=np.uint8)
            dyn_upper = np.array([h_hi, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, dyn_lower, dyn_upper)
            color_mask[red_dominant] = 0

        g_dominant = (g16 >= r16 - 5) & (g16 >= b16 + 5)
        color_mask[~g_dominant] = 0

    elif mode == "red":
        lower1 = np.array([0, 70, 40], dtype=np.uint8)
        upper1 = np.array([15, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, 70, 40], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        color_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2),
        )
    elif mode == "blue":
        lower = np.array([90, 40, 40], dtype=np.uint8)
        upper = np.array([140, 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower, upper)
    else:
        color_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10,
        )

    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    color_score = color_mask.astype(np.float32) / 255.0

    # 2) Edge score gated by color for colored modes
    edges = cv2.Canny(gray, 40, 120)
    edges_blur = cv2.GaussianBlur(edges, (5, 5), 0)
    edge_score = edges_blur.astype(np.float32) / 255.0
    if mode in {"green", "red", "blue"}:
        edge_score *= color_score

    # 3) Suppress vertical rails (columns that are "on" most of the image)
    if h >= 4 and w >= 2:
        col_on_frac = (color_score > 0).mean(axis=0)
        rail_cols = col_on_frac > 0.40
        if np.any(rail_cols):
            color_score[:, rail_cols] *= 0.01
            edge_score[:, rail_cols] *= 0.01

    # 4) Centerline boost via distance transform on ink pixels
    ink_mask = (color_score > 0.35) & (edge_score > 0.15)
    if not np.any(ink_mask):
        ink_mask = (color_score > 0.2) & (edge_score > 0.05)

    bin_for_dt = ink_mask.astype(np.uint8)
    if np.any(bin_for_dt):
        dist = cv2.distanceTransform(bin_for_dt, cv2.DIST_L2, 5)
        center_score = dist.astype(np.float32)
        maxd = float(center_score.max())
        if maxd > 0:
            center_score /= maxd
    else:
        center_score = np.zeros_like(color_score, dtype=np.float32)

    # 5) Combine scores; for colored modes strongly emphasize the centerline ridge
    if mode in {"green", "red", "blue"}:
        prob = 0.2 * color_score + 0.1 * edge_score + 0.7 * center_score
        prob = prob ** 2
    else:
        prob = 0.3 * color_score + 0.4 * edge_score + 0.3 * center_score

    # 6) Gate with binary preprocessed mask to suppress gridlines
    try:
        cleaned_binary = preprocess_curve_track(roi_bgr, mode=mode)
        if cleaned_binary is not None and cleaned_binary.size == prob.size:
            cleaned_score = cleaned_binary.astype(np.float32) / 255.0
            gate = 0.05 + 0.95 * cleaned_score
            prob *= gate
    except Exception:
        pass

    maxp = float(prob.max())
    if maxp > 0:
        prob = prob / maxp
    prob = np.clip(prob, 1e-4, 1.0).astype(np.float32)

    return (prob * 255.0).astype(np.uint8)


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
    
    # Heuristics for "colored curve":
    # - Saturation > 60 (to avoid gray grid lines)
    # - Value < 220 (to avoid white paper background, though paper is usually low sat)
    # - Value > 50 (to avoid pure black noise)
    mask = (saturation > 60) & (value > 50) & (value < 230)
    
    valid_hues = hsv[:, :, 0][mask]
    
    if len(valid_hues) < 50:
        return None  # Not enough colored pixels to determine
    
    # Histograms for hues (0-179 in OpenCV)
    hist, bins = np.histogram(valid_hues, bins=18, range=(0, 180))
    peak_idx = np.argmax(hist)
    
    # If the peak is not dominant enough, might be mixed
    if hist[peak_idx] < len(valid_hues) * 0.3:
        return None
        
    peak_hue = (bins[peak_idx] + bins[peak_idx+1]) / 2
    
    # Map to our standard ranges
    # Red: 0-10 or 170-180
    if peak_hue < 15 or peak_hue > 165:
        return "red"
    # Green: 40-90
    elif 35 < peak_hue < 95:
        return "green"
    # Blue: 90-140
    elif 85 < peak_hue < 145:
        return "blue"
        
    return None

def pick_curve_x_per_row(mask, min_run=2):
    h, w = mask.shape
    xs = np.full(h, np.nan, dtype=np.float32)

    max_span = max(25, w // 8)
    search_radius = max(30, w // 6)

    # 1. First pass: trust only unambiguous rows (ink span narrow enough to be the curve, not a gridline)
    for y in range(h):
        idx = np.flatnonzero(mask[y, :] > 0)
        if idx.size >= min_run and (idx[-1] - idx[0]) <= max_span:
            xs[y] = float(np.median(idx))

    # 2. Interpolate anchors → continuous guide path
    s = pd.Series(xs)
    guide = s.interpolate(limit_direction='both').to_numpy()

    # 3. Second pass: for rows with wide ink spans (gridlines/annotations), pick the
    #    pixels CLOSEST to the guide (not median-of-all-nearby which oscillates).
    for y in range(h):
        if np.isnan(xs[y]) and np.isfinite(guide[y]):
            idx = np.flatnonzero(mask[y, :] > 0)
            if idx.size >= 1:
                # Find pixels within search radius
                nearby = idx[np.abs(idx - guide[y]) <= search_radius]
                if nearby.size >= 1:
                    # Among those, find the closest one(s) to the guide
                    distances = np.abs(nearby - guide[y])
                    min_dist = distances.min()
                    # Take median of pixels within 3px of the closest (handles thick ink)
                    closest_cluster = nearby[distances <= min_dist + 3]
                    xs[y] = float(np.median(closest_cluster))

    # 4. Outlier removal: clear picks that deviate too far from their local neighbours.
    outlier_threshold = max(20, w // 8)
    local_med = (pd.Series(xs)
                 .rolling(21, min_periods=3, center=True)
                 .median()
                 .to_numpy(dtype=np.float32))
    xs[np.abs(xs - local_med) > outlier_threshold] = np.nan

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
            
    # Filter peaks that are too close to each other (e.g., edges of the same thick line)
    w = image_array.shape[1]
    min_dist = w * 0.05  # Assume a track is at least 5% of image width
    
    filtered_peaks = []
    for p in peaks:
        if not filtered_peaks or (p - filtered_peaks[-1]) >= min_dist:
            filtered_peaks.append(p)
            
    # Group into tracks (a track is between two consecutive lines)
    if len(filtered_peaks) >= 2:
        tracks = [(int(filtered_peaks[i]), int(filtered_peaks[i+1])) for i in range(len(filtered_peaks)-1)]
    else:
        # Fallback: divide into 3 equal sections
        section_width = w // 3
        tracks = [(int(i*section_width), int((i+1)*section_width)) for i in range(3)]
    
    return tracks
