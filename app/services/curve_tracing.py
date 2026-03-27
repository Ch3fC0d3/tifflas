
import numpy as np
import cv2
import pandas as pd
import math
from typing import Tuple, Optional, List

# Try to import the Numba-optimized tracer
try:
    import fast_tracer
except ImportError:
    # Fallback or local import if running from different context
    try:
        from ... import fast_tracer
    except ImportError:
        fast_tracer = None
        print("Warning: fast_tracer module not found. Curve tracing will be slow or fail.")

# ----------------------------
# Curve Tracing Functions
# ----------------------------

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

    # Use fast_tracer module if available, otherwise we might need a pure python fallback
    # (The original code relied on fast_tracer existing)
    if not fast_tracer:
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
    # ink strokes over their edges.
    if bin_mask.any():
        _dist = cv2.distanceTransform(bin_mask.astype(np.uint8), cv2.DIST_L2, 3)
        _d_max = _dist.max()
        if _d_max > 0:
            _dist_norm = (_dist / _d_max).astype(np.float32)
            live_score = live_score * (0.7 + 0.3 * _dist_norm)
            live_score = np.clip(live_score, eps, 1.0)

    cost = -np.log(live_score)

    # Soft rail penalty: down-weight columns that stay on for many rows
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
    xs = np.full_like(xs_fwd, np.nan)
    confidence = np.zeros_like(conf_fwd)
    
    for y in range(h):
        v1 = xs_fwd[y]
        v2 = xs_bwd[y]
        valid1 = np.isfinite(v1)
        valid2 = np.isfinite(v2)
        
        if valid1 and valid2:
            if abs(v1 - v2) > 5:
                p1 = prob[y, int(min(w-1, max(0, v1)))]
                p2 = prob[y, int(min(w-1, max(0, v2)))]
                xs[y] = v1 if p1 > p2 else v2
            else:
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

    def _skeletonize_binary(bin_img):
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
            best_score = -1e9
            idx_rel = peaks[0]
            for pk in peaks:
                pk_val = row[pk]
                pk_x = start + pk
                dist = abs(pk_x - prev_x)
                
                norm_dist = dist / float(s_rad) # 0 to 1
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

def trace_curve_multiscale(curve_mask, scale_min, scale_max, curve_type="GR", max_step=3, smooth_lambda=0.1, curv_lambda=0.0, hot_side=None, snap_threshold=1.2):
    """
    Enhanced multi-scale curve tracing with 5 scales and weighted fusion.
    """
    if curve_mask is None or curve_mask.size == 0:
        return np.array([]), np.array([])
    
    # For GR curves, we want to allow very sharp peaks, so we lower the smoothing significantly.
    if curve_type.upper() == "GR" and smooth_lambda > 0.01:
        smooth_lambda = 0.001
    snap_threshold = float(np.clip(snap_threshold, 1.0, 2.5))
    
    h, w = curve_mask.shape
    if h < 4 or w < 4:
        return trace_curve_with_dp(curve_mask, scale_min, scale_max, curve_type, max_step, smooth_lambda, curv_lambda, hot_side)
    
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
                "curv_lambda": max(0.000001, curv_lambda * scale * jaggedness_factor)
            }
        else:
            return {
                "smooth_lambda": smooth_lambda * scale,
                "max_step": max(1, int(max_step * scale)),
                "rail_threshold": 0.1 * scale,
                "curv_lambda": curv_lambda * scale
            }
    
    # Calculate jaggedness factor for parameter tuning
    prob = curve_mask.astype(np.float32) / 255.0
    gray = (prob * 255).astype(np.uint8)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    jaggedness_factor = max(0.5, min(2.0, 1.0 + edge_density * 5))
    
    if len(valid_scales) < 2:
        return trace_curve_with_dp(curve_mask, scale_min, scale_max, curve_type, max_step, smooth_lambda, curv_lambda, hot_side)
    
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
        params = get_scale_params(scale, curve_type, jaggedness_factor)
        
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
    
    # Weighted fusion of multi-scale results
    # Prefer results with higher confidence
    final_xs = np.full(h, np.nan, dtype=np.float32)
    final_conf = np.zeros(h, dtype=np.float32)
    
    for y in range(h):
        best_conf = -1.0
        best_val = np.nan
        
        # Check agreement between scales
        values = []
        confs = []
        for i, scale in enumerate(valid_scales):
            if i < len(all_xs) and np.isfinite(all_xs[i][y]):
                values.append(all_xs[i][y])
                confs.append(all_confs[i][y])
        
        if not values:
            continue
            
        # Weighted average based on confidence
        total_weight = sum(confs)
        if total_weight > 0:
            weighted_sum = sum(v * c for v, c in zip(values, confs))
            final_xs[y] = weighted_sum / total_weight
            final_conf[y] = max(confs) # Use max confidence
        else:
            final_xs[y] = values[0]
            final_conf[y] = 0.0
            
    return final_xs, final_conf


def trace_curve_direct_centerline(mask, threshold=10):
    """Find the centerline by picking the peak intensity in each row.
    Uses intensity-weighted centroid around the peak for sub-pixel accuracy.

    Args:
        mask: Grayscale probability map (0-255)
        threshold: Minimum intensity to consider as ink

    Returns:
        Array of x-coordinates (one per row), np.nan where no ink found
    """
    h, w = mask.shape[:2]
    xs = np.full(h, np.nan, dtype=np.float32)

    for y in range(h):
        row = mask[y, :].astype(np.float32)
        valid = row >= threshold
        if not np.any(valid):
            continue
        max_intensity = row[valid].max()
        if max_intensity <= threshold:
            continue
        peak_mask = row >= max_intensity * 0.95
        peak_indices = np.where(peak_mask)[0]
        if len(peak_indices) == 0:
            continue
        weights = row[peak_indices] ** 3
        wsum = weights.sum()
        if wsum > 0:
            xs[y] = float((peak_indices * weights).sum() / wsum)
        else:
            xs[y] = float(peak_indices[len(peak_indices) // 2])

    return xs
