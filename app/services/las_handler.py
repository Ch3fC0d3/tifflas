
import numpy as np
import io
import json
from typing import List, Dict, Optional, Any

# Optional LAS validator
try:
    import lasio
    LASIO_AVAILABLE = True
except ImportError:
    LASIO_AVAILABLE = False

CURVE_KEYWORDS = {
    "GR":   ["GR", "GAMMA RAY"],
    "RES":  ["RES", "RESISTIVITY", "ILD", "LLD", "LWD RES"],
    "RHOB": ["RHOB", "DENSITY", "RHO B", "BULK DENSITY"],
    "NPHI": ["NPHI", "NEUTRON POROSITY", "NEUT", "PHI N"],
    "PEF":  ["PEF", "PHOTOELECTRIC", "PE"],
}

MISSING_MARKERS = [-999.25, -999.0, -9999.0, 999.25]

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
        "min": float(v.min()) if len(v) > 0 else None,
        "max": float(v.max()) if len(v) > 0 else None,
        "mean": float(v.mean()) if len(v) > 0 else None,
        "std": float(v.std()) if len(v) > 0 else None,
        "p05": float(np.percentile(v, 5)) if len(v) > 0 else None,
        "p50": float(np.median(v)) if len(v) > 0 else None,
        "p95": float(np.percentile(v, 95)) if len(v) > 0 else None,
    }

    # Count "spikes" (value changes > 3 std dev in one step)
    if len(v) > 10:
        diffs = np.abs(np.diff(v))
        std_diff = np.std(diffs)
        if std_diff > 1e-6:
            spikes = (diffs > 5 * std_diff).sum()
            features["pct_spikes"] = float(spikes) / len(v) * 100.0
        else:
            features["pct_spikes"] = 0.0
    else:
        features["pct_spikes"] = 0.0
    
    features["pct_missing"] = (1.0 - features["num_valid"]/max(1, features["num_samples"])) * 100.0
    
    return features

def summarize_las_curves_from_str(las_text, depth_mnemonics=("DEPT", "DEPTH")):
    """Read LAS content from a string and compute features for each non-depth curve."""
    if not LASIO_AVAILABLE:
        return None

    try:
        las = lasio.read(io.StringIO(las_text))
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
