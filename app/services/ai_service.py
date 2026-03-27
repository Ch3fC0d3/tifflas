
import os
import json
import requests
from typing import Optional, Dict, Any

# Optional imports for AI clients
try:
    import openai
except ImportError:
    openai = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

# Load credentials from environment
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID") or os.getenv("OPENAI_MODEL")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID") or "models/gemini-2.0-flash"

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
    if OPENAI_API_KEY and OPENAI_MODEL_ID and openai:
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
    if not HF_API_TOKEN or not HF_MODEL_ID or not InferenceClient:
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
    if OPENAI_API_KEY and OPENAI_MODEL_ID and openai:
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
    if not HF_API_TOKEN or not HF_MODEL_ID or not InferenceClient:
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
    if OPENAI_API_KEY and OPENAI_MODEL_ID and openai:
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
    if not HF_API_TOKEN or not HF_MODEL_ID or not InferenceClient:
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
    if OPENAI_API_KEY and OPENAI_MODEL_ID and openai:
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
    if not HF_API_TOKEN or not HF_MODEL_ID or not InferenceClient:
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
