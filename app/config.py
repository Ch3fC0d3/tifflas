
import os
from pathlib import Path
from typing import List

# ----------------------------
# Environment & Paths
# ----------------------------

def _default_curve_trace_model_candidates() -> List[Path]:
    # Assuming this file is in app/config.py, repo_dir is up one level
    repo_dir = Path(__file__).resolve().parent.parent
    volume_mount = str(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") or "").strip()
    volume_dir = Path(volume_mount) if volume_mount else None
    
    candidates = []
    if volume_dir:
        candidates.extend([
            volume_dir / 'models' / 'testtiflas_black_seg_v2_pairs_wvgs.pt',
            volume_dir / 'models' / 'testtiflas_black_seg_v2_pairs.pt',
            volume_dir / 'models' / 'testtiflas_black_seg_v2_captures.pt',
            volume_dir / 'models' / 'testtiflas_black_seg_v2.pt',
        ])
        
    candidates.extend([
        repo_dir / 'models' / 'testtiflas_black_seg_v2_pairs_wvgs.pt',
        Path.cwd() / 'models' / 'testtiflas_black_seg_v2_pairs_wvgs.pt',
        repo_dir / 'models' / 'testtiflas_black_seg_v2_pairs.pt',
        Path.cwd() / 'models' / 'testtiflas_black_seg_v2_pairs.pt',
        repo_dir / 'models' / 'testtiflas_black_seg_v2_captures.pt',
        Path.cwd() / 'models' / 'testtiflas_black_seg_v2_captures.pt',
        repo_dir / 'models' / 'testtiflas_black_seg_v2.pt',
        Path.cwd() / 'models' / 'testtiflas_black_seg_v2.pt',
        repo_dir / 'deploy_models' / 'testtiflas_black_seg_v2_pairs_wvgs.pt',
        Path.cwd() / 'deploy_models' / 'testtiflas_black_seg_v2_pairs_wvgs.pt',
        repo_dir / 'curve_trace_model.pt',
        Path.cwd() / 'curve_trace_model.pt',
    ])
    return candidates


def resolve_default_curve_trace_model_path() -> str:
    env_path = os.environ.get("CURVE_TRACE_MODEL_PATH")
    if env_path:
        return env_path

    candidates = _default_curve_trace_model_candidates()
    for p in candidates:
        try:
            if p.exists():
                return str(p)
        except Exception:
            continue
    return str(candidates[0])


def experimental_black_ai_enabled() -> bool:
    value = str(os.environ.get("TURBOTIFFLAS_ENABLE_EXPERIMENTAL_BLACK_AI", "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def training_captures_base_dir() -> Path:
    repo_dir = Path(__file__).resolve().parent.parent
    explicit = str(os.environ.get("TURBOTIFFLAS_TRAINING_CAPTURES_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser()

    volume_mount = str(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") or "").strip()
    if volume_mount:
        return Path(volume_mount) / 'training_captures'

    return repo_dir / 'training_captures'

# ----------------------------
# Constants
# ----------------------------

CURVE_TYPE_DEFAULTS = {
    "GR":   {"mnemonic": "GR",   "unit": "API"},
    "RHOB": {"mnemonic": "RHOB", "unit": "G/CC"},
    "NPHI": {"mnemonic": "NPHI", "unit": "V/V"},
    "DT":   {"mnemonic": "DTC",  "unit": "US/F"},
    "DTC":  {"mnemonic": "DTC",  "unit": "US/F"},
    "CALI": {"mnemonic": "CALI", "unit": "IN"},
    "SP":   {"mnemonic": "SP",   "unit": "MV"},
}

try:
    CURVE_TRACE_UPSCALE = float(os.environ.get("CURVE_TRACE_UPSCALE", "2.0"))
except Exception:
    CURVE_TRACE_UPSCALE = 2.0
CURVE_TRACE_UPSCALE = max(1.0, min(4.0, CURVE_TRACE_UPSCALE))

APP_VERSION = os.environ.get("APP_VERSION", "dev")
APP_BUILD_TIME = os.environ.get("APP_BUILD_TIME", "unknown")
SECRET_KEY = os.environ.get("SECRET_KEY", "tiflas-dev-secret-key-change-in-prod")

# ----------------------------
# Auth + Billing (Stripe)
# ----------------------------

APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:5000")

_default_auth_db_path = Path.cwd() / "data" / "auth_billing.db"
AUTH_DB_PATH = os.environ.get("AUTH_DB_PATH", str(_default_auth_db_path))

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_MONTHLY = os.environ.get("STRIPE_PRICE_MONTHLY", "")
STRIPE_PRICE_ANNUAL = os.environ.get("STRIPE_PRICE_ANNUAL", "")
