"""
SAFER Microservices — Shared Configuration

Centralized configuration for all services.
Database abstraction layer supports SQLite (dev) and PostgreSQL (prod).
"""

import os
from pathlib import Path

# ─── Environment Detection ──────────────────────────────────────────────────
IS_PRODUCTION = os.getenv("RENDER", "") == "true" or os.getenv("IS_PRODUCTION", "") == "true"

# ─── Paths ──────────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent          # Root of repository
MODEL_DIR = BACKEND_ROOT / "MODEL AI"

# ─── Service Ports ──────────────────────────────────────────────────────────
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))
SCORING_PORT = int(os.getenv("SCORING_PORT", "8001"))
GRAPH_PORT = int(os.getenv("GRAPH_PORT", "8002"))
DATA_PORT = int(os.getenv("DATA_PORT", "8003"))

# ─── Service URLs (internal communication) ──────────────────────────────────
SCORING_SERVICE_URL = os.getenv("SCORING_SERVICE_URL", f"http://localhost:{SCORING_PORT}")
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", f"http://localhost:{GRAPH_PORT}")
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", f"http://localhost:{DATA_PORT}")

# ─── Database ──────────────────────────────────────────────────────────────
# SQLite for development, swap to PostgreSQL by changing this env var:
#   DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/safer
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BACKEND_ROOT / 'safer.db'}"
)

# ─── CORS ───────────────────────────────────────────────────────────────────
CORS_ORIGINS = [
    # Local development
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    # Cloudflare Pages production URLs
    "https://safer-fds.pages.dev",
    "https://*.safer-fds.pages.dev",
    # Render.com backend (same-origin API calls)
    "https://safer-api.onrender.com",
    # HuggingFace Spaces (alternative deployment)
    "https://*.hf.space",
]

# Allow additional origins via env
_extra = os.getenv("CORS_EXTRA_ORIGINS", "")
if _extra:
    CORS_ORIGINS.extend(_extra.split(","))

# ─── Model Files ────────────────────────────────────────────────────────────
XGB_MODEL_PATH = MODEL_DIR / "xgb_model.json"
LGB_MODEL_PATH = MODEL_DIR / "lgb_model.txt"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# ─── Feature Configuration (must match training pipeline) ───────────────────
CATEGORICAL_COLS = [
    "sender_bank", "receiver_bank", "payment_rail",
    "ewallet_provider", "merchant", "merchant_category",
    "channel", "device_type", "device_brand",
]

BINARY_COLS = [
    "is_velocity_anomaly", "is_geo_mismatch", "is_off_hours",
    "is_high_value_for_rail", "is_suspicious_ip", "is_risky_merchant",
    "is_new_account", "has_failed_attempts", "is_device_mismatch",
    "is_sim_swap", "is_unusual_beneficiary", "is_new_device",
]

NUMERIC_COLS = [
    "amount", "account_age_days", "velocity_count", "geo_distance_km",
]

# Columns to drop before feeding to ML model (non-feature columns)
DROP_COLS = [
    "id", "timestamp", "sender_name", "sender_account", "sender_city",
    "sender_province", "sender_lat", "sender_lng",
    "receiver_name", "receiver_account", "receiver_city",
    "receiver_province", "receiver_lat", "receiver_lng",
    "ip_address", "device_fingerprint", "is_fraud",
]

# ─── Severity Thresholds ────────────────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "critical": 80,
    "high": 60,
    "medium": 35,
    "low": 0,
}

def get_severity(score: int) -> str:
    """Convert numeric risk score (0-100) to severity label."""
    if score >= SEVERITY_THRESHOLDS["critical"]:
        return "critical"
    elif score >= SEVERITY_THRESHOLDS["high"]:
        return "high"
    elif score >= SEVERITY_THRESHOLDS["medium"]:
        return "medium"
    return "low"
