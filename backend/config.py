"""
config.py
---------
Centralized, environment-driven configuration. Nothing here is a secret —
API keys and provider selection are read from the environment so they never
need to be hard-coded or committed.

Every setting has a safe local default so the app runs out of the box on a
laptop with no external services configured.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


# ---------------------------------------------------------------------------
# Model / weights
# ---------------------------------------------------------------------------

WEIGHTS_PATH: Path = _env_path("TRIAGE_WEIGHTS_PATH", MODELS_DIR / "lesion_classifier.pt")

# ---------------------------------------------------------------------------
# Similarity search (FAISS)
# ---------------------------------------------------------------------------

SIMILARITY_INDEX_DIR: Path = _env_path("TRIAGE_SIMILARITY_INDEX_DIR", MODELS_DIR / "similarity_index")
SIMILARITY_INDEX_FILE: Path = SIMILARITY_INDEX_DIR / "index.faiss"
SIMILARITY_METADATA_FILE: Path = SIMILARITY_INDEX_DIR / "metadata.json"
SIMILARITY_TOP_K: int = int(os.environ.get("TRIAGE_SIMILARITY_TOP_K", "4"))

# ---------------------------------------------------------------------------
# AI assistant (optional external LLM)
# ---------------------------------------------------------------------------

# AI_PROVIDER: "" or "local"  -> deterministic local fallback only (default, no network calls)
#              "openai"       -> use an OpenAI-compatible chat completions endpoint
AI_PROVIDER: str = os.environ.get("AI_PROVIDER", "local").strip().lower()
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ASSISTANT_TIMEOUT_SECONDS: float = float(os.environ.get("TRIAGE_ASSISTANT_TIMEOUT", "8"))


def assistant_external_configured() -> bool:
    return AI_PROVIDER == "openai" and bool(OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

MAX_IMAGE_MB: float = float(os.environ.get("TRIAGE_MAX_IMAGE_MB", "12"))
MAX_IMAGE_BYTES: int = int(MAX_IMAGE_MB * 1024 * 1024)
ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

_default_origins = (
    "http://127.0.0.1:8000,http://localhost:8000,"
    "http://127.0.0.1:5500,http://localhost:5500"
)
CORS_ORIGINS = [o.strip() for o in os.environ.get("TRIAGE_CORS_ORIGINS", _default_origins).split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

APP_VERSION = "0.2.2"
DISABLE_GRADCAM = _env_bool("TRIAGE_DISABLE_GRADCAM", False)
