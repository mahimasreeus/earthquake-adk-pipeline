"""Central configuration: loads .env and exposes typed settings.

Every setting comes from an environment variable (optionally via a .env file
at the project root) so no credentials, bucket names, or project IDs are ever
hardcoded in the source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"

# Load .env from the project root (no-op if it doesn't exist).
load_dotenv(PROJECT_ROOT / ".env")

USGS_QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_FALLBACK_FEED_URL = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson"
)

VALID_BACKENDS = ("local", "gcs", "s3")


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Snapshot of all runtime settings."""

    gemini_model: str
    use_vertexai: bool
    days: int
    min_magnitude: float
    storage_backend: str
    gcs_bucket: str | None
    s3_bucket: str | None
    aws_region: str | None
    output_dir: Path


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes")


def get_settings() -> Settings:
    """Read settings from the environment (call at runtime, not import time,
    so CLI flags that set env vars are respected)."""
    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    if backend not in VALID_BACKENDS:
        raise ConfigError(
            f"STORAGE_BACKEND={backend!r} is invalid; use one of {VALID_BACKENDS}."
        )
    output_dir = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()
    return Settings(
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        use_vertexai=_get_bool("GOOGLE_GENAI_USE_VERTEXAI"),
        days=int(os.getenv("PIPELINE_DAYS", "30")),
        min_magnitude=float(os.getenv("PIPELINE_MIN_MAGNITUDE", "2.5")),
        storage_backend=backend,
        gcs_bucket=os.getenv("GCS_BUCKET") or None,
        s3_bucket=os.getenv("S3_BUCKET") or None,
        aws_region=os.getenv("AWS_REGION") or None,
        output_dir=output_dir,
    )


def validate_llm_credentials() -> None:
    """Fail fast with a helpful message if Gemini credentials are missing."""
    if _get_bool("GOOGLE_GENAI_USE_VERTEXAI"):
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            raise ConfigError(
                "GOOGLE_GENAI_USE_VERTEXAI=TRUE but GOOGLE_CLOUD_PROJECT is not set. "
                "Set it in .env (see .env.example)."
            )
        return
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key or key.startswith("your-"):
        raise ConfigError(
            "GOOGLE_API_KEY not set — copy .env.example to .env and add your "
            "Gemini API key from https://aistudio.google.com/apikey"
        )
