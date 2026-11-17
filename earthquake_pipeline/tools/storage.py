"""Storage: one small interface, three backends (local, GCS, S3).

The agent tools only call `get_backend()` and the three interface methods, so
switching STORAGE_BACKEND never changes pipeline code.
"""

from __future__ import annotations

import logging
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..config import ConfigError, Settings, get_settings
from .common import error_result, require_state

logger = logging.getLogger(__name__)

DATASET = "earthquakes"
CONTENT_TYPES = {".geojson": "application/geo+json", ".csv": "text/csv",
                 ".json": "application/json", ".html": "text/html; charset=utf-8",
                 ".parquet": "application/vnd.apache.parquet"}


def guess_content_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return CONTENT_TYPES.get(suffix) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def object_prefix(zone: str, run_date: str, run_id: str) -> str:
    """Data-lake style partitioned prefix, e.g.
    processed/earthquakes/run_date=2026-11-16/run_id=20261116T224500Z"""
    return f"{zone}/{DATASET}/run_date={run_date}/run_id={run_id}"


LATEST_REPORT_KEY = f"reports/{DATASET}/latest/index.html"


class StorageBackend(ABC):
    """Minimal interface every backend implements."""

    name = "base"

    @abstractmethod
    def upload_bytes(self, data: bytes, key: str, content_type: str) -> str:
        """Store `data` at `key`; return the object's URI."""

    @abstractmethod
    def upload_file(self, path: str | Path, key: str, content_type: str | None = None) -> str:
        """Store a local file at `key`; return the object's URI."""

    @abstractmethod
    def get_public_url_or_uri(self, key: str) -> str:
        """Return the URI (gs://, s3://, or local path) for `key`."""


class LocalBackend(StorageBackend):
    """Writes into ./output/ with the same key layout as the cloud buckets —
    lets anyone run the pipeline without cloud credentials."""

    name = "local"

    def __init__(self, root: Path):
        self.root = Path(root)

    def upload_bytes(self, data: bytes, key: str, content_type: str) -> str:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return self.get_public_url_or_uri(key)

    def upload_file(self, path, key, content_type=None) -> str:
        return self.upload_bytes(Path(path).read_bytes(), key, content_type or "")

    def get_public_url_or_uri(self, key: str) -> str:
        return str((self.root / key).resolve())


class GCSBackend(StorageBackend):
    """Google Cloud Storage. Auth via Application Default Credentials
    (`gcloud auth application-default login`) or GOOGLE_APPLICATION_CREDENTIALS."""

    name = "gcs"

    def __init__(self, bucket: str):
        from google.cloud import storage  # imported lazily: only needed for gcs

        self.bucket_name = bucket
        self.bucket = storage.Client().bucket(bucket)

    def upload_bytes(self, data: bytes, key: str, content_type: str) -> str:
        blob = self.bucket.blob(key)
        if key.endswith("latest/index.html"):
            blob.cache_control = "no-cache"  # always serve the newest report
        blob.upload_from_string(data, content_type=content_type)
        return self.get_public_url_or_uri(key)

    def upload_file(self, path, key, content_type=None) -> str:
        return self.upload_bytes(Path(path).read_bytes(), key,
                                 content_type or guess_content_type(path))

    def get_public_url_or_uri(self, key: str) -> str:
        return f"gs://{self.bucket_name}/{key}"


class S3Backend(StorageBackend):
    """AWS S3. Auth via the standard boto3 chain (~/.aws/credentials, env vars,
    or an IAM role)."""

    name = "s3"

    def __init__(self, bucket: str, region: str | None = None):
        import boto3  # imported lazily: only needed for s3

        self.bucket_name = bucket
        self.client = boto3.client("s3", region_name=region)

    def upload_bytes(self, data: bytes, key: str, content_type: str) -> str:
        self.client.put_object(Bucket=self.bucket_name, Key=key, Body=data,
                               ContentType=content_type)
        return self.get_public_url_or_uri(key)

    def upload_file(self, path, key, content_type=None) -> str:
        return self.upload_bytes(Path(path).read_bytes(), key,
                                 content_type or guess_content_type(path))

    def get_public_url_or_uri(self, key: str) -> str:
        return f"s3://{self.bucket_name}/{key}"


def get_backend(settings: Settings | None = None) -> StorageBackend:
    """Factory driven by STORAGE_BACKEND=local|gcs|s3."""
    settings = settings or get_settings()
    if settings.storage_backend == "gcs":
        if not settings.gcs_bucket or settings.gcs_bucket.startswith("your-"):
            raise ConfigError("STORAGE_BACKEND=gcs but GCS_BUCKET is not set in .env.")
        return GCSBackend(settings.gcs_bucket)
    if settings.storage_backend == "s3":
        if not settings.s3_bucket or settings.s3_bucket.startswith("your-"):
            raise ConfigError("STORAGE_BACKEND=s3 but S3_BUCKET is not set in .env.")
        return S3Backend(settings.s3_bucket, settings.aws_region)
    return LocalBackend(settings.output_dir)


def _upload_dir(state, local_dir: Path, zone: str) -> list[str]:
    """Upload every file in `local_dir` (skipping _private files) under the
    zone's partitioned prefix; return the URIs written."""
    backend = get_backend()
    prefix = object_prefix(zone, state["run_date"], state["run_id"])
    uris = []
    for path in sorted(local_dir.iterdir()):
        if path.is_file() and not path.name.startswith("_"):
            uris.append(backend.upload_file(path, f"{prefix}/{path.name}"))
            logger.info("Uploaded %s", uris[-1])
    return uris


# --- ADK tools ------------------------------------------------------------
def upload_raw(tool_context=None) -> dict[str, Any]:
    """Upload the raw GeoJSON for this run to the configured storage backend
    under raw/earthquakes/run_date=.../run_id=.../.

    Returns:
        A status dict with the backend name and the URIs written.
    """
    try:
        state = tool_context.state
        raw_path = require_state(state, "raw_path", "run_id")[0]
        uris = _upload_dir(state, Path(raw_path).parent, "raw")
        state["raw_uris"] = uris
        return {"status": "success", "backend": get_settings().storage_backend, "uris": uris}
    except Exception as exc:
        return error_result("upload_raw", exc)


def upload_processed(tool_context=None) -> dict[str, Any]:
    """Upload all processed outputs (clean events, aggregates, KPIs, quality
    report) to processed/earthquakes/run_date=.../run_id=.../.

    Returns:
        A status dict with the backend name and the URIs written.
    """
    try:
        state = tool_context.state
        run_dir = require_state(state, "run_dir", "kpis_path", "quality_path")[0]
        uris = _upload_dir(state, Path(run_dir) / "processed", "processed")
        state["processed_uris"] = uris
        return {"status": "success", "backend": get_settings().storage_backend,
                "file_count": len(uris), "uris": uris}
    except Exception as exc:
        return error_result("upload_processed", exc)
