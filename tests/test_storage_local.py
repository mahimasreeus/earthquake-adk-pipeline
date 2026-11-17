"""Tests for the local storage backend and the upload tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from earthquake_pipeline.config import ConfigError
from earthquake_pipeline.tools import storage as s
from earthquake_pipeline.tools.common import ensure_run_context, write_json


def test_object_prefix_layout():
    assert (s.object_prefix("raw", "2026-11-16", "20261116T120000Z")
            == "raw/earthquakes/run_date=2026-11-16/run_id=20261116T120000Z")


def test_local_backend_writes_bytes_and_files(tmp_path):
    backend = s.LocalBackend(tmp_path)
    uri = backend.upload_bytes(b"hello", "a/b/c.txt", "text/plain")
    assert Path(uri).read_bytes() == b"hello"

    src = tmp_path / "src.csv"
    src.write_text("x\n1\n")
    uri = backend.upload_file(src, "processed/x.csv")
    assert Path(uri).read_text() == "x\n1\n"
    assert backend.get_public_url_or_uri("processed/x.csv") == uri


def test_get_backend_selects_local(local_env):
    backend = s.get_backend()
    assert isinstance(backend, s.LocalBackend)
    assert backend.root == local_env


def test_get_backend_requires_bucket(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "gcs")
    monkeypatch.setenv("GCS_BUCKET", "")
    with pytest.raises(ConfigError, match="GCS_BUCKET"):
        s.get_backend()


def test_content_types():
    assert s.guess_content_type("x.geojson") == "application/geo+json"
    assert s.guess_content_type("index.html").startswith("text/html")


def test_upload_raw_tool_uses_partitioned_keys(tool_context, local_env):
    state = tool_context.state
    run_dir = ensure_run_context(state)
    state["raw_path"] = str(write_json(run_dir / "raw" / "earthquakes.geojson", {"features": []}))

    result = s.upload_raw(tool_context)
    assert result["status"] == "success"
    expected = (local_env / "raw" / "earthquakes" / f"run_date={state['run_date']}"
                / f"run_id={state['run_id']}" / "earthquakes.geojson")
    assert result["uris"] == [str(expected.resolve())]
    assert expected.exists()
    assert state["raw_uris"] == result["uris"]


def test_upload_processed_reports_missing_state(tool_context):
    ensure_run_context(tool_context.state)
    result = s.upload_processed(tool_context)
    assert result["status"] == "error"
