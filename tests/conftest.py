"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "sample_earthquakes.geojson"


@pytest.fixture
def geojson() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def local_env(tmp_path, monkeypatch) -> Path:
    """Point the pipeline at a temp output dir with the local backend."""
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def tool_context(local_env):
    """Stand-in for ADK's ToolContext: tools only use `.state`."""
    return SimpleNamespace(state={})
