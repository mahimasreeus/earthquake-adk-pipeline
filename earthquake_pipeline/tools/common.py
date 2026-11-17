"""Helpers shared by every tool: run context, file paths, result dicts.

Agents never see the data itself. Tools write files into a run-scoped working
directory (output/runs/<run_id>/) and keep only small values (paths, counts,
run_id) in ADK session state, which is a plain dict-like object.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, MutableMapping

from ..config import get_settings

logger = logging.getLogger(__name__)

# Session-state keys used across agents (one place, so names never drift).
RUN_ID = "run_id"
RUN_DATE = "run_date"
RUN_DIR = "run_dir"


def new_run_id() -> str:
    """UTC timestamp like 20261116T224500Z — sortable and unique per run."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_run_context(state: MutableMapping[str, Any]) -> Path:
    """Create run_id / run_date / run_dir in state if this is a new run.

    main.py seeds these up front; under `adk web` the first tool call creates
    them. Returns the run's working directory.
    """
    if not state.get(RUN_ID):
        run_id = new_run_id()
        state[RUN_ID] = run_id
        state[RUN_DATE] = f"{run_id[0:4]}-{run_id[4:6]}-{run_id[6:8]}"
    if not state.get(RUN_DIR):
        state[RUN_DIR] = str(get_settings().output_dir / "runs" / state[RUN_ID])
    run_dir = Path(state[RUN_DIR])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def require_state(state: MutableMapping[str, Any], *keys: str) -> list[Any]:
    """Return state values for `keys`, raising a clear error if any is missing
    (e.g. a step was skipped or an earlier tool failed)."""
    missing = [k for k in keys if not state.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing pipeline state {missing}; run the earlier step(s) first."
        )
    return [state[k] for k in keys]


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def error_result(step: str, exc: Exception) -> dict[str, Any]:
    """Uniform error payload so agents can see what failed and react."""
    logger.exception("%s failed", step)
    return {"status": "error", "step": step, "error_type": type(exc).__name__,
            "error": str(exc)}
