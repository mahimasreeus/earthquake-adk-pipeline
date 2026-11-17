"""Reporting: render a self-contained HTML report with Jinja2 and publish it."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import TEMPLATES_DIR, get_settings
from .common import error_result, read_json, require_state
from .processing import DEPTH_LABELS
from .storage import LATEST_REPORT_KEY, get_backend, object_prefix

logger = logging.getLogger(__name__)

MAX_MAP_POINTS = 5000  # keeps the HTML small even for large windows
# Sequential blue ramp (light -> dark) for the band x depth heatmap cells.
HEAT_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe records (NaN -> None)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def _heatmap(band_depth: pd.DataFrame) -> dict[str, Any]:
    """Pivot band x depth counts into rows of cells with a ramp color each."""
    pivot = band_depth.pivot(index="magnitude_band", columns="depth_class",
                             values="event_count")
    pivot = pivot.reindex(index=list(dict.fromkeys(band_depth["magnitude_band"])),
                          columns=DEPTH_LABELS).fillna(0).astype(int)
    peak = max(int(pivot.values.max()), 1)
    rows = []
    for band, counts in pivot.iterrows():
        cells = []
        for depth, count in counts.items():
            step = 0 if count == 0 else 1 + round((len(HEAT_RAMP) - 2) * count / peak)
            cells.append({"depth": depth, "count": int(count), "bg": HEAT_RAMP[step],
                          "dark_text": step < 4, "empty": count == 0})
        rows.append({"band": band, "cells": cells, "total": int(counts.sum())})
    return {"columns": DEPTH_LABELS, "rows": rows,
            "column_totals": [int(v) for v in pivot.sum(axis=0)]}


def report_keys(state) -> dict[str, str]:
    prefix = object_prefix("reports", state["run_date"], state["run_id"])
    return {"run": f"{prefix}/index.html", "latest": LATEST_REPORT_KEY}


def build_report_context(state, summary_text: str) -> dict[str, Any]:
    """Collect everything the template needs from the run's files."""
    processed = Path(state["run_dir"]) / "processed"
    events = pd.read_csv(processed / "events_clean.csv")
    daily = pd.read_csv(processed / "agg_daily.csv")
    backend = get_backend()
    keys = report_keys(state)

    points = events.sort_values("magnitude", ascending=False).head(MAX_MAP_POINTS)
    return {
        "summary_text": summary_text.strip(),
        "run": {
            "run_id": state["run_id"], "run_date": state["run_date"],
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "backend": get_settings().storage_backend,
            "source_url": state.get("source_url"),
            "start": state.get("query_start"), "end": state.get("query_end"),
            "min_magnitude": state.get("min_magnitude"),
        },
        "kpis": read_json(processed / "kpis.json"),
        "quality": read_json(processed / "quality_report.json"),
        "daily": _records(daily),
        "regions": _records(pd.read_csv(processed / "agg_regions.csv")),
        "heatmap": _heatmap(pd.read_csv(processed / "agg_band_depth.csv")),
        "top_events": _records(pd.read_csv(processed / "top_events.csv")),
        "points": _records(points[["latitude", "longitude", "depth_km", "magnitude",
                                   "place", "time"]].round(3)),
        "points_truncated": len(events) > MAX_MAP_POINTS,
        "uris": {
            "raw": state.get("raw_uris", []),
            "processed": state.get("processed_uris", []),
            "report": [backend.get_public_url_or_uri(keys["run"]),
                       backend.get_public_url_or_uri(keys["latest"])],
        },
    }


def render_report(context: dict[str, Any]) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR),
                      autoescape=select_autoescape(["html", "j2"]))
    return env.get_template("report.html.j2").render(**context)


# --- ADK tools ------------------------------------------------------------
def get_report_data(tool_context=None) -> dict[str, Any]:
    """Return a small JSON digest of this run's results for writing the
    summary: KPIs, top 5 regions, magnitude band x depth counts, quality
    status, and the query window. Call this before writing the summary.

    Returns:
        A status dict with the digest (never the full dataset).
    """
    try:
        state = tool_context.state
        run_dir = require_state(state, "run_dir", "kpis_path")[0]
        processed = Path(run_dir) / "processed"
        quality = read_json(processed / "quality_report.json")
        band = pd.read_csv(processed / "agg_band_depth.csv")
        events = pd.read_csv(processed / "events_clean.csv")
        # Name the specific flagged events so the LLM never has to guess which
        # event a tsunami flag belongs to.
        flagged = events[events["tsunami"] == 1].sort_values("magnitude", ascending=False)
        return {
            "status": "success",
            "window": {"start": state.get("query_start"),
                       "end_exclusive": state.get("query_end"),
                       "min_magnitude": state.get("min_magnitude")},
            "kpis": read_json(processed / "kpis.json"),
            "top_regions": _records(pd.read_csv(processed / "agg_regions.csv").head(5)),
            "band_depth_counts": _records(band[band["event_count"] > 0]),
            "tsunami_flagged_events": _records(flagged[["magnitude", "place", "date"]].head(10)),
            "quality": {k: quality[k] for k in ("status", "issues", "raw_rows", "clean_rows")},
        }
    except Exception as exc:
        return error_result("get_report_data", exc)


def generate_html_report(summary_text: str, tool_context=None) -> dict[str, Any]:
    """Render the HTML report (charts, tables, data quality & lineage) with
    the given plain-English summary embedded in the "Agent summary" section.

    Writes the page into the run folder and to output/latest/index.html.

    Args:
        summary_text: A 3-5 sentence plain-English summary of notable patterns.

    Returns:
        A status dict with the local report paths.
    """
    try:
        state = tool_context.state
        run_dir = require_state(state, "run_dir", "kpis_path", "quality_path")[0]
        if not summary_text or len(summary_text.strip()) < 20:
            raise ValueError("summary_text is empty; write a 3-5 sentence summary first.")
        html = render_report(build_report_context(state, summary_text))

        run_report = Path(run_dir) / "report" / "index.html"
        latest = get_settings().output_dir / "latest" / "index.html"
        for path in (run_report, latest):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")

        state["report_path"] = str(run_report)
        state["local_latest_report"] = str(latest)
        return {"status": "success", "report_path": str(run_report),
                "local_latest_report": str(latest), "size_kb": round(len(html) / 1024, 1)}
    except Exception as exc:
        return error_result("generate_html_report", exc)


def upload_report(tool_context=None) -> dict[str, Any]:
    """Upload the generated HTML report to storage, both under this run's
    partition and as reports/earthquakes/latest/index.html.

    Returns:
        A status dict with the report URIs.
    """
    try:
        state = tool_context.state
        (report_path,) = require_state(state, "report_path")
        backend = get_backend()
        html = Path(report_path).read_bytes()
        keys = report_keys(state)
        uris = [backend.upload_bytes(html, keys["run"], "text/html; charset=utf-8"),
                backend.upload_bytes(html, keys["latest"], "text/html; charset=utf-8")]
        state["report_uris"] = uris
        return {"status": "success", "backend": backend.name, "uris": uris,
                "local_latest_report": state.get("local_latest_report")}
    except Exception as exc:
        return error_result("upload_report", exc)
