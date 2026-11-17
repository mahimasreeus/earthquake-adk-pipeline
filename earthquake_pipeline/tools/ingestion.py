"""Ingestion: pull earthquake events from the USGS GeoJSON API."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

from ..config import USGS_FALLBACK_FEED_URL, USGS_QUERY_URL
from .common import ensure_run_context, error_result, write_json

logger = logging.getLogger(__name__)

USER_AGENT = "earthquake-adk-pipeline/1.0 (course project)"
USGS_MAX_EVENTS = 20000  # the FDSN API rejects queries that match more than this


class IngestionError(RuntimeError):
    """Raised when data cannot be fetched from USGS."""


def resolve_window(days: int, start_date: str = "", end_date: str = "") -> tuple[date, date]:
    """Turn either explicit dates or 'last N days' into a (start, end) pair.

    end is exclusive and defaults to tomorrow (UTC), so 'today' is included.
    """
    today = datetime.now(timezone.utc).date()
    end = date.fromisoformat(end_date) if end_date else today + timedelta(days=1)
    if start_date:
        start = date.fromisoformat(start_date)
    else:
        if not 1 <= int(days) <= 366:
            raise ValueError("days must be between 1 and 366")
        start = end - timedelta(days=int(days))
    if start >= end:
        raise ValueError(f"start_date {start} must be before end_date {end}")
    return start, end


def _get_json(url: str, params: dict | None, retries: int, timeout: int) -> dict:
    """GET with retry + exponential backoff (1s, 2s, 4s...)."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            logger.warning("USGS request attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise IngestionError(f"USGS API unreachable after {retries} attempts: {last_exc}")


def fetch_geojson(start: date, end: date, min_magnitude: float,
                  retries: int = 3, timeout: int = 60) -> tuple[dict, str]:
    """Fetch a FeatureCollection; returns (geojson, source_url_used).

    Falls back to the 30-day summary feed (filtered client-side) if the query
    API fails, so a flaky endpoint doesn't kill the whole run.
    """
    params = {"format": "geojson", "starttime": start.isoformat(),
              "endtime": end.isoformat(), "minmagnitude": min_magnitude,
              "orderby": "time"}
    try:
        data = _get_json(USGS_QUERY_URL, params, retries, timeout)
        return data, requests.Request("GET", USGS_QUERY_URL, params=params).prepare().url
    except IngestionError as primary_exc:
        logger.warning("Primary query failed; trying summary feed fallback.")
        try:
            data = _get_json(USGS_FALLBACK_FEED_URL, None, retries, timeout)
        except IngestionError:
            raise primary_exc
        lo = datetime.combine(start, datetime.min.time(), timezone.utc).timestamp() * 1000
        hi = datetime.combine(end, datetime.min.time(), timezone.utc).timestamp() * 1000
        data["features"] = [
            f for f in data.get("features", [])
            if lo <= (f["properties"].get("time") or 0) < hi
            and (f["properties"].get("mag") or -99) >= min_magnitude
        ]
        return data, USGS_FALLBACK_FEED_URL


# ---------------------------------------------------------------------------
# ADK tool
# ---------------------------------------------------------------------------
def fetch_earthquake_data(days: int, min_magnitude: float, start_date: str = "",
                          end_date: str = "", tool_context=None) -> dict[str, Any]:
    """Download earthquake events from the USGS API and save the raw GeoJSON.

    Use `days` for relative windows ("last 14 days"). Only pass start_date /
    end_date when the user gave explicit calendar dates.

    Args:
        days: Size of the look-back window in days, ending today (UTC). 1-366.
        min_magnitude: Minimum magnitude to include, e.g. 2.5 or 4.0.
        start_date: Optional explicit start date, YYYY-MM-DD (inclusive).
        end_date: Optional explicit end date, YYYY-MM-DD (exclusive).

    Returns:
        A status dict with the record count, query window, and raw file path.
    """
    try:
        state = tool_context.state
        run_dir = ensure_run_context(state)
        if not -1.0 <= float(min_magnitude) <= 10.0:
            raise ValueError("min_magnitude must be between -1 and 10")
        start, end = resolve_window(days, start_date, end_date)

        geojson, source_url = fetch_geojson(start, end, float(min_magnitude))
        count = len(geojson.get("features", []))
        raw_path = write_json(run_dir / "raw" / "earthquakes.geojson", geojson)
        logger.info("Fetched %d events (%s to %s, M>=%s)", count, start, end, min_magnitude)

        state["query_start"] = start.isoformat()
        state["query_end"] = end.isoformat()
        state["min_magnitude"] = float(min_magnitude)
        state["source_url"] = source_url
        state["raw_path"] = str(raw_path)
        state["raw_count"] = count

        result = {"status": "success", "run_id": state["run_id"],
                  "record_count": count, "start_date": start.isoformat(),
                  "end_date_exclusive": end.isoformat(),
                  "min_magnitude": float(min_magnitude),
                  "used_fallback_feed": source_url == USGS_FALLBACK_FEED_URL,
                  "raw_path": str(raw_path)}
        if count == 0:
            result["warning"] = "USGS returned zero events for this window."
        if count >= USGS_MAX_EVENTS:
            result["warning"] = "Result may be truncated at the USGS 20,000-event limit."
        return result
    except Exception as exc:  # surfaced to the agent, never crashes the run
        return error_result("fetch_earthquake_data", exc)
