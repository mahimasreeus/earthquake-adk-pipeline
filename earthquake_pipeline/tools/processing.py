"""Processing: flatten, clean, enrich, aggregate, and quality-check the data.

Pure pandas functions (easy to unit test) at the top; thin ADK tool wrappers
at the bottom that read/write files and session state.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import error_result, read_json, require_state, write_json

logger = logging.getLogger(__name__)

# --- Reference data --------------------------------------------------------
US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico",
}
# Other abbreviations USGS uses in place strings.
REGION_ALIASES = {"MX": "Mexico", "B.C.": "Mexico", "P.R.": "Puerto Rico",
                  "U.S. Virgin Islands": "U.S. Virgin Islands"}

# "12 km SSW of " prefix that USGS puts before the nearest named place.
_DISTANCE_PREFIX = re.compile(r"^\s*\d+(\.\d+)?\s*km\s+[NSEW]{1,3}\s+of\s+", re.I)

MAGNITUDE_BINS = [-np.inf, 2.5, 4.0, 5.0, 6.0, np.inf]
MAGNITUDE_LABELS = ["Micro (<2.5)", "Minor (2.5-3.9)", "Light (4.0-4.9)",
                    "Moderate (5.0-5.9)", "Strong (6.0+)"]
DEPTH_BINS = [-np.inf, 70, 300, np.inf]
DEPTH_LABELS = ["Shallow (<70 km)", "Intermediate (70-300 km)", "Deep (>300 km)"]

EVENT_COLUMNS = ["id", "time", "magnitude", "mag_type", "place", "region",
                 "longitude", "latitude", "depth_km", "significance", "tsunami",
                 "felt", "alert", "status", "event_type", "url"]


# --- Pure transformation functions ----------------------------------------
def parse_region(place: Any) -> str:
    """Derive a region from a USGS place string.

    "10 km SSW of Volcano, Hawaii" -> "Hawaii"; "5 km N of Ridgecrest, CA"
    -> "California"; "Fiji region" -> "Fiji"; "south of the Fiji Islands"
    -> "South of the Fiji Islands".
    """
    if not isinstance(place, str) or not place.strip():
        return "Unknown"
    text = place.split(",")[-1].strip() if "," in place else _DISTANCE_PREFIX.sub("", place).strip()
    if text in US_STATES:
        return US_STATES[text]
    if text in REGION_ALIASES:
        return REGION_ALIASES[text]
    text = re.sub(r"\s+region$", "", text, flags=re.I).strip()
    return text[:1].upper() + text[1:] if text else "Unknown"


def flatten_features(geojson: dict) -> pd.DataFrame:
    """GeoJSON FeatureCollection -> one flat row per event (no filtering)."""
    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties") or {}
        coords = ((feat.get("geometry") or {}).get("coordinates") or []) + [None] * 3
        rows.append({
            "id": feat.get("id"), "time": props.get("time"),
            "magnitude": props.get("mag"), "mag_type": props.get("magType"),
            "place": props.get("place"), "longitude": coords[0],
            "latitude": coords[1], "depth_km": coords[2],
            "significance": props.get("sig"), "tsunami": props.get("tsunami"),
            "felt": props.get("felt"), "alert": props.get("alert"),
            "status": props.get("status"), "event_type": props.get("type"),
            "url": props.get("url"),
        })
    df = pd.DataFrame(rows, columns=[c for c in EVENT_COLUMNS if c != "region"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    for col in ("magnitude", "longitude", "latitude", "depth_km", "significance", "felt"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_events(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Filter to valid earthquakes; return (clean_df, step log for lineage)."""
    steps: list[dict[str, Any]] = []

    def log(name: str, before: int, after: int, **extra: Any) -> None:
        steps.append({"step": name, "rows_before": before, "rows_after": after,
                      "rows_removed": before - after, **extra})

    n = len(df)
    is_quake = df["event_type"].fillna("").str.lower().eq("earthquake")
    dropped_types = df.loc[~is_quake, "event_type"].fillna("unknown").value_counts().to_dict()
    df = df[is_quake]
    log("drop_non_earthquake_events", n, len(df), removed_by_type=dropped_types)

    n = len(df)
    df = df.dropna(subset=["magnitude", "longitude", "latitude", "time"])
    log("drop_null_magnitude_or_coordinates", n, len(df))

    n = len(df)
    df = df.drop_duplicates(subset="id", keep="first")
    log("drop_duplicate_ids", n, len(df))

    return df.reset_index(drop=True), steps


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add region, date, hour, magnitude_band, depth_class; tidy dtypes."""
    df = df.copy()
    df["region"] = df["place"].map(parse_region)
    df["date"] = df["time"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["time"].dt.hour
    df["magnitude_band"] = pd.cut(df["magnitude"], MAGNITUDE_BINS,
                                  labels=MAGNITUDE_LABELS, right=False).astype(str)
    # Negative depths (events above sea level, e.g. volcanoes) count as shallow.
    df["depth_class"] = pd.cut(df["depth_km"].fillna(0), DEPTH_BINS,
                               labels=DEPTH_LABELS, right=False).astype(str)
    df["tsunami"] = df["tsunami"].fillna(0).astype(int)
    df["felt"] = df["felt"].fillna(0).astype(int)
    return df[EVENT_COLUMNS + ["date", "hour", "magnitude_band", "depth_class"]]


def agg_daily(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Daily count, mean and max magnitude. Days with no events show 0."""
    daily = (df.groupby("date")["magnitude"]
               .agg(event_count="count", mean_magnitude="mean", max_magnitude="max"))
    if start and end:  # fill gaps so the time series has every day in the window
        all_days = pd.date_range(start, end, inclusive="left").strftime("%Y-%m-%d")
        daily = daily.reindex(all_days)
        daily["event_count"] = daily["event_count"].fillna(0).astype(int)
    daily.index.name = "date"
    return daily.round(2).reset_index()


def agg_regions(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Top regions by event count with magnitude and depth stats."""
    out = (df.groupby("region")
             .agg(event_count=("id", "count"), avg_magnitude=("magnitude", "mean"),
                  max_magnitude=("magnitude", "max"), avg_depth_km=("depth_km", "mean"))
             .sort_values(["event_count", "max_magnitude"], ascending=False)
             .head(top_n))
    return out.round(2).reset_index()


def agg_band_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Event count for every magnitude band x depth class (zeros included)."""
    grid = pd.MultiIndex.from_product([MAGNITUDE_LABELS, DEPTH_LABELS],
                                      names=["magnitude_band", "depth_class"])
    counts = df.groupby(["magnitude_band", "depth_class"]).size().reindex(grid, fill_value=0)
    out = counts.rename("event_count").reset_index()
    # Hide the Micro band unless the query actually included sub-2.5 events.
    if out.loc[out["magnitude_band"] == MAGNITUDE_LABELS[0], "event_count"].sum() == 0:
        out = out[out["magnitude_band"] != MAGNITUDE_LABELS[0]]
    return out.reset_index(drop=True)


def top_events(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["time", "magnitude", "mag_type", "place", "region", "depth_km",
            "tsunami", "alert", "significance", "url"]
    out = df.sort_values(["magnitude", "time"], ascending=[False, False]).head(n)[cols]
    out = out.copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M UTC")
    return out.reset_index(drop=True)


def compute_kpis(df: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers for the report and the reporting agent."""
    if df.empty:
        return {"total_events": 0}
    strongest = df.loc[df["magnitude"].idxmax()]
    busiest = daily.loc[daily["event_count"].idxmax()]
    return {
        "total_events": int(len(df)),
        "strongest_event": {
            "magnitude": float(strongest["magnitude"]),
            "place": strongest["place"] if isinstance(strongest["place"], str) else "Unknown location",
            "time": strongest["time"].strftime("%Y-%m-%d %H:%M UTC"),
            "depth_km": round(float(strongest["depth_km"]), 1),
            "tsunami_flag": bool(strongest["tsunami"]),
            "pager_alert": strongest["alert"] if isinstance(strongest["alert"], str) else None,
        },
        "pct_shallow": round(100 * (df["depth_class"] == DEPTH_LABELS[0]).mean(), 1),
        "tsunami_flagged": int(df["tsunami"].sum()),
        "busiest_day": {"date": busiest["date"], "event_count": int(busiest["event_count"])},
        "avg_magnitude": round(float(df["magnitude"].mean()), 2),
        "events_m5_plus": int((df["magnitude"] >= 5).sum()),
        "distinct_regions": int(df["region"].nunique()),
        "orange_or_red_alerts": int(df["alert"].isin(["orange", "red"]).sum()),
    }


def build_quality_report(raw_count: int, steps: list[dict], df: pd.DataFrame,
                         min_magnitude: float | None) -> dict[str, Any]:
    """Explicit, simple data-quality checks. status: pass | warn | fail."""
    issues: list[str] = []
    mag_min = float(df["magnitude"].min()) if not df.empty else None
    mag_max = float(df["magnitude"].max()) if not df.empty else None

    if df.empty:
        issues.append("FAIL: zero rows after cleaning.")
    else:
        if mag_max > 10 or mag_min < -2:
            issues.append(f"WARN: magnitude range {mag_min}..{mag_max} outside physical bounds.")
        if min_magnitude is not None and mag_min < min_magnitude - 0.05:
            issues.append(f"WARN: min magnitude {mag_min} below requested {min_magnitude}.")
        if not df["latitude"].between(-90, 90).all() or not df["longitude"].between(-180, 180).all():
            issues.append("WARN: coordinates outside valid lat/lon range.")
        if df["id"].duplicated().any():
            issues.append("FAIL: duplicate ids remain after de-duplication.")
    removed = raw_count - len(df)
    if raw_count and removed / raw_count > 0.25:
        issues.append(f"WARN: {removed} of {raw_count} raw rows ({removed / raw_count:.0%}) were dropped.")

    status = "fail" if any(i.startswith("FAIL") for i in issues) else ("warn" if issues else "pass")
    by_step = {s["step"]: s["rows_removed"] for s in steps}
    return {
        "status": status,
        "issues": issues,
        "raw_rows": raw_count,
        "clean_rows": int(len(df)),
        "non_earthquake_removed": by_step.get("drop_non_earthquake_events", 0),
        "nulls_dropped": by_step.get("drop_null_magnitude_or_coordinates", 0),
        "duplicates_removed": by_step.get("drop_duplicate_ids", 0),
        "magnitude_range": [mag_min, mag_max],
        "depth_range_km": ([float(df["depth_km"].min()), float(df["depth_km"].max())]
                           if not df.empty else [None, None]),
        "steps": steps,
    }


# --- ADK tools ------------------------------------------------------------
def _processed_dir(state) -> Path:
    path = Path(state["run_dir"]) / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def clean_and_transform(tool_context=None) -> dict[str, Any]:
    """Flatten the raw GeoJSON, drop invalid/non-earthquake rows, de-duplicate,
    and add derived fields (region, date, hour, magnitude_band, depth_class).

    Reads the raw file saved by the ingestion step. Writes events_clean.csv
    (and .parquet when pyarrow is installed).

    Returns:
        A status dict with row counts before/after each cleaning step.
    """
    try:
        state = tool_context.state
        (raw_path,) = require_state(state, "raw_path")
        raw_df = flatten_features(read_json(raw_path))
        clean, steps = clean_events(raw_df)
        clean = add_derived_fields(clean)

        out = _processed_dir(state)
        clean_csv = out / "events_clean.csv"
        clean.to_csv(clean_csv, index=False)
        written = [str(clean_csv)]
        try:
            clean.to_parquet(out / "events_clean.parquet", index=False)
            written.append(str(out / "events_clean.parquet"))
        except ImportError:
            logger.info("pyarrow not installed; skipping Parquet output.")
        write_json(out / "_cleaning_steps.json", steps)

        state["clean_path"] = str(clean_csv)
        state["raw_rows"] = len(raw_df)
        state["clean_rows"] = len(clean)
        return {"status": "success", "raw_rows": len(raw_df), "clean_rows": len(clean),
                "steps": [{k: s[k] for k in ("step", "rows_before", "rows_after")} for s in steps],
                "files": written}
    except Exception as exc:
        return error_result("clean_and_transform", exc)


def compute_aggregations(tool_context=None) -> dict[str, Any]:
    """Compute the five aggregate outputs from the cleaned events:
    daily time series, top 15 regions, magnitude band x depth class counts,
    top 10 strongest events, and summary KPIs (kpis.json).

    Returns:
        A status dict with the output file names and the headline KPIs.
    """
    try:
        state = tool_context.state
        (clean_path,) = require_state(state, "clean_path")
        df = _read_clean(clean_path)
        out = _processed_dir(state)

        daily = agg_daily(df, state.get("query_start"), state.get("query_end"))
        outputs = {
            "agg_daily.csv": daily,
            "agg_regions.csv": agg_regions(df),
            "agg_band_depth.csv": agg_band_depth(df),
            "top_events.csv": top_events(df),
        }
        for name, frame in outputs.items():
            frame.to_csv(out / name, index=False)
        kpis = compute_kpis(df, daily)
        write_json(out / "kpis.json", kpis)

        state["kpis_path"] = str(out / "kpis.json")
        return {"status": "success", "files": list(outputs) + ["kpis.json"], "kpis": kpis}
    except Exception as exc:
        return error_result("compute_aggregations", exc)


def run_quality_checks(tool_context=None) -> dict[str, Any]:
    """Run data-quality checks on the cleaned data and save quality_report.json.

    Checks: zero rows, magnitude within physical bounds and >= requested
    minimum, valid coordinates, no duplicate ids, share of rows dropped.

    Returns:
        A status dict whose `quality_status` is "pass", "warn", or "fail",
        plus the list of issues. Do not continue the pipeline on "fail".
    """
    try:
        state = tool_context.state
        clean_path, run_dir = require_state(state, "clean_path", "run_dir")
        df = _read_clean(clean_path)
        steps = read_json(Path(run_dir) / "processed" / "_cleaning_steps.json")
        report = build_quality_report(int(state.get("raw_count") or state.get("raw_rows") or 0),
                                      steps, df, state.get("min_magnitude"))
        path = write_json(_processed_dir(state) / "quality_report.json", report)

        state["quality_path"] = str(path)
        state["quality_status"] = report["status"]
        return {"status": "success", "quality_status": report["status"],
                "issues": report["issues"], "raw_rows": report["raw_rows"],
                "clean_rows": report["clean_rows"],
                "magnitude_range": report["magnitude_range"]}
    except Exception as exc:
        return error_result("run_quality_checks", exc)
