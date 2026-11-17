"""Unit tests for the transformation, aggregation, and quality logic."""

from __future__ import annotations

import pytest

from earthquake_pipeline.tools import processing as p
from earthquake_pipeline.tools.common import ensure_run_context, write_json


@pytest.mark.parametrize("place, expected", [
    ("45 km NW of Anchorage, Alaska", "Alaska"),
    ("5 km N of Ridgecrest, CA", "California"),
    ("30 km E of Ensenada, B.C., MX", "Mexico"),
    ("Fiji region", "Fiji"),
    ("south of the Fiji Islands", "South of the Fiji Islands"),
    ("12 km SSW of Somewhere", "Somewhere"),
    (None, "Unknown"),
    ("", "Unknown"),
])
def test_parse_region(place, expected):
    assert p.parse_region(place) == expected


@pytest.fixture
def clean(geojson):
    df, steps = p.clean_events(p.flatten_features(geojson))
    return p.add_derived_fields(df), steps


def test_flatten_keeps_every_feature(geojson):
    df = p.flatten_features(geojson)
    assert len(df) == 13
    assert str(df["time"].dt.tz) == "UTC"


def test_cleaning_steps_remove_expected_rows(clean):
    df, steps = clean
    removed = {s["step"]: s["rows_removed"] for s in steps}
    assert removed == {"drop_non_earthquake_events": 2,
                       "drop_null_magnitude_or_coordinates": 2,
                       "drop_duplicate_ids": 1}
    assert len(df) == 8
    assert df["id"].is_unique
    assert set(df["event_type"]) == {"earthquake"}


def test_derived_fields(clean):
    df, _ = clean
    row = df.set_index("id")
    assert row.loc["us1", "magnitude_band"] == "Strong (6.0+)"
    assert row.loc["us1", "depth_class"] == "Deep (>300 km)"
    assert row.loc["us4", "depth_class"] == "Intermediate (70-300 km)"
    assert row.loc["hv1", "depth_class"] == "Shallow (<70 km)"  # negative depth
    assert row.loc["ci1", "region"] == "California"
    assert row.loc["ci1", "magnitude_band"] == "Minor (2.5-3.9)"
    assert row.loc["ak1", "date"] == "2026-11-01" and row.loc["ak1", "hour"] == 3


def test_agg_daily_fills_missing_days(clean):
    df, _ = clean
    daily = p.agg_daily(df, "2026-10-31", "2026-11-04")
    assert list(daily["date"]) == ["2026-10-31", "2026-11-01", "2026-11-02", "2026-11-03"]
    assert list(daily["event_count"]) == [0, 3, 4, 1]
    assert daily.loc[1, "max_magnitude"] == 6.4


def test_agg_regions_sorted_by_count(clean):
    df, _ = clean
    regions = p.agg_regions(df)
    assert regions.iloc[0]["region"] == "Alaska"
    assert regions.iloc[0]["event_count"] == 2
    assert regions["event_count"].is_monotonic_decreasing


def test_agg_band_depth_is_complete_grid(clean):
    df, _ = clean
    grid = p.agg_band_depth(df)
    assert grid["event_count"].sum() == len(df)
    assert len(grid) == 4 * 3  # Micro band hidden when empty
    cell = grid[(grid.magnitude_band == "Strong (6.0+)") & (grid.depth_class == "Deep (>300 km)")]
    assert int(cell["event_count"].iloc[0]) == 1


def test_top_events_and_kpis(clean):
    df, _ = clean
    top = p.top_events(df, n=3)
    assert list(top["magnitude"]) == [6.4, 5.2, 4.8]
    kpis = p.compute_kpis(df, p.agg_daily(df))
    assert kpis["total_events"] == 8
    assert kpis["strongest_event"]["magnitude"] == 6.4
    assert kpis["tsunami_flagged"] == 1
    assert kpis["strongest_event"]["tsunami_flag"] is True  # us1 in the fixture
    assert kpis["busiest_day"] == {"date": "2026-11-02", "event_count": 4}
    assert kpis["pct_shallow"] == 50.0  # 4 of 8 (incl. a negative depth)


def test_quality_report_pass_and_fail(clean):
    df, steps = clean
    ok = p.build_quality_report(13, steps, df, min_magnitude=2.5)
    assert ok["status"] == "warn"  # 5/13 rows dropped (>25%) is flagged
    assert ok["duplicates_removed"] == 1 and ok["nulls_dropped"] == 2
    assert ok["magnitude_range"] == [2.7, 6.4]

    empty = p.build_quality_report(0, [], df.iloc[0:0], min_magnitude=2.5)
    assert empty["status"] == "fail"


def test_processing_tools_chain(geojson, tool_context):
    """The three ADK tool functions run in order using only session state."""
    state = tool_context.state
    run_dir = ensure_run_context(state)
    state["raw_path"] = str(write_json(run_dir / "raw" / "earthquakes.geojson", geojson))
    state["raw_count"] = 13
    state["min_magnitude"] = 2.5

    assert p.clean_and_transform(tool_context)["clean_rows"] == 8
    agg = p.compute_aggregations(tool_context)
    assert agg["status"] == "success" and agg["kpis"]["total_events"] == 8
    qc = p.run_quality_checks(tool_context)
    assert qc["status"] == "success" and qc["quality_status"] in ("pass", "warn")
    assert (run_dir / "processed" / "quality_report.json").exists()


def test_tool_reports_error_instead_of_raising(tool_context):
    result = p.clean_and_transform(tool_context)  # no raw_path in state yet
    assert result["status"] == "error"
    assert "raw_path" in result["error"]
