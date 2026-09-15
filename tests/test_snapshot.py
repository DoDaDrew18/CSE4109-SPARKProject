"""Tests for the snapshot store and point-in-time vintage loader.

These pin down the leakage guarantee the whole evaluation rests on: a vintage
dated D must contain exactly what was published on or before D, never a value
that only existed later. The backfill scenario below uses the shape real NSSP
county data has -- a low first print that is revised upward for weeks.
"""

import pandas as pd
import pytest

from src.snapshot import (
    REQUIRED_COLUMNS,
    SchemaError,
    SnapshotExistsError,
    SnapshotStore,
)

STL_COUNTY = "29189"
STL_CITY = "29510"
COOK = "17031"

WEEK1 = "2026-01-04"
WEEK2 = "2026-01-11"

# Publication dates. Week 1 is first reported on the 14th, then revised twice.
ISSUE_A = "2026-01-14"
ISSUE_B = "2026-01-21"
ISSUE_C = "2026-01-28"


def rows(*triples):
    """Build a snapshot frame from (geo, time_value, value, issue) tuples."""
    return pd.DataFrame(
        [
            {"geo_value": g, "time_value": t, "value": v, "issue": i}
            for g, t, v, i in triples
        ]
    )


@pytest.fixture
def store(tmp_path):
    return SnapshotStore(tmp_path / "snapshots")


@pytest.fixture
def backfilled(store):
    """A store where week 1 is revised upward across three issues.

    issue 2026-01-14  week1 = 4.1                    (first, incomplete print)
    issue 2026-01-21  week1 = 5.6, week2 = 3.9       (week1 revised up)
    issue 2026-01-28  week1 = 6.0, week2 = 5.1       (both revised up)
    """
    store.write_snapshot("nssp", ISSUE_A, rows((STL_COUNTY, WEEK1, 4.1, ISSUE_A)))
    store.write_snapshot(
        "nssp",
        ISSUE_B,
        rows(
            (STL_COUNTY, WEEK1, 5.6, ISSUE_B),
            (STL_COUNTY, WEEK2, 3.9, ISSUE_B),
        ),
    )
    store.write_snapshot(
        "nssp",
        ISSUE_C,
        rows(
            (STL_COUNTY, WEEK1, 6.0, ISSUE_C),
            (STL_COUNTY, WEEK2, 5.1, ISSUE_C),
        ),
    )
    return store


def value_at(vintage, geo, time_value):
    match = vintage[
        (vintage["geo_value"] == geo)
        & (vintage["time_value"] == pd.Timestamp(time_value))
    ]
    assert len(match) == 1, f"expected exactly one row for {geo} {time_value}"
    return float(match["value"].iloc[0])


# -- writing ---------------------------------------------------------------


def test_available_issues_lists_writes_in_order(backfilled):
    assert backfilled.available_issues("nssp") == [
        pd.Timestamp(ISSUE_A),
        pd.Timestamp(ISSUE_B),
        pd.Timestamp(ISSUE_C),
    ]
    assert backfilled.sources() == ["nssp"]


def test_rewriting_identical_snapshot_is_a_noop(store):
    frame = rows((STL_COUNTY, WEEK1, 4.1, ISSUE_A))
    first = store.write_snapshot("nssp", ISSUE_A, frame)
    second = store.write_snapshot("nssp", ISSUE_A, frame)
    assert first == second
    assert store.available_issues("nssp") == [pd.Timestamp(ISSUE_A)]


def test_conflicting_rewrite_is_rejected(store):
    store.write_snapshot("nssp", ISSUE_A, rows((STL_COUNTY, WEEK1, 4.1, ISSUE_A)))
    with pytest.raises(SnapshotExistsError):
        store.write_snapshot(
            "nssp", ISSUE_A, rows((STL_COUNTY, WEEK1, 9.9, ISSUE_A))
        )
    # The stored value is untouched by the failed write.
    assert value_at(store.load_vintage("nssp", ISSUE_A), STL_COUNTY, WEEK1) == 4.1


def test_missing_required_column_is_rejected(store):
    frame = rows((STL_COUNTY, WEEK1, 4.1, ISSUE_A)).drop(columns=["value"])
    with pytest.raises(SchemaError, match="value"):
        store.write_snapshot("nssp", ISSUE_A, frame)


def test_issue_before_reference_week_is_rejected(store):
    """Swapped date columns must fail loudly, not invert the vintage logic."""
    frame = rows((STL_COUNTY, ISSUE_A, 4.1, WEEK1))
    with pytest.raises(SchemaError, match="precedes"):
        store.write_snapshot("nssp", WEEK1, frame)


# -- vintage semantics -----------------------------------------------------


def test_vintage_before_first_issue_is_empty_and_typed(backfilled):
    vintage = backfilled.load_vintage("nssp", "2026-01-07")
    assert vintage.empty
    assert list(vintage.columns) == list(REQUIRED_COLUMNS)
    assert vintage["value"].dtype == "float64"
    assert vintage["time_value"].dtype == "datetime64[ns]"


def test_vintage_hides_weeks_published_later(backfilled):
    """On the 14th, week 2 had not been published, so it must not appear."""
    vintage = backfilled.load_vintage("nssp", ISSUE_A)
    assert set(vintage["time_value"]) == {pd.Timestamp(WEEK1)}
    assert pd.Timestamp(WEEK2) not in set(vintage["time_value"])


def test_latest_issue_wins_for_revised_week(backfilled):
    vintage = backfilled.load_vintage("nssp", ISSUE_C)
    assert value_at(vintage, STL_COUNTY, WEEK1) == 6.0
    assert value_at(vintage, STL_COUNTY, WEEK2) == 5.1
    assert len(vintage) == 2, "one row per county-week, not one per revision"


def test_as_of_boundary_is_inclusive(backfilled):
    """A snapshot issued on as_of itself is visible; the next one is not."""
    on_issue = backfilled.load_vintage("nssp", ISSUE_B)
    assert value_at(on_issue, STL_COUNTY, WEEK1) == 5.6

    day_before = backfilled.load_vintage("nssp", "2026-01-20")
    assert value_at(day_before, STL_COUNTY, WEEK1) == 4.1


def test_backfill_first_print_is_below_settled_value(backfilled):
    """The leak this module prevents: scoring week 1 against 6.0 on the 14th."""
    as_known_then = value_at(backfilled.load_vintage("nssp", ISSUE_A), STL_COUNTY, WEEK1)
    settled = value_at(backfilled.latest_vintage("nssp"), STL_COUNTY, WEEK1)
    assert as_known_then == 4.1
    assert settled == 6.0
    assert as_known_then < settled


def test_revision_to_one_county_leaves_others_alone(store):
    store.write_snapshot(
        "nssp",
        ISSUE_A,
        rows(
            (STL_COUNTY, WEEK1, 4.1, ISSUE_A),
            (STL_CITY, WEEK1, 7.2, ISSUE_A),
            (COOK, WEEK1, 3.3, ISSUE_A),
        ),
    )
    store.write_snapshot("nssp", ISSUE_B, rows((STL_COUNTY, WEEK1, 5.6, ISSUE_B)))

    vintage = store.load_vintage("nssp", ISSUE_B)
    assert value_at(vintage, STL_COUNTY, WEEK1) == 5.6
    assert value_at(vintage, STL_CITY, WEEK1) == 7.2
    assert value_at(vintage, COOK, WEEK1) == 3.3


def test_revision_history_is_ordered_oldest_first(backfilled):
    history = backfilled.revision_history("nssp", STL_COUNTY, WEEK1)
    assert list(history["value"]) == [4.1, 5.6, 6.0]
    assert list(history["issue"]) == [
        pd.Timestamp(ISSUE_A),
        pd.Timestamp(ISSUE_B),
        pd.Timestamp(ISSUE_C),
    ]
