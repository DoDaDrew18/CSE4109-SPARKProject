"""Offline tests for the NWSS wastewater ingester. No network.

The fixtures mirror the real export: county names (not FIPS) joined with
", ", one site serving both St. Louis County and St. Louis city, the v3
export's thousands separators, and two CDC vintages where the second
revises a value and adds a late site -- the revision story we measured.
"""

import io

import pandas as pd
import pytest

from src.fips import FipsLookup
from src.ingest_nwss import (
    FEATURE_COLUMNS,
    WVAL_CAP,
    ingest_frame,
    parse_export,
    site_fips,
    to_snapshots,
    weekly_features,
)
from src.snapshot import SnapshotStore

REFERENCE = pd.DataFrame({
    "state": ["MO", "MO", "MO", "KS", "IL"],
    "fips": ["29189", "29510", "29095", "20209", "17031"],
    "name": ["St. Louis County", "St. Louis city", "Jackson County",
             "Wyandotte County", "Cook County"],
})

HEADER = ("State/Territory,Counties_Served,Site,Population_Served,Source,Site_WVAL,"
          "Site_WVAL_Category,Date_Included_In_WVAL,Week_End,Pathogen_Target,date_updated")


def csv(updated, *rows):
    """Export text from (counties, site, pop, wval, week_end, pathogen) tuples."""
    lines = [HEADER]
    for counties, site, pop, wval, week, pathogen in rows:
        lines.append(f'Missouri,"{counties}",{site},"{pop}",State_Territory,"{wval}",'
                     f"Low,2022-01-01,{week},{pathogen},{updated}")
    return "\n".join(lines) + "\n"


WEEK0, WEEK1, WEEK2 = "2025-12-27", "2026-01-03", "2026-01-10"   # Saturdays
FIRST = csv("2026-01-09 11:00",
            ("Saint Louis", "ID:1", "300,000", "6.0", WEEK0, "SARS-CoV-2"),
            ("Saint Louis", "ID:1", "300,000", "4.0", WEEK1, "SARS-CoV-2"),
            ("Saint Louis, Saint Louis City", "ID:2", "100,000", "8.0", WEEK1, "SARS-CoV-2"),
            ("Saint Louis", "ID:1", "300,000", "2.0", WEEK1, "RSV"))
SECOND = csv("2026-01-16 11:00",
             ("Saint Louis", "ID:1", "300,000", "5.0", WEEK1, "SARS-CoV-2"),   # revised
             ("Saint Louis, Saint Louis City", "ID:2", "100,000", "8.0", WEEK1, "SARS-CoV-2"),
             ("Saint Louis", "ID:3", "50,000", "1,000.5", WEEK1, "SARS-CoV-2"),  # late site, huge
             ("Saint Louis", "ID:1", "300,000", "3.0", WEEK2, "SARS-CoV-2"),
             ("Saint Louis", "ID:1", "300,000", "2.0", WEEK1, "Influenza A virus"))


@pytest.fixture
def lookup():
    return FipsLookup(REFERENCE)


@pytest.fixture
def root(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    for text in (FIRST, SECOND):
        ingest_frame(store, parse_export(io.StringIO(text)))
    return tmp_path / "snapshots"


def test_parse_strips_thousands_separators():
    frame = parse_export(io.StringIO(SECOND))
    assert frame.loc[frame["site"] == "ID:3", "site_wval"].item() == 1000.5
    assert frame["population_served"].iloc[0] == 300000
    assert (frame["week_end"].dt.dayofweek == 5).all()     # Saturdays


def test_snapshots_are_per_pathogen_keyed_by_site_and_week_start():
    snaps = to_snapshots(parse_export(io.StringIO(FIRST)))
    assert set(snaps) == {"nwss_covid", "nwss_rsv"}
    covid = snaps["nwss_covid"]
    assert set(covid["geo_value"]) == {"ID:1", "ID:2"}
    week1 = covid[covid["week_end"] == WEEK1]
    assert (week1["time_value"] == pd.Timestamp("2025-12-28")).all()   # Sunday
    assert (covid["issue"] == pd.Timestamp("2026-01-09")).all()        # date_updated


def test_mixed_vintage_is_refused():
    mixed = FIRST + SECOND.split("\n", 1)[1]
    with pytest.raises(ValueError, match="date_updated"):
        to_snapshots(parse_export(io.StringIO(mixed)))


def test_site_fips_explodes_multi_county_and_crosses_state_lines(lookup):
    assert site_fips("Missouri", "Saint Louis, Saint Louis City", lookup) == ("29189", "29510")
    assert site_fips("Missouri", "Jackson, Wyandotte", lookup) == ("29095", "20209")
    missing = []
    assert site_fips("Missouri", "Atlantis", lookup, missing) == ()
    assert missing == [("Missouri", "Atlantis")]
    assert site_fips("Missouri", None, lookup) == ()


def test_population_weighted_mean_and_shared_site(root, lookup):
    feats = weekly_features("2026-01-09", root=root, lookup=lookup)
    assert list(feats.columns) == FEATURE_COLUMNS
    county = feats[(feats["fips"] == "29189") & (feats["week_end"] == WEEK1)].iloc[0]
    # (4*300k + 8*100k) / 400k: the shared site counts with its full population.
    assert county["ww_covid"] == pytest.approx(5.0)
    assert county["ww_rsv"] == pytest.approx(2.0)
    assert pd.isna(county["ww_flu"])
    assert county["ww_n_sites"] == 2
    city = feats[feats["fips"] == "29510"].iloc[0]
    assert city["ww_covid"] == pytest.approx(8.0) and city["ww_n_sites"] == 1


def test_as_of_hides_later_vintages(root, lookup):
    """On Jan 9 week 2, the revision and the late site do not exist yet."""
    early = weekly_features("2026-01-09", "29189", root=root, lookup=lookup)
    assert list(early["week_end"]) == [pd.Timestamp(WEEK0), pd.Timestamp(WEEK1)]
    assert weekly_features("2026-01-08", root=root, lookup=lookup).empty

    late = weekly_features("2026-01-16", "29189", root=root, lookup=lookup)
    assert list(late["week_end"]) == [pd.Timestamp(w) for w in (WEEK0, WEEK1, WEEK2)]
    week1 = late.iloc[1]
    # 5.0 revised value, 8.0 shared site, late site capped at WVAL_CAP.
    expected = (5 * 300_000 + 8 * 100_000 + WVAL_CAP * 50_000) / 450_000
    assert week1["ww_covid"] == pytest.approx(expected)
    assert week1["ww_n_sites"] == 3
    assert week1["ww_flu"] == pytest.approx(2.0)


def test_proxy_fills_uncovered_weeks_with_first_prints(root, lookup):
    """Before the first vintage (Jan 9) strict is empty; the proxy shows week 0
    once its lag has passed (Dec 27 + 6 days), and never touches weeks an
    honest vintage could hold -- so neither the late site ID:3 nor the flu
    row first posted on Jan 16 leaks into week 1 on Jan 9."""
    assert weekly_features("2026-01-05", root=root, lookup=lookup).empty
    assert weekly_features("2026-01-01", "29189", root=root, lookup=lookup,
                           allow_proxy=True).empty           # lag not yet elapsed
    proxy = weekly_features("2026-01-05", "29189", root=root, lookup=lookup,
                            allow_proxy=True)
    assert list(proxy["week_end"]) == [pd.Timestamp(WEEK0)]
    assert proxy["ww_covid"].iloc[0] == pytest.approx(6.0)

    strict = weekly_features("2026-01-09", "29189", root=root, lookup=lookup)
    filled = weekly_features("2026-01-09", "29189", root=root, lookup=lookup,
                             allow_proxy=True)
    pd.testing.assert_frame_equal(strict, filled)


def test_rerun_of_same_vintage_is_a_noop(root):
    store = SnapshotStore(root)
    ingest_frame(store, parse_export(io.StringIO(FIRST)))
    assert len(store.available_issues("nwss_covid")) == 2


def test_fips_filter_and_empty_result_is_typed(root, lookup):
    feats = weekly_features("2026-01-16", ["17031"], root=root, lookup=lookup)
    assert feats.empty and list(feats.columns) == FEATURE_COLUMNS
