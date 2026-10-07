"""Offline tests for the county_week table (and the NSSP backfill it reads).

The synthetic store below has the shape real NSSP county data has: a week is
first printed one week after it ends, low, and revised upward for a few
weeks. Every test pins down one part of the timing contract in
``src/align.py``: the target week is unpublished on its own as_of date,
history values are the ones known then, and labels are only the answer key.
"""

import sys
import types
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src import align
from src.align import LeakError, assert_no_leak, build_county_week, nssp_vintage
from src.backfill_nssp import epiweek_of, epiweeks, latest_completed_epiweek, vintage_from_history
from src.snapshot import SnapshotStore

COOK, STL = "17031", "29510"
FIRST_SUNDAY = pd.Timestamp("2026-01-04")          # epiweek 202601
SIGNAL_BASE = {"nssp_covid": 1.0, "nssp_influenza": 2.0, "nssp_rsv": 0.5}


def final_value(source, fips, week):
    """Settled value of week ``week`` (0-based): a smooth rise per county."""
    return SIGNAL_BASE[source] + 0.1 * week + (0.5 if fips == STL else 0.0)


def printed_value(source, fips, week, lag):
    """Value printed ``lag`` weeks after the week ended: 70% at lag 1, final from lag 4."""
    return round(final_value(source, fips, week) * min(1.0, 0.6 + 0.1 * lag), 6)


def make_store(root, n_weeks=20, fips=(COOK, STL)):
    """Write one vintage per Saturday; vintage j holds weeks 0..j-1.

    The issue for vintage j is the Saturday ending week j, so on that date
    week j itself (the nowcast target) is not yet published.
    """
    store = SnapshotStore(root)
    for j in range(1, n_weeks + 1):
        issue = FIRST_SUNDAY + timedelta(weeks=j, days=6)
        for source in SIGNAL_BASE:
            rows = [{"geo_value": f, "time_value": FIRST_SUNDAY + timedelta(weeks=w),
                     "value": printed_value(source, f, w, j - w), "issue": issue}
                    for f in fips for w in range(j)]
            store.write_snapshot(source, issue, pd.DataFrame(rows))
    return store


def saturday(week):
    return FIRST_SUNDAY + timedelta(weeks=week, days=6)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return make_store(tmp_path_factory.mktemp("store") / "snapshots")


@pytest.fixture(scope="module")
def table(store):
    return build_county_week([saturday(w) for w in range(1, 21)], fips=(COOK, STL),
                             store_root=store.root, history=6, feature_modules={})


def test_target_week_is_unpublished_and_lag1_is_first_print(table):
    as_of = saturday(10)
    rows = table[(table["as_of"] == as_of) & (table["fips"] == COOK)].set_index("lag")
    assert rows.loc[0, "week_end"] == as_of
    assert np.isnan(rows.loc[0, "ed_resp"])                      # nowcast target
    first = sum(printed_value(s, COOK, 9, 1) for s in SIGNAL_BASE)
    assert rows.loc[1, "ed_resp"] == pytest.approx(first)
    assert rows.loc[1, "ed_resp_first"] == pytest.approx(first)


def test_history_values_are_as_known_on_as_of_not_final(table):
    as_of = saturday(10)
    row = table[(table["as_of"] == as_of) & (table["fips"] == STL) & (table["lag"] == 2)].iloc[0]
    known = sum(printed_value(s, STL, 8, 2) for s in SIGNAL_BASE)
    final = sum(printed_value(s, STL, 8, 12) for s in SIGNAL_BASE)
    assert row["ed_resp"] == pytest.approx(known)
    assert row["ed_resp_final"] == pytest.approx(final)
    assert row["ed_resp"] < row["ed_resp_final"]


def test_columns_and_one_row_per_key(table):
    for col in ("fips", "metro", "week_end", "as_of", "ed_covid", "ed_flu", "ed_rsv",
                "ed_resp", "ed_resp_final", "ed_resp_first"):
        assert col in table
    assert not table.duplicated(["fips", "week_end", "as_of"]).any()
    assert set(table["lag"]) == set(range(7))
    assert (table.loc[table["fips"] == STL, "metro"] == "stl").all()


def test_resp_is_missing_when_any_signal_is_missing():
    frame = pd.DataFrame({"ed_covid": [1.0, 1.0], "ed_flu": [2.0, np.nan], "ed_rsv": [0.5, 0.5]})
    assert align._sum_resp(frame).tolist()[0] == pytest.approx(3.5)
    assert np.isnan(align._sum_resp(frame).tolist()[1])


def test_midweek_as_of_cannot_see_its_own_week(store):
    wednesday = saturday(10) - timedelta(days=3)
    vint = nssp_vintage(store, wednesday, (COOK,))
    assert vint["week_end"].max() == saturday(8)                 # vintage of the 9th Saturday
    t = build_county_week([wednesday], fips=(COOK,), store_root=store.root, history=3,
                          feature_modules={})
    assert t.loc[t["lag"] == 0, "week_end"].iloc[0] == saturday(10)
    assert t.loc[t["lag"] == 0, "ed_resp"].isna().all()


def _fake_module(monkeypatch, name, frame_fn):
    module = types.ModuleType(name)
    module.weekly_features = frame_fn
    monkeypatch.setitem(sys.modules, name, module)


def test_feature_loader_rows_after_as_of_raise(store, monkeypatch):
    def leaky(as_of, fips=None):
        as_of = pd.Timestamp(as_of)
        return pd.DataFrame({"fips": [COOK], "week_end": [as_of + timedelta(days=7)],
                             "temp_mean_f": [30.0]})
    _fake_module(monkeypatch, "fake_noaa", leaky)
    with pytest.raises(LeakError):
        build_county_week([saturday(5)], fips=(COOK,), store_root=store.root,
                          feature_modules={"fake_noaa": ("temp_mean_f",)})


def test_feature_loader_merges_and_missing_module_is_skipped(store, monkeypatch, caplog):
    def ok(as_of, fips=None):
        weeks = [pd.Timestamp(as_of) - timedelta(weeks=k) for k in range(3)]
        return pd.DataFrame({"fips": [COOK] * 3, "week_end": weeks,
                             "temp_mean_f": [30.0, 31.0, 32.0]})
    _fake_module(monkeypatch, "fake_noaa", ok)
    t = build_county_week([saturday(5)], fips=(COOK,), store_root=store.root, history=4,
                          feature_modules={"fake_noaa": ("temp_mean_f",),
                                           "no_such_pkg.features": ("news_n",)})
    assert "news_n" not in t
    assert t.set_index("lag").loc[0, "temp_mean_f"] == 30.0
    assert t.set_index("lag").loc[4, "temp_mean_f"] != t.set_index("lag").loc[4, "temp_mean_f"]
    assert "no_such_pkg.features" in caplog.text


def test_assert_no_leak_catches_future_values(table):
    bad = table.copy()
    bad.loc[bad.index[0], "week_end"] = bad.loc[bad.index[0], "as_of"] + timedelta(days=7)
    bad.loc[bad.index[0], "ed_covid"] = 1.0
    with pytest.raises(LeakError):
        assert_no_leak(bad)


# -- backfill helpers (feed the store this table reads) ------------------------

def test_epiweek_helpers_cross_year_end():
    assert epiweek_of("2026-01-04") == 202601
    assert epiweek_of("2025-12-27") == 202552
    assert epiweek_of("2024-12-29") == 202501
    assert epiweeks(202451, 202502) == [202451, 202452, 202501, 202502]
    from datetime import date
    assert latest_completed_epiweek(date(2026, 10, 6)) == 202639     # Tuesday
    assert latest_completed_epiweek(date(2026, 10, 3)) == 202638     # that Saturday


def test_vintage_from_history_matches_as_of_rule():
    """Newest issue at or before as_of wins; weeks first issued later are absent."""
    history = pd.DataFrame({
        "geo_value": ["17031"] * 4,
        "time_value": pd.to_datetime(["2026-01-04", "2026-01-04", "2026-01-04", "2026-01-11"]),
        "value": [1.0, 1.5, 1.7, 2.0],
        "delphi_issue": [202602, 202603, 202605, 202603],
    })
    v = vintage_from_history(history, 202604).set_index("time_value")
    assert v.loc[pd.Timestamp("2026-01-04"), "value"] == 1.5
    assert len(vintage_from_history(history, 202602)) == 1
