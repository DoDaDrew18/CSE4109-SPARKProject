"""Offline tests for the scoring functions: gap closed and the paired test."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (dedupe_series, gap_closed, gap_closed_total, ladder_tests, metrics,
                          summarize, wilcoxon_paired)


def test_gap_closed_reference_points():
    first, final = np.array([4.0, 4.0, 4.0]), np.array([5.0, 5.0, 5.0])
    pred = np.array([5.0, 4.0, 3.0])            # exact, same as first print, worse
    np.testing.assert_allclose(gap_closed(pred, first, final), [1.0, 0.0, -1.0])
    # Overshooting by half the gap closes half of it.
    assert gap_closed([5.5], [4.0], [5.0])[0] == pytest.approx(0.5)


def test_gap_closed_undefined_when_first_print_was_final():
    out = gap_closed([5.0, 6.0, np.nan], [5.0, 5.0, 4.0], [5.0, 5.0, 5.0])
    assert np.isnan(out).all()
    assert np.isnan(gap_closed_total([5.0, 6.0], [5.0, 5.0], [5.0, 5.0]))


def test_gap_closed_total_is_error_ratio():
    pred, first, final = [5.0, 3.0], [4.0, 4.0], [5.0, 5.0]
    # sum |pred - final| = 2, sum |first - final| = 2
    assert gap_closed_total(pred, first, final) == pytest.approx(0.0)
    assert gap_closed_total([5.0, np.nan], first, final) == pytest.approx(1.0)


def test_metrics_basic():
    frame = pd.DataFrame({"pred": [1.0, 2.0, 3.0], "ed_resp_final": [1.0, 2.0, 4.0],
                          "ed_resp_first": [0.5, 1.5, 3.0]})
    m = metrics(frame)
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert m["r2"] == pytest.approx(1 - 1 / (((np.array([1, 2, 4]) - 7 / 3) ** 2).sum()))
    assert m["gap_closed"] == pytest.approx(1 - 1 / 2)


def test_wilcoxon_detects_consistent_improvement():
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 2, 40)
    better = base * 0.5
    r = wilcoxon_paired(better, base)
    assert r["n"] == 40 and r["median_diff"] < 0 and r["pvalue"] < 0.001


def test_wilcoxon_ties_and_missing_pairs():
    r = wilcoxon_paired([1.0, 2.0, np.nan], [1.0, 2.0, 3.0])
    assert r["n"] == 2 and r["pvalue"] == 1.0 and r["median_diff"] == 0.0
    assert wilcoxon_paired([], [])["n"] == 0


def test_summary_groups_and_ladder_tests():
    rows = []
    for fips, metro in (("29510", "stl"), ("17031", "chi")):
        for w in range(12):
            week = pd.Timestamp("2026-01-10") + pd.Timedelta(weeks=w)
            final = 3.0 + w * 0.1
            for rung, pred in (("0_naive", final - 0.5), ("1_ar", final - 0.1)):
                rows.append({"rung": rung, "fips": fips, "metro": metro, "week_end": week,
                             "pred": pred, "ed_resp_final": final, "ed_resp_first": final - 0.4})
    scored = pd.DataFrame(rows)
    table = summarize(scored)
    assert {"all", "home", "comparison", "metro:stl", "metro:chi"} <= set(table["group"])
    ar = table[(table["group"] == "all") & (table["rung"] == "1_ar")].iloc[0]
    assert ar["gap_closed"] == pytest.approx(0.75)
    tests = ladder_tests(scored)
    assert tests.iloc[0]["rung"] == "1_ar" and tests.iloc[0]["vs"] == "0_naive"
    assert tests.iloc[0]["pvalue"] < 0.001


def test_dedupe_series_collapses_identical_counties_only():
    base = {"rung": "1_ar", "metro": "stl", "week_end": pd.Timestamp("2026-07-04"),
            "pred": 2.0, "ed_resp_final": 2.5, "ed_resp_first": 2.2}
    scored = pd.DataFrame([{**base, "fips": "29510"}, {**base, "fips": "29189"},
                           {**base, "fips": "17031", "metro": "chi"}])
    assert len(dedupe_series(scored)) == 2


def test_official_coverage_counts_dates_and_first_print_lags():
    from src.evaluate import official_coverage
    as_of = pd.to_datetime(["2026-01-10", "2026-01-17", "2026-01-24"])
    rows = []
    for d in as_of:
        for lag in range(3):
            week = d - pd.Timedelta(weeks=lag)
            # 29095 is dark on the first date, then week 2026-01-10 first appears at lag 2.
            value = np.nan if lag == 0 or (d == as_of[0]) else 1.0
            rows.append({"fips": "29095", "metro": "kc", "as_of": d, "week_end": week,
                         "lag": lag, "ed_resp": value})
    cov = official_coverage(pd.DataFrame(rows)).iloc[0]
    assert cov["as_of_dates"] == 3 and cov["dates_with_data"] == 2
    assert str(cov["first_as_of"]) == "2026-01-17"
    assert cov["first_print_lag_max"] == 2
