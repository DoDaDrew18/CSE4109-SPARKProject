"""Offline tests for the result checks: sanity, Holm, small groups, news shuffle."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import holm
from src.validate_results import (OriginCache, adjust_tests, ceiling_of, check_predictions,
                                  inject_news, inject_power,
                                  placebo_pvalue, shuffle_news, small_groups)


def test_holm_matches_hand_computation():
    # m = 4: sorted 0.01, 0.02, 0.03, 0.5 -> x4, x3, x2, x1 -> 0.04, 0.06, 0.06, 0.5
    np.testing.assert_allclose(holm([0.03, 0.01, 0.5, 0.02]), [0.06, 0.04, 0.5, 0.06])


def test_holm_is_monotone_capped_and_skips_nan():
    out = holm([0.4, np.nan, 0.3])
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [0.6, 0.6])   # 0.3*2=0.6, max(0.4, 0.6)
    assert holm([0.9, 0.8])[0] == 1.0
    assert np.isnan(holm([np.nan])).all()


def test_check_predictions_flags_each_problem():
    preds = pd.DataFrame({"rung": "r", "fips": "1", "as_of": "2025-01-04",
                          "pred": [1.0, np.nan, -0.1, np.inf, 50.0],
                          "ed_resp_final": 2.0})
    issues = check_predictions(preds, ceiling=40.0)
    assert sorted(issues["problem"].str.split().str[0]) == \
        ["above", "missing", "negative", "non-finite"]
    assert (issues.loc[issues["problem"] == "missing", "severity"] == "warning").all()
    assert (issues["severity"] == "error").sum() == 3
    assert check_predictions(preds.iloc[:1], ceiling=40.0).empty


def test_ceiling_uses_both_answer_key_columns():
    a = pd.DataFrame({"ed_resp_final": [3.0, np.nan], "ed_resp_first": [4.0, 1.0]})
    b = pd.DataFrame({"ed_resp_final": [5.0]})
    assert ceiling_of(a, b) == pytest.approx(10.0)


def test_adjust_tests_families_per_group():
    tests = pd.DataFrame({"group": ["a", "a", "b"], "rung": "x", "vs": "y",
                          "pvalue": [0.02, 0.03, 0.02]})
    out = adjust_tests(tests, "src", by="group")
    np.testing.assert_allclose(out["pvalue_holm"], [0.04, 0.04, 0.02])
    assert out["significant"].all() and not out["lost_significance"].any()
    flat = adjust_tests(tests.drop(columns="group"), "src")
    np.testing.assert_allclose(flat["pvalue_holm"], [0.06, 0.06, 0.06])
    assert flat["lost_significance"].all()


def test_small_groups():
    metrics = pd.DataFrame({"group": ["all", "all", "home", "home"], "rung": ["a", "b"] * 2,
                            "n": [400, 400, 9, 9]})
    out = small_groups(metrics, "m")
    assert out.to_dict("records") == [{"source": "m", "group": "home", "n": 9}]


def _news_table():
    weeks = pd.date_range("2025-01-04", periods=6, freq="7D")
    rows = [{"fips": f, "week_end": w, "as_of": w + pd.Timedelta(weeks=lag), "lag": lag,
             "news_n": float(i + 10 * j), "news_concern": float(i % 4), "ed_resp": 1.0}
            for j, f in enumerate(("1", "2")) for i, w in enumerate(weeks) for lag in (0, 1)]
    return pd.DataFrame(rows)


def test_shuffle_news_keeps_each_county_distribution():
    table = _news_table()
    out = shuffle_news(table, np.random.default_rng(1))
    assert list(out.columns) == list(table.columns) and len(out) == len(table)
    for f in ("1", "2"):
        before = table[table["fips"] == f].drop_duplicates("week_end")
        after = out[out["fips"] == f].drop_duplicates("week_end")
        assert sorted(before["news_n"]) == sorted(after["news_n"])
        # count and concern travel together
        pairs = set(zip(before["news_n"], before["news_concern"]))
        assert set(zip(after["news_n"], after["news_concern"])) == pairs
    # every as_of/lag row of a week gets the same shuffled values
    assert (out.groupby(["fips", "week_end"])["news_n"].nunique() == 1).all()
    assert not out["news_n"].equals(table["news_n"])
    assert out["ed_resp"].equals(table["ed_resp"])


def test_origin_cache_replays_first_run_models(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, history):
            pass

        def model(self, origin, *args, **kwargs):
            calls.append(origin)
            return f"fit@{origin.date()}"

    monkeypatch.setattr("src.validate_results.hist_mod.Refitter", Fake)
    cache = OriginCache(None)
    first = [cache.model(pd.Timestamp(d)) for d in ("2025-01-04", "2025-01-11")]
    again = [cache.model(pd.Timestamp(d)) for d in ("2025-01-04", "2025-01-11")]
    assert first == again and len(calls) == 2


def test_placebo_pvalue():
    runs = pd.DataFrame({"news": ["real"] + ["shuffled"] * 4,
                         "gain_mae": [0.03, 0.01, 0.04, -0.02, 0.0]})
    assert placebo_pvalue(runs) == pytest.approx(2 / 5)


def _peak_table():
    """Two counties, 20 weeks; county 1 rises sharply in weeks 5 and 12."""
    weeks = pd.date_range("2025-01-04", periods=20, freq="7D")
    rows = []
    for f in ("1", "2"):
        final = np.ones(20)
        if f == "1":
            final[5:] *= 3.0
            final[12:] *= 2.0
        for i, w in enumerate(weeks):
            for lag in (0, 1):
                rows.append({"fips": f, "week_end": w, "as_of": w + pd.Timedelta(weeks=lag),
                             "lag": lag, "ed_resp": 1.0, "ed_resp_final": final[i],
                             "news_n": 1.0 if i in (2, 3) else np.nan,
                             "news_concern": 2.0 if i in (2, 3) else np.nan})
    return pd.DataFrame(rows)


def test_inject_news_full_precision_lands_on_peaks():
    table = _peak_table()
    out = inject_news(table, 1.0, np.random.default_rng(0))
    assert list(out.columns) == list(table.columns) and len(out) == len(table)
    weekly = out[out["fips"] == "1"].drop_duplicates("week_end").reset_index(drop=True)
    marked = set(weekly.index[weekly["news_n"] > 0])
    assert marked == {5, 12}                       # same count as the real news (2)
    assert (out.groupby(["fips", "week_end"])["news_n"].nunique(dropna=False) == 1).all()
    assert out["ed_resp"].equals(table["ed_resp"])


def test_inject_news_scale_and_zero_precision():
    table = _peak_table()
    out = inject_news(table, 0.0, np.random.default_rng(0), scale=3)
    for f in ("1", "2"):
        weekly = out[out["fips"] == f].drop_duplicates("week_end")
        assert (weekly["news_n"] > 0).sum() == 6
        assert set(weekly.loc[weekly["news_n"] > 0, "news_concern"]) == {2.0}


def test_inject_power_counts_only_significant_gains():
    runs = pd.DataFrame({"scale": 1.0, "precision": [1.0] * 4, "rep": range(4), "n": 50,
                         "gain_mae": [0.1, 0.1, -0.1, 0.1], "pvalue": [0.01, 0.2, 0.01, 0.04]})
    out = inject_power(runs)
    assert out.loc[0, "power"] == pytest.approx(0.5) and out.loc[0, "reps"] == 4
