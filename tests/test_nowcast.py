"""Offline tests for the rolling-origin nowcast ladder.

The guarantees under test: a model fitted at origin D learns only from
values known on D (never the final values, never a later vintage), features
never read the answer key, and rungs without data are skipped, not faked.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src import nowcast
from src.align import build_county_week
from src.nowcast import LADDER, available_rungs, features, rolling_origin, training_set
from tests.test_align import COOK, STL, make_store, saturday

N_WEEKS = 30


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    store = make_store(tmp_path_factory.mktemp("store") / "snapshots", n_weeks=N_WEEKS)
    return build_county_week([saturday(w) for w in range(1, N_WEEKS + 1)],
                             fips=(COOK, STL), store_root=store.root, history=20,
                             feature_modules={})


def test_training_labels_are_values_known_at_origin(table):
    feats = features(table)
    origin = saturday(20)
    train = training_set(table, feats, origin, label_lag=2, window=10)
    assert train["as_of"].max() <= origin - timedelta(weeks=2)
    assert train["as_of"].min() >= origin - timedelta(weeks=10)
    # Week 18 is 2 weeks old at the origin: its label is the lag-2 print,
    # which is below the final value the scorer will use.
    row = train[(train["fips"] == COOK) & (train["as_of"] == saturday(18))].iloc[0]
    known = table[(table["as_of"] == origin) & (table["week_end"] == saturday(18))
                  & (table["fips"] == COOK)].iloc[0]
    assert row["label"] == pytest.approx(known["ed_resp"])
    assert row["label"] < known["ed_resp_final"]


def test_rolling_origin_never_uses_the_future(table):
    """Predictions at origin D are identical whether or not later vintages exist."""
    origin = saturday(22)
    full = rolling_origin(table, label_lag=2, window=12, min_train=5)
    cut = rolling_origin(table[table["as_of"] <= origin], label_lag=2, window=12, min_train=5)
    a = full[full["as_of"] == origin].set_index(["rung", "fips"])["pred"]
    b = cut[cut["as_of"] == origin].set_index(["rung", "fips"])["pred"]
    assert len(a) == len(b) > 0
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_labels_never_reach_the_features(table):
    poisoned = table.assign(ed_resp_final=table["ed_resp_final"] * 1000,
                            ed_resp_first=-1.0)
    assert not {"ed_resp_final", "ed_resp_first"} & set(features(poisoned).columns)
    a = rolling_origin(table, label_lag=2, window=12, min_train=5)["pred"]
    b = rolling_origin(poisoned, label_lag=2, window=12, min_train=5)["pred"]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_naive_is_last_published_value(table):
    preds = rolling_origin(table, rungs=[LADDER[0]])
    origin = saturday(15)
    row = preds[(preds["as_of"] == origin) & (preds["fips"] == STL)].iloc[0]
    lag1 = table[(table["as_of"] == origin) & (table["fips"] == STL) & (table["lag"] == 1)]
    assert row["pred"] == pytest.approx(lag1["ed_resp"].iloc[0])


def test_ar_beats_naive_on_a_steady_upward_trend(table):
    """The synthetic series rises and first prints are 30% low; AR should learn both."""
    preds = rolling_origin(table, label_lag=4, window=15, min_train=10)
    late = preds[preds["as_of"] >= saturday(20)]
    err = (late["pred"] - late["ed_resp_final"]).abs().groupby(late["rung"]).mean()
    assert err["1_ar_season"] < err["0_naive"]


def test_rungs_without_data_are_skipped(table):
    feats = features(table)
    assert [r.name for r in available_rungs(feats)] == ["0_naive", "1_ar_season", "n0_mean",
                                                           "n_season"]
    with_weather = table.assign(temp_mean_f=40.0, dewpoint_f=30.0, precip_in=0.1)
    names = [r.name for r in available_rungs(features(with_weather))]
    assert names == ["0_naive", "1_ar_season", "2_weather", "n0_mean", "n_season",
                     "n1_weather"]


def test_scoreable_drops_unsettled_weeks_and_unpaired_rows(table):
    preds = rolling_origin(table, label_lag=2, window=12, min_train=5)
    scored = nowcast.scoreable(preds, settle_weeks=4)
    assert scored["week_end"].max() <= preds["as_of"].max() - timedelta(weeks=4)
    counts = scored.groupby(["fips", "as_of"])["rung"].nunique()
    assert (counts == scored["rung"].nunique()).all()


def test_counties_without_official_data_get_only_no_official_rungs(table):
    """Like Missouri before June 2026: no ED value on the date, so no ladder rung."""
    dark = table.copy()
    mask = (dark["fips"] == STL) & (dark["as_of"] < saturday(18))
    dark.loc[mask, ["ed_covid", "ed_flu", "ed_rsv", "ed_resp"]] = np.nan
    dark = dark.assign(temp_mean_f=40.0 + dark["week_end"].dt.isocalendar().week.astype(float))
    preds = rolling_origin(dark, label_lag=2, window=12, min_train=5)
    gap = preds[(preds["fips"] == STL) & (preds["as_of"] < saturday(18))]
    assert len(gap) and set(gap["rung"]) <= {"n0_mean", "n_season", "n1_weather"}
    assert not gap["has_official"].any()
    scored = nowcast.scoreable(preds, settle_weeks=4, official=False)
    assert set(scored["rung"]) == {"n0_mean", "n_season", "n1_weather"}
    assert (scored["fips"] == STL).all()


def test_season_harmonics_follow_the_target_week(table):
    feats = features(table)
    row = feats[feats["as_of"] == saturday(10)].iloc[0]
    doy = saturday(10).dayofyear
    assert row["season_s1"] == pytest.approx(np.sin(2 * np.pi * doy / 365.25))
    assert row["season_c2"] == pytest.approx(np.cos(4 * np.pi * doy / 365.25))


def test_leave_one_metro_out_never_sees_the_held_out_metro(table):
    """Scrambling Chicago's ED values cannot move Chicago's held-out predictions."""
    scrambled = table.copy()
    chi = scrambled["fips"] == COOK
    scrambled.loc[chi, "ed_resp"] = scrambled.loc[chi, "ed_resp"] * 7.0
    a = nowcast.leave_one_metro_out(table, holdout=("chi",), label_lag=2, window=12, min_train=5)
    b = nowcast.leave_one_metro_out(scrambled, holdout=("chi",), label_lag=2, window=12,
                                    min_train=5)
    assert len(a) and set(a["fips"]) == {COOK} and set(a["held_out"]) == {"chi"}
    np.testing.assert_allclose(a["pred"].to_numpy(), b["pred"].to_numpy())
