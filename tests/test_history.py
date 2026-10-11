"""Offline tests for the national-history GBM: what it may learn from on a date."""

import numpy as np
import pandas as pd

from src import history


def make_history(n_weeks=60, hsas=(("1", "Illinois", ("17031",)),
                                   ("2", "Missouri", ("29510", "29189")),
                                   ("3", "Ohio", ("39049",)))):
    weeks = pd.date_range("2025-01-04", periods=n_weeks, freq="7D")
    rows = []
    for hsa, state, fips in hsas:
        for i, w in enumerate(weeks):
            rows.append({"hsa": hsa, "state": state, "week_end": w, "fips_list": fips,
                         "resp": 2 + np.sin(i / 6) + int(hsa) * 0.1})
    return pd.DataFrame(rows)


def test_examples_need_three_consecutive_prior_weeks():
    hist = make_history(10)
    hist = hist.drop(hist[(hist["hsa"] == "1") & (hist["week_end"] == hist["week_end"][5])].index)
    ex = history.examples(hist)
    # 7 per complete HSA; HSA 1 loses the hole week and the 3 weeks after it
    assert (ex["hsa"] == "3").sum() == 7
    assert (ex["hsa"] == "1").sum() == 7 - 4
    row = ex[ex["hsa"] == "3"].iloc[0]
    src = hist[hist["hsa"] == "3"].reset_index(drop=True)
    assert np.isclose(row["y"], np.log(src["resp"][3]) - np.log(src["resp"][2]))
    assert np.isclose(row["d1"], np.log(src["resp"][2]) - np.log(src["resp"][1]))


def test_pool_only_settled_weeks_and_no_missouri_before_published():
    ex = history.examples(make_history())
    origin = pd.Timestamp("2025-10-04")
    p = history.pool(ex, origin)
    assert p["week_end"].max() <= origin - pd.Timedelta(weeks=history.SETTLE_WEEKS)
    assert "Missouri" not in set(p["state"])
    later = history.pool(ex, history.MISSOURI_FIRST_PUBLISHED + pd.Timedelta(weeks=20))
    assert "Missouri" in set(later["state"])


def test_pool_can_exclude_hsas_containing_study_counties():
    ex = history.examples(make_history())
    p = history.pool(ex, "2026-12-26", exclude_fips=("29189",))
    assert set(p["hsa"]) == {"1", "3"}


def test_refitter_reuses_model_within_window():
    ex = history.examples(make_history(80, hsas=tuple(
        (str(i), "Ohio", (f"39{i:03d}",)) for i in range(40))))
    r = history.Refitter(ex, every=4)
    a = r.model("2026-06-06")
    assert a is not None
    assert r.model("2026-06-27") is a                 # 3 weeks later: reused
    assert r.model("2026-07-04") is not a             # 4 weeks later: refit


def test_every_model_fits_and_predicts_finite_values():
    ex = history.examples(make_history(80, hsas=tuple(
        (str(i), "Ohio", (f"39{i:03d}",)) for i in range(40))))
    feats = pd.DataFrame({"last": [2.0, 3.0], "y_lag2": [1.8, np.nan], "y_lag3": [1.5, 3.1],
                          "as_of": pd.to_datetime(["2026-01-03", "2026-01-10"])})
    for kind in history.MODELS:
        model = history.fit(ex, model=kind)
        pred = history.predict(model, feats)
        assert np.isfinite(pred).all() and (pred > 0).all(), kind
