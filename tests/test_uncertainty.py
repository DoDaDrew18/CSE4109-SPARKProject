"""Calibrated ranges, analog outlook, refit rules, and the Qwen-first LLM default."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import history as hist
from src import uncertainty as unc
from src.app import llm


def _history(n_hsa=60, n_weeks=170, seed=0) -> pd.DataFrame:
    """Seasonal HSA series with persistent momentum, in ``history.load``'s shape."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2022-10-01", periods=n_weeks, freq="7D")
    t = np.arange(n_weeks)
    rows = []
    for h in range(n_hsa):
        phase = rng.uniform(-0.5, 0.5)
        x, d, wave = 0.0, 0.0, np.zeros(n_weeks)
        for i in range(n_weeks):                 # momentum persists, level slowly reverts
            d = 0.7 * d + rng.normal(0, 0.06)
            x = 0.95 * x + d
            wave[i] = x
        level = np.exp(1.0 + 0.6 * np.sin(2 * np.pi * t / 52 + phase) + wave + rng.normal(0, 0.03, n_weeks))
        rows.append(pd.DataFrame({"hsa": h, "state": "Illinois", "week_end": weeks, "resp": level,
                                  "fips_list": [(f"17{h:03d}",)] * n_weeks}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def rows():
    return hist.examples(_history())


@pytest.fixture(scope="module")
def fitted(rows):
    origin = rows["week_end"].max() - pd.Timedelta(weeks=20)
    return origin, unc.Probabilistic(hist.pool(rows, origin))


def test_range_holds_about_80_percent_of_unseen_weeks(rows, fitted):
    origin, model = fitted
    future = rows[(rows["week_end"] > origin) & (rows["week_end"] <= origin + pd.Timedelta(weeks=12))]
    lo, hi = model.log_range(future)
    cover = np.mean((future["y"] >= lo) & (future["y"] <= hi))
    assert 0.70 <= cover <= 0.92
    assert np.all(lo <= hi) and model.n_cal >= unc.MIN_CAL


def test_range_brackets_the_value_scale(fitted):
    origin, model = fitted
    feats = pd.DataFrame({"last": [3.0, 0.5], "y_lag2": [2.5, 0.6], "y_lag3": [2.0, np.nan],
                          "as_of": [origin, origin]})
    lo, hi = model.range(feats)
    assert np.all(lo > 0) and np.all(lo < hi)
    assert lo[0] < 3.0 * 1.6 and hi[0] > 3.0 / 1.6        # around the last value, not off in space


def test_outlook_uses_only_settled_weeks_and_distinct_areas(rows, fitted):
    origin, model = fitted
    feats = pd.DataFrame({"last": [3.0], "y_lag2": [2.4], "y_lag3": [2.0], "as_of": [origin]})
    o = model.outlook(feats)[0]
    assert o.n == unc.K and 0 <= o.p_fall <= 1 - o.p_rise <= 1
    assert len({e["hsa"] for e in o.examples}) == len(o.examples) == 3
    newest = origin - pd.Timedelta(weeks=hist.SETTLE_WEEKS)
    assert all(pd.Timestamp(e["week_end"]) <= newest for e in o.examples)


def test_outlook_separates_rising_from_falling(rows, fitted):
    origin, model = fitted
    up = pd.DataFrame({"last": [3.0], "y_lag2": [2.2], "y_lag3": [1.6], "as_of": [origin]})
    down = pd.DataFrame({"last": [1.6], "y_lag2": [2.2], "y_lag3": [3.0], "as_of": [origin]})
    assert model.outlook(up)[0].median_change > model.outlook(down)[0].median_change


def test_prob_refitter_never_reuses_a_later_fit_for_an_earlier_date(rows):
    refit = unc.ProbRefitter(rows)
    late = rows["week_end"].max() - pd.Timedelta(weeks=10)
    a = refit.model(late)
    assert refit.model(late + pd.Timedelta(weeks=1)) is a           # forward reuse is honest
    b = refit.model(late - pd.Timedelta(weeks=30))
    assert b is not a and b.cal_from < a.cal_from


def test_point_refitter_never_reuses_a_later_fit_for_an_earlier_date(rows):
    refit = hist.Refitter(rows)
    late = rows["week_end"].max() - pd.Timedelta(weeks=10)
    a = refit.model(late)
    assert refit.model(late + pd.Timedelta(weeks=1)) is a
    assert refit.model(late - pd.Timedelta(weeks=30)) is not a


def test_load_metrics_missing_file_is_none(tmp_path):
    assert unc.load_metrics(tmp_path / "nope.csv") is None


class _Tags:
    def __init__(self, names):
        self.names = names

    def json(self):
        return {"models": [{"name": n} for n in self.names]}


def test_backend_prefers_qwen_even_with_an_anthropic_key(monkeypatch):
    monkeypatch.delenv("SPARK_APP_LLM", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: _Tags(["qwen3:8b"]))
    assert llm.backend() == "ollama" and llm.model_name() == "qwen3:8b"


def test_backend_without_qwen_uses_rules_not_claude(monkeypatch):
    monkeypatch.delenv("SPARK_APP_LLM", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: _Tags(["llama3:8b"]))
    assert llm.backend() == "template"
    monkeypatch.setenv("SPARK_APP_LLM", "claude")
    assert llm.backend() == "claude"


def test_outlook_worker_reports_probability_and_named_analogs(rows, fitted, monkeypatch):
    from src.app import workers
    from src.app.region import Region
    origin, model = fitted

    class _Refit:
        def model(self, as_of):
            return model

    weeks = pd.date_range(end=origin - pd.Timedelta(days=7), periods=3, freq="7D")
    pub = pd.DataFrame({"fips": "17031", "week_end": weeks, "resp_pct": [2.0, 2.6, 3.2]})
    monkeypatch.setattr(workers, "_prob_refitter", lambda: _Refit())
    monkeypatch.setattr(workers, "nssp_vintage", lambda fips, as_of: (pub, {"issue": str(origin.date())}))
    monkeypatch.setattr(workers, "_hsa_label", lambda fl: f"HSA of {fl[0]}")
    reg = Region(["17031"], {"17031": "Cook County, IL"}, 5_000_000, (-87.7, 41.8), "Cook County, IL")
    out = workers.run_outlook(workers.Context(reg, origin))
    assert len(out) == 1 and out[0].category == "outlook"
    d = out[0].data
    assert 0 <= d["p_rise"] <= 1 and len(d["analogs"]) == 3 and "HSA of" in out[0].claim
    assert out[0].known_by == str(origin.date())
