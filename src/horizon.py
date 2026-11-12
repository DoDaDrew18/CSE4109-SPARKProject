"""Age-aware nowcast: jump as many weeks as the newest published value is old.

The history model (``src.history``) learns a one-week change and is applied
to the newest published value. But that value is often two or more weeks old
(about a fifth of scored county-weeks), and ``nowcast.features`` back-fills a
missing week from the one before it, so on exactly those stale weeks the
model sees two identical values and zero momentum.

This variant fixes both, with nothing tuned on the backtest:

* features come from the three newest *distinct* published weeks, and
* the model is trained to jump ``h`` weeks (h = 1..``H_MAX``, a feature),
  where ``h`` is how old the newest published value is on the date.

Same settings, leak rules and refit schedule as ``history`` (pool: weeks
settled ``SETTLE_WEEKS`` before the origin; no Missouri before it was
published). Scored on the same rows as the ladder; a second, pre-specified
variant averages the four nonlinear history rungs already in
``predictions.csv``.

    python -m src.horizon      # → data/results/metrics_horizon.csv

Result (2026-11-12, both variants fixed before they were scored; 462
county-weeks): neither beats the gradient-boosting rung (skill +18.9%).
Age-aware +17.6% (worse than rung 6, p < 0.001); ensemble +19.3% (vs rung 6
p = 0.066, Holm 0.13: a tie). Kept as a negative result.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import evaluate
from src import history as hist

log = logging.getLogger(__name__)

H_MAX = 4
FEATURES = ("level", "d1", "d2", "h", "season_s1", "season_c1", "season_s2", "season_c2")
NONLINEAR = ("6_gbm_history", "7_xgb_history", "8_rf_history", "9_svm_history")
OUT = Path("data/results/metrics_horizon.csv")


def design(v1, v2, v3, h, target_week) -> pd.DataFrame:
    """Features for log(y_target / v1), where v1 is ``h`` weeks before the target."""
    l1, l2, l3 = hist._log(v1), hist._log(v2), hist._log(v3)
    x = pd.DataFrame({"level": l1, "d1": l1 - l2, "d2": l2 - l3,
                      "h": np.minimum(np.asarray(h, float), H_MAX)})
    x[list(FEATURES[4:])] = hist.season(target_week)
    return x


def examples(history: pd.DataFrame, h_max: int = H_MAX) -> pd.DataFrame:
    """One training row per (HSA, target week, h): three consecutive weeks ending h weeks earlier."""
    h_ = history.sort_values(["hsa", "week_end"])
    g = h_.groupby("hsa")
    out = []
    for h in range(1, h_max + 1):
        v = {k: g["resp"].shift(h + k) for k in (0, 1, 2)}
        span = (h_["week_end"] - g["week_end"].shift(h + 2)).dt.days
        ok = (span == 7 * (h + 2)) & v[0].notna() & v[1].notna() & v[2].notna()
        x = design(v[0][ok], v[1][ok], v[2][ok], np.full(ok.sum(), h), h_.loc[ok, "week_end"])
        x.index = h_.index[ok]
        x["y"] = hist._log(h_.loc[ok, "resp"]) - hist._log(v[0][ok])
        out.append(pd.concat([h_.loc[ok, ["hsa", "state", "week_end", "fips_list"]], x], axis=1))
    return pd.concat(out, ignore_index=True)


class Refitter:
    """Fit at an origin, reuse forward for ``every`` weeks (as ``history.Refitter``)."""

    def __init__(self, rows: pd.DataFrame, every: int = hist.REFIT_WEEKS):
        self.rows, self.every, self._at, self._model = rows, every, None, None

    def model(self, origin):
        origin = pd.Timestamp(origin)
        if self._model is None or not self._at <= origin < self._at + pd.Timedelta(weeks=self.every):
            train = hist.pool(self.rows, origin)
            if len(train) < hist.MIN_POOL:
                return None
            self._model = HistGradientBoostingRegressor(**hist.GBM).fit(
                train[list(FEATURES)].to_numpy(float), train["y"].to_numpy(float))
            self._at = origin
        return self._model


def published(table: pd.DataFrame) -> pd.DataFrame:
    """Per (fips, as_of): age ``h`` of the newest published value and the three newest distinct weeks."""
    from src.nowcast import LABEL_COLUMNS, _pivot
    t = table.drop(columns=[c for c in LABEL_COLUMNS if c in table.columns])
    resp = _pivot(t, "ed_resp", range(1, int(t["lag"].max()) + 1))
    arr, lags = resp.to_numpy(float), np.array(resp.columns, int)
    rows = []
    for (fips, as_of), r in zip(resp.index, arr):
        idx = np.flatnonzero(~np.isnan(r))
        if not len(idx):
            continue
        vals = r[idx]
        rows.append({"fips": fips, "as_of": as_of, "h": int(lags[idx[0]]), "v1": vals[0],
                     "v2": vals[1] if len(vals) > 1 else vals[0],
                     "v3": vals[2] if len(vals) > 2 else (vals[1] if len(vals) > 1 else vals[0])})
    return pd.DataFrame(rows)


def predict(model, pub: pd.DataFrame) -> np.ndarray:
    target = pub["as_of"] + pd.to_timedelta((5 - pub["as_of"].dt.dayofweek) % 7, unit="D")
    x = design(pub["v1"], pub["v2"], pub["v3"], pub["h"], target)
    return np.exp(hist._log(pub["v1"]) + model.predict(x[list(FEATURES)].to_numpy(float)))


def run(table: pd.DataFrame, history: pd.DataFrame, preds: pd.DataFrame):
    """Score both pre-specified variants on the ladder's own rows; returns (scored rows, summary)."""
    from src.nowcast import scoreable
    base = evaluate.dedupe_series(scoreable(preds, official=True))
    keys = base[base["rung"] == "0_naive"][["fips", "as_of"]]
    pub = published(table).merge(keys, on=["fips", "as_of"])
    refit, out = Refitter(examples(history)), []
    for origin, part in pub.groupby("as_of"):
        model = refit.model(origin)
        if model is not None:
            out.append(part.assign(pred=predict(model, part)))
    age = pd.concat(out)
    meta = base[base["rung"] == "0_naive"].drop(columns=["rung", "pred"])
    new = [meta.merge(age[["fips", "as_of", "pred"]], on=["fips", "as_of"]).assign(rung="13_gbm_age")]
    ens = (base[base["rung"].isin(NONLINEAR)].groupby(["fips", "as_of"])["pred"].mean()
           .rename("pred").reset_index())
    new.append(meta.merge(ens, on=["fips", "as_of"]).assign(rung="14_ensemble"))
    scored = pd.concat([base, *new], ignore_index=True)
    n = scored.groupby(["fips", "as_of"])["rung"].transform("nunique")
    scored = scored[n == n.max()]
    summary = evaluate.summarize(scored)
    tests = []
    for rung in ("13_gbm_age", "14_ensemble"):
        a = scored[scored["rung"] == rung].sort_values(["fips", "as_of"])
        for vs in ("0_naive", "6_gbm_history"):
            b = scored[scored["rung"] == vs].sort_values(["fips", "as_of"])
            w = evaluate.wilcoxon_paired((a["pred"] - a["ed_resp_final"]).abs().to_numpy(),
                                         (b["pred"] - b["ed_resp_final"]).abs().to_numpy())
            tests.append({"rung": rung, "vs": vs, **w})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = evaluate.holm(tests["pvalue"]) if "pvalue" in tests else np.nan
    scored = scored.merge(age[["fips", "as_of", "h"]], on=["fips", "as_of"], how="left")
    return scored, (summary, tests)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--table", default="data/county_week.parquet")
    p.add_argument("--history", default=str(hist.DEFAULT_CSV))
    p.add_argument("--preds", default="data/results/predictions.csv")
    a = p.parse_args()
    preds = pd.read_csv(a.preds, dtype={"fips": str}, parse_dates=["as_of", "week_end"])
    scored, (summary, tests) = run(pd.read_parquet(a.table), hist.load(a.history), preds)
    summary.to_csv(OUT, index=False)
    tests.to_csv(OUT.with_name("wilcoxon_horizon.csv"), index=False)
    scored.to_csv(OUT.with_name("predictions_horizon.csv"), index=False)
    show = summary[summary["group"].isin(["all"]) | summary["group"].str.startswith("metro:")]
    print(show.pivot(index="rung", columns="group", values="skill").round(3).to_string())
    print(summary[summary["group"] == "all"][["rung", "n", "mae", "skill"]].round(4).to_string(index=False))
    print(tests.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
