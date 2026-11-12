"""Can the nowcast beat +19%? A pre-registered dev/test search.

Protocol (fixed 2026-11-12 before any variant was scored)
---------------------------------------------------------
* **Dev** = scored county-weeks with origin before 2025-07-01 (2024-25
  season, 212 rows). **Test** = origin 2025-07-01 onward (250 rows). Same
  rows, deduplication and final values as the ladder (``nowcast.scoreable``).
* Each candidate changes one thing in rung ``6_gbm_history`` and is scored on
  **dev only**. Every single change that lowers dev MAE is then combined
  into one model. The final pick is whichever of {best single change, the
  combination} has the lowest dev MAE.
* The final pick is scored **once** on test against rung 6 (paired Wilcoxon).
  Every candidate's dev number is reported, including the losers.

Candidates
----------
``revision``     backfill correction: the newest values the model sees are
                 first prints that are later revised (median +2%, 29% by more
                 than 5%). At origin D, for each lag k, the correction is the
                 median log(value as published on D / value as first published
                 at lag k) over study-county weeks that had settled by D
                 (9+ weeks old on D, within the 52-week vintage). Only the
                 vintage of D and earlier is read. Applied to every input lag.
``revision_phase`` the same, estimated separately for rising and falling
                 series (sign of the newest weekly change), since backfill is
                 larger on the way up.
``l1``           absolute-error loss: the scoring metric is MAE, which the
                 median minimises, while rung 6 fits the mean (squared error).
``trend``        Flusion-style trend features: the 4- and 6-week local linear
                 slope and a 4-week mean level, besides the last two changes.

    python -m src.improve          # stage 1, dev only → data/results/metrics_improve.csv
    python -m src.improve --final  # stage 2: combine, lock on dev, test once

Result (run 2026-11-12)
-----------------------
Dev skill vs carry-forward: rung 6 +17.2%; revision +18.3%; revision_phase
+18.2%; l1 +20.6%; trend +9.8%; revision+l1 +21.3% (locked pick).
Test (2025-26, 250 county-weeks, scored once): rung 6 +21.1%, revision+l1
+24.4%. All 462: +18.9% -> +22.7% (includes the dev rows used to choose).
Not significant on test: paired Wilcoxon p = 0.70; week-block bootstrap of
the MAE difference -0.019 [-0.051, +0.011]. The new model wins on 46% of
rows; the gain is at peak weeks (top 20% of values: -0.084 per row, others
-0.003), the onsets rung 6 lagged.
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

SPLIT = pd.Timestamp("2025-07-01")
SETTLED_LAG = 9
MAX_INPUT_LAG = 8
DEDUP_FIPS = ("29510",)                 # St. Louis city: identical copy of St. Louis County
TREND = ("level", "d1", "d2", "slope4", "slope6", "mean4", "season_s1", "season_c1", "season_s2", "season_c2")
OUT = Path("data/results/metrics_improve.csv")


# ── inputs from the vintage ─────────────────────────────────────────────────────

def vintage(table: pd.DataFrame) -> pd.DataFrame:
    """(fips, as_of) x lag 0..52 of ``ed_resp`` as published on ``as_of`` (labels never read)."""
    from src.nowcast import LABEL_COLUMNS, _pivot
    t = table.drop(columns=[c for c in LABEL_COLUMNS if c in table.columns])
    return _pivot(t, "ed_resp", range(0, int(t["lag"].max()) + 1))


def inputs(piv: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Newest ``n`` input values per row, back-filled as ``nowcast.features`` does, with source lags."""
    vals = piv[list(range(1, piv.columns.max() + 1))]
    arr = vals.to_numpy(float)
    lags = np.array(vals.columns)
    out = {}
    for k in range(1, n + 1):
        sub = arr[:, k - 1:]
        first = np.where(np.isnan(sub), np.inf, np.arange(sub.shape[1])).min(axis=1)
        ok = np.isfinite(first)
        idx = np.where(ok, first, 0).astype(int)
        out[f"v{k}"] = np.where(ok, sub[np.arange(len(sub)), idx], np.nan)
        out[f"src{k}"] = np.where(ok, lags[k - 1 + idx], 0)
    return pd.DataFrame(out, index=piv.index).reset_index()


def revision_table(table: pd.DataFrame) -> pd.DataFrame:
    """Every (early print, later print) pair of one county-week: the raw material for corrections."""
    t = table.loc[~table["fips"].isin(DEDUP_FIPS) & table["ed_resp"].notna(),
                  ["fips", "as_of", "week_end", "lag", "ed_resp"]]
    early = t[t["lag"].between(1, MAX_INPUT_LAG)].rename(columns={"as_of": "early_as_of", "lag": "k",
                                                                  "ed_resp": "early"})
    late = t[t["lag"] >= SETTLED_LAG].rename(columns={"as_of": "late_as_of", "lag": "late_lag",
                                                      "ed_resp": "late"})
    pairs = early.merge(late, on=["fips", "week_end"])
    pairs = pairs[(pairs["early"] > hist.LOG_FLOOR) & (pairs["late"] > hist.LOG_FLOOR)]
    pairs["r"] = np.log(pairs["late"] / pairs["early"])
    return pairs


def corrections(pairs: pd.DataFrame, origin, phase: pd.Series | None = None) -> dict:
    """Median log revision per input lag (and phase), from pairs fully known on ``origin``."""
    p = pairs[(pairs["late_as_of"] == origin) & (pairs["early_as_of"] <= origin)]
    if phase is not None:
        p = p.assign(up=p.set_index(["fips", "early_as_of"]).index.map(phase).to_numpy())
        return {(k, up): float(g["r"].median()) for (k, up), g in p.groupby(["k", "up"]) if len(g) >= 10}
    return {k: float(g["r"].median()) for k, g in p.groupby("k") if len(g) >= 10}


def apply(inp: pd.DataFrame, corr_by_origin: dict, phased: bool, n: int = 6) -> pd.DataFrame:
    out = inp.copy()
    up = (np.log(np.clip(out["v1"], hist.LOG_FLOOR, None)) - np.log(np.clip(out["v2"], hist.LOG_FLOOR, None))) > 0
    for k in range(1, n + 1):
        c = [corr_by_origin.get(o, {}).get((s, u) if phased else s, 0.0) if s < SETTLED_LAG else 0.0
             for o, s, u in zip(out["as_of"], out[f"src{k}"], up)]
        out[f"v{k}"] = out[f"v{k}"] * np.exp(np.asarray(c, float))
    return out


# ── models ──────────────────────────────────────────────────────────────────────

def _slope(logs: np.ndarray) -> np.ndarray:
    """OLS slope per row of log values ordered oldest → newest (per week)."""
    w = logs.shape[1]
    t = np.arange(w) - (w - 1) / 2
    return (logs * t).sum(axis=1) / (t ** 2).sum()


def trend_design(vals: np.ndarray, week_end) -> pd.DataFrame:
    """``vals``: newest-first values, 6 columns. Same as history.design plus trend features."""
    lg = hist._log(vals)
    x = pd.DataFrame({"level": lg[:, 0], "d1": lg[:, 0] - lg[:, 1], "d2": lg[:, 1] - lg[:, 2],
                      "slope4": _slope(lg[:, 3::-1]), "slope6": _slope(lg[:, ::-1]),
                      "mean4": lg[:, :4].mean(axis=1) - lg[:, 0]})
    x[list(TREND[6:])] = hist.season(week_end)
    return x


def trend_examples(history: pd.DataFrame) -> pd.DataFrame:
    h = history.sort_values(["hsa", "week_end"])
    g = h.groupby("hsa")
    lags = np.column_stack([g["resp"].shift(k) for k in range(1, 7)])
    span = (h["week_end"] - g["week_end"].shift(6)).dt.days.to_numpy()
    ok = (span == 42) & ~np.isnan(lags).any(axis=1)
    x = trend_design(lags[ok], h.loc[ok, "week_end"])
    x.index = h.index[ok]
    x["y"] = hist._log(h.loc[ok, "resp"]) - hist._log(lags[ok, 0])
    return pd.concat([h.loc[ok, ["hsa", "state", "week_end", "fips_list"]], x], axis=1)


class Fits:
    """One model per (variant, refit block), refit every ``hist.REFIT_WEEKS`` going forward."""

    def __init__(self, rows: pd.DataFrame, columns, loss: str = "squared_error"):
        self.rows, self.columns, self.loss = rows, list(columns), loss
        self._at, self._model = None, None

    def model(self, origin):
        origin = pd.Timestamp(origin)
        if self._model is None or not self._at <= origin < self._at + pd.Timedelta(weeks=hist.REFIT_WEEKS):
            train = hist.pool(self.rows, origin)
            if len(train) < hist.MIN_POOL:
                return None
            self._model = HistGradientBoostingRegressor(loss=self.loss, **hist.GBM).fit(
                train[self.columns].to_numpy(float), train["y"].to_numpy(float))
            self._at = origin
        return self._model


def _target_week(as_of: pd.Series) -> pd.Series:
    return as_of + pd.to_timedelta((5 - as_of.dt.dayofweek) % 7, unit="D")


def predict_rows(inp: pd.DataFrame, fits: Fits, trend: bool) -> np.ndarray:
    out = np.full(len(inp), np.nan)
    for origin, idx in inp.groupby("as_of").groups.items():
        part = inp.loc[idx]
        model = fits.model(origin)
        if model is None:
            continue
        vals = part[[f"v{k}" for k in range(1, 7)]].to_numpy(float)
        if trend:
            x = trend_design(vals, _target_week(part["as_of"]))[list(TREND)]
        else:
            x = hist.design(vals[:, 0], vals[:, 1], vals[:, 2], _target_week(part["as_of"]))[list(hist.FEATURES)]
        out[inp.index.get_indexer(idx)] = np.exp(hist._log(vals[:, 0]) + model.predict(x.to_numpy(float)))
    return out


# ── the search ──────────────────────────────────────────────────────────────────

def run(table: pd.DataFrame, history: pd.DataFrame, preds: pd.DataFrame, final: str | None = None):
    from src.nowcast import scoreable
    base = evaluate.dedupe_series(scoreable(preds, official=True))
    keys = base[base["rung"] == "0_naive"].drop(columns=["rung", "pred"]).reset_index(drop=True)
    inp = inputs(vintage(table)).merge(keys[["fips", "as_of"]], on=["fips", "as_of"]).reset_index(drop=True)

    pairs = revision_table(table)
    origins = sorted(inp["as_of"].unique())
    piv1 = inp.set_index(["fips", "as_of"])
    phase_all = inputs(vintage(table)).set_index(["fips", "as_of"])
    phase = (np.log(phase_all["v1"].clip(lower=hist.LOG_FLOOR)) - np.log(phase_all["v2"].clip(lower=hist.LOG_FLOOR))) > 0
    corr = {o: corrections(pairs, o) for o in origins}
    corr_phase = {o: corrections(pairs, o, phase) for o in origins}

    rows = hist.examples(history)
    trows = trend_examples(history)
    variants = {
        "6_gbm_history (rerun)": dict(fits=Fits(rows, hist.FEATURES), inp=inp, trend=False),
        "revision": dict(fits=Fits(rows, hist.FEATURES), inp=apply(inp, corr, False), trend=False),
        "revision_phase": dict(fits=Fits(rows, hist.FEATURES), inp=apply(inp, corr_phase, True), trend=False),
        "l1": dict(fits=Fits(rows, hist.FEATURES, "absolute_error"), inp=inp, trend=False),
        "trend": dict(fits=Fits(trows, TREND), inp=inp, trend=True),
    }
    scored = keys.copy()
    scored["naive"] = keys.merge(base[base["rung"] == "0_naive"][["fips", "as_of", "pred"]],
                                 on=["fips", "as_of"])["pred"].to_numpy()
    scored["gbm6"] = keys.merge(base[base["rung"] == "6_gbm_history"][["fips", "as_of", "pred"]],
                                on=["fips", "as_of"])["pred"].to_numpy()
    for name, v in variants.items():
        log.info("variant %s", name)
        p = pd.Series(predict_rows(v["inp"], v["fits"], v["trend"]), index=pd.MultiIndex.from_frame(inp[["fips", "as_of"]]))
        scored[name] = scored.set_index(["fips", "as_of"]).index.map(p).to_numpy(float)
    scored["split"] = np.where(scored["as_of"] < SPLIT, "dev", "test")
    return scored, corr


def mae(s: pd.DataFrame, col: str) -> float:
    return float((s[col] - s["ed_resp_final"]).abs().mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Dev/test search for a better nowcast.")
    p.add_argument("--table", default="data/county_week.parquet")
    p.add_argument("--history", default=str(hist.DEFAULT_CSV))
    p.add_argument("--preds", default="data/results/predictions.csv")
    p.add_argument("--final", action="store_true", help="stage 2: combine dev winners, test the pick once")
    a = p.parse_args()
    if a.final:
        stage2(pd.read_parquet(a.table), hist.load(a.history))
        return
    preds = pd.read_csv(a.preds, dtype={"fips": str}, parse_dates=["as_of", "week_end"])
    scored, corr = run(pd.read_parquet(a.table), hist.load(a.history), preds)
    scored.to_csv(OUT.with_name("predictions_improve.csv"), index=False)
    dev = scored[scored["split"] == "dev"]
    cols = [c for c in scored.columns if c not in ("fips", "as_of", "week_end", "metro", "home", "has_official",
                                                     "n_train", "ed_resp_final", "ed_resp_first", "split")]
    table = pd.DataFrame([{"variant": c, "dev_mae": mae(dev, c), "dev_skill": 1 - mae(dev, c) / mae(dev, "naive")}
                          for c in cols])
    table.to_csv(OUT, index=False)
    print(table.round(4).to_string(index=False))
    last = max(corr)
    print("revision corrections at", pd.Timestamp(last).date(), {k: round(v, 3) for k, v in sorted(corr[last].items())})



# ── stage 2: combine dev winners, lock the pick on dev, then test once ─────────

def stage2(table: pd.DataFrame, history: pd.DataFrame, path=OUT.with_name("predictions_improve.csv")):
    """Combination of the dev winners (``revision`` + ``l1``; ``revision_phase`` is an
    alternative to ``revision`` and lost to it on dev). Pick by dev MAE, then score
    the pick on test against rung 6. Test numbers are computed only after the pick."""
    scored = pd.read_csv(path, dtype={"fips": str}, parse_dates=["as_of", "week_end"])
    inp = inputs(vintage(table)).merge(scored[["fips", "as_of"]], on=["fips", "as_of"]).reset_index(drop=True)
    pairs = revision_table(table)
    corr = {o: corrections(pairs, o) for o in sorted(inp["as_of"].unique())}
    p = pd.Series(predict_rows(apply(inp, corr, False), Fits(hist.examples(history), hist.FEATURES,
                                                               "absolute_error"), False),
                  index=pd.MultiIndex.from_frame(inp[["fips", "as_of"]]))
    scored["revision+l1"] = scored.set_index(["fips", "as_of"]).index.map(p).to_numpy(float)
    scored.to_csv(path, index=False)

    dev, test = scored[scored["split"] == "dev"], scored[scored["split"] == "test"]
    pick = min(("l1", "revision+l1"), key=lambda c: mae(dev, c))
    print(f"dev MAE: l1 {mae(dev, 'l1'):.4f}, revision+l1 {mae(dev, 'revision+l1'):.4f} -> locked pick: {pick}")

    w = evaluate.wilcoxon_paired((test[pick] - test["ed_resp_final"]).abs().to_numpy(),
                                 (test["gbm6"] - test["ed_resp_final"]).abs().to_numpy())
    rows = [{"split": s, "n": len(d), "naive": mae(d, "naive"), "gbm6": mae(d, "gbm6"), "pick": mae(d, pick),
             "skill_gbm6": 1 - mae(d, "gbm6") / mae(d, "naive"), "skill_pick": 1 - mae(d, pick) / mae(d, "naive")}
            for s, d in (("test (held out)", test), ("all 462 (incl. dev)", scored))]
    res = pd.DataFrame(rows)
    print(f"\nTEST, the one pre-registered comparison: {pick} vs rung 6 (paired Wilcoxon):", w)
    print(res.round(4).to_string(index=False))
    by_metro = pd.DataFrame([{"metro": m, "n": len(d), "skill_gbm6": 1 - mae(d, "gbm6") / mae(d, "naive"),
                              "skill_pick": 1 - mae(d, pick) / mae(d, "naive")} for m, d in test.groupby("metro")])
    print("\ntest by metro:\n" + by_metro.round(3).to_string(index=False))
    other = {c: round(1 - mae(test, c) / mae(test, "naive"), 4)
             for c in ("revision", "revision_phase", "l1", "trend", "revision+l1")}
    print("\nfor the record, test skill of every candidate (NOT used to choose):", other)
    res.assign(pick=pick, wilcoxon_p=w.get("pvalue"), median_diff=w.get("median_diff")).to_csv(
        OUT.with_name("metrics_improve_final.csv"), index=False)
    return pick, res, w


if __name__ == "__main__":
    main()
