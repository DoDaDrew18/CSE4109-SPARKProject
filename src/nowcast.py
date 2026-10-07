"""Nowcast this week's respiratory ED share with a ladder of simple models.

Usage, from the repo root (after ``python -m src.align``):

    python -m src.nowcast                    # writes data/results/*.csv

The ladder. Each rung adds one source to the one below, so the difference
between neighbours is that source's contribution:

  0 naive        last published value carried forward (no fitting)
  1 ar_season    linear regression on lagged ``ed_resp`` as known on the date,
                 plus week-of-year harmonics (so rung 2 tests weather *beyond*
                 the calendar, not weather as a stand-in for the season)
  2 +weather     rung 1 + NOAA temperature / dew point / precipitation
  3 +wastewater  rung 2 + NWSS viral levels
  4 +news        rung 3 + article count and concern

A rung whose columns are absent from ``county_week`` (loader not built yet,
or no data) is skipped, not faked.

What is predicted, from what
----------------------------
At origin D (an ``as_of`` date, a Saturday) the target is ``ed_resp`` for the
week ending D (``lag == 0`` in ``county_week``). NSSP's first print arrives a
week later, so the newest value the model sees is lag 1 -- itself a first
print that will be revised upward.

The honest training label
-------------------------
Scoring uses the final value, but a model trained at D cannot have seen
final values: for recent weeks they did not exist yet. Training on
``ed_resp_final`` would leak every later revision into the past. So the
training set at origin D is built only from the vintage of D:

* examples are past origins D' with ``label_lag <= (D - D')/7 <= window``;
* each example's features are what was known on D' (the ``as_of == D'`` slice);
* its label is the value of week D' **as known on D** (row
  ``week_end == D', as_of == D``).

``label_lag`` trades recency against settledness: lag-1 values are the
incomplete first prints, so training on them teaches the model to predict
first prints, not final values. Waiting a few weeks gives nearly-settled
labels at the cost of skipping the most recent weeks. The default (3) is
justified by the revision profile in ``src.evaluate.revision_profile``.

Pooled, not per-county
----------------------
With a one-year window each county has at most ~50 honest examples; a
per-county regression with up to a dozen features would mostly fit noise.
The study counties are Midwest/South metros whose respiratory seasons move
together, so one pooled model with a per-metro intercept (a metro indicator)
shares the dynamics and keeps each metro's typical drift. The model predicts
the *log ratio* of this week to the last published value; ridge shrinkage
pulls the slopes toward zero, i.e. toward the naive rung, when the data say
little. St. Louis city (29510) and County (29189) are one series in NSSP, so
training and scoring keep one copy.

No official data on the date
----------------------------
Missouri (St. Louis, Kansas City) is absent from every Delphi NSSP issue
before 202623 (June 2026) and then appears back-filled. On those dates there
is no ``last`` value, so the ladder cannot run; this is the situation where
other sources would matter most. The ``n*`` rungs nowcast the level from
weather / wastewater / news alone, trained on the counties that did have
honest labels, and are scored separately on county-weeks without official
data, against ``n0_mean`` (the pooled training mean) and ``n_season``
(week-of-year harmonics + metro levels); ``n1_weather`` adds weather to
``n_season``, so its paired test is weather-over-season. Because Missouri has
no training labels on those dates, this is already an out-of-metro test;
:func:`leave_one_metro_out` repeats it for the four metros that do have data.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.align import LABEL_COLUMNS, OUTPUT as TABLE_PATH
from src.study import METROS

log = logging.getLogger(__name__)

AR_LAGS = 3
LABEL_LAG = 3
WINDOW = 52
MIN_TRAIN = 30
RIDGE = 1.0
LOG_FLOOR = 0.05      # % of ED visits; below NSSP's reporting resolution
SETTLE_WEEKS = 8
RESULTS = Path("data/results")

WEATHER = ("temp_mean_f", "dewpoint_f", "precip_in")
WASTEWATER = ("ww_covid", "ww_flu", "ww_rsv")
NEWS = ("news_n", "news_concern")


@dataclass(frozen=True)
class Rung:
    name: str
    groups: tuple               # feature groups on top of the AR lags
    no_official: bool = False   # uses no ED values at all


LADDER = (
    Rung("0_naive", ()),
    Rung("1_ar_season", ("season",)),
    Rung("2_weather", ("season", "weather")),
    Rung("3_wastewater", ("season", "weather", "wastewater")),
    Rung("4_news", ("season", "weather", "wastewater", "news")),
    # For county-weeks with no official ED value on the date at all.
    Rung("n0_mean", (), no_official=True),
    Rung("n_season", ("season",), no_official=True),
    Rung("n1_weather", ("season", "weather"), no_official=True),
    Rung("n2_wastewater", ("season", "weather", "wastewater"), no_official=True),
    Rung("n3_news", ("season", "weather", "wastewater", "news"), no_official=True),
)
ORDER = [r.name for r in LADDER]
SEASON = ("season_s1", "season_c1", "season_s2", "season_c2")


# -- features ------------------------------------------------------------------

def _pivot(table: pd.DataFrame, column: str, lags) -> pd.DataFrame:
    wide = table.pivot_table(index=["fips", "as_of"], columns="lag", values=column,
                             aggfunc="first", dropna=False)
    return wide.reindex(columns=list(lags))


def features(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (fips, as_of): everything a model may use at that origin.

    Built only from rows of the same ``as_of`` slice, so a feature can never
    contain a value published after its origin. Label columns are never read.

    * ``last``     newest published ``ed_resp`` (any lag >= 1); the naive rung.
    * ``y_lag{k}`` ``ed_resp`` at lag k, filled from older lags when the
                   archive skipped a week, so a hole does not drop the origin.
    * weather      target week (lag 0) and the week before.
    * wastewater   newest reading within lags 0..3 (sites report late).
    * news         target week and the week before; no articles means 0.
    """
    table = table.drop(columns=[c for c in LABEL_COLUMNS if c in table.columns])
    max_lag = int(table["lag"].max())
    resp = _pivot(table, "ed_resp", range(1, max_lag + 1))
    filled = resp.bfill(axis=1)          # lag k <- newest of lags >= k
    out = pd.DataFrame(index=resp.index)
    out["last"] = filled[1]
    for k in range(1, AR_LAGS + 1):
        out[f"y_lag{k}"] = filled[k] if k in filled else np.nan

    for col in WEATHER:
        if col in table:
            w = _pivot(table, col, (0, 1))
            out[f"{col}_0"], out[f"{col}_1"] = w[0], w[1]
    for col in WASTEWATER:
        if col in table:
            out[f"{col}_recent"] = _pivot(table, col, range(0, 4)).bfill(axis=1)[0]
    for col in NEWS:
        if col in table:
            n = _pivot(table, col, (0, 1)).fillna(0.0)
            out[f"{col}_0"], out[f"{col}_1"] = n[0], n[1]
    meta = table.drop_duplicates(["fips", "as_of"]).set_index(["fips", "as_of"])
    out["metro"] = meta["metro"]
    out = out.reset_index()
    # Calendar position of the target week: known in advance, so not a leak.
    # Two annual harmonics let the model learn a seasonal curve, so any gain
    # from weather has to come from beyond "it is January".
    doy = (out["as_of"] + pd.to_timedelta((5 - out["as_of"].dt.dayofweek) % 7, unit="D")
           ).dt.dayofyear.to_numpy(float)
    for k in (1, 2):
        out[f"season_s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        out[f"season_c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return out


def group_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Feature columns per group, keeping only those with any data."""
    def present(prefixes):
        return [c for c in frame.columns
                if c.startswith(prefixes) and frame[c].notna().any()]
    return {
        "ar": [f"y_lag{k}" for k in range(1, AR_LAGS + 1)],
        "season": [c for c in SEASON if c in frame.columns],
        "weather": present(tuple(f"{c}_" for c in WEATHER)),
        "wastewater": present(tuple(f"{c}_" for c in WASTEWATER)),
        "news": present(tuple(f"{c}_" for c in NEWS)),
    }


def available_rungs(feats: pd.DataFrame) -> list[Rung]:
    """Rungs whose every feature group has data; the rest are skipped."""
    groups = group_columns(feats)
    keep = []
    for rung in LADDER:
        missing = [g for g in rung.groups if not groups[g]]
        if missing:
            log.warning("skipping rung %s: no data for %s", rung.name, ", ".join(missing))
            continue
        keep.append(rung)
    return keep


# -- honest training sets ------------------------------------------------------

def training_set(table: pd.DataFrame, feats: pd.DataFrame, origin,
                 label_lag: int = LABEL_LAG, window: int = WINDOW) -> pd.DataFrame:
    """Examples a model fitted at ``origin`` is allowed to learn from.

    Labels come from the ``as_of == origin`` slice only (values as known at
    the origin), for weeks ``label_lag..window`` weeks old. Features come from
    each example's own, earlier, as_of slice. Raises if anything is dated
    after the origin -- the rolling-origin guarantee.
    """
    origin = pd.Timestamp(origin)
    lo, hi = origin - timedelta(weeks=window), origin - timedelta(weeks=label_lag)
    labels = table.loc[(table["as_of"] == origin) & table["week_end"].between(lo, hi),
                       ["fips", "week_end", "ed_resp"]]
    labels = labels.rename(columns={"week_end": "as_of", "ed_resp": "label"})
    train = feats.merge(labels, on=["fips", "as_of"], how="inner")
    train = train[train["label"].notna() & train["last"].notna()]
    # St. Louis city and county are one NSSP series; keep one copy so the
    # pooled fit does not weight St. Louis double.
    train = train.drop_duplicates(["metro", "as_of", "label", "last"])
    if (train["as_of"] > hi).any():
        raise AssertionError("training example dated after origin - label_lag")
    return train


# -- models ------------------------------------------------------------------

def _log(values) -> np.ndarray:
    """Log of a percentage, floored so a reported 0.0 does not become -inf."""
    return np.log(np.clip(np.asarray(values, dtype=float), LOG_FLOOR, None))


def _design(frame: pd.DataFrame, columns, metros, center=None):
    """Feature matrix plus metro indicators; NaNs imputed with training means.

    ED lag columns enter on the log scale, matching the log-ratio target.
    """
    x = frame[list(columns)].astype(float).copy()
    for c in columns:
        if c.startswith("y_lag"):
            x[c] = _log(x[c])
    if center is None:
        center = x.mean()
    x = x.fillna(center).fillna(0.0)
    dummies = [(frame["metro"] == m).to_numpy(float) for m in metros[1:]]
    return np.column_stack([x.to_numpy(), *dummies]) if dummies else x.to_numpy(), center


def _ridge(x_tr, y, x_te, n_feat: int, ridge: float) -> np.ndarray:
    """Ridge on standardized features; intercept and indicators unpenalized."""
    mu, sd = x_tr[:, :n_feat].mean(axis=0), x_tr[:, :n_feat].std(axis=0)
    sd[sd == 0] = 1.0
    x_tr, x_te = x_tr.copy(), x_te.copy()
    x_tr[:, :n_feat] = (x_tr[:, :n_feat] - mu) / sd
    x_te[:, :n_feat] = (x_te[:, :n_feat] - mu) / sd
    x_tr = np.column_stack([np.ones(len(x_tr)), x_tr])
    x_te = np.column_stack([np.ones(len(x_te)), x_te])
    penalty = np.zeros(x_tr.shape[1])
    penalty[1:1 + n_feat] = ridge
    gram = x_tr.T @ x_tr + np.diag(penalty) + 1e-8 * np.eye(x_tr.shape[1])
    return x_te @ np.linalg.solve(gram, x_tr.T @ y)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, columns,
                ridge: float | None = None) -> np.ndarray:
    """Ladder rungs: ridge regression of log(label / last), pooled, metro intercepts.

    Respiratory activity grows and decays multiplicatively, so the one-week
    change is modeled as a log ratio. (A level-scale version was tried first
    and lost to the naive rung by ~20% MAE; see the report for that caveat.)
    With little signal the slopes shrink to zero and the prediction falls
    back to ``last`` times the metro's average weekly ratio.
    """
    ridge = RIDGE if ridge is None else ridge
    metros = sorted(METROS)
    x_tr, center = _design(train, columns, metros)
    x_te, _ = _design(test, columns, metros, center)
    y = _log(train["label"]) - _log(train["last"])
    return np.exp(_log(test["last"]) + _ridge(x_tr, y, x_te, len(columns), ridge))


def fit_predict_no_official(train: pd.DataFrame, test: pd.DataFrame, columns,
                            ridge: float | None = None) -> np.ndarray:
    """Nowcast the level with no ED values at all: log(label) on the other sources.

    For county-weeks with no official data on the prediction date (Missouri
    before June 2026). Metro indicators are one-hot for *every* metro and
    penalized like the other slopes: a metro with training rows gets its own
    level, while a metro with none (exactly the ones this model exists for)
    keeps a zero indicator and falls back to the pooled intercept instead of
    silently inheriting a reference metro's level. An empty ``columns`` gives
    the pooled training mean on the log scale -- the baseline to beat.
    """
    ridge = RIDGE if ridge is None else ridge
    y = _log(train["label"])
    if not columns:
        return np.full(len(test), float(np.exp(y.mean())))
    x_tr, center = _design(train, columns, [])
    x_te, _ = _design(test, columns, [], center)
    metros = sorted(METROS)
    hot = lambda f: np.column_stack([(f["metro"] == m).to_numpy(float) for m in metros])
    x_tr, x_te = np.column_stack([x_tr, hot(train)]), np.column_stack([x_te, hot(test)])
    return np.exp(_ridge(x_tr, y, x_te, x_tr.shape[1], ridge))


def rolling_origin(table: pd.DataFrame, rungs=None, label_lag: int = LABEL_LAG,
                   window: int = WINDOW, min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """Fit every rung at every origin; return one row per (rung, fips, origin).

    Ladder rungs predict county-weeks that have some official value on the
    date (``has_official``); the no-official rungs predict every county-week,
    so they can be scored on the rows the ladder cannot touch. The answer key
    (``ed_resp_final``, ``ed_resp_first``) is joined after prediction.
    """
    feats = features(table)
    rungs = available_rungs(feats) if rungs is None else rungs
    groups = group_columns(feats)
    keys = table.loc[table["lag"] == 0, ["fips", "as_of", "week_end", "metro", "home",
                                          *LABEL_COLUMNS]]
    out = []
    for origin in sorted(feats["as_of"].unique()):
        test = feats[feats["as_of"] == origin]
        official = test["last"].notna().to_numpy()
        train = training_set(table, feats, origin, label_lag, window)
        for rung in rungs:
            other = [c for g in rung.groups for c in groups[g]]
            if rung.name == "0_naive":
                rows, pred = official, test.loc[official, "last"].to_numpy(float)
            elif len(train) < min_train:
                continue
            elif rung.no_official:
                rows, pred = np.ones(len(test), bool), fit_predict_no_official(train, test, other)
            else:
                rows = official
                pred = fit_predict(train, test[official], groups["ar"] + other)
            if not rows.any():
                continue
            out.append(pd.DataFrame({"rung": rung.name, "fips": test["fips"].to_numpy()[rows],
                                     "as_of": origin, "pred": pred,
                                     "has_official": official[rows], "n_train": len(train)}))
    if not out:
        return pd.DataFrame(columns=["rung", "fips", "as_of", "pred", "has_official", "n_train"])
    preds = pd.concat(out, ignore_index=True)
    return preds.merge(keys, on=["fips", "as_of"], how="left")


def leave_one_metro_out(table: pd.DataFrame, holdout=("chi", "ind", "lou", "mem"),
                        label_lag: int = LABEL_LAG, window: int = WINDOW,
                        min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """No-official rungs for each metro as if it had never reported.

    For held-out metro M at every origin: train on the other metros' honest
    examples only, predict M's target week without any of M's ED values.
    This asks the Missouri question -- how well can we nowcast a place with
    no official data? -- on metros where the answer can be scored all year.
    """
    feats = features(table)
    rungs = [r for r in available_rungs(feats) if r.no_official]
    groups = group_columns(feats)
    keys = table.loc[table["lag"] == 0, ["fips", "as_of", "week_end", "metro", "home",
                                          *LABEL_COLUMNS]]
    out = []
    for origin in sorted(feats["as_of"].unique()):
        train_all = training_set(table, feats, origin, label_lag, window)
        for metro in holdout:
            train = train_all[train_all["metro"] != metro]
            test = feats[(feats["as_of"] == origin) & (feats["metro"] == metro)]
            if test.empty or len(train) < min_train:
                continue
            for rung in rungs:
                cols = [c for g in rung.groups for c in groups[g]]
                out.append(pd.DataFrame({
                    "rung": rung.name, "fips": test["fips"].to_numpy(), "as_of": origin,
                    "pred": fit_predict_no_official(train, test, cols),
                    "has_official": False, "n_train": len(train), "held_out": metro}))
    if not out:
        return pd.DataFrame(columns=["rung", "fips", "as_of", "pred", "has_official",
                                     "n_train", "held_out"])
    return pd.concat(out, ignore_index=True).merge(keys, on=["fips", "as_of"], how="left")


def scoreable(preds: pd.DataFrame, settle_weeks: int = SETTLE_WEEKS,
              latest_issue=None, official: bool = True) -> pd.DataFrame:
    """Rows to score: settled final, the requested subset, predicted by every rung.

    The newest weeks' "final" values are still first prints; scoring against
    them would reward predicting first prints. ``official`` picks the ladder
    comparison (county-weeks with some official value on the date, ladder
    rungs only) or the no-official comparison (county-weeks with none,
    no-official rungs only). Requiring every rung keeps comparisons paired.
    """
    latest_issue = pd.Timestamp(latest_issue or preds["as_of"].max())
    names = {r.name for r in LADDER if r.no_official != official}
    rows = preds[preds["rung"].isin(names) & (preds["has_official"] == official)
                 & (preds["week_end"] <= latest_issue - timedelta(weeks=settle_weeks))
                 & preds["ed_resp_final"].notna()]
    if rows.empty:
        return rows.reset_index(drop=True)
    n_rungs = rows["rung"].nunique()
    full = rows.groupby(["fips", "as_of"])["rung"].transform("nunique") == n_rungs
    return rows[full].reset_index(drop=True)


def _print(title: str, frame: pd.DataFrame) -> None:
    print(f"\n== {title}")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(frame.to_string(index=False) if not frame.empty else "(nothing to score)")


def main() -> None:
    from src import evaluate

    parser = argparse.ArgumentParser(description="Run the nowcast ladder.")
    parser.add_argument("--table", default=str(TABLE_PATH))
    parser.add_argument("--out", default=str(RESULTS))
    parser.add_argument("--label-lag", type=int, default=LABEL_LAG)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--settle-weeks", type=int, default=SETTLE_WEEKS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    table = pd.read_parquet(args.table)
    preds = rolling_origin(table, label_lag=args.label_lag, window=args.window)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "predictions.csv", index=False)

    results = {}
    for name, official in (("ladder", True), ("no_official", False)):
        scored = evaluate.dedupe_series(scoreable(preds, args.settle_weeks, official=official))
        baseline = "0_naive" if official else "n0_mean"
        metrics = evaluate.summarize(scored, baseline) if not scored.empty else pd.DataFrame()
        tests = (evaluate.ladder_tests(scored, baseline, ORDER)
                 if not scored.empty else pd.DataFrame())
        metrics.to_csv(out / f"metrics_{name}.csv", index=False)
        tests.to_csv(out / f"wilcoxon_{name}.csv", index=False)
        results[name] = (scored, metrics, tests)

    lomo = leave_one_metro_out(table, label_lag=args.label_lag, window=args.window)
    lomo.to_csv(out / "predictions_lomo.csv", index=False)
    lomo_scored = evaluate.dedupe_series(scoreable(lomo, args.settle_weeks, official=False))
    lomo_metrics = evaluate.summarize(lomo_scored, "n0_mean")
    lomo_metrics = lomo_metrics[lomo_metrics["group"].str.startswith(("all", "metro:"))]
    lomo_tests = pd.concat([evaluate.ladder_tests(part, "n0_mean", ORDER).assign(group=m)
                            for m, part in [("all", lomo_scored),
                                            *lomo_scored.groupby("metro")]],
                           ignore_index=True)
    lomo_metrics.to_csv(out / "metrics_lomo.csv", index=False)
    lomo_tests.to_csv(out / "wilcoxon_lomo.csv", index=False)
    results["leave_one_metro_out"] = (lomo_scored, lomo_metrics, lomo_tests)

    coverage = evaluate.official_coverage(table)
    revisions = evaluate.revision_profile(table)
    coverage.to_csv(out / "official_coverage.csv", index=False)
    revisions.to_csv(out / "revision_profile.csv", index=False)

    print(f"label_lag={args.label_lag}, window={args.window}, settle={args.settle_weeks}w; "
          f"St. Louis city/county deduplicated (identical NSSP series)")
    _print("official coverage per county (as_of Saturdays with any ED value in lags 1-4)",
           coverage)
    _print("revision profile: value known at lag k vs final", revisions)
    for name, (scored, metrics, tests) in results.items():
        span = (f"{scored['week_end'].min().date()} .. {scored['week_end'].max().date()}, "
                f"{scored[['fips', 'as_of']].drop_duplicates().shape[0]} county-weeks"
                if not scored.empty else "none")
        _print(f"{name}: metrics vs final ({span})", metrics)
        _print(f"{name}: paired Wilcoxon on |error|", tests)
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
