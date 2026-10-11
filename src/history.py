"""Learn week-to-week ED dynamics from every US Health Service Area's settled history.

The ladder's ridge rungs learn from the study metros alone: at most ~150
honest examples per origin. CDC's county table (``rdmq-nq56``) carries the
same ED signal for ~675 reporting Health Service Areas (HSAs) back to
October 2022, which is enough to fit a nonlinear model of how respiratory
activity moves from one week to the next.

Usage, from the repo root (CSV from data.cdc.gov/d/rdmq-nq56, or Box):

    python -m src.nowcast --history raw/cdc_nssp/rdmq-nq56_2026-10-10.csv

Why this is (almost) honest
---------------------------
The CSV is one vintage, pulled 2026-10-10: every week in it is already
revised. At origin D the model may only use weeks ending at least
``SETTLE_WEEKS`` before D; by then the revision profile
(``data/results/revision_profile.csv``) shows under 2% of values still move
more than 5%, so those weeks looked on D almost exactly as they do today.
Missouri is excluded before 2026-06-13 because NSSP published nothing for it
until then (README, finding 2). What remains is an approximation, and we say
so: a week's value may have moved by a fraction of a percent after D.

What is learned
---------------
One example per HSA-week: the log ratio of this week to last week, from the
level and the two most recent weekly log changes (momentum and its change)
plus week-of-year harmonics. At prediction time the same features come from
the vintage known on D, and the predicted ratio is applied to the newest
published value, so the model is a learned correction to carrying it
forward.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

log = logging.getLogger(__name__)

DEFAULT_CSV = Path("raw/cdc_nssp/rdmq-nq56_2026-10-10.csv")
SETTLE_WEEKS = 8
LOG_FLOOR = 0.05
MISSOURI_FIRST_PUBLISHED = pd.Timestamp("2026-06-13")
MIN_POOL = 1000
REFIT_WEEKS = 4
SIGNALS = ("percent_visits_covid", "percent_visits_influenza", "percent_visits_rsv")
FEATURES = ("level", "d1", "d2", "season_s1", "season_c1", "season_s2", "season_c2")
GBM = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
           random_state=0)
SVM_SAMPLE = 10_000     # RBF SVR scales ~quadratically; fit on a random subsample


def _make(name: str):
    """A fresh regressor for the weekly log ratio. Same features for every model.

    Settings are fixed in advance, not tuned on the backtest: tuning on the
    scored weeks would leak the answer key into model choice.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    if name == "gbm":
        return HistGradientBoostingRegressor(**GBM)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                            min_child_weight=200, subsample=0.8, colsample_bytree=1.0,
                            tree_method="hist", n_jobs=-1, random_state=0)
    if name == "rf":
        return RandomForestRegressor(n_estimators=150, min_samples_leaf=100, max_features=0.6,
                                     n_jobs=-1, random_state=0)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.02))
    if name == "linear":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    raise ValueError(f"unknown model {name!r}")


MODELS = ("gbm", "xgb", "rf", "svm", "linear")


def _log(values) -> np.ndarray:
    return np.log(np.clip(np.asarray(values, dtype=float), LOG_FLOOR, None))


def load(path=DEFAULT_CSV) -> pd.DataFrame:
    """One row per HSA-week: ``hsa``, ``state``, ``week_end``, ``resp``, ``fips_list``.

    Counties in one HSA carry identical values, so each HSA is kept once.
    ``resp`` is COVID + flu + RSV; weeks missing any of the three are dropped
    (``percent_visits_combined`` ended in 2024). State total rows are dropped.
    """
    raw = pd.read_csv(path, usecols=["week_end", "geography", "county", "fips",
                                     "hsa_nci_id", *SIGNALS], dtype={"fips": str})
    raw = raw[raw["county"] != "All"].copy()
    raw["fips"] = raw["fips"].str.zfill(5)
    raw["week_end"] = pd.to_datetime(raw["week_end"])
    raw["resp"] = raw[list(SIGNALS)].sum(axis=1, min_count=len(SIGNALS))
    members = raw.groupby("hsa_nci_id")["fips"].agg(lambda s: tuple(sorted(set(s))))
    out = (raw.dropna(subset=["resp"])
              .drop_duplicates(["hsa_nci_id", "week_end"])
              .rename(columns={"hsa_nci_id": "hsa", "geography": "state"})
              [["hsa", "state", "week_end", "resp"]])
    out["fips_list"] = out["hsa"].map(members)
    log.info("history: %d HSA-weeks, %d HSAs, %s..%s", len(out), out["hsa"].nunique(),
             out["week_end"].min().date(), out["week_end"].max().date())
    return out.sort_values(["hsa", "week_end"]).reset_index(drop=True)


def season(week_end) -> np.ndarray:
    """Two annual harmonics of the week's day of year (as ``src.nowcast``)."""
    doy = pd.DatetimeIndex(week_end).dayofyear.to_numpy(float)
    return np.column_stack([f(2 * np.pi * k * doy / 365.25)
                            for k in (1, 2) for f in (np.sin, np.cos)])


def design(lag1, lag2, lag3, week_end) -> pd.DataFrame:
    """Features for predicting log(y_t / y_{t-1}) from the three prior weeks."""
    l1, l2, l3 = _log(lag1), _log(lag2), _log(lag3)
    x = pd.DataFrame({"level": l1, "d1": l1 - l2, "d2": l2 - l3})
    x[list(FEATURES[3:])] = season(week_end)
    return x


def examples(hist: pd.DataFrame) -> pd.DataFrame:
    """Training rows: one per HSA-week with three consecutive prior weeks."""
    h = hist.sort_values(["hsa", "week_end"])
    g = h.groupby("hsa")
    lags = {k: g["resp"].shift(k) for k in (1, 2, 3)}
    span = (h["week_end"] - g["week_end"].shift(3)).dt.days
    ok = (span == 21) & lags[1].notna() & lags[2].notna() & lags[3].notna()
    x = design(lags[1][ok], lags[2][ok], lags[3][ok], h.loc[ok, "week_end"])
    x.index = h.index[ok]
    x["y"] = _log(h.loc[ok, "resp"]) - _log(lags[1][ok])
    return pd.concat([h.loc[ok, ["hsa", "state", "week_end", "fips_list"]], x], axis=1)


def pool(rows: pd.DataFrame, origin, settle_weeks: int = SETTLE_WEEKS,
         exclude_fips=()) -> pd.DataFrame:
    """Examples a model fitted at ``origin`` may learn from.

    Weeks ending at least ``settle_weeks`` before the origin (so nearly
    final on the date), no Missouri before it was first published, and
    optionally no HSA containing ``exclude_fips`` (a pure-transfer check).
    """
    origin = pd.Timestamp(origin)
    keep = rows["week_end"] <= origin - pd.Timedelta(weeks=settle_weeks)
    if origin < MISSOURI_FIRST_PUBLISHED:
        keep &= rows["state"] != "Missouri"
    if len(exclude_fips):
        bad = set(exclude_fips)
        keep &= ~rows["fips_list"].map(lambda fs: bool(bad.intersection(fs)))
    return rows[keep]


def fit(train: pd.DataFrame, columns=FEATURES, model: str = "gbm"):
    if len(train) < MIN_POOL:
        return None
    if model == "svm" and len(train) > SVM_SAMPLE:
        train = train.sample(SVM_SAMPLE, random_state=0)
    return _make(model).fit(train[list(columns)].to_numpy(float), train["y"].to_numpy(float))


def predict(model, feats: pd.DataFrame, columns=FEATURES) -> np.ndarray:
    """Nowcast from ``src.nowcast.features`` rows: last value times the learned ratio.

    Holes in the AR lags are filled with the newest published value (no
    momentum rather than a dropped origin).
    """
    last = feats["last"]
    x = design(last, feats["y_lag2"].fillna(last), feats["y_lag3"].fillna(last),
               feats["as_of"])
    return np.exp(_log(last) + model.predict(x[list(columns)].to_numpy(float)))


class Refitter:
    """Fit at an origin, reuse for ``every`` weeks, then refit.

    A model fitted at an earlier origin only saw data that was older still,
    so reusing it is honest; it just trails the newest settled weeks a bit.
    Refitting each origin costs ~5x the run time for no visible change.
    """

    def __init__(self, rows: pd.DataFrame, every: int = REFIT_WEEKS):
        self.rows, self.every, self._cache = rows, every, {}

    def model(self, origin, columns=FEATURES, exclude_fips=(), kind: str = "gbm"):
        origin = pd.Timestamp(origin)
        key = (kind, tuple(columns), tuple(exclude_fips))
        fitted_at, model = self._cache.get(key, (None, None))
        if model is None or origin - fitted_at >= pd.Timedelta(weeks=self.every):
            model = fit(pool(self.rows, origin, exclude_fips=exclude_fips), columns, kind)
            if model is not None:
                self._cache[key] = (origin, model)
        return model
