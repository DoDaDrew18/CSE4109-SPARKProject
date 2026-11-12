"""How sure is the nowcast? Calibrated ranges and an analog outlook.

The gradient-boosting nowcast (``src.history``) gives one number. A public
health reader needs two more things: how wide the plausible range is, and
how likely activity is to be rising. Both are learned from the same
national HSA history, at the same origin, with the same leak rules
(``history.pool``: weeks settled 8+ weeks before the date, no Missouri
before it was published).

Range: conformalized quantile regression (CQR)
----------------------------------------------
Two gradient-boosting models learn the 10th and 90th percentiles of the
weekly log change from the same features as the point model, so the range
widens in winter surges and narrows in quiet summers. The newest
``CAL_WEEKS`` settled weeks are held out; their errors set one additive
correction so that the range would have held the truth in 80% of those
held-out HSA-weeks (Romano, Patterson & Candès 2019). The guarantee assumes
the held-out weeks look like the target week; ``backtest`` checks it on the
study's honest vintage backtest instead of trusting it.

Outlook: analog ensemble
------------------------
The ``K`` HSA-weeks in national history whose level, momentum and time of
year are closest to the area's newest published weeks: what happened the
next week in those situations? The share that rose by 10% or more is the
rise probability, and the closest analogs are named, so the reader can see
the evidence (Delle Monache et al. 2013 use the same idea for weather).

Backtest:  python -m src.uncertainty    # → data/results/metrics_uncertainty.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

from src import history as hist

log = logging.getLogger(__name__)

__all__ = ["Probabilistic", "ProbRefitter", "row_features", "backtest", "load_metrics",
           "ALPHA", "RISE", "METRICS"]

ALPHA = 0.2                   # 80% range
CAL_WEEKS = 12                # newest settled weeks held out to calibrate the range
K = 60                        # analogs per query
RISE = np.log(1.10)           # "rising" = up 10% or more week over week
METRICS = Path("data/results/metrics_uncertainty.csv")
QGBM = {k: v for k, v in hist.GBM.items()}
MIN_CAL = 200


def row_features(feats: pd.DataFrame) -> pd.DataFrame:
    """The point model's design matrix for ``last`` / ``y_lag2`` / ``y_lag3`` / ``as_of`` rows."""
    last = feats["last"]
    return hist.design(last, feats["y_lag2"].fillna(last), feats["y_lag3"].fillna(last),
                       feats["as_of"])[list(hist.FEATURES)]


@dataclass
class Outlook:
    p_rise: float              # share of analogs up >= 10% the next week
    p_fall: float              # share down >= 10%
    median_change: float       # median next-week change of the analogs, as a fraction
    n: int
    examples: list[dict]       # closest analogs, one per HSA


class Probabilistic:
    """Interval models and analog index fitted on one origin's pool."""

    def __init__(self, pool: pd.DataFrame, alpha: float = ALPHA, cal_weeks: int = CAL_WEEKS,
                 k: int = K):
        cols = list(hist.FEATURES)
        weeks = np.sort(pool["week_end"].unique())
        cut = weeks[-cal_weeks] if len(weeks) > cal_weeks else weeks[-1]
        train, cal = pool[pool["week_end"] < cut], pool[pool["week_end"] >= cut]
        if len(train) < hist.MIN_POOL or len(cal) < MIN_CAL:
            raise ValueError(f"pool too small for calibrated ranges ({len(train)} train, {len(cal)} cal)")
        x_tr, y_tr = train[cols].to_numpy(float), train["y"].to_numpy(float)
        self.lo = HistGradientBoostingRegressor(loss="quantile", quantile=alpha / 2, **QGBM).fit(x_tr, y_tr)
        self.hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - alpha / 2, **QGBM).fit(x_tr, y_tr)
        x_cal, y_cal = cal[cols].to_numpy(float), cal["y"].to_numpy(float)
        scores = np.maximum(self.lo.predict(x_cal) - y_cal, y_cal - self.hi.predict(x_cal))
        n = len(scores)
        self.q = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
        self.alpha, self.n_cal, self.cal_from = alpha, n, pd.Timestamp(cut)

        x_all = pool[cols].to_numpy(float)
        self.mu, self.sd = x_all.mean(0), x_all.std(0) + 1e-9
        self.k = min(k, len(pool))
        self.nn = NearestNeighbors(n_neighbors=self.k).fit((x_all - self.mu) / self.sd)
        self.ref = pool[["hsa", "state", "week_end", "fips_list", "y"]].reset_index(drop=True)
        self.base_rise = float((pool["y"] >= RISE).mean())

    def log_range(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated 80% range of the weekly log change for design rows ``x``."""
        v = x[list(hist.FEATURES)].to_numpy(float)
        lo, hi = self.lo.predict(v) - self.q, self.hi.predict(v) + self.q
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def range(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated 80% range of this week's value (% of ED visits)."""
        lo, hi = self.log_range(row_features(feats))
        base = hist._log(feats["last"])
        return np.exp(base + lo), np.exp(base + hi)

    def outlook(self, feats: pd.DataFrame, n_examples: int = 3) -> list[Outlook]:
        """Analog outlook for each row of ``feats``."""
        z = (row_features(feats).to_numpy(float) - self.mu) / self.sd
        dist, idx = self.nn.kneighbors(z)
        out = []
        for row in idx:
            y = self.ref["y"].to_numpy()[row]
            seen, examples = set(), []
            for i in row:
                r = self.ref.iloc[i]
                if r["hsa"] in seen:
                    continue
                seen.add(r["hsa"])
                examples.append({"hsa": r["hsa"], "state": r["state"], "fips_list": list(r["fips_list"]),
                                 "week_end": pd.Timestamp(r["week_end"]).date().isoformat(),
                                 "change": float(np.exp(r["y"]) - 1)})
                if len(examples) == n_examples:
                    break
            out.append(Outlook(float((y >= RISE).mean()), float((y <= -RISE).mean()),
                               float(np.exp(np.median(y)) - 1), len(y), examples))
        return out


class ProbRefitter:
    """``history.Refitter`` for ``Probabilistic``: refit every ``every`` weeks.

    A fit is reused only for origins on or after the origin it was fitted at,
    so asking about an earlier date never reuses a model that saw later weeks.
    """

    def __init__(self, rows: pd.DataFrame, every: int = hist.REFIT_WEEKS):
        self.rows, self.every = rows, every
        self._fitted_at, self._model = None, None

    def model(self, origin) -> Probabilistic | None:
        origin = pd.Timestamp(origin)
        fresh = self._model is not None and \
            self._fitted_at <= origin < self._fitted_at + pd.Timedelta(weeks=self.every)
        if not fresh:
            try:
                self._model, self._fitted_at = Probabilistic(hist.pool(self.rows, origin)), origin
            except ValueError as exc:
                log.info("no calibrated range at %s: %s", origin.date(), exc)
                return None
        return self._model


# ── honest check on the study's vintage backtest ────────────────────────────

def _auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def backtest(table: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score ranges and rise probabilities on the vintaged study backtest.

    Same rows as rung ``6_gbm_history``: every origin, every study county
    with an official value on the date, features from that date's vintage,
    scored against the final value. Returns (per-row predictions, summary).
    """
    from src.nowcast import features
    feats = features(table)
    keys = table.loc[table["lag"] == 0, ["fips", "as_of", "metro", "ed_resp_final"]]
    refit, point_refit = ProbRefitter(rows), hist.Refitter(rows)
    out = []
    for origin in sorted(feats["as_of"].unique()):
        test = feats[(feats["as_of"] == origin) & feats["last"].notna()]
        if test.empty:
            continue
        model, point = refit.model(origin), point_refit.model(origin)
        if model is None or point is None:
            continue
        lo, hi = model.range(test)
        look = model.outlook(test, n_examples=0)
        out.append(pd.DataFrame({
            "fips": test["fips"].to_numpy(), "as_of": origin, "last": test["last"].to_numpy(float),
            "pred": hist.predict(point, test), "lo": lo, "hi": hi,
            "p_rise": [o.p_rise for o in look], "base_rise": model.base_rise}))
    pred = pd.concat(out, ignore_index=True).merge(keys, on=["fips", "as_of"], how="left")
    pred = pred[pred["ed_resp_final"].notna()].drop_duplicates(["metro", "as_of", "last", "ed_resp_final"])
    truth = pred["ed_resp_final"].to_numpy(float)
    rose = pd.Series(hist._log(truth) - hist._log(pred["last"]) >= RISE, index=pred.index)
    pred["covered"] = (truth >= pred["lo"]) & (truth <= pred["hi"])
    pred["rose"] = rose
    brier = float(np.mean((pred["p_rise"] - rose) ** 2))
    brier_base = float(np.mean((pred["base_rise"] - rose) ** 2))
    summary = pd.DataFrame([{
        "n": len(pred), "target_coverage": 1 - ALPHA, "coverage": float(pred["covered"].mean()),
        "median_width": float((pred["hi"] - pred["lo"]).median()),
        "median_width_rel": float(((pred["hi"] - pred["lo"]) / pred["pred"]).median()),
        "rise_rate": float(rose.mean()), "brier_rise": brier, "brier_base": brier_base,
        "brier_skill": 1 - brier / brier_base if brier_base > 0 else float("nan"),
        "auc_rise": _auc(pred["p_rise"].to_numpy(float), rose.to_numpy(bool))}])
    by_metro = pred.groupby("metro").agg(n=("covered", "size"), coverage=("covered", "mean"),
                                         rise_rate=("rose", "mean")).reset_index()
    return pred, pd.concat([summary.assign(metro="all"), by_metro], ignore_index=True)


def load_metrics(path: Path = METRICS) -> dict | None:
    """The overall backtest row, for findings to quote; ``None`` before it is run."""
    try:
        m = pd.read_csv(path)
        return m[m["metro"] == "all"].iloc[0].to_dict()
    except (OSError, IndexError, KeyError, pd.errors.EmptyDataError):
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Backtest calibrated ranges and analog rise probabilities.")
    p.add_argument("--table", default="data/county_week.parquet")
    p.add_argument("--history", default=str(hist.DEFAULT_CSV))
    p.add_argument("--out", default=str(METRICS))
    a = p.parse_args()
    pred, summary = backtest(pd.read_parquet(a.table), hist.examples(hist.load(a.history)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(a.out, index=False)
    pred.to_csv(Path(a.out).with_name("predictions_uncertainty.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
