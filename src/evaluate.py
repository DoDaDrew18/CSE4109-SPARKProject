"""Score nowcasts against the final values, honestly and in pairs.

Every metric compares a prediction with ``ed_resp_final`` -- the settled value
nobody had on the prediction date -- because that is what the nowcast is
trying to recover. Two reference points make the numbers interpretable:

* **gap closed** = 1 - |pred - final| / |first - final|. The first print is
  what waiting one week buys for free; a nowcast that closes a positive share
  of the first print's error is better than waiting. It is undefined when the
  first print was already final (no gap to close); those weeks are dropped
  from the per-row version rather than counted as 0 or 1. The headline
  number is the aggregate 1 - sum|pred - final| / sum|first - final|, which
  is stable when individual gaps are tiny.
* **paired Wilcoxon** on per county-week absolute errors between two rungs.
  Errors from the same week are strongly correlated (a flu peak is hard for
  every model), so an unpaired comparison of MAEs would mostly measure the
  season, not the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.study import HOME_METRO

# Gaps smaller than this (percentage points of ED visits) count as "no gap".
GAP_EPS = 1e-9


def gap_closed(pred, first, final) -> np.ndarray:
    """Per-row share of the first print's error that the prediction removes.

    1 = exactly final, 0 = no better than the first print, negative = worse.
    NaN where the first print equals the final value (nothing to close) or
    any input is missing.
    """
    pred, first, final = (np.asarray(a, dtype=float) for a in (pred, first, final))
    gap = np.abs(first - final)
    out = np.full(gap.shape, np.nan)
    ok = np.isfinite(pred) & np.isfinite(gap) & (gap > GAP_EPS)
    out[ok] = 1.0 - np.abs(pred[ok] - final[ok]) / gap[ok]
    return out


def gap_closed_total(pred, first, final) -> float:
    """Aggregate gap closed: 1 - sum|pred - final| / sum|first - final|.

    Rows missing any input are dropped. NaN if the first prints had no error
    at all over the rows scored.
    """
    pred, first, final = (np.asarray(a, dtype=float) for a in (pred, first, final))
    ok = np.isfinite(pred) & np.isfinite(first) & np.isfinite(final)
    gap = np.abs(first[ok] - final[ok]).sum()
    if gap <= GAP_EPS:
        return float("nan")
    return float(1.0 - np.abs(pred[ok] - final[ok]).sum() / gap)


def metrics(frame: pd.DataFrame, pred: str = "pred", final: str = "ed_resp_final",
            first: str = "ed_resp_first") -> dict:
    """MAE, RMSE, R^2 vs final, and gap closed, for one set of predictions."""
    p, y = frame[pred].to_numpy(float), frame[final].to_numpy(float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if not len(y):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan,
                "gap_closed": np.nan, "gap_closed_median": np.nan}
    err = p - y
    sst = ((y - y.mean()) ** 2).sum()
    per_row = gap_closed(frame[pred], frame[first], frame[final])
    return {
        "n": int(ok.sum()),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "r2": float(1 - (err ** 2).sum() / sst) if sst > 0 else np.nan,
        "gap_closed": gap_closed_total(frame[pred], frame[first], frame[final]),
        "gap_closed_median": float(np.nanmedian(per_row)) if np.isfinite(per_row).any() else np.nan,
    }


def wilcoxon_paired(err_a, err_b) -> dict:
    """Paired Wilcoxon signed-rank test on two rungs' absolute errors.

    ``median_diff`` < 0 means rung A's errors are smaller. Pairs with a
    missing error are dropped; if every pair ties, there is no evidence
    either way and the p-value is 1 (scipy would raise instead).
    """
    a, b = np.asarray(err_a, dtype=float), np.asarray(err_b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    diff = a - b
    result = {"n": int(len(diff)), "median_diff": float(np.median(diff)) if len(diff) else np.nan,
              "statistic": np.nan, "pvalue": np.nan}
    if not len(diff):
        return result
    if np.all(diff == 0):
        result["pvalue"] = 1.0
        return result
    test = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    result["statistic"], result["pvalue"] = float(test.statistic), float(test.pvalue)
    return result


def dedupe_series(scored: pd.DataFrame) -> pd.DataFrame:
    """Drop county-weeks that are exact copies of another county in the same metro.

    NSSP reports St. Louis city (29510) and St. Louis County (29189) as one
    series -- every published value is identical -- so their predictions and
    labels coincide too. Counting both would weight St. Louis double in the
    pooled metrics and hand the Wilcoxon test pairs that are not independent.
    """
    # Key on the labels, not the prediction: county-level features (wastewater
    # sites, news) can make the two copies' predictions differ slightly, and
    # dropping a copy for one rung but not another would unpair the rungs.
    cols = ["rung", "metro", "week_end", "ed_resp_final", "ed_resp_first"]
    scored = scored.sort_values([c for c in ("fips", "rung", "week_end") if c in scored])
    return scored.drop_duplicates(subset=[c for c in cols if c in scored]).reset_index(drop=True)


def _groups(scored: pd.DataFrame):
    """(group label, subset) pairs: all, home vs comparison, each metro."""
    yield "all", scored
    yield "home", scored[scored["metro"] == HOME_METRO]
    yield "comparison", scored[scored["metro"] != HOME_METRO]
    for metro, part in scored.groupby("metro"):
        yield f"metro:{metro}", part


def summarize(scored: pd.DataFrame, baseline: str = "0_naive") -> pd.DataFrame:
    """Metrics per rung per group; tidy, one row per (group, rung).

    ``skill`` = 1 - MAE / MAE(baseline) on the same rows. The README's gap
    closed is measured against the *first print*, which arrives a week after
    the nowcast is made and is usually within a few percent of final, so for
    a lag-0 nowcast it is strongly negative by construction; ``skill`` is the
    comparison against what was actually available on the date.
    """
    rows = []
    for label, part in _groups(scored):
        base = part[part["rung"] == baseline]
        base_mae = (base["pred"] - base["ed_resp_final"]).abs().mean() if len(base) else np.nan
        for rung, sub in part.groupby("rung"):
            m = metrics(sub)
            rows.append({"group": label, "rung": rung, **m,
                         "skill": 1 - m["mae"] / base_mae if base_mae else np.nan})
    return pd.DataFrame(rows)


def ladder_tests(scored: pd.DataFrame, baseline: str = "0_naive", order=None) -> pd.DataFrame:
    """Wilcoxon of each rung against the baseline and against the rung below.

    ``order`` is the ladder order (default: sorted names); "the rung below"
    is the previous present rung in that order.
    """
    err = scored.assign(abs_err=(scored["pred"] - scored["ed_resp_final"]).abs())
    wide = err.pivot_table(index=["fips", "week_end"], columns="rung", values="abs_err")
    rungs = [r for r in (order or sorted(wide.columns)) if r in wide.columns]
    rows = []
    for below, rung in zip(rungs, rungs[1:]):
        for other in dict.fromkeys((baseline, below)):
            if other in wide and other != rung:
                rows.append({"rung": rung, "vs": other,
                             **wilcoxon_paired(wide[rung], wide[other])})
    return pd.DataFrame(rows, columns=["rung", "vs", "n", "median_diff", "statistic", "pvalue"])


def official_coverage(table: pd.DataFrame, recent_lags: int = 4) -> pd.DataFrame:
    """Per county: on how many prediction dates was there recent official data?

    ``dates_with_data`` counts as_of Saturdays with any ED value at lags
    1..``recent_lags``; ``first_as_of`` is the first such date. The first-print
    lag of a week is the lag at which it first appears across the as_of grid
    (1 = on time); its distribution shows the archive holes per county.
    """
    rows = []
    for fips, part in table.groupby("fips"):
        recent = part[part["lag"].between(1, recent_lags)]
        has = recent.groupby("as_of")["ed_resp"].apply(lambda s: s.notna().any())
        seen = part[part["ed_resp"].notna() & (part["lag"] >= 1)]
        first_lag = seen.groupby("week_end")["lag"].min()
        # Only weeks that could have been first printed on time inside the grid.
        first_lag = first_lag[first_lag.index + pd.Timedelta(weeks=1) >= part["as_of"].min()]
        rows.append({
            "fips": fips, "metro": part["metro"].iloc[0], "as_of_dates": int(len(has)),
            "dates_with_data": int(has.sum()),
            "first_as_of": has[has].index.min().date() if has.any() else None,
            "weeks_first_printed": int(len(first_lag)),
            "first_print_lag1": float((first_lag == 1).mean()) if len(first_lag) else np.nan,
            "first_print_lag_median": float(first_lag.median()) if len(first_lag) else np.nan,
            "first_print_lag_max": int(first_lag.max()) if len(first_lag) else None,
        })
    return pd.DataFrame(rows)


def revision_profile(table: pd.DataFrame, settle_weeks: int = 8) -> pd.DataFrame:
    """How far a value known at lag k sits from the final value, per lag.

    This is the evidence for ``src.nowcast.LABEL_LAG``: training labels taken
    at a lag where the median relative revision is already small are nearly
    final, while lag-1 labels would teach the model to predict first prints.
    """
    settled = table["week_end"] <= table["as_of"].max() - pd.Timedelta(weeks=settle_weeks)
    known = table[settled & (table["lag"] >= 1) & table["ed_resp"].notna()
                  & table["ed_resp_final"].notna()]
    rel = (known["ed_resp"] - known["ed_resp_final"]) / known["ed_resp_final"].where(
        known["ed_resp_final"] != 0)
    frame = known.assign(rel=rel, abs_rel=rel.abs())
    out = frame.groupby("lag").agg(n=("rel", "size"), median_rel=("rel", "median"),
                                   median_abs_rel=("abs_rel", "median"),
                                   share_over_5pct=("abs_rel", lambda s: float((s > 0.05).mean())))
    return out.reset_index().query("lag <= 12")
