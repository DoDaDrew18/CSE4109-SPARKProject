"""Check the nowcast results before anyone reads them as findings.

Usage, from the repo root (after ``python -m src.nowcast``):

    python -m src.validate_results                 # checks 1-3, seconds
    python -m src.validate_results --placebo 19    # + shuffled-news test, ~1 min per run

Writes ``data/results/validation_report.md`` plus the tables behind it, and
exits 1 if any prediction fails a hard check. ``src.evaluate`` scores the
models; this module asks whether the scores can be believed:

1. **Predictions are sane.** Every prediction is a share of ED visits, so it
   must be finite and non-negative, and one far above anything ever observed
   is a model blowing up, not a forecast. Hard failures: non-finite,
   negative, or above ``CEILING_FACTOR`` x the largest observed value.
   Missing predictions are a warning (ARIMA skips origins with too little
   history by design), counted per rung.
2. **p-values survive multiple testing.** Each ``wilcoxon_*.csv`` (per group
   for leave-one-metro-out) and the ablation table is one family of tests;
   Holm-adjusted p-values are reported next to the raw ones. A rung "beats"
   another only if the adjusted p is below ``ALPHA``.
3. **Groups are big enough.** Metric rows scored on fewer than ``MIN_N``
   county-weeks are listed: their MAE and R^2 are anecdotes, not results.
4. **News helps because of what the articles say** (``--placebo K``). The
   news features are shuffled across weeks within each county -- same
   counts and concern levels, wrong timing -- and rungs 11 and 12 are rerun
   K times. If the real news gain is no bigger than the shuffled gains, the
   improvement of rung 12 over 11 is extra parameters, not information.
   p = (1 + #shuffles with gain >= real gain) / (K + 1); K = 19 gives a
   smallest possible p of 0.05.
5. **The test could see a real effect at all** (``--inject``). The mirror
   image of the placebo: real news is replaced by *synthetic* news whose
   timing is known to carry information, and rungs 11 and 12 are rerun. Each
   county gets ``scale`` x as many article-weeks as it really has (same
   sparsity at scale 1); a share ``precision`` of them land on weeks where
   the final ED share rose most (sampled from the county's top-decile
   week-over-week rises), the rest on random weeks. Every synthetic
   article-week has the same count and concern (the real corpus medians), so
   only the timing is informative. Power = share of repeats where rung 12
   beats 11 with Wilcoxon p < ``ALPHA``. If even precision 1 at scale 1 is
   rarely detected, the real-news test was underpowered and its null (or
   its p = 0.036) says little either way; precision 0 is a built-in
   placebo and should show power near ``ALPHA``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate
from src import history as hist_mod
from src import nowcast
from src.align import OUTPUT as TABLE_PATH

log = logging.getLogger(__name__)

CEILING_FACTOR = 2.0
MIN_N = 30
ALPHA = 0.05
NEWS_COLUMNS = ("news_n", "news_concern")
PLACEBO_RUNGS = ("11_gbm_local", "12_gbm_news")


# -- 1. predictions ---------------------------------------------------------------

def check_predictions(preds: pd.DataFrame, ceiling: float, source: str = "") -> pd.DataFrame:
    """One row per problem prediction: severity, problem, and the row itself."""
    pred = preds["pred"].to_numpy(float)
    problems = [
        ("warning", "missing", np.isnan(pred)),
        ("error", "non-finite", np.isinf(pred)),
        ("error", "negative", pred < 0),
        ("error", f"above {ceiling:.1f} ({CEILING_FACTOR:g}x observed max)",
         np.isfinite(pred) & (pred > ceiling)),
    ]
    cols = [c for c in ("rung", "fips", "as_of", "pred", "ed_resp_final") if c in preds]
    out = [preds.loc[mask, cols].assign(severity=sev, problem=what, source=source)
           for sev, what, mask in problems if mask.any()]
    if not out:
        return pd.DataFrame(columns=["source", "severity", "problem", *cols])
    return pd.concat(out, ignore_index=True)[["source", "severity", "problem", *cols]]


def ceiling_of(*frames: pd.DataFrame) -> float:
    """``CEILING_FACTOR`` x the largest ED share in any answer-key column."""
    seen = [f[c].max() for f in frames for c in ("ed_resp_final", "ed_resp_first") if c in f]
    return CEILING_FACTOR * float(np.nanmax(seen))


# -- 2. multiple testing ----------------------------------------------------------

def adjust_tests(tests: pd.DataFrame, source: str, by: str | None = None) -> pd.DataFrame:
    """Add ``pvalue_holm`` and ``significant``; one family per file or per ``by``."""
    tests = tests.copy()
    tests["pvalue_holm"] = np.nan
    families = tests.groupby(by).groups.values() if by else [tests.index]
    for idx in families:
        tests.loc[idx, "pvalue_holm"] = evaluate.holm(tests.loc[idx, "pvalue"])
    tests["significant"] = tests["pvalue_holm"] < ALPHA
    tests["lost_significance"] = (tests["pvalue"] < ALPHA) & ~tests["significant"]
    return tests.assign(source=source)


# -- 3. sample sizes --------------------------------------------------------------

def small_groups(metrics: pd.DataFrame, source: str) -> pd.DataFrame:
    """Metric groups scored on fewer than ``MIN_N`` county-weeks."""
    if metrics.empty or "n" not in metrics:
        return pd.DataFrame(columns=["source", "group", "n"])
    key = "group" if "group" in metrics else "variant"
    small = metrics[metrics["n"] < MIN_N].groupby(key, as_index=False)["n"].max()
    return small.rename(columns={key: "group"}).assign(source=source)[["source", "group", "n"]]


# -- 4. shuffled news -------------------------------------------------------------

class OriginCache:
    """A ``Refitter`` whose models are remembered per origin.

    Reusing a plain ``Refitter`` for a second run would hand early origins a
    model fitted at a later one (a leak). Here the first run fits exactly as
    usual and every later run gets, for each origin, the model the first run
    used there. The GBM never sees news, so the shuffles cannot change it.
    """

    def __init__(self, history: pd.DataFrame):
        self.refit, self._seen = hist_mod.Refitter(history), {}

    def model(self, origin, *args, **kwargs):
        key = (pd.Timestamp(origin), args, tuple(sorted(kwargs.items())))
        if key not in self._seen:
            self._seen[key] = self.refit.model(origin, *args, **kwargs)
        return self._seen[key]


def shuffle_news(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute each county's weekly news values across weeks.

    ``news_n`` and ``news_concern`` move together, and every as_of/lag row
    of a week gets the same shuffled values, so the per-county distribution
    is unchanged and only the timing is broken.
    """
    weekly = table.drop_duplicates(["fips", "week_end"])[["fips", "week_end", *NEWS_COLUMNS]]
    parts = []
    for _, part in weekly.groupby("fips"):
        values = part[list(NEWS_COLUMNS)].to_numpy()[rng.permutation(len(part))]
        parts.append(part[["fips", "week_end"]].assign(**dict(zip(NEWS_COLUMNS, values.T))))
    shuffled = pd.concat(parts, ignore_index=True)
    return table.drop(columns=list(NEWS_COLUMNS)).merge(shuffled, on=["fips", "week_end"],
                                                        how="left")[table.columns]


def news_comparison(table: pd.DataFrame, refitter, settle_weeks: int) -> dict:
    """Rung 12 vs 11 on settled official county-weeks.

    ``gain_mae`` = MAE(11) - MAE(12) (positive: news helps), plus the paired
    Wilcoxon test of rung 12's absolute errors against rung 11's.
    """
    rungs = [r for r in nowcast.LADDER if r.name in PLACEBO_RUNGS]
    preds = nowcast.rolling_origin(table, rungs=rungs, refitter=refitter)
    scored = evaluate.dedupe_series(nowcast.scoreable(preds, settle_weeks))
    err = scored.assign(err=(scored["pred"] - scored["ed_resp_final"]).abs()).pivot_table(
        index=["fips", "as_of"], columns="rung", values="err")
    base, news = err[PLACEBO_RUNGS[0]], err[PLACEBO_RUNGS[1]]
    test = evaluate.wilcoxon_paired(news, base)
    return {"gain_mae": float(base.mean() - news.mean()), "n": int(len(err)),
            "median_diff": test["median_diff"], "pvalue": test["pvalue"]}


def news_gain(table: pd.DataFrame, refitter, settle_weeks: int) -> tuple[float, int]:
    """MAE(rung 11) - MAE(rung 12) on settled official county-weeks, and n."""
    result = news_comparison(table, refitter, settle_weeks)
    return result["gain_mae"], result["n"]


def placebo_news(table: pd.DataFrame, history: pd.DataFrame, k: int, seed: int = 0,
                 settle_weeks: int = nowcast.SETTLE_WEEKS) -> pd.DataFrame:
    """Real news gain (run 0) and ``k`` gains with shuffled news."""
    cache, rng = OriginCache(history), np.random.default_rng(seed)
    rows = []
    for run in range(k + 1):
        frame = table if run == 0 else shuffle_news(table, rng)
        gain, n = news_gain(frame, cache, settle_weeks)
        rows.append({"run": run, "news": "real" if run == 0 else "shuffled",
                     "gain_mae": gain, "n": n})
        log.info("placebo run %d/%d: gain %.4f", run, k, gain)
    return pd.DataFrame(rows)


def placebo_pvalue(runs: pd.DataFrame) -> float:
    real = runs.loc[runs["news"] == "real", "gain_mae"].iloc[0]
    fake = runs.loc[runs["news"] == "shuffled", "gain_mae"]
    return float((1 + (fake >= real).sum()) / (len(fake) + 1))


# -- 5. synthetic news (injection-recovery) ---------------------------------------

RISE_QUANTILE = 0.9      # "peak weeks" = a county's top decile of week-over-week rises


def _weekly(table: pd.DataFrame) -> pd.DataFrame:
    cols = ["fips", "week_end", "ed_resp_final", *[c for c in NEWS_COLUMNS if c in table]]
    return table.drop_duplicates(["fips", "week_end"])[cols].sort_values(["fips", "week_end"])


def inject_news(table: pd.DataFrame, precision: float, rng: np.random.Generator,
                scale: float = 1.0, news_n: float | None = None,
                concern: float | None = None) -> pd.DataFrame:
    """Replace the news columns with synthetic news of known informativeness.

    Per county: ``k = round(scale x real article-weeks)`` (at least 1);
    ``round(precision x k)`` of them go to weeks sampled from the county's
    top-``RISE_QUANTILE`` week-over-week log rises in the final ED share
    (topped up with the next-largest rises if the decile is too small), the
    rest to random other weeks. All get the same ``news_n`` and ``concern``
    (default: the real corpus medians); every other week has no news, as in
    the real table. The answer key is read only to *place* the synthetic
    signal: this tests the pipeline, never feeds a reported model.
    """
    weekly = _weekly(table)
    real = weekly[weekly["news_n"].fillna(0) > 0] if "news_n" in weekly else weekly.iloc[:0]
    news_n = float(real["news_n"].median()) if news_n is None and len(real) else (news_n or 1.0)
    concern = (float(real["news_concern"].median()) if concern is None and len(real)
               else (concern or 2.0))
    parts = []
    for fips, part in weekly.groupby("fips"):
        part = part.reset_index(drop=True)
        k = max(1, int(round(scale * (real["fips"] == fips).sum())))
        k = min(k, len(part))
        rise = np.log(part["ed_resp_final"].where(part["ed_resp_final"] > 0)).diff()
        ranked = rise.dropna().sort_values(ascending=False).index.to_numpy()
        n_hit = min(int(round(precision * k)), len(ranked))
        pool = ranked[:max(n_hit, int(np.ceil((1 - RISE_QUANTILE) * len(ranked))))]
        hits = rng.choice(pool, n_hit, replace=False) if n_hit else np.array([], int)
        rest = np.setdiff1d(part.index.to_numpy(), hits)
        other = rng.choice(rest, min(k - n_hit, len(rest)), replace=False)
        chosen = np.concatenate([hits, other]).astype(int)
        out = part[["fips", "week_end"]].assign(news_n=np.nan, news_concern=np.nan)
        out.loc[chosen, "news_n"], out.loc[chosen, "news_concern"] = news_n, concern
        parts.append(out)
    fake = pd.concat(parts, ignore_index=True)
    base = table.drop(columns=[c for c in NEWS_COLUMNS if c in table])
    columns = list(table.columns) + [c for c in NEWS_COLUMNS if c not in table]
    return base.merge(fake, on=["fips", "week_end"], how="left")[columns]


def inject_recovery(table: pd.DataFrame, history: pd.DataFrame, precisions=(0.0, 0.5, 1.0),
                    scales=(1.0,), reps: int = 5, seed: int = 0,
                    settle_weeks: int = nowcast.SETTLE_WEEKS, refitter=None) -> pd.DataFrame:
    """One row per (scale, precision, repeat): gain, n, Wilcoxon p."""
    cache, rng = refitter or OriginCache(history), np.random.default_rng(seed)
    rows = []
    for scale in scales:
        for precision in precisions:
            for rep in range(reps):
                frame = inject_news(table, precision, rng, scale)
                result = news_comparison(frame, cache, settle_weeks)
                rows.append({"scale": scale, "precision": precision, "rep": rep, **result})
                log.info("inject scale %g precision %g rep %d: gain %.4f p %.3g",
                         scale, precision, rep, result["gain_mae"], result["pvalue"])
    return pd.DataFrame(rows)


def inject_power(runs: pd.DataFrame) -> pd.DataFrame:
    """Per (scale, precision): mean/median gain and power at ``ALPHA``."""
    hit = (runs["pvalue"] < ALPHA) & (runs["gain_mae"] > 0)
    return (runs.assign(detected=hit)
            .groupby(["scale", "precision"], as_index=False)
            .agg(reps=("rep", "size"), gain_mean=("gain_mae", "mean"),
                 gain_median=("gain_mae", "median"), p_median=("pvalue", "median"),
                 power=("detected", "mean"), n=("n", "max")))


# -- report -----------------------------------------------------------------------

def _md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_none_\n\n"
    cell = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "|" + " --- |" * len(frame.columns)]
    lines += ["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n\n"


def report(issues, tests, small, placebo, injected=None) -> str:
    errors = issues[issues["severity"] == "error"]
    missing = (issues[issues["problem"] == "missing"]
               .groupby(["source", "rung"], as_index=False).size()
               .rename(columns={"size": "missing"}))
    lost = tests[tests["lost_significance"]][["source", "rung", "vs", "n", "median_diff",
                                              "pvalue", "pvalue_holm"]]
    out = ["# Result validation\n\n",
           f"Generated by `python -m src.validate_results`. Hard failures: **{len(errors)}**.\n\n",
           "## 1. Implausible predictions (errors)\n\n", _md(errors),
           "## Missing predictions (warnings)\n\n", _md(missing),
           f"## 2. Significant before Holm correction, not after (alpha = {ALPHA})\n\n",
           f"{int(tests['significant'].sum())} of {int(tests['pvalue'].notna().sum())} tests "
           "remain significant after correction; all adjusted p-values are in "
           "`validation_tests.csv`.\n\n", _md(lost),
           f"## 3. Groups scored on fewer than {MIN_N} county-weeks\n\n",
           "Treat these metrics as anecdotes.\n\n", _md(small),
           "## 4. Shuffled-news placebo\n\n"]
    if placebo is None:
        out.append("_not run; use `--placebo 19`_\n")
    else:
        real = placebo.loc[placebo["news"] == "real", "gain_mae"].iloc[0]
        fake = placebo.loc[placebo["news"] == "shuffled", "gain_mae"]
        out.append(f"MAE gain of rung 12 over 11 with real news: **{real:.4f}**; with "
                   f"shuffled news: median {fake.median():.4f}, max {fake.max():.4f} "
                   f"over {len(fake)} shuffles. Placebo p = **{placebo_pvalue(placebo):.3f}**.\n")
    out.append("\n## 5. Synthetic-news recovery (power)\n\n")
    if injected is None:
        out.append("_not run; use `--inject`_\n")
    else:
        out.append("Real news replaced by synthetic article-weeks; `precision` = share placed "
                   "on top-decile ED rises, `scale` = article-weeks relative to the real "
                   f"corpus. Power = share of repeats with rung 12 better than 11 at "
                   f"p < {ALPHA}. Precision 0 is a placebo (expect power near {ALPHA}).\n\n")
        out.append(_md(inject_power(injected)))
    return "".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate the nowcast results.")
    parser.add_argument("--results", default=str(nowcast.RESULTS))
    parser.add_argument("--table", default=str(TABLE_PATH))
    parser.add_argument("--history", default=str(hist_mod.DEFAULT_CSV))
    parser.add_argument("--placebo", type=int, default=0, metavar="K",
                        help="shuffled-news runs (0 = skip; 19 for p >= 0.05 resolution)")
    parser.add_argument("--inject", action="store_true",
                        help="synthetic-news injection-recovery test (power of rung 12 vs 11)")
    parser.add_argument("--inject-precision", default="0,0.5,1",
                        help="comma list: share of synthetic article-weeks on peak rises")
    parser.add_argument("--inject-scale", default="1",
                        help="comma list: article-weeks as a multiple of the real corpus")
    parser.add_argument("--inject-reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = Path(args.results)

    preds = {name: pd.read_csv(res / f"{name}.csv")
             for name in ("predictions", "predictions_lomo") if (res / f"{name}.csv").exists()}
    if not preds:
        log.error("no predictions in %s; run python -m src.nowcast first", res)
        return 2
    ceiling = ceiling_of(*preds.values())
    issues = pd.concat([check_predictions(p, ceiling, name) for name, p in preds.items()],
                       ignore_index=True)

    tests, small = [], []
    for path in sorted(res.glob("wilcoxon_*.csv")):
        frame = pd.read_csv(path)
        if not frame.empty:
            tests.append(adjust_tests(frame, path.stem, "group" if "group" in frame else None))
    if (res / "metrics_ablation.csv").exists():
        abl = pd.read_csv(res / "metrics_ablation.csv").rename(columns={"variant": "rung"})
        tests.append(adjust_tests(abl[["rung", "vs", "n", "pvalue"]].dropna(subset=["pvalue"]),
                                  "metrics_ablation"))
    for path in sorted(res.glob("metrics_*.csv")):
        small.append(small_groups(pd.read_csv(path), path.stem))
    tests = pd.concat(tests, ignore_index=True)
    small = pd.concat(small, ignore_index=True)

    placebo = injected = None
    if args.placebo or args.inject:
        if not Path(args.history).exists():
            log.error("placebo/inject need the CDC history CSV at %s", args.history)
            return 2
        table = pd.read_parquet(args.table)
        if not set(NEWS_COLUMNS) <= set(table.columns):
            log.error("no news columns in %s; build it with python -m src.align --news-source",
                      args.table)
            return 2
        history = hist_mod.examples(hist_mod.load(args.history))
        cache = OriginCache(history)
        if args.placebo:
            placebo = placebo_news(table, history, args.placebo, args.seed)
            placebo.to_csv(res / "placebo_news.csv", index=False)
        if args.inject:
            floats = lambda text: tuple(float(v) for v in text.split(","))
            injected = inject_recovery(table, history, floats(args.inject_precision),
                                       floats(args.inject_scale), args.inject_reps,
                                       args.seed, refitter=cache)
            injected.to_csv(res / "inject_news.csv", index=False)

    issues.to_csv(res / "validation_issues.csv", index=False)
    tests.to_csv(res / "validation_tests.csv", index=False)
    text = report(issues, tests, small, placebo, injected)
    (res / "validation_report.md").write_text(text)
    print(text)
    print(f"wrote {res}/validation_report.md")
    return 1 if (issues["severity"] == "error").any() else 0


if __name__ == "__main__":
    sys.exit(main())
