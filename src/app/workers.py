"""Data workers: each reads one source for the selected counties, as of a date.

Ported in spirit from CalHacks 13's ``lib/agents/workers/*.ts``: a worker has
an id, a display name, a one-line description (the planner reads it) and a
``run(ctx)`` that returns findings. What changed for a study: every finding
carries ``known_by``, the date its information was public, and
``validate.check_finding`` drops any finding whose ``known_by`` is after the
question's date. A worker that cannot honestly answer for that date says so
in a finding instead of quietly using newer data.

Workers
-------
ed_visits    NSSP % of ED visits (COVID, flu, RSV) as published on the date
nowcast      the study's best model (HistGBM on national history) for this week,
             with a calibrated 80% range (``src.uncertainty``, conformalized quantile GBM)
outlook      probability of a 10%+ rise from the nearest analogs in national history
hotspots     Getis-Ord Gi* of the region against its neighbors, same vintage
wastewater   CDC NWSS levels from honest vintages (from 2026-04-17), else a labeled proxy
weather      Open-Meteo ERA5 temperature / dew point / humidity, 7-day availability lag
air_quality  Open-Meteo CAMS PM2.5 and ozone (model-based, not vintaged)
news         the study's LLM-labeled local articles + a live GDELT title search
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from src.app.data import nssp_vintage
from src.app.region import Region, counties
from src.study import METROS, week_end_of

log = logging.getLogger(__name__)

__all__ = ["Finding", "Context", "Worker", "WORKERS", "run_worker"]

_UA = {"User-Agent": "CSE4109-SPARK/region-ask"}
HTTP_TIMEOUT = 20


@dataclass
class Finding:
    worker: str
    claim: str
    category: str
    known_by: str                         # ISO date the information was public
    source: str
    source_url: str = ""
    severity: float = 0.3                 # 0 calm .. 1 alarming
    confidence: float = 0.7
    fips: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    id: str = ""

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Context:
    region: Region
    as_of: pd.Timestamp
    question: str = ""
    log: Callable[[str], None] = lambda msg: None
    emit_layer: Callable[[dict], None] = lambda layer: None


@dataclass
class Worker:
    id: str
    name: str
    description: str
    run: Callable[[Context], list[Finding]]
    sources: tuple[str, ...] = ()


def _pct(x: float) -> str:
    return f"{x:.2f}%"


def _weighted(frame: pd.DataFrame, column: str, region: Region) -> float:
    pop = counties().set_index("fips")["population"].astype(float)
    w = frame["fips"].map(pop).fillna(1.0)
    ok = frame[column].notna()
    return float(np.average(frame.loc[ok, column], weights=w[ok])) if ok.any() else float("nan")


def _county_name(region: Region, fips: str) -> str:
    return region.names.get(fips) or fips


# ── ED visits ────────────────────────────────────────────────────────────────

def run_ed(ctx: Context) -> list[Finding]:
    region, out = ctx.region, []
    ctx.log(f"Reading NSSP ED data for {len(region.fips)} counties as published on "
            f"{ctx.as_of.date()}")
    wide, meta = nssp_vintage(region.fips, ctx.as_of)
    ctx.log(f"Issue {meta['issue']} (epiweek {meta['epiweek']}) from {meta['source']}; "
            f"{wide['fips'].nunique()} counties with data")
    url = "https://data.cdc.gov/Public-Health-Surveillance/NSSP-Emergency-Department-Visit-Trajectories-by-St/rdmq-nq56"
    if meta.get("note"):
        out.append(Finding("ed_visits", meta["note"], "data_gap", meta["issue"], "CDC NSSP via Delphi Epidata",
                           "https://cmu-delphi.github.io/delphi-epidata/", severity=0.2, confidence=0.95))
    if meta.get("error"):
        out.append(Finding("ed_visits", f"Delphi request failed ({meta['error'][:100]}); only the study's "
                           "saved snapshots were available, which cover 7 study counties.", "data_gap",
                           meta["issue"], "CDC NSSP via Delphi Epidata", severity=0.2, confidence=0.9))
    if meta["no_data"]:
        names = [_county_name(region, f) for f in meta["no_data"][:6]]
        more = len(meta["no_data"]) - len(names)
        mo = any(f.startswith("29") for f in meta["no_data"]) and ctx.as_of < pd.Timestamp("2026-06-13")
        out.append(Finding(
            "ed_visits", f"{len(meta['no_data'])} of {len(region.fips)} counties had no NSSP ED data "
            f"published as of {meta['issue']}: {', '.join(names)}{f' and {more} more' if more else ''}"
            + (". NSSP published nothing for Missouri counties until June 2026." if mo else "."),
            "data_gap", meta["issue"], "CDC NSSP via Delphi Epidata", url, severity=0.2,
            confidence=0.95, fips=meta["no_data"], data={"no_data": meta["no_data"]}))
    if wide.empty:
        return out

    latest_week = wide["week_end"].max()
    lag_days = (ctx.as_of - latest_week).days
    last = wide[wide["week_end"] == latest_week]
    ctx.emit_layer({"kind": "ed", "week_end": latest_week.date().isoformat(),
                    "values": {f: round(float(v), 2) for f, v in zip(last["fips"], last["resp_pct"])}})

    agg = wide.groupby("week_end").apply(
        lambda g: pd.Series({c: _weighted(g, c, region) for c in ("resp_pct", "covid", "influenza", "rsv")}),
        include_groups=False).sort_index()
    now = agg.iloc[-1]
    prior = agg.iloc[-4] if len(agg) >= 4 else agg.iloc[0]
    change = (now["resp_pct"] / prior["resp_pct"] - 1) * 100 if prior["resp_pct"] > 0 else float("nan")
    lead = max(("covid", "influenza", "rsv"), key=lambda c: now[c])
    trend = ("rising" if change > 15 else "falling" if change < -15 else "roughly flat")
    sev = min(0.95, 0.15 + now["resp_pct"] / 15 + max(0.0, change) / 200)
    out.append(Finding(
        "ed_visits",
        f"Region (population-weighted): {_pct(now['resp_pct'])} of ED visits were respiratory "
        f"(COVID {_pct(now['covid'])}, flu {_pct(now['influenza'])}, RSV {_pct(now['rsv'])}) in the week "
        f"ending {latest_week.date()}, {trend} ({change:+.0f}% vs 3 weeks earlier). "
        f"{lead.capitalize() if lead != 'influenza' else 'Flu'} is the largest share.",
        "ed", meta["issue"], "CDC NSSP via Delphi Epidata", url, severity=round(sev, 2), confidence=0.85,
        fips=sorted(last["fips"]), data={"series": {d.date().isoformat(): round(float(v), 3)
                                                    for d, v in agg["resp_pct"].items()},
                                         "change_pct_3w": None if np.isnan(change) else round(change, 1)}))
    out.append(Finding(
        "ed_visits", f"The newest ED week published on {ctx.as_of.date()} ended {latest_week.date()}, "
        f"{lag_days} days earlier; early values are often revised upward later (at the Jan 2026 peak "
        f"19% of counties' first print moved by 10% or more).",
        "data_gap", meta["issue"], "CDC NSSP via Delphi Epidata (revision study in this repo)", url,
        severity=0.15, confidence=0.9, data={"lag_days": lag_days}))

    for fips in region.fips[:5]:
        s = wide[wide["fips"] == fips].set_index("week_end")["resp_pct"]
        if len(s) < 2:
            continue
        ch = (s.iloc[-1] / s.iloc[-4] - 1) * 100 if len(s) >= 4 and s.iloc[-4] > 0 else float("nan")
        out.append(Finding(
            "ed_visits", f"{_county_name(region, fips)}: {_pct(s.iloc[-1])} respiratory ED share "
            f"(week ending {s.index[-1].date()})" + ("" if np.isnan(ch) else f", {ch:+.0f}% vs 3 weeks earlier"),
            "ed", meta["issue"], "CDC NSSP via Delphi Epidata", url,
            severity=round(min(0.9, 0.1 + s.iloc[-1] / 15 + max(0.0, 0 if np.isnan(ch) else ch) / 200), 2),
            confidence=0.8, fips=[fips], data={"value": round(float(s.iloc[-1]), 3)}))
    return out


# ── Nowcast ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _history_rows():
    from src import history as hist
    path = Path(hist.DEFAULT_CSV)
    if not path.exists():
        return None
    return hist.examples(hist.load(path))


@lru_cache(maxsize=1)
def _refitter():
    from src import history as hist
    rows = _history_rows()
    return None if rows is None else hist.Refitter(rows)


@lru_cache(maxsize=1)
def _prob_refitter():
    from src.uncertainty import ProbRefitter
    rows = _history_rows()
    return None if rows is None else ProbRefitter(rows)


MAX_NOWCAST_GAP_DAYS = 21     # target week vs newest published week; backtest lags were 7-14 days
_SRC_HIST = "This study (src/history.py)"


def _published(ctx: Context, worker: str):
    """The area's published series for the nowcast and outlook, or a data-gap finding.

    Returns ``(feats, region_feats, target, meta, gaps)``: one feature row
    per county and one for the population-weighted area series (``last``,
    ``y_lag2``, ``y_lag3``, ``as_of``), or ``gaps`` explaining why there is none.
    """
    wide, meta = nssp_vintage(ctx.region.fips, ctx.as_of)
    wide = wide[wide["resp_pct"].notna()] if not wide.empty else wide
    if wide.empty:
        return None, None, None, meta, [Finding(
            worker, f"No {worker}: no county in the area had ED data published as of {meta['issue']}.",
            "data_gap", meta["issue"], _SRC_HIST, severity=0.2, confidence=0.9)]
    target = week_end_of(ctx.as_of)
    gap = (target - wide["week_end"].max()).days
    if gap > MAX_NOWCAST_GAP_DAYS:
        return None, None, None, meta, [Finding(
            worker, f"No {worker}: the newest published ED week ended {wide['week_end'].max().date()}, "
            f"{gap} days before the target week. The model learns a one-step correction for the usual "
            "1-2 week reporting lag; projecting further would be a forecast it was never tested on.",
            "data_gap", meta["issue"], _SRC_HIST, severity=0.2, confidence=0.9)]

    def row(s: np.ndarray, fips: str) -> dict:
        return {"fips": fips, "last": s[-1], "y_lag2": s[-2] if len(s) > 1 else np.nan,
                "y_lag3": s[-3] if len(s) > 2 else np.nan, "as_of": ctx.as_of}

    feats = pd.DataFrame([row(g.sort_values("week_end")["resp_pct"].to_numpy(float), fips)
                          for fips, g in wide.groupby("fips")])
    pop = counties().set_index("fips")["population"].astype(float)
    w = wide.assign(w=wide["fips"].map(pop).fillna(1.0).clip(lower=1.0))
    area = (w.assign(wx=w["resp_pct"] * w["w"]).groupby("week_end")[["wx", "w"]].sum()
              .pipe(lambda t: t["wx"] / t["w"]).sort_index())
    return feats, pd.DataFrame([row(area.to_numpy(float), "area")]), target, meta, []


def _backtest_note() -> str:
    from src.uncertainty import load_metrics
    m = load_metrics()
    if not m:
        return ""
    return (f" In the study's vintage backtest the 80% range held the final value in "
            f"{m['coverage']:.0%} of {int(m['n'])} county-weeks.")


def run_nowcast(ctx: Context) -> list[Finding]:
    from src import history as hist
    refit = _refitter()
    if refit is None:
        ctx.log("No CDC history CSV (raw/cdc_nssp/); the nowcast model cannot be fitted.")
        return []
    feats, area, target, meta, gaps = _published(ctx, "nowcast")
    if gaps:
        return gaps
    ctx.log(f"Fitting gradient boosting on national history settled 8+ weeks before {ctx.as_of.date()}")
    model = refit.model(ctx.as_of)
    if model is None:
        return []
    feats["pred"] = hist.predict(model, feats)
    last_w, pred_w = _weighted(feats, "last", ctx.region), _weighted(feats, "pred", ctx.region)
    change = (pred_w / last_w - 1) * 100 if last_w > 0 else float("nan")

    ctx.log("Fitting calibrated 10th/90th-percentile models (conformalized quantile regression)")
    prob = _prob_refitter().model(ctx.as_of)
    layer = {"kind": "nowcast", "week_end": target.date().isoformat(),
             "values": {f: round(float(v), 2) for f, v in zip(feats["fips"], feats["pred"])}}
    data = {"target_week": target.date().isoformat(), "pred": round(pred_w, 3), "last": round(last_w, 3),
            "per_county": {f: round(float(p), 3) for f, p in zip(feats["fips"], feats["pred"])}}
    span = ""
    if prob is not None:
        lo, hi = prob.range(feats)
        layer["lo"] = {f: round(float(v), 2) for f, v in zip(feats["fips"], lo)}
        layer["hi"] = {f: round(float(v), 2) for f, v in zip(feats["fips"], hi)}
        a_lo, a_hi = (float(v[0]) for v in prob.range(area))
        a_lo, a_hi = min(a_lo, pred_w), max(a_hi, pred_w)
        data.update(range_80=[round(a_lo, 3), round(a_hi, 3)], range_calibrated_on=f"{prob.n_cal} HSA-weeks "
                    f"from {prob.cal_from.date()}")
        span = f" (80% range {_pct(a_lo)}-{_pct(a_hi)})"
    ctx.emit_layer(layer)
    return [Finding(
        "nowcast",
        f"Nowcast for the week ending {target.date()}: {_pct(pred_w)}{span} respiratory ED share for the region "
        f"({change:+.0f}% vs the newest published {_pct(last_w)}). In the study's backtest this model beat "
        f"carrying the last value forward by 19% (MAE) on four metros, but it lags sharp peak onsets."
        + (_backtest_note() if span else ""),
        "nowcast", ctx.as_of.date().isoformat(), "This study: HistGBM on CDC rdmq-nq56 history (src/history.py)",
        severity=round(min(0.95, 0.15 + pred_w / 15 + max(0.0, change) / 150), 2), confidence=0.6,
        fips=feats["fips"].tolist(), data=data)]


def _hsa_label(fips_list: list[str]) -> str:
    c = counties().set_index("fips")
    known = [f for f in fips_list if f in c.index]
    if not known:
        return "an unnamed HSA"
    top = c.loc[known].sort_values("population", ascending=False).iloc[0]
    more = len(known) - 1
    return f"{top['name']}, {top['state']}" + (f" (+{more} count{'y' if more == 1 else 'ies'})" if more else "")


def run_outlook(ctx: Context) -> list[Finding]:
    from src.uncertainty import RISE, load_metrics
    if _prob_refitter() is None:
        ctx.log("No CDC history CSV (raw/cdc_nssp/); the outlook cannot be fitted.")
        return []
    feats, area, target, meta, gaps = _published(ctx, "outlook")
    if gaps:
        return gaps
    ctx.log(f"Searching national history (675 HSAs, settled 8+ weeks before {ctx.as_of.date()}) for analogs")
    prob = _prob_refitter().model(ctx.as_of)
    if prob is None:
        return []
    o = prob.outlook(area)[0]
    near = "; ".join(f"{_hsa_label(e['fips_list'])}, week ending {e['week_end']} ({e['change']:+.0%})"
                     for e in o.examples)
    m = load_metrics()
    track = (f" Backtest: these probabilities beat the base rate (Brier skill {m['brier_skill']:+.0%}, "
             f"AUC {m['auc_rise']:.2f}, {int(m['n'])} county-weeks)." if m else "")
    rise_pct = round((np.exp(RISE) - 1) * 100)
    claim = (f"Outlook from {o.n} analogs: in the most similar past situations across US Health Service Areas "
             f"(same level, momentum and time of year), ED share rose {rise_pct}%+ the following week "
             f"{o.p_rise:.0%} of the time and fell {rise_pct}%+ {o.p_fall:.0%} of the time (median change "
             f"{o.median_change:+.0%}; base rate of a rise in that history {prob.base_rise:.0%}). "
             f"Closest: {near}." + track)
    return [Finding(
        "outlook", claim, "outlook", ctx.as_of.date().isoformat(),
        "This study: analog ensemble (k-nearest neighbors) on CDC rdmq-nq56 history (src/uncertainty.py)",
        severity=round(min(0.95, 0.1 + 0.8 * o.p_rise), 2), confidence=0.6, fips=feats["fips"].tolist(),
        data={"p_rise": round(o.p_rise, 3), "p_fall": round(o.p_fall, 3),
              "median_change": round(o.median_change, 3), "base_rise": round(prob.base_rise, 3),
              "target_week": target.date().isoformat(), "analogs": o.examples})]


# ── Hot spots ────────────────────────────────────────────────────────────────

NEIGHBOR_PAD_DEG = 1.5
MAX_CONTEXT = 140


def run_hotspots(ctx: Context) -> list[Finding]:
    from src.spatial.hotspots import gi_star
    shapes = counties()
    mine = shapes[shapes["fips"].isin(ctx.region.fips)]
    minx, miny, maxx, maxy = mine.total_bounds
    p = NEIGHBOR_PAD_DEG
    ctxt = shapes.cx[minx - p:maxx + p, miny - p:maxy + p]
    if len(ctxt) > MAX_CONTEXT:   # keep the nearest by centroid distance
        cx, cy = ctx.region.center
        d = (ctxt["lon"] - cx) ** 2 + (ctxt["lat"] - cy) ** 2
        keep = set(ctxt.loc[d.nsmallest(MAX_CONTEXT).index, "fips"]) | set(ctx.region.fips)
        ctxt = ctxt[ctxt["fips"].isin(keep)]
    ctx.log(f"Comparing against {len(ctxt) - len(mine)} neighboring counties (same vintage)")
    wide, meta = nssp_vintage(ctxt["fips"].tolist(), ctx.as_of)
    if wide.empty:
        return []
    week = wide["week_end"].max()
    vals = wide[wide["week_end"] == week][["fips", "resp_pct"]]
    if len(vals) < 8:
        return [Finding("hotspots", f"Too few counties around the area had data on {meta['issue']} "
                        f"({len(vals)}) for a hot-spot test.", "data_gap", meta["issue"],
                        "Getis-Ord Gi* (src/spatial/hotspots.py)", severity=0.1, confidence=0.9)]
    gi = gi_star(ctxt, vals, permutations=999, conus_only=False)
    inside = gi[gi["fips"].isin(ctx.region.fips)]
    hot = inside[inside["class"].str.startswith("Hot")]
    cold = inside[inside["class"].str.startswith("Cold")]
    ctx.emit_layer({"kind": "hotspots", "week_end": week.date().isoformat(),
                    "classes": dict(zip(gi["fips"], gi["class"]))})
    mine_mean = _weighted(vals[vals["fips"].isin(ctx.region.fips)], "resp_pct", ctx.region)
    around = vals[~vals["fips"].isin(ctx.region.fips)]["resp_pct"]
    rel = (f"; the area averages {_pct(mine_mean)} vs a median of {_pct(around.median())} in "
           f"{len(around)} surrounding counties") if len(around) else ""
    if len(hot):
        claim = (f"Hot spot: {len(hot)} count{'y' if len(hot) == 1 else 'ies'} in the area "
                 f"({', '.join(_county_name(ctx.region, f) for f in hot['fips'][:4])}) sit in a statistically "
                 f"significant cluster of high respiratory ED share (Gi*, FDR-corrected) for the week ending "
                 f"{week.date()}{rel}.")
        sev = 0.75
    elif len(cold):
        claim = (f"Cold spot: {len(cold)} count{'y' if len(cold) == 1 else 'ies'} in the area form a "
                 f"significant low cluster for the week ending {week.date()}{rel}.")
        sev = 0.15
    else:
        claim = f"No significant hot or cold spot in the area for the week ending {week.date()}{rel}."
        sev = 0.3
    return [Finding("hotspots", claim, "hotspot", meta["issue"],
                    "Getis-Ord Gi* on NSSP (src/spatial/hotspots.py)", severity=sev, confidence=0.7,
                    fips=hot["fips"].tolist() or cold["fips"].tolist(),
                    data={"hot": hot["fips"].tolist(), "cold": cold["fips"].tolist(),
                          "n_context": int(len(vals))})]


# ── Wastewater ───────────────────────────────────────────────────────────────

HONEST_NWSS = pd.Timestamp("2026-04-17")


def run_wastewater(ctx: Context) -> list[Finding]:
    from src.ingest_nwss import weekly_features
    url = "https://data.cdc.gov/Public-Health-Surveillance/NWSS-Wastewater-Viral-Activity-Levels/atcp-73re"
    proxy = False
    try:
        ww = weekly_features(ctx.as_of, ctx.region.fips)
        if ww.empty and ctx.as_of < HONEST_NWSS:
            ctx.log("No honest NWSS vintage before 2026-04-17; using the labeled proxy")
            ww, proxy = weekly_features(ctx.as_of, ctx.region.fips, allow_proxy=True), True
    except Exception as exc:
        ctx.log(f"Wastewater store unavailable: {exc}")
        return []
    if ww.empty:
        return [Finding("wastewater", "No NWSS wastewater site reported for these counties as of "
                        f"{ctx.as_of.date()}.", "data_gap", ctx.as_of.date().isoformat(), "CDC NWSS", url,
                        severity=0.1, confidence=0.8)]
    ww = ww.sort_values("week_end")
    latest = ww["week_end"].max()
    now = ww[ww["week_end"] == latest]
    before = ww[ww["week_end"] == latest - timedelta(weeks=2)]
    parts = []
    for col, label in (("ww_covid", "COVID"), ("ww_flu", "flu A"), ("ww_rsv", "RSV")):
        v = now[col].mean()
        if np.isnan(v):
            continue
        b = before[col].mean() if len(before) else float("nan")
        arrow = "" if np.isnan(b) else (" ↑" if v > b * 1.25 else " ↓" if v < b * 0.8 else " →")
        parts.append(f"{label} {v:.1f}{arrow}")
    note = (" PROXY: honest wastewater vintages start 2026-04-17; this uses later CDC files under the "
            "publication-lag rule and may include revisions." if proxy else "")
    known = (latest + timedelta(days=6)).date().isoformat() if not proxy else ctx.as_of.date().isoformat()
    return [Finding("wastewater", f"Wastewater viral activity (WVAL, county mean of "
                    f"{int(now['ww_n_sites'].sum())} site reports) for the week ending {latest.date()}: "
                    f"{', '.join(parts) or 'no values'} (arrows: vs 2 weeks earlier).{note}",
                    "wastewater", min(known, ctx.as_of.date().isoformat()), "CDC NWSS (src/ingest_nwss.py)",
                    url, severity=0.4, confidence=0.45 if proxy else 0.7, fips=sorted(now["fips"]),
                    data={"proxy": proxy, "week_end": latest.date().isoformat()})]


# ── Weather ──────────────────────────────────────────────────────────────────

WEATHER_LAG_DAYS = 7     # the study treats weather as public a week after the fact


def run_weather(ctx: Context) -> list[Finding]:
    lon, lat = ctx.region.center
    end = ctx.as_of - timedelta(days=WEATHER_LAG_DAYS)
    start = end - timedelta(days=27)
    ctx.log(f"Open-Meteo ERA5 at {lat:.2f}, {lon:.2f}, {start.date()} to {end.date()}")
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=HTTP_TIMEOUT, headers=_UA,
                     params={"latitude": f"{lat:.3f}", "longitude": f"{lon:.3f}",
                             "start_date": start.date().isoformat(), "end_date": end.date().isoformat(),
                             "daily": "temperature_2m_mean,dew_point_2m_mean,relative_humidity_2m_mean",
                             "temperature_unit": "fahrenheit", "timezone": "auto"})
    r.raise_for_status()
    d = pd.DataFrame(r.json()["daily"]).dropna()
    if d.empty:
        return []
    last, prev = d.tail(7), d.iloc[:-7]
    t, dp, rh = last["temperature_2m_mean"].mean(), last["dew_point_2m_mean"].mean(), last["relative_humidity_2m_mean"].mean()
    dt = t - prev["temperature_2m_mean"].mean() if len(prev) else float("nan")
    cold_dry = t < 45 and dp < 32
    return [Finding("weather", f"Week ending {last['time'].iloc[-1]}: mean {t:.0f}°F "
                    f"({dt:+.0f}°F vs the 3 weeks before), dew point {dp:.0f}°F, humidity {rh:.0f}%."
                    + (" Cold, dry air favors flu transmission; note the study found weather added no "
                       "accuracy beyond seasonality." if cold_dry else ""),
                    "weather", min(end.date().isoformat(), ctx.as_of.date().isoformat()),
                    "Open-Meteo Historical Weather (ERA5)", "https://open-meteo.com/en/docs/historical-weather-api",
                    severity=0.45 if cold_dry else 0.2, confidence=0.75,
                    data={"temp_f": round(t, 1), "dewpoint_f": round(dp, 1), "rh": round(rh, 1)})]


# ── Air quality ──────────────────────────────────────────────────────────────

def run_air(ctx: Context) -> list[Finding]:
    lon, lat = ctx.region.center
    end = ctx.as_of - timedelta(days=1)
    start = end - timedelta(days=13)
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=HTTP_TIMEOUT, headers=_UA,
                     params={"latitude": f"{lat:.3f}", "longitude": f"{lon:.3f}", "hourly": "pm2_5,ozone",
                             "start_date": start.date().isoformat(), "end_date": end.date().isoformat(),
                             "timezone": "auto"})
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"]).dropna()
    if h.empty:
        return []
    h["day"] = h["time"].str[:10]
    daily = h.groupby("day")[["pm2_5", "ozone"]].mean()
    last = daily.tail(7)
    pm, oz, worst = last["pm2_5"].mean(), last["ozone"].mean(), daily["pm2_5"].max()
    bad = worst > 35
    return [Finding("air_quality", f"Past week PM2.5 averaged {pm:.1f} µg/m³ (worst day in 2 weeks "
                    f"{worst:.0f}), ozone {oz:.0f} µg/m³" + (". Smoke-level particulates can worsen "
                    "respiratory ED visits." if bad else ", not at levels that drive respiratory ED visits."),
                    "air", end.date().isoformat(), "Open-Meteo Air Quality (CAMS model, not monitor data)",
                    "https://open-meteo.com/en/docs/air-quality-api", severity=0.5 if bad else 0.15,
                    confidence=0.6, data={"pm25": round(pm, 1), "ozone": round(oz, 1)})]


# ── News ─────────────────────────────────────────────────────────────────────

NEWS_WINDOW_DAYS = 28
ILLNESS_QUERY = "(flu OR influenza OR COVID OR RSV OR \"respiratory illness\")"
CONCERN = {0: "no concern", 1: "mention", 2: "rising", 3: "surge"}


def _local_news(ctx: Context) -> list[Finding]:
    labels_path, catalog_path = Path("data/labels/llm_labels_grounded.parquet"), Path("labels/articles.csv")
    if not labels_path.exists() or not catalog_path.exists():
        return []
    lab = pd.read_parquet(labels_path)
    lab = lab[lab["run"] == lab["run"].min()].drop_duplicates("article_id")
    cat = pd.read_csv(catalog_path, dtype={"fips_guess": str})
    df = lab.merge(cat[["article_id", "url", "title", "outlet", "metro"]], on="article_id", how="left")
    df["published"] = pd.to_datetime(df["published"])
    metros = {m.key for m in METROS.values() if set(m.counties) & set(ctx.region.fips)}
    here = df["fips"].isin(ctx.region.fips) | (df["fips"].isna() & df["metro"].isin(metros))
    window = (df["published"] <= ctx.as_of) & (df["published"] > ctx.as_of - timedelta(days=NEWS_WINDOW_DAYS))
    hits = df[here & window & df["relevant"].fillna(False).astype(bool)].sort_values("published")
    out = []
    for _, a in hits.tail(6).iterrows():
        c = int(a["concern"]) if pd.notna(a["concern"]) else 0
        quote = a.get("concern_quote") if isinstance(a.get("concern_quote"), str) else ""
        out.append(Finding("news", f"{a['outlet']} ({a['published'].date()}): \"{a['title']}\". LLM label: "
                           f"{CONCERN.get(c, c)}{f', {a.illness}' if isinstance(a.get('illness'), str) else ''}"
                           + (f"; quote: \"{quote[:160]}\"" if quote else ""),
                           "news", a["published"].date().isoformat(), f"{a['outlet']} (labeled by Qwen3 8B, quote-checked)",
                           a["url"], severity=round(0.15 + 0.25 * c, 2), confidence=0.55,
                           fips=[a["fips"]] if isinstance(a["fips"], str) else [],
                           data={"concern": c, "article_id": a["article_id"]}))
    if metros and not out:
        out.append(Finding("news", f"No labeled local article about respiratory illness in the "
                           f"{NEWS_WINDOW_DAYS} days before {ctx.as_of.date()} (study corpus).", "news",
                           ctx.as_of.date().isoformat(), "Study news corpus", severity=0.15, confidence=0.5))
    return out


def _gdelt_news(ctx: Context) -> list[Finding]:
    from src.news.collect import GdeltClient, GdeltRateLimited, RateLimiter
    place = ctx.region.names[ctx.region.fips[0]].split(",")[0].replace(" County", "").replace(" city", "")
    query = f"{ILLNESS_QUERY} \"{place}\" sourcecountry:US sourcelang:english"
    client = GdeltClient(limiter=RateLimiter(0), retries=0, timeout=HTTP_TIMEOUT)
    start = ctx.as_of - timedelta(days=21)
    ctx.log(f"GDELT title search: {query}")
    try:
        arts = client.search(query, start=start, end=ctx.as_of + timedelta(hours=23), max_records=25)
    except GdeltRateLimited:
        ctx.log("GDELT is rate-limiting this IP; skipping the live search")
        return []
    out = []
    for a in arts[:6]:
        seen = pd.Timestamp(a.get("seendate") or ctx.as_of)
        if seen.tz is not None:
            seen = seen.tz_convert(None)
        if seen.normalize() > ctx.as_of:
            continue
        out.append(Finding("news", f"Headline ({a.get('domain', '')}, {seen.date()}): \"{a.get('title', '')}\" "
                           "(unlabeled; title only)", "news", seen.date().isoformat(),
                           f"GDELT DOC API ({a.get('domain', '')})", a.get("url", ""), severity=0.25,
                           confidence=0.35))
    return out


def run_news(ctx: Context) -> list[Finding]:
    out = _local_news(ctx)
    try:
        out += _gdelt_news(ctx)
    except Exception as exc:
        ctx.log(f"GDELT search failed: {exc}")
    return out


WORKERS: dict[str, Worker] = {w.id: w for w in (
    Worker("ed_visits", "ED Watch", "Share of emergency department visits for COVID, flu and RSV in each county, "
           "as published on the date, with the 3-week trend and counties that had no data.", run_ed,
           ("CDC NSSP",)),
    Worker("nowcast", "Nowcaster", "The study's best model: estimates this week's respiratory ED share from the "
           "published series and national history (fills the 1-2 week reporting lag).", run_nowcast,
           ("CDC NSSP", "CDC rdmq-nq56")),
    Worker("outlook", "Outlook", "Probability that the area's ED share is rising (up 10%+) this week, from the "
           "most similar past situations in national history, naming the closest matches. Use for 'is it "
           "rising / getting worse / will it peak' questions.", run_outlook, ("CDC NSSP", "CDC rdmq-nq56")),
    Worker("hotspots", "Cluster Scan", "Getis-Ord Gi* test of whether the area is a significant hot or cold "
           "spot relative to surrounding counties.", run_hotspots, ("CDC NSSP",)),
    Worker("wastewater", "Sewer Sentinel", "CDC wastewater viral activity (COVID, flu A, RSV) for sewersheds in "
           "the area; an early signal that can lead ED visits.", run_wastewater, ("CDC NWSS",)),
    Worker("weather", "Front Line", "Recent temperature, dew point and humidity (cold, dry air favors flu).",
           run_weather, ("Open-Meteo ERA5",)),
    Worker("air_quality", "Clean Air Patrol", "Recent PM2.5 and ozone (smoke and pollution episodes raise "
           "respiratory ED visits).", run_air, ("Open-Meteo CAMS",)),
    Worker("news", "News Desk", "Local news about respiratory illness before the date: the study's labeled "
           "articles plus a live GDELT headline search.", run_news, ("Local news", "GDELT")),
)}


def run_worker(worker: Worker, ctx: Context, timeout: float = 60.0) -> list[Finding]:
    """Run one worker with a time limit; errors become a log line and no findings."""
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(worker.run, ctx)
    try:
        return future.result(timeout=timeout) or []
    except TimeoutError:
        ctx.log(f"timed out after {timeout:.0f}s")
    except Exception as exc:
        log.exception("worker %s failed", worker.id)
        ctx.log(f"failed: {str(exc)[:160]}")
    finally:
        pool.shutdown(wait=False)   # a hung request must not hold the answer hostage
    return []
