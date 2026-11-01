"""NSSP ED data for any set of counties, exactly as public on a chosen date.

The study's snapshot store only holds the seven study counties (and three
all-county weeks for the map). The app answers for wherever the user draws,
so it asks Delphi for those counties ``as_of`` the right epiweek and caches
the answer under ``raw/app_cache/nssp/`` (gitignored with the rest of
``raw/``).

Which vintage a date sees
-------------------------
Same convention as ``ingest_nssp`` / ``backfill_nssp``: a vintage is stamped
with the Saturday ending its Delphi issue epiweek, and ``load_vintage(D)``
returns stamps on or before D. So on date D the newest visible issue is the
epiweek ending on the last Saturday on or before D (``visible_issue``).
Asking Delphi for that epiweek's ``as_of`` reproduces it for any county.

Verified against Delphi (Nov 2026)
----------------------------------
* A request for ~5 named counties returns in ~2 s; an all-county request for
  4 weeks took 74 s. So we fetch the selection (plus neighbors for the
  hot-spot check) and never the whole country per question.
* Missouri counties come back empty for any ``as_of`` before June 2026
  (README finding 2). That emptiness is the true answer for that date, so it
  is cached as "fetched, no data" rather than retried.
* Anonymous access is capped at 60 requests/hour. One question costs 3
  requests (one per signal) for counties not already cached; set
  ``DELPHI_API_KEY`` to lift the cap.
* **Delphi's covidcast API is retired for data after 2026-09-22** ("as_of is
  entirely after the cutoff"; replaced by cast-api / epidata-etl). Its last
  usable issue is epiweek 202638 (Saturday 2026-09-26). For later dates we
  read CDC's own table (``src.ingest_cdc_nssp``, sources ``cdc_nssp_*``)
  from the snapshot store when a vintage on/before the date exists. Those
  are HSA values (counties in one HSA share a number). Without one, we use
  the last Delphi issue and say the data stops there.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.ingest_nssp import fetch_signal
from src.snapshot import SnapshotStore, coerce_date
from src.spatial.hotspots import respiratory_share
from src.spatial.map import epiweek_of
from src.study import STUDY_FIPS

log = logging.getLogger(__name__)

__all__ = ["CACHE_DIR", "HISTORY_WEEKS", "SIGNALS", "DELPHI_LAST_ISSUE", "visible_issue", "nssp_vintage"]

CACHE_DIR = Path("raw/app_cache/nssp")
DELPHI_LAST_ISSUE = pd.Timestamp("2026-09-26")     # epiweek 202638, the last as_of Delphi accepts
_LOCK = threading.Lock()          # workers share the cache file; serialize read-fetch-write
CDC_SOURCES = {"covid": "cdc_nssp_covid", "influenza": "cdc_nssp_influenza", "rsv": "cdc_nssp_rsv"}
HISTORY_WEEKS = 16
SIGNALS = {"covid": "pct_ed_visits_covid", "influenza": "pct_ed_visits_influenza",
           "rsv": "pct_ed_visits_rsv"}
_COLUMNS = ["fips", "signal", "week_start", "value"]


def visible_issue(as_of) -> pd.Timestamp:
    """Saturday ending the newest NSSP issue public on ``as_of``."""
    d = coerce_date(as_of)
    return d - timedelta(days=(d.weekday() - 5) % 7)


def _cache_path(issue: pd.Timestamp, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"asof={epiweek_of(issue)}.parquet"


def _read_cache(path: Path) -> tuple[pd.DataFrame, set[str]]:
    """Cached rows plus the set of FIPS already asked about (with or without data)."""
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS), set()
    frame = pd.read_parquet(path)
    asked = set(frame["fips"])
    return frame[frame["signal"] != "_asked"][_COLUMNS], asked


def _write_cache(path: Path, rows: pd.DataFrame, asked: set[str]) -> None:
    marks = pd.DataFrame({"fips": sorted(asked), "signal": "_asked",
                          "week_start": pd.NaT, "value": float("nan")})
    out = pd.concat([rows[_COLUMNS], marks], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)


def _fetch(fips: list[str], issue: pd.Timestamp, fetch=fetch_signal) -> pd.DataFrame:
    ew = epiweek_of(issue)
    start = epiweek_of(issue - timedelta(weeks=HISTORY_WEEKS))
    key = os.environ.get("DELPHI_API_KEY")
    parts = []
    for name, signal in SIGNALS.items():
        raw = fetch(signal, f"{start}-{ew}", ",".join(fips), as_of=ew, api_key=key)
        if len(raw):
            parts.append(pd.DataFrame({"fips": raw["geo_value"], "signal": name,
                                       "week_start": pd.to_datetime(raw["time_value"]),
                                       "value": raw["value"]}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_COLUMNS)


def _local(fips: list[str], issue: pd.Timestamp, root="raw/snapshots") -> pd.DataFrame:
    """Study counties from the snapshot store: the offline fallback."""
    store = SnapshotStore(root)
    try:
        wide = respiratory_share(store, as_of=issue)
    except Exception as exc:  # store missing or empty on a fresh clone
        log.info("local snapshots unavailable: %s", exc)
        return pd.DataFrame(columns=_COLUMNS)
    wide = wide[wide["fips"].isin(fips)]
    return wide.melt(id_vars=["fips", "week_start"], value_vars=list(SIGNALS),
                     var_name="signal", value_name="value")[_COLUMNS]


def _cdc(fips: list[str], as_of: pd.Timestamp, root="raw/snapshots") -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """CDC rdmq-nq56 vintages saved by ``ingest_cdc_nssp``, newest stamp after Delphi's last."""
    store = SnapshotStore(root)
    try:
        issues = [i for i in store.available_issues(CDC_SOURCES["covid"])
                  if DELPHI_LAST_ISSUE < i <= as_of]
    except Exception:
        return pd.DataFrame(columns=_COLUMNS), None
    if not issues:
        return pd.DataFrame(columns=_COLUMNS), None
    parts = []
    for name, source in CDC_SOURCES.items():
        v = store.load_vintage(source, as_of)
        v = v[v["geo_value"].isin(fips)]
        parts.append(pd.DataFrame({"fips": v["geo_value"], "signal": name,
                                   "week_start": pd.to_datetime(v["time_value"]), "value": v["value"]}))
    return pd.concat(parts, ignore_index=True), max(issues)


def nssp_vintage(fips, as_of, cache_dir=CACHE_DIR, fetch=fetch_signal,
                 local_root="raw/snapshots") -> tuple[pd.DataFrame, dict]:
    """Weekly COVID/flu/RSV/total % of ED visits for ``fips`` as public on ``as_of``.

    Returns ``(wide, meta)``: ``wide`` has ``fips, week_start, week_end,
    covid, influenza, rsv, resp_pct`` (a week is kept only when all three
    signals are present, as in ``hotspots.respiratory_share``), and ``meta``
    says which issue was used, where it came from, and which counties had no
    published data at all on that date.
    """
    fips = sorted({str(f).zfill(5) for f in ([fips] if isinstance(fips, str) else fips)})
    issue = visible_issue(as_of)
    note = None
    if issue > DELPHI_LAST_ISSUE:
        cdc_rows, cdc_issue = _cdc(fips, coerce_date(as_of), local_root)
        if cdc_issue is not None:
            return _finish(cdc_rows, fips, cdc_issue,
                           f"CDC rdmq-nq56 vintage saved {cdc_issue.date()} (HSA values)", None, None)
        note = (f"Delphi stopped serving NSSP after 2026-09-22 and no newer CDC vintage is saved "
                f"(run python -m src.ingest_cdc_nssp weekly), so ED data stops at the issue of "
                f"{DELPHI_LAST_ISSUE.date()}; anything CDC published after that is missing here.")
        issue = DELPHI_LAST_ISSUE
    with _LOCK:
        path = _cache_path(issue, cache_dir)
        rows, asked = _read_cache(path)
        missing = [f for f in fips if f not in asked]
        source, error = "cache", None
        if missing:
            try:
                new = _fetch(missing, issue, fetch)
                rows = pd.concat([rows, new], ignore_index=True)
                asked |= set(missing)
                _write_cache(path, rows, asked)
                source = "delphi"
            except Exception as exc:
                error = str(exc)[:200]
                log.warning("Delphi fetch failed (%s); falling back to local snapshots", error)
                local = _local([f for f in missing if f in STUDY_FIPS], issue, local_root)
                rows = pd.concat([rows, local], ignore_index=True)
                source = "local snapshots of the 7 study counties (Delphi failed)"
    return _finish(rows, fips, issue, source, error, note)


def _finish(rows: pd.DataFrame, fips: list[str], issue: pd.Timestamp, source: str,
            error: str | None, note: str | None) -> tuple[pd.DataFrame, dict]:
    rows = rows[rows["fips"].isin(fips)]
    rows = rows.assign(value=pd.to_numeric(rows["value"], errors="coerce")).dropna(subset=["value"])
    if rows.empty:
        wide = pd.DataFrame(columns=["fips", "week_start", "week_end", *SIGNALS, "resp_pct"])
    else:
        wide = rows.pivot_table(index=["fips", "week_start"], columns="signal",
                                values="value", aggfunc="last").reset_index()
        for name in SIGNALS:
            if name not in wide:
                wide[name] = float("nan")
        wide["week_start"] = pd.to_datetime(wide["week_start"])
        wide["week_end"] = wide["week_start"] + timedelta(days=6)
        wide["resp_pct"] = wide[list(SIGNALS)].sum(axis=1, min_count=len(SIGNALS))
        wide = wide.dropna(subset=["resp_pct"])
        # The vintage rule, checked rather than assumed: nothing after the issue.
        wide = wide[wide["week_end"] <= issue]
        wide = wide.sort_values(["fips", "week_start"]).reset_index(drop=True)
    meta = {"issue": issue.date().isoformat(), "epiweek": epiweek_of(issue),
            "source": source, "error": error, "note": note,
            "no_data": [f for f in fips if f not in set(wide["fips"])]}
    return wide, meta
