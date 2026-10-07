"""Rebuild one NSSP vintage per week for the study counties, from 202416 on.

The weekly protocol (``python -m src.ingest_nssp`` every week) only covers the
future. For the retrospective backtest we need what was public on every past
prediction date, so this script walks epiweeks from 202416 -- the first issue
Delphi archived -- to the latest completed one and asks Delphi for each week's
data ``as_of`` that epiweek. Each call writes one snapshot per signal stamped
with the Saturday ending the as_of week, exactly as ``ingest()`` does for a
single ``--as-of`` run. Usage, from the repo root:

    python -m src.backfill_nssp                 # 202416 .. latest completed
    python -m src.backfill_nssp --end 202610    # stop early

Two methods, same snapshots:

* ``history`` (default): pull every archived issue per county with Delphi's
  ``issues=`` parameter and rebuild each week's vintage locally with the rule
  Delphi's ``as_of`` applies server-side (newest issue <= as_of per
  county-week). 21 requests in total.
* ``as-of``: call ``ingest(as_of=week)`` once per week -- the same code path as
  the weekly protocol, but ~800 requests. Anonymous Delphi access is capped at
  60 requests/hour (HTTP 429, ``Retry-After`` ~1h), so this needs an API key.

Vintages written by either method are byte-for-byte comparable: the history
method re-submits weeks already on disk, and the store refuses any that differ.

Resumable: a week whose three signals are all already on disk is skipped, so
an interrupted run picks up where it stopped. A rewrite that would change an
existing snapshot (for example, an all-county vintage already stored under the
same issue date) is refused by the store; we log it and move on rather than
abort, because the existing file is the one other people already used.

Archive holes: some as_of weeks return nothing for a signal, and some weeks'
first archived print is several weeks late. Both are reported by
:func:`coverage`, not papered over -- a hole means the backtest genuinely could
not have seen that week on time.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.ingest_nssp import API, LIVE_SIGNALS, _UA, ingest, mmwr_week_start
from src.snapshot import SnapshotExistsError, SnapshotStore
from src.study import STUDY_FIPS

log = logging.getLogger(__name__)

FIRST_ARCHIVED_EPIWEEK = 202416
# History requested in every vintage. Starting well before the archive lets
# the earliest vintages carry lagged values for the model's history window
# (those early weeks are already-revised values, which is what was public).
HISTORY_START_EPIWEEK = 202340

SOURCES = tuple("nssp_" + s.removeprefix("pct_ed_visits_") for s in LIVE_SIGNALS)


def epiweek_of(day) -> int:
    """MMWR epiweek (e.g. 202601) containing ``day``.

    An MMWR week belongs to the year that holds its Wednesday, which is the
    same rule as "week 1 contains January 4th" used by ``mmwr_week_start``.
    """
    d = pd.Timestamp(day).date()
    sunday = d - timedelta(days=(d.weekday() + 1) % 7)
    year = (sunday + timedelta(days=3)).year
    return year * 100 + (sunday - mmwr_week_start(year * 100 + 1)).days // 7 + 1


def latest_completed_epiweek(today: date | None = None) -> int:
    """The newest epiweek whose Saturday is strictly before ``today``.

    The current week is excluded because its vintage would be stamped with a
    Saturday that has not happened yet.
    """
    today = today or date.today()
    return epiweek_of(today - timedelta(days=(today.weekday() + 1) % 7 + 1))


def epiweeks(start: int, end: int) -> list[int]:
    """Every epiweek from ``start`` to ``end`` inclusive, across year ends."""
    out, day = [], mmwr_week_start(start)
    while (week := epiweek_of(day)) <= end:
        out.append(week)
        day += timedelta(weeks=1)
    return out


def issue_for(epiweek: int) -> pd.Timestamp:
    """The snapshot issue date ``ingest(as_of=epiweek)`` stamps."""
    return pd.Timestamp(mmwr_week_start(epiweek) + timedelta(days=6))


class RateLimited(RuntimeError):
    """Delphi answered 429. Anonymous access is capped at 60 requests/hour."""

    def __init__(self, retry_after: float):
        super().__init__(f"rate limited; retry after {retry_after:.0f}s")
        self.retry_after = retry_after


def _get(params: dict) -> dict:
    response = requests.get(API, params=params, timeout=180, headers=_UA)
    if response.status_code == 429:
        raise RateLimited(float(response.headers.get("retry-after", 3600)))
    if response.status_code == 401:
        # "Requested too many multiples for anonymous queries": without a key,
        # at most two of geo/time/issue may be lists or ranges.
        raise RuntimeError(f"Delphi refused the query (401): {response.text[:200]}")
    response.raise_for_status()
    return response.json()


def fetch_history(signal: str, geo_values: str, time_range: str, issues: tuple[int, int],
                  get=_get) -> pd.DataFrame:
    """Every archived row of ``signal`` for issues in ``issues`` (inclusive).

    Same columns as ``ingest_nssp.fetch_signal``. A truncated response
    (``result == 2``) is split in half and retried rather than accepted.
    """
    weeks = epiweeks(*issues)
    payload = get({"data_source": "nssp", "signal": signal, "time_type": "week",
                   "geo_type": "county", "geo_values": geo_values,
                   "time_values": time_range, "issues": f"{issues[0]}-{issues[1]}"})
    if payload.get("result") == -2:
        return pd.DataFrame()
    if payload.get("result") == 2 and len(weeks) > 1:
        mid = len(weeks) // 2
        return pd.concat([fetch_history(signal, geo_values, time_range, (weeks[0], weeks[mid - 1]), get),
                          fetch_history(signal, geo_values, time_range, (weeks[mid], weeks[-1]), get)],
                         ignore_index=True)
    if payload.get("result") != 1:
        raise RuntimeError(f"{signal} issues {issues}: {payload.get('message')}")
    raw = pd.DataFrame(payload["epidata"])
    return pd.DataFrame({
        "geo_value": raw["geo_value"].astype(str).str.zfill(5),
        "time_value": raw["time_value"].map(mmwr_week_start).map(pd.Timestamp),
        "value": pd.to_numeric(raw["value"], errors="coerce"),
        "delphi_issue": raw["issue"].astype("int64"),
    })


def vintage_from_history(history: pd.DataFrame, as_of: int) -> pd.DataFrame:
    """What ``fetch_signal(..., as_of=as_of)`` returns, rebuilt locally.

    Delphi's ``as_of`` keeps, per (county, week), the row with the newest
    issue at or before ``as_of``. Doing the same here from the full issue
    history gives an identical frame for a fraction of the requests.
    """
    if history.empty:
        return history
    known = history[history["delphi_issue"] <= as_of]
    known = known.sort_values(["geo_value", "time_value", "delphi_issue"], kind="mergesort")
    known = known.drop_duplicates(["geo_value", "time_value"], keep="last")
    return known.reset_index(drop=True)


def _write(store: SnapshotStore, source: str, issue, frame: pd.DataFrame) -> str:
    """Write one vintage; log and report "exists" if the store refuses it.

    Dates are cast to millisecond precision first because that is what a
    parquet round trip returns: the store's "same content is a no-op" check
    compares dtypes too, and a second-precision frame never equals its own
    file read back under pandas 3.
    """
    frame = frame.assign(issue=pd.Timestamp(issue))
    for col in ("time_value", "issue"):
        frame[col] = frame[col].astype("datetime64[ms]")
    try:
        store.write_snapshot(source, issue, frame)
        return "ok"
    except SnapshotExistsError as exc:
        log.warning("%s: %s", source, exc)
        return "exists"


def backfill_from_history(store: SnapshotStore, start: int = FIRST_ARCHIVED_EPIWEEK,
                          end: int | None = None, fips=STUDY_FIPS,
                          get=_get, wait=time.sleep, pause: float = 1.0) -> pd.DataFrame:
    """Default method: pull each county's issue history, write one vintage per week.

    One request per county per signal (21 for the study counties): anonymous
    Delphi access allows at most two ranged dimensions per query, so time
    and issue are ranges and counties go one at a time. A 429 waits out the
    server's ``Retry-After`` and resumes.
    """
    end = end or latest_completed_epiweek()
    weeks = epiweeks(start, end)
    records = []
    for signal in LIVE_SIGNALS:
        source = "nssp_" + signal.removeprefix("pct_ed_visits_")
        on_disk = set(store.available_issues(source))
        if all(issue_for(w) in on_disk for w in weeks):
            continue
        parts = []
        for geo in fips:
            while True:
                try:
                    parts.append(fetch_history(signal, geo, f"{HISTORY_START_EPIWEEK}-{end}",
                                               (start, end), get))
                    break
                except RateLimited as exc:
                    log.warning("%s; sleeping", exc)
                    wait(exc.retry_after + 5)
            log.info("%s %s: %d rows", signal, geo, len(parts[-1]))
            wait(pause)
        history = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        # Rewrite weeks already on disk too: the store treats identical content
        # as a no-op and refuses different content, so this doubles as a check
        # that the rebuilt vintages match ones fetched with as_of directly.
        for week in weeks:
            frame = vintage_from_history(history, week)
            if frame.empty:
                records.append({"epiweek": week, "source": source, "status": "empty"})
                continue
            status = _write(store, source, issue_for(week), frame)
            records.append({"epiweek": week, "source": source, "status": status,
                            "rows": len(frame)})
    return pd.DataFrame(records, columns=["epiweek", "source", "status", "rows"])


def backfill(store: SnapshotStore, start: int = FIRST_ARCHIVED_EPIWEEK,
             end: int | None = None, fips=STUDY_FIPS, pause: float = 0.5,
             ingest_fn=ingest) -> pd.DataFrame:
    """Alternative method: one ``ingest(as_of=week)`` call per week.

    The same code path as the weekly protocol, but ~2 requests per signal per
    week; without a Delphi API key it hits the 60/hour cap after ~10 weeks.
    ``ingest_fn`` is injectable so tests can run without the network.
    """
    end = end or latest_completed_epiweek()
    geo_values = ",".join(fips)
    on_disk = {s: set(store.available_issues(s)) for s in SOURCES}
    records = []
    for week in epiweeks(start, end):
        issue = issue_for(week)
        if all(issue in on_disk[s] for s in SOURCES):
            records.append({"epiweek": week, "issue": issue, "status": "skipped"})
            continue
        try:
            _, written = ingest_fn(store, time_range=f"{HISTORY_START_EPIWEEK}-{week}",
                                   geo_values=geo_values, as_of=week)
            status = "ok" if all(written.values()) else "partial"
        except SnapshotExistsError as exc:
            log.warning("epiweek %s: %s", week, exc)
            written, status = {}, "exists"
        records.append({"epiweek": week, "issue": issue, "status": status, **written})
        log.info("epiweek %s -> %s %s", week, status, written)
        time.sleep(pause)
    return pd.DataFrame(records)


def coverage(store: SnapshotStore) -> pd.DataFrame:
    """Per signal: vintages on disk, and weeks whose first print came late.

    ``late_first_prints`` lists reference weeks (by week_end) whose first
    appearance in any vintage was more than one week after the week ended --
    the archive holes a backtest has to live with.
    """
    rows = []
    for source in SOURCES:
        issues = store.available_issues(source)
        first_seen = {}
        for issue in issues:
            frame = store._read_file(store.path_for(source, issue))
            for tv in frame["time_value"].unique():
                first_seen.setdefault(pd.Timestamp(tv), issue)
        late = sorted(
            (tv + timedelta(days=6)).date().isoformat()
            for tv, iss in first_seen.items()
            if tv >= pd.Timestamp(mmwr_week_start(FIRST_ARCHIVED_EPIWEEK))
            and (iss - (tv + timedelta(days=6))).days > 7
        )
        rows.append({"source": source, "vintages": len(issues),
                     "first": issues[0].date() if issues else None,
                     "last": issues[-1].date() if issues else None,
                     "late_first_prints": late})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill weekly NSSP vintages.")
    parser.add_argument("--store", default="raw/snapshots")
    parser.add_argument("--start", type=int, default=FIRST_ARCHIVED_EPIWEEK)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--method", choices=("history", "as-of"), default="history",
                        help="history: few issue-range requests (default); "
                             "as-of: one ingest() per week (needs an API key)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    store = SnapshotStore(args.store)
    run = backfill_from_history if args.method == "history" else backfill
    result = run(store, args.start, args.end)
    if not result.empty:
        print(result.groupby(["status"]).size().to_string())
    with pd.option_context("display.max_colwidth", 200, "display.width", 200):
        print(coverage(store).to_string(index=False))


if __name__ == "__main__":
    main()
