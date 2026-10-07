"""Pull NSSP county-level respiratory ED visit percentages into the snapshot store.

This is the target variable: the share of emergency department visits that are
respiratory, by county, by week. Usage, from the repo root:

    python -m src.ingest_nssp                  # today's vintage, all counties
    python -m src.ingest_nssp --as-of 202607   # what was public in epiweek 2026-07

What we verified against the live Delphi API (Oct 2026):

* ``pct_ed_visits_combined`` is DISCONTINUED at epiweek 202439. The live
  signals are ``pct_ed_visits_covid``, ``pct_ed_visits_influenza`` and
  ``pct_ed_visits_rsv`` (~2,400 counties each). We store the three separately;
  summing them into one respiratory share is a modeling decision.

* Delphi archives NSSP revisions from issue 202416 (mid-April 2024) onward,
  with first prints at a 1-week lag. Before that, only values first seen in
  April 2024 exist, so honest backtests can only score weeks from then on.
  The archive has holes (e.g. week 202540's first archived print is at lag 7).

* Revisions are material: at the flu peak (week 202601), 19% of counties'
  first prints were later revised by >=10%, mostly upward (1,370 up vs 215 down).

* ``as_of`` must be an EPIWEEK (202607) for weekly signals. A calendar date
  (20260215) is read as an enormous epiweek and silently returns the LATEST
  data -- a leak, not an error. ``_check_epiweek`` refuses it.
"""

from __future__ import annotations

import argparse
import re
from datetime import date, timedelta

import pandas as pd
import requests

from src.snapshot import SnapshotStore

API = "https://api.delphi.cmu.edu/epidata/covidcast/"

LIVE_SIGNALS = ("pct_ed_visits_covid", "pct_ed_visits_influenza", "pct_ed_visits_rsv")
DISCONTINUED = {"pct_ed_visits_combined": "ends at epiweek 202439"}

_UA = {"User-Agent": "CSE4109-SPARK/ingest"}


def mmwr_week_start(epiweek: int) -> date:
    """The Sunday an epiweek (e.g. 202601) begins. Week 1 contains January 4th."""
    year, week = divmod(int(epiweek), 100)
    jan4 = date(year, 1, 4)
    week1_start = jan4 - timedelta(days=(jan4.weekday() + 1) % 7)
    return week1_start + timedelta(weeks=week - 1)


def _check_epiweek(value) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"\d{6}", text) or not 1 <= int(text) % 100 <= 53:
        raise ValueError(
            f"as_of must be an epiweek like 202607, got {value!r}. A calendar "
            f"date is silently read by Delphi as a huge epiweek and returns the "
            f"LATEST data, which leaks the future into a backtest."
        )
    return int(text)


def _year_chunks(time_range: str) -> list[str]:
    """Split '202240-202652' by year so no single request hits the row cap."""
    start, end = (int(x) for x in time_range.split("-"))
    return [
        f"{start if y == start // 100 else y * 100 + 1}-"
        f"{end if y == end // 100 else y * 100 + 53}"
        for y in range(start // 100, end // 100 + 1)
    ]


def fetch_signal(signal: str, time_range: str, geo_values: str = "*",
                 as_of=None, api_key: str | None = None) -> pd.DataFrame:
    """One NSSP signal, all weeks in ``time_range``, as known at ``as_of``."""
    if signal in DISCONTINUED:
        raise ValueError(f"{signal} is discontinued ({DISCONTINUED[signal]})")

    frames = []
    for chunk in _year_chunks(time_range):
        params = {"data_source": "nssp", "signal": signal, "time_type": "week",
                  "geo_type": "county", "geo_values": geo_values,
                  "time_values": chunk}
        if as_of is not None:
            params["as_of"] = _check_epiweek(as_of)
        if api_key:
            params["api_key"] = api_key

        response = requests.get(API, params=params, timeout=120, headers=_UA)
        response.raise_for_status()
        payload = response.json()
        if payload.get("result") == -2:       # "no results" is an answer
            continue
        if payload.get("result") != 1:        # 2 = truncated: fail loudly
            raise RuntimeError(f"{signal} {chunk}: {payload.get('message')}")
        frames.append(pd.DataFrame(payload["epidata"]))

    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    return pd.DataFrame({
        "geo_value": raw["geo_value"].astype(str).str.zfill(5),
        "time_value": raw["time_value"].map(mmwr_week_start).map(pd.Timestamp),
        "value": pd.to_numeric(raw["value"], errors="coerce"),
        # Delphi's own per-row issue, kept for reference. The snapshot's issue
        # is the date the whole vintage was knowable.
        "delphi_issue": raw["issue"].astype("int64"),
    })


def ingest(store: SnapshotStore, time_range: str = "202240-202652",
           geo_values: str = "*", as_of=None, api_key: str | None = None,
           today: date | None = None) -> tuple[date, dict]:
    """Write one snapshot per signal. Returns (issue date, rows per signal).

    A historical vintage is stamped with the Saturday that ends its as_of week:
    data published at any point in that week is certainly public by then, so a
    model dated that Saturday can see it without peeking.
    """
    if as_of is not None:
        issue = mmwr_week_start(_check_epiweek(as_of)) + timedelta(days=6)
    else:
        issue = today or date.today()

    written = {}
    for signal in LIVE_SIGNALS:
        frame = fetch_signal(signal, time_range, geo_values, as_of, api_key)
        if frame.empty:
            written[signal] = 0
            continue
        frame["issue"] = pd.Timestamp(issue)
        source = "nssp_" + signal.removeprefix("pct_ed_visits_")
        store.write_snapshot(source, issue, frame)
        written[signal] = len(frame)
    return issue, written


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull NSSP county ED data.")
    parser.add_argument("--store", default="raw/snapshots")
    parser.add_argument("--time-range", default="202240-202652")
    parser.add_argument("--counties", default="*", help="comma-separated FIPS, or *")
    parser.add_argument("--as-of", default=None, help="epiweek, e.g. 202607")
    parser.add_argument("--api-key", default=None, help="free Delphi key")
    args = parser.parse_args()

    issue, written = ingest(SnapshotStore(args.store), args.time_range,
                            args.counties, args.as_of, args.api_key)
    print(f"vintage issue date: {issue}")
    for signal, count in written.items():
        print(f"  {signal:<26} {count:>9,} rows")


if __name__ == "__main__":
    main()
