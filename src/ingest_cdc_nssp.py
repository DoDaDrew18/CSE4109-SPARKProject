"""Pull CDC's own NSSP county table (data.cdc.gov ``rdmq-nq56``) into the snapshot store.

Why a second NSSP source: Delphi relays this same CDC feed, but on 2026-10-10
Delphi's newest week ended Sep 12 while CDC's ended Oct 3 -- four weeks
fresher, and freshness is the point of nowcasting.

What CDC does not give us is history: the table is overwritten in place every
week, so the only vintages that will ever exist are the ones we save. Run it
weekly, from the repo root:

    python -m src.ingest_cdc_nssp

Two facts about the table that matter downstream:

* The values are Health Service Area (HSA) values, not county values
  (``trend_source`` is "HSA" on every row). Counties in one HSA carry the same
  number -- St. Louis city and County, Cook and DuPage. About 950 HSAs back
  ~3,100 counties; ``hsa_nci_id`` is stored so a model can count each HSA once.
* ``percent_visits_combined`` ended 2024-09-28 here too, so COVID, flu and RSV
  are stored separately, as in ``src.ingest_nssp``.

Snapshots use the same layout as the Delphi ingester -- ``time_value`` is the
Sunday a week starts (CDC's ``week_end`` minus 6 days) -- under the sources
``cdc_nssp_covid``, ``cdc_nssp_influenza`` and ``cdc_nssp_rsv``.
"""

from __future__ import annotations

import argparse
import io
from datetime import date

import pandas as pd
import requests

from src.snapshot import SnapshotStore

DATASET = "https://data.cdc.gov/resource/rdmq-nq56"
COLUMNS = ["week_end", "geography", "county", "fips", "hsa_nci_id",
           "percent_visits_covid", "percent_visits_influenza", "percent_visits_rsv"]
SIGNALS = {
    "percent_visits_covid": "cdc_nssp_covid",
    "percent_visits_influenza": "cdc_nssp_influenza",
    "percent_visits_rsv": "cdc_nssp_rsv",
}
PAGE = 200_000
_UA = {"User-Agent": "CSE4109-SPARK/ingest"}


def fetch_table() -> pd.DataFrame:
    """The whole table as CDC serves it today, paged, with a row-count check."""
    expected = int(requests.get(f"{DATASET}.json", params={"$select": "count(*)"},
                                timeout=60, headers=_UA).json()[0]["count"])
    pages, offset = [], 0
    while True:
        response = requests.get(f"{DATASET}.csv", timeout=300, headers=_UA, params={
            "$select": ",".join(COLUMNS),
            # A total order, so pages neither overlap nor skip rows.
            "$order": "week_end,geography,county,fips",
            "$limit": PAGE, "$offset": offset,
        })
        response.raise_for_status()
        page = pd.read_csv(io.StringIO(response.text), dtype=str)
        pages.append(page)
        if len(page) < PAGE:
            break
        offset += PAGE

    table = pd.concat(pages, ignore_index=True)
    if len(table) != expected:
        raise RuntimeError(f"paged {len(table):,} rows but CDC reports {expected:,}")
    return table


def to_snapshots(table: pd.DataFrame, issue) -> dict[str, pd.DataFrame]:
    """Split CDC's wide table into one snapshot frame per signal.

    Drops the state rollup rows (``county == "All"``) and rows where a signal
    is blank, which CDC uses for suppressed or unreported HSAs.
    """
    rows = table[(table["county"] != "All") & table["fips"].notna()]
    base = pd.DataFrame({
        "geo_value": rows["fips"].str.split(".").str[0].str.zfill(5),
        "time_value": pd.to_datetime(rows["week_end"]).dt.normalize() - pd.Timedelta(days=6),
        "hsa_nci_id": rows["hsa_nci_id"],
    })
    frames = {}
    for column, source in SIGNALS.items():
        frame = base.assign(value=pd.to_numeric(rows[column], errors="coerce"),
                            issue=pd.Timestamp(issue))
        frames[source] = frame[frame["value"].notna()].reset_index(drop=True)
    return frames


def ingest(store: SnapshotStore, issue: date | None = None) -> tuple[date, dict]:
    issue = issue or date.today()
    written = {}
    for source, frame in to_snapshots(fetch_table(), issue).items():
        store.write_snapshot(source, issue, frame)
        written[source] = len(frame)
    return issue, written


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull CDC's NSSP county table.")
    parser.add_argument("--store", default="raw/snapshots")
    args = parser.parse_args()

    issue, written = ingest(SnapshotStore(args.store))
    print(f"vintage issue date: {issue}")
    for source, count in written.items():
        print(f"  {source:<20} {count:>9,} rows")


if __name__ == "__main__":
    main()
