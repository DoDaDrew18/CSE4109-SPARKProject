"""Pull CDC NWSS wastewater viral activity levels into the snapshot store.

Wastewater is our care-seeking-independent early signal: a sewershed sheds
virus whether or not anyone goes to an ED. Usage, from the repo root:

    python -m src.ingest_nwss              # this week's vintage, nationwide
    python -m src.ingest_nwss --archive    # + every past vintage on the Wayback Machine
    python -m src.ingest_nwss --coverage   # print study-county coverage only

Source: Socrata dataset ``atcp-73re`` ("CDC Wastewater Viral Activity Level
for SARS-CoV-2, Influenza A and RSV"). One row per site x week x pathogen.
What we verified against the live API and archived copies (Oct 2026):

* Columns: state_territory (full name), counties_served (county NAMES joined
  with ", ", e.g. "Saint Louis, Saint Louis City"; 136 rows for one Colorado
  site are blank), site ("ID:1126", stable anonymous plant id),
  population_served, source (State_Territory / CDC_Verily / WastewaterSCAN
  ..., can change for a site over time), site_wval, site_wval_category
  (Very Low .. Very High), date_included_in_wval, week_end, pathogen_target
  ("SARS-CoV-2", "Influenza A virus", "RSV" -- flu is Influenza A only),
  date_updated. (site, week_end, pathogen_target) is unique.

* WVAL is CDC's site-relative level: the week's concentration measured
  against that site's own baseline, so ~1 is baseline and levels are
  comparable across sites with different lab methods. Flu and RSV sit at
  exactly 1.0 in ~58% of rows. The tail is absurd (max 2.7e21 for RSV; the
  99th percentile is ~36-62), so sites are capped at ``WVAL_CAP`` before
  averaging.

* History: week_end runs 2022-01-01 .. present for COVID and flu, from
  2022-03-05 for RSV; ~566k rows, ~1,780 sites. It is a full history, not a
  rolling window. week_end is always a Saturday, i.e. already our
  ``week_end`` convention.

* THERE IS A REVISION STORY, AND date_updated DOES NOT TELL IT. Every row of
  a pull carries the same date_updated (the timestamp of that Friday's CDC
  processing run), because CDC re-posts the whole history each week. We
  recovered five full vintages (2026-04-17, 07-10, 08-07, 09-04, 10-02) from
  the Wayback Machine and diffed them:
    - routine month-to-month: 3-8% of site-weeks change value, ~2% by >=10%,
      a little more in the most recent 4 weeks;
    - Aug 14, 2026: CDC changed the WVAL method and recomputed ALL history.
      Between the 08-07 and 09-04 vintages 48% of site-weeks changed, 17% by
      >=10%, and 31% changed category -- including 2022 values;
    - late sites: a week's first print has 15-25% fewer site rows than it
      settles at (week 2026-08-01: 2,414 rows on 08-07, 2,991 on 09-04).
  So a value is NOT fixed once published, and we vintage it exactly like
  NSSP: one snapshot per CDC processing run.

* Publication lag: each vintage's newest week_end is the Saturday 6 days
  before its Friday date_updated, in all five vintages
  (``PUBLICATION_LAG_DAYS``).

* Other NWSS datasets: the per-pathogen sample-level concentration sets
  (j9g8-acpt COVID from 2020, ymmh-divb flu A from 2021, 45cq-cw4i RSV from
  2022) do carry county_fips, but they are raw lab concentrations (method-
  dependent units) and are ALSO re-posted wholesale with one date_updated.
  2ew6-ywp6 / g653-rqe2 are COVID-only and stopped updating Aug 2026. WVAL
  (atcp-73re) is the only weekly, cross-site-comparable source for all
  three pathogens, so it is the one we use.

Why the snapshot issue is ``date_updated``'s date and not the pull date:
date_updated is when CDC posted that exact vintage (it went public that
Friday morning), and a Wayback copy captured weeks later still describes
that Friday. Stamping the pull date would hide data that was public. The
snapshot key is (site, week start Sunday) -- site, not county, because a
site serves several counties and the name->FIPS mapping is a choice we may
improve; snapshots keep exactly what CDC posted and counties are attached
at read time in :func:`weekly_features`.

Honest coverage therefore starts at the first archived vintage, 2026-04-17.
CDC created atcp-73re on 2026-03-04 and nothing earlier is archived, so
for an as_of before that there is no record of what WVAL said. Backtests
from April 2024 can opt into ``allow_proxy=True`` (see weekly_features), which
is not leak-free and must be reported as such.
"""

from __future__ import annotations

import argparse
import gzip
import io
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.fips import DEFAULT_PATH as FIPS_REFERENCE, FipsLookup
from src.snapshot import SnapshotStore, coerce_date

__all__ = ["DATASET", "PATHOGENS", "PUBLICATION_LAG_DAYS", "WVAL_CAP",
           "FEATURE_COLUMNS", "parse_export", "vintage_issue", "to_snapshots",
           "site_fips", "ingest_frame", "ingest_live", "ingest_archive",
           "weekly_features", "coverage"]

DATASET = "atcp-73re"
CSV_URL = f"https://data.cdc.gov/api/views/{DATASET}/rows.csv?accessType=DOWNLOAD"
# Both export endpoints the Wayback Machine has full copies of. The
# /resource/ endpoint is not listed: its captures are the 1,000-row default page.
ARCHIVE_PREFIXES = (f"data.cdc.gov/api/views/{DATASET}/rows.csv*",
                    f"data.cdc.gov/api/v3/views/{DATASET}/export.csv*")
CDX_URL = "https://web.archive.org/cdx/search/cdx"

PATHOGENS = {"SARS-CoV-2": "covid", "Influenza A virus": "flu", "RSV": "rsv"}
PUBLICATION_LAG_DAYS = 6
WVAL_CAP = 100.0
FEATURE_COLUMNS = ["fips", "week_end", "ww_covid", "ww_flu", "ww_rsv", "ww_n_sites"]

# A full vintage has ~500k rows; anything far smaller is a truncated page.
_MIN_VINTAGE_ROWS = 100_000
_UA = {"User-Agent": "CSE4109-SPARK/ingest"}


def source_name(short: str) -> str:
    """Snapshot source for a pathogen short name: 'flu' -> 'nwss_flu'."""
    return f"nwss_{short}"


# -- parsing ---------------------------------------------------------------


def parse_export(data) -> pd.DataFrame:
    """Read a CSV export (path, bytes, or gzip bytes) into typed columns.

    The legacy ``rows.csv`` and the v3 ``export.csv`` differ only cosmetically
    (v3 writes thousands separators, "694,399.71"), so both go through here.
    """
    if isinstance(data, (bytes, bytearray)):
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        data = io.BytesIO(data)
    raw = pd.read_csv(data, dtype=str, keep_default_na=False, na_values=[""])
    raw.columns = [c.strip().lower().replace("/", "_").replace(" ", "_")
                   for c in raw.columns]

    def number(col):
        return pd.to_numeric(raw[col].str.replace(",", "", regex=False), errors="coerce")

    return pd.DataFrame({
        "state_territory": raw["state_territory"].str.strip(),
        "counties_served": raw["counties_served"].str.strip(),
        "site": raw["site"].str.strip(),
        "population_served": number("population_served"),
        "data_source": raw["source"],
        "site_wval": number("site_wval"),
        "site_wval_category": raw["site_wval_category"],
        "date_included_in_wval": raw["date_included_in_wval"],
        "week_end": pd.to_datetime(raw["week_end"]),
        "pathogen_target": raw["pathogen_target"].str.strip(),
        "date_updated": raw["date_updated"],
    })


def vintage_issue(frame: pd.DataFrame) -> pd.Timestamp:
    """The CDC processing date of a pull; refuses a frame mixing two runs."""
    stamps = frame["date_updated"].dropna().unique()
    if len(stamps) != 1:
        raise ValueError(f"expected one date_updated per vintage, got {list(stamps)[:5]}")
    return coerce_date(stamps[0])


def to_snapshots(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split one vintage into a snapshot frame per pathogen, keyed by site.

    ``time_value`` is the Sunday starting the MMWR week, as for NSSP; the
    Saturday ``week_end`` rides along as an extra column.
    """
    issue = vintage_issue(frame)
    unknown = set(frame["pathogen_target"]) - set(PATHOGENS)
    if unknown:
        raise ValueError(f"unexpected pathogen_target values {sorted(unknown)}")

    out = {}
    for target, short in PATHOGENS.items():
        rows = frame[frame["pathogen_target"] == target]
        if rows.empty:
            continue
        snap = pd.DataFrame({
            "geo_value": rows["site"].to_numpy(),
            "time_value": (rows["week_end"] - timedelta(days=6)).to_numpy(),
            "value": rows["site_wval"].to_numpy(),
            "issue": issue,
            "week_end": rows["week_end"].to_numpy(),
        })
        for col in ("state_territory", "counties_served", "population_served",
                    "data_source", "site_wval_category", "date_included_in_wval",
                    "date_updated"):
            snap[col] = rows[col].to_numpy()
        out[source_name(short)] = snap
    return out


# -- writing ---------------------------------------------------------------


def ingest_frame(store: SnapshotStore, frame: pd.DataFrame, raw_dir=None,
                 raw_bytes: bytes | None = None) -> tuple[pd.Timestamp, dict]:
    """Write one vintage. Returns (issue, rows per source)."""
    issue = vintage_issue(frame)
    if raw_dir is not None and raw_bytes is not None:
        path = Path(raw_dir) / f"{DATASET}_updated={issue.date()}.csv.gz"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw_bytes if raw_bytes[:2] == b"\x1f\x8b"
                             else gzip.compress(raw_bytes))
    written = {}
    for source, snap in to_snapshots(frame).items():
        store.write_snapshot(source, issue, snap)
        written[source] = len(snap)
    return issue, written


def ingest_live(store: SnapshotStore, raw_dir="raw/nwss") -> tuple[pd.Timestamp, dict]:
    """Today's vintage, nationwide (one ~7 MB gzip download)."""
    response = requests.get(CSV_URL, timeout=300, headers=_UA)
    response.raise_for_status()
    body = response.content
    return ingest_frame(store, parse_export(body), raw_dir, body)


def archive_captures() -> list[tuple[str, str]]:
    """(timestamp, original URL) of every distinct full export on the Wayback Machine."""
    captures = []
    for prefix in ARCHIVE_PREFIXES:
        response = requests.get(CDX_URL, timeout=120, headers=_UA, params={
            "url": prefix, "output": "json", "filter": "statuscode:200",
            "collapse": "digest"})
        response.raise_for_status()
        rows = response.json()
        captures += [(r[1], r[2]) for r in rows[1:]]
    return sorted(captures)


def ingest_archive(store: SnapshotStore, raw_dir="raw/nwss") -> list[tuple]:
    """Backfill every archived vintage. Returns [(issue, rows per source)].

    A vintage already on disk (e.g. from our own live pull) is never
    re-derived from a secondhand copy, so run order doesn't matter.
    """
    done = set(store.available_issues(source_name("covid")))
    results = []
    for stamp, original in archive_captures():
        response = requests.get(f"https://web.archive.org/web/{stamp}id_/{original}",
                                timeout=300, headers=_UA)
        if not response.ok:
            continue
        frame = parse_export(response.content)
        if len(frame) < _MIN_VINTAGE_ROWS:
            continue
        issue = vintage_issue(frame)
        if issue in done:        # the same CDC run captured twice
            continue
        done.add(issue)
        results.append(ingest_frame(store, frame, raw_dir, response.content))
    return results


# -- reading ---------------------------------------------------------------


def site_fips(state: str, counties_served, lookup: FipsLookup,
              unmapped: list | None = None) -> tuple[str, ...]:
    """FIPS of every county a site serves.

    Names are looked up in the site's own state first. Sewersheds cross
    state lines (a Missouri site lists Wyandotte, KS), so a miss falls back to
    the one state nationwide with that county name; if several states have
    it ("Montgomery" for a DC plant) the county is dropped and recorded in
    ``unmapped`` rather than guessed.
    """
    if not isinstance(counties_served, str) or not counties_served.strip():
        return ()
    found = []
    for name in (n.strip() for n in counties_served.split(",")):
        if not name:
            continue
        try:
            found.append(lookup.county_fips(state, name))
            continue
        except KeyError:
            pass
        states = lookup.states_with(name)
        if len(states) == 1:
            found.append(lookup.county_fips(states[0], name))
        elif unmapped is not None:
            unmapped.append((state, name))
    return tuple(dict.fromkeys(found))


def _first_prints(store: SnapshotStore, source: str) -> pd.DataFrame:
    """Each (site, week) as first seen in any stored vintage."""
    frames = [pd.read_parquet(store.path_for(source, i))
              for i in store.available_issues(source)]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True).sort_values("issue", kind="mergesort")
    return combined.drop_duplicates(["geo_value", "time_value"], keep="first")


def _site_weeks(store: SnapshotStore, source: str, as_of: pd.Timestamp,
                allow_proxy: bool) -> pd.DataFrame:
    rows = store.load_vintage(source, as_of)
    if not allow_proxy:
        return rows
    # Proxy: weeks newer than anything the last vintage on/before as_of could
    # hold (the gap before the first archived vintage, or between two sparse
    # ones) are taken from the EARLIEST vintage that has them (least
    # revised), once the measured lag says they would have been out. Weeks an
    # honest vintage could hold are never touched, so late-arriving sites and
    # later pathogens don't leak into them. The proxy reproduces the lag, not
    # the revisions, so it is not leak-free.
    first = _first_prints(store, source)
    if first.empty:
        return rows
    week_end = pd.to_datetime(first["week_end"])
    lag = timedelta(days=PUBLICATION_LAG_DAYS)
    keep = week_end + lag <= as_of
    # Any NWSS vintage counts: a run that posted no rows for this pathogen
    # still shows the pathogen had nothing public for those weeks.
    honest = sorted(i for short in PATHOGENS.values()
                    for i in store.available_issues(source_name(short)) if i <= as_of)
    if honest:
        keep &= week_end + lag > honest[-1]
    extra = first[keep]
    if rows.empty or extra.empty:
        return extra if rows.empty else rows
    return pd.concat([rows, extra], ignore_index=True)


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame({
        "fips": pd.Series(dtype="object"),
        "week_end": pd.Series(dtype="datetime64[ns]"),
        "ww_covid": pd.Series(dtype="float64"),
        "ww_flu": pd.Series(dtype="float64"),
        "ww_rsv": pd.Series(dtype="float64"),
        "ww_n_sites": pd.Series(dtype="int64"),
    })


def weekly_features(as_of, fips=None, root="raw/snapshots", *,
                    allow_proxy: bool = False, lookup: FipsLookup | None = None,
                    reference=FIPS_REFERENCE) -> pd.DataFrame:
    """County-week wastewater levels as public on ``as_of``.

    Columns ``fips, week_end, ww_covid, ww_flu, ww_rsv, ww_n_sites``. Each
    level is the population-weighted mean of capped site WVALs over every site
    serving the county that week. A site serving several counties counts in
    each with its FULL population: NWSS does not say how a sewershed splits
    across counties, and splitting evenly would invent a precision we don't
    have. ``ww_n_sites`` counts distinct sites reporting any pathogen.

    Strict by default: only vintages CDC posted on/before ``as_of``, so for
    ``as_of`` before the first stored vintage (2026-04-17 with the archive
    backfill) the result is empty. ``allow_proxy=True`` fills older weeks from
    later vintages under the publication-lag rule; label such results.
    """
    as_of = coerce_date(as_of)
    store = SnapshotStore(root)
    lookup = lookup or FipsLookup.from_file(reference)
    wanted = None if fips is None else {str(f).zfill(5) for f in
                                        ([fips] if isinstance(fips, str) else fips)}

    parts = []
    for short in PATHOGENS.values():
        rows = _site_weeks(store, source_name(short), as_of, allow_proxy)
        if len(rows):
            parts.append(rows.assign(pathogen=short))
    if not parts:
        return _empty_features()
    rows = pd.concat(parts, ignore_index=True)

    sites = rows[["state_territory", "counties_served"]].drop_duplicates()
    sites["fips"] = [site_fips(s, c, lookup) for s, c in sites.itertuples(index=False)]
    rows = rows.merge(sites, on=["state_territory", "counties_served"], how="left")
    rows = rows.explode("fips").dropna(subset=["fips"])
    if wanted is not None:
        rows = rows[rows["fips"].isin(wanted)]
    rows = rows.dropna(subset=["value"])
    if rows.empty:
        return _empty_features()

    rows["week_end"] = pd.to_datetime(rows["week_end"])
    rows["level"] = rows["value"].clip(upper=WVAL_CAP)
    rows["weight"] = rows["population_served"].fillna(0).clip(lower=0)
    return _aggregate(rows)


def _aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted mean per county-week-pathogen, pivoted wide."""
    keys = ["fips", "week_end"]

    def weighted(group):
        w = group["weight"].to_numpy(dtype=float)
        if w.sum() <= 0:                 # no population known: plain mean
            w = np.ones_like(w)
        return float(np.average(group["level"].to_numpy(dtype=float), weights=w))

    levels = (rows.groupby(keys + ["pathogen"])[["level", "weight"]]
              .apply(weighted).rename("v").reset_index()
              .pivot(index=keys, columns="pathogen", values="v"))
    levels = levels.reindex(columns=list(PATHOGENS.values()))
    levels.columns = [f"ww_{c}" for c in levels.columns]
    n_sites = rows.groupby(keys)["geo_value"].nunique().rename("ww_n_sites")
    out = levels.join(n_sites).reset_index()
    out["fips"] = out["fips"].astype(str)
    out["ww_n_sites"] = out["ww_n_sites"].astype("int64")
    return out[FEATURE_COLUMNS].sort_values(keys).reset_index(drop=True)


def coverage(fips, root="raw/snapshots", reference=FIPS_REFERENCE) -> pd.DataFrame:
    """Per study county: sites, weeks per pathogen, first/last week (latest vintage)."""
    store = SnapshotStore(root)
    issues = store.available_issues(source_name("covid"))
    if not issues:
        return pd.DataFrame()
    feats = weekly_features(issues[-1], fips, root, reference=reference)
    rows = []
    for f in fips:
        sub = feats[feats["fips"] == f]
        row = {"fips": f, "max_sites": int(sub["ww_n_sites"].max()) if len(sub) else 0}
        for p in PATHOGENS.values():
            have = sub.dropna(subset=[f"ww_{p}"])
            row[f"weeks_{p}"] = len(have)
            row[f"first_{p}"] = have["week_end"].min().date() if len(have) else None
            row[f"last_{p}"] = have["week_end"].max().date() if len(have) else None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull CDC NWSS wastewater WVAL.")
    parser.add_argument("--store", default="raw/snapshots")
    parser.add_argument("--raw", default="raw/nwss")
    parser.add_argument("--archive", action="store_true",
                        help="also backfill past vintages from the Wayback Machine")
    parser.add_argument("--coverage", action="store_true",
                        help="only print study-county coverage")
    args = parser.parse_args()
    store = SnapshotStore(args.store)

    if not args.coverage:
        issue, written = ingest_live(store, args.raw)
        print(f"vintage issue date: {issue.date()}")
        for source, count in written.items():
            print(f"  {source:<12} {count:>9,} rows")
        if args.archive:
            for issue, written in ingest_archive(store, args.raw):
                print(f"archived vintage {issue.date()}: {written}")

    from src.study import STUDY_FIPS
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(coverage(list(STUDY_FIPS), args.store).to_string(index=False))


if __name__ == "__main__":
    main()
