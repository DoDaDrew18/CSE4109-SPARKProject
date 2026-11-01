"""Pull modeled air quality (PM2.5, ozone) for the study counties and roll it up by MMWR week.

Fills the "EPA AirNow: deferred" row of the data table without an API key:
Open-Meteo's Air Quality API serves the Copernicus Atmosphere Monitoring
Service (CAMS) model at any point, hourly, with an archive. Usage, from the
repo root:

    python -m src.ingest_airquality                          # 2024-01-01 .. yesterday
    python -m src.ingest_airquality --start 2025-10-01 --end 2026-03-31

Downstream code reads only ``weekly_features(as_of)``, never the raw files.
The same source powers the PM2.5 check of the Coordinate (CalHacks 13) app
(``web/src/lib/agents/workers/envBurden.ts``), where it was used for the
last 7 days only; here we pull the archive.

What we verified against the live service (Nov 1 2026)
------------------------------------------------------
* No key, one request per county per calendar year (``start_date`` /
  ``end_date``), hourly ``pm2_5`` and ``ozone`` in ug/m3. With
  ``timezone=America/Chicago`` the hours are local, so daily means line up
  with the NOAA local-day summaries.
* **The archive starts 2022-08-03 18:00** (local) at St. Louis; earlier
  hours come back as nulls, not an error. Every study week (Apr 2024 on) is
  covered.
* The point is snapped to a coarse model grid. The St. Louis city and
  County centroids return **identical** daily series over the whole
  archive (both reported as 38.6, -90.2/-90.4), so, as with NSSP, the two
  St. Louis counties are one air-quality series. Small moves can still
  switch cells: (38.63, -90.20) gives PM2.5 3.0 at 2024-01-01 00:00 where
  the city centroid gives 3.3, so the points are pinned in ``COUNTY_POINTS``.
* ``past_days`` responses run up to *today* and include forecast hours, so
  the newest days are model forecasts, not analyses.

This is model output, not a measurement, and not vintaged
---------------------------------------------------------
CAMS values are a chemical-transport model run with data assimilation, and
Open-Meteo does not publish dated versions of its archive: a day's value
may be replaced after the fact (forecast -> analysis) and we cannot see
what it was on a past date. So, as with weather, the only point-in-time
rule is a fixed lag: a day counts only ``AVAILABILITY_LAG_DAYS`` (7) after
it ended, by which time it is no longer a forecast. Report this source as
"modeled, available-after-lag", never as a reconstructed vintage. Storage
follows ``src.ingest_noaa``: one file per county, rewritten in place (new
pull wins on overlapping dates) with a ``pulled_at`` column.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.ingest_noaa import _merge_write, _session, _year_chunks
from src.study import STUDY_FIPS, week_end_of

__all__ = ["AQ_API", "AVAILABILITY_LAG_DAYS", "DEFAULT_START", "COUNTY_POINTS",
           "WEEKLY_COLUMNS", "fetch_county", "ingest", "load_daily", "weekly_features"]

AQ_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
AVAILABILITY_LAG_DAYS = 7
DEFAULT_START = date(2024, 1, 1)
ARCHIVE_START = date(2022, 8, 3)
MIN_HOURS = 18          # a day with fewer non-null hours has no daily mean

# Equal-area centroids of the Census 2023 county boundaries
# (``src.spatial.shapes.add_centroids``), and each county's local time zone.
COUNTY_POINTS: dict[str, tuple[float, float, str]] = {
    "29510": (38.635825, -90.245112, "America/Chicago"),     # St. Louis city
    "29189": (38.640618, -90.443634, "America/Chicago"),     # St. Louis County
    "29095": (39.008411, -94.346185, "America/Chicago"),     # Jackson County, MO
    "17031": (41.839601, -87.816203, "America/Chicago"),     # Cook County, IL
    "18097": (39.781654, -86.138485, "America/Indiana/Indianapolis"),  # Marion, IN
    "47157": (35.184184, -89.894079, "America/Chicago"),     # Shelby County, TN
    "21111": (38.187184, -85.659230, "America/New_York"),    # Jefferson County, KY
}

HOURLY = ("pm2_5", "ozone")
WEEKLY_COLUMNS = ["fips", "week_end", "pm25", "ozone", "aq_n_days"]


def _parse(payload: dict, fips: str) -> pd.DataFrame:
    """Hourly response -> one row per local day: mean PM2.5 / ozone, hours seen."""
    hourly = payload.get("hourly") or {}
    frame = pd.DataFrame({"time": pd.to_datetime(hourly.get("time", [])),
                          **{k: pd.to_numeric(pd.Series(hourly.get(k, []), dtype="object"),
                                              errors="coerce") for k in HOURLY}})
    if frame.empty:
        return pd.DataFrame(columns=["fips", "date", "pm25", "ozone", "aq_hours"])
    frame["date"] = frame["time"].dt.normalize()
    daily = frame.groupby("date").agg(pm25=("pm2_5", "mean"), ozone=("ozone", "mean"),
                                      aq_hours=("pm2_5", "count")).reset_index()
    short = daily["aq_hours"] < MIN_HOURS
    daily.loc[short, ["pm25", "ozone"]] = float("nan")
    daily.insert(0, "fips", fips)
    return daily


def fetch_county(fips: str, start: date, end: date,
                 session: requests.Session | None = None) -> pd.DataFrame:
    """Daily means for one county over [start, end], one request per year."""
    lat, lon, tz = COUNTY_POINTS[fips]
    session = session or _session()
    frames = []
    for lo, hi in _year_chunks(max(start, ARCHIVE_START), end):
        resp = session.get(AQ_API, params={
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "hourly": ",".join(HOURLY),
            "start_date": lo.isoformat(), "end_date": hi.isoformat(), "timezone": tz,
        }, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("error"):
            raise ValueError(f"Open-Meteo air quality {fips} {lo}..{hi}: {payload.get('reason')}")
        frames.append(_parse(payload, fips))
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=["fips", "date", "pm25", "ozone", "aq_hours"])
    return pd.concat(frames, ignore_index=True)


def ingest(root="raw/airquality", start: date = DEFAULT_START, end: date | None = None,
           fips=None, session: requests.Session | None = None) -> dict:
    """Pull every study county into ``root``. Returns stored days per county."""
    end = end or date.today() - timedelta(days=1)
    fips = list(fips or STUDY_FIPS)
    session = session or _session()
    pulled_at = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    written = {}
    for f in fips:
        daily = fetch_county(f, start, end, session)
        daily["pulled_at"] = pulled_at
        written[f] = len(_merge_write(Path(root) / f"{f}.parquet", daily))
    return written


def load_daily(root="raw/airquality", fips=None) -> pd.DataFrame:
    """Every stored day for the given counties (all study counties by default)."""
    fips = list(fips or STUDY_FIPS)
    frames = [pd.read_parquet(p) for f in fips if (p := Path(root) / f"{f}.parquet").exists()]
    if not frames:
        raise FileNotFoundError(f"no air quality files under {root}; run "
                                f"`python -m src.ingest_airquality` first")
    out = pd.concat(frames, ignore_index=True)
    out["fips"] = out["fips"].astype(str).str.zfill(5)
    return out


def weekly_features(as_of, fips=None, root="raw/airquality",
                    lag_days: int = AVAILABILITY_LAG_DAYS,
                    daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weekly mean PM2.5 and ozone per study county, using days public by ``as_of``.

    A day counts only if ``date + lag_days <= as_of``; partial weeks are kept
    with ``aq_n_days`` (days with a PM2.5 mean). ``daily`` lets tests pass
    rows in directly instead of reading ``root``.
    """
    if fips is None:
        fips = list(STUDY_FIPS)
    elif isinstance(fips, str):
        fips = [fips]
    unknown = [f for f in fips if f not in COUNTY_POINTS]
    if unknown:
        raise KeyError(f"not study counties: {unknown}")
    if daily is None:
        daily = load_daily(root, fips)
    days = daily[daily["fips"].isin(fips)]
    cutoff = pd.Timestamp(as_of).normalize() - pd.Timedelta(days=lag_days)
    days = days[pd.to_datetime(days["date"]) <= cutoff]
    if days.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    days = days.assign(week_end=pd.to_datetime(days["date"]).map(week_end_of))
    weekly = days.groupby(["fips", "week_end"]).agg(
        pm25=("pm25", "mean"), ozone=("ozone", "mean"), aq_n_days=("pm25", "count"),
    ).reset_index()
    weekly["aq_n_days"] = weekly["aq_n_days"].astype("int64")
    return weekly[WEEKLY_COLUMNS].sort_values(["fips", "week_end"]).reset_index(drop=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Pull CAMS air quality (Open-Meteo).")
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--end", default=None, help="default: yesterday")
    parser.add_argument("--root", default="raw/airquality")
    args = parser.parse_args(argv)

    end = date.fromisoformat(args.end) if args.end else None
    written = ingest(args.root, date.fromisoformat(args.start), end)
    daily = load_daily(args.root, list(written))
    for f, count in written.items():
        rows = daily[(daily["fips"] == f) & daily["pm25"].notna()]
        print(f"  {f}  {count:>5,} days  {rows['date'].min():%Y-%m-%d} .. "
              f"{rows['date'].max():%Y-%m-%d}  PM2.5 mean {rows['pm25'].mean():.1f}, "
              f"ozone mean {rows['ozone'].mean():.1f} ug/m3")


if __name__ == "__main__":
    main()
