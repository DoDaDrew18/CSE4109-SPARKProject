"""Pull daily airport weather for the six study metros and roll it up by MMWR week.

Hypothesis 1 is that cold, dry weeks come before flu/RSV rises, so we need
temperature, precipitation and -- above all -- dew point per metro per week.
Usage, from the repo root:

    python -m src.ingest_noaa                               # 2024-01-01 .. yesterday
    python -m src.ingest_noaa --start 2025-10-01 --end 2026-03-31

Downstream code reads only ``weekly_features(as_of)``, never the raw files.

What we verified against the live services (Oct 6 2026)
-------------------------------------------------------
* Every station id in ``src.study.METROS`` is the intended airport. NCEI's
  ``includeStationName`` returns Lambert-St. Louis, Kansas City Intl, Chicago
  O'Hare, Indianapolis Intl, Memphis Intl and Louisville Intl, and each one
  has a full daily record from 2024-01-01 on (TMAX/TMIN/PRCP/AWND >99.8%).

* **TAVG does not exist** for any of the six stations (Lambert's TAVG ended
  in 2005). ``temp_mean_f`` is therefore (TMAX+TMIN)/2. TAVG is still
  requested, and preferred if it ever shows up.

* **NCEI humidity stops on 2024-12-31.** ADPT (mean dew point), RHAV/RHMN/RHMX,
  AWBT and ASLP are present for every day of 2024 at all six stations and
  for no day after, both in the Access service and in the raw GHCN-Daily
  by-station files. NCEI's other humidity datasets don't fill the gap:
  ``global-hourly`` and ``global-summary-of-the-day`` return ``[]`` through
  the Access service, and their file archives stop on 2025-08-27 with no
  2026 directory. That would leave both study flu seasons with no dew point.

* So humidity comes from the **Iowa Environmental Mesonet (IEM) ASOS
  archive**: the same NWS airport sensors (METARs), no key, current to
  today. We average the routine hourly METARs over each local day. Checked
  against NCEI for STL in 2024 (366 days), the IEM daily mean dew point
  matches ADPT with r=0.999 (mean difference -0.06F, sd 0.61F), and mean RH
  matches RHAV with r=0.997 (difference +0.16, sd 0.96). We use IEM for
  every day, including 2024, so the series has no break at a source switch.
  ADPT/RHAV are kept in the raw file and fill in only if IEM is missing.

* ``dataTypes`` with a type the station lacks (TAVG) or that doesn't exist
  at all (BOGUS) is NOT an error. The service returns 200 and leaves the key
  out of each row, and asking only for absent types returns rows that have
  just DATE and STATION. A missing field therefore can't be told apart from
  a typo, so ``_parse_ncei`` always emits every column, NaN when absent.

* ``units=standard`` gives degrees F and inches as clean strings ("36",
  "0.00"). Without it you get tenths of C and tenths of mm, padded with
  spaces ("   22"). We always send ``units=standard`` and strip whitespace.
  RHAV comes padded even in standard units ("   76").

* **Availability lag.** On 2026-10-06 the Access service's latest day for
  all six stations was 2026-09-29 (7 days old). The raw GHCN-Daily file
  (modified 2026-10-07 00:08 UTC) reached 2026-10-03, so the underlying
  feed runs about 3 days behind and the Access service lags it by a few
  more. We use ``AVAILABILITY_LAG_DAYS = 7``, the slower of the two, which
  is the conservative choice. IEM is current to the same day, but one lag
  for every weather field keeps a week's temperature and dew point in step.
  A 7-day lag costs little because weather is expected to lead illness by
  1-3 weeks anyway.

Why plain per-station parquet and not the snapshot store
--------------------------------------------------------
``SnapshotStore`` keeps one immutable vintage per issue date because NSSP
values are revised for weeks. Daily weather is effectively never revised
after its quality checks, so the only point-in-time question is *when a day
became public*, and a fixed lag answers that. Weekly vintages would store
near-identical copies of the whole record each week. Instead each station
gets one file that is rewritten in place (new pull wins on overlapping
dates, older dates kept), with a ``pulled_at`` column, and ``weekly_features``
enforces ``as_of`` itself.
"""

from __future__ import annotations

import argparse
import io
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.study import METROS, STUDY_FIPS, week_end_of

NCEI_API = "https://www.ncei.noaa.gov/access/services/data/v1"
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

AVAILABILITY_LAG_DAYS = 7
DEFAULT_START = date(2024, 1, 1)

# GHCN element -> our column. Every one is requested; absent ones become NaN.
NCEI_FIELDS = {
    "TMAX": "tmax_f", "TMIN": "tmin_f", "TAVG": "tavg_f", "PRCP": "prcp_in",
    "ADPT": "adpt_f", "RHAV": "rhav_pct", "AWND": "awnd_mph",
}
HUMIDITY_COLUMNS = ("dewpoint_f", "rh_mean")

# GHCN station -> (IEM ASOS id, local time zone). The time zone makes IEM's
# local-day averages line up with GHCN's local-day summaries.
ASOS = {
    "USW00013994": ("STL", "America/Chicago"),
    "USW00003947": ("MCI", "America/Chicago"),
    "USW00094846": ("ORD", "America/Chicago"),
    "USW00093819": ("IND", "America/Indiana/Indianapolis"),
    "USW00013893": ("MEM", "America/Chicago"),
    "USW00093821": ("SDF", "America/New_York"),
}

WEEKLY_COLUMNS = ["fips", "week_end", "temp_mean_f", "temp_min_f", "temp_max_f",
                  "dewpoint_f", "rh_mean", "precip_in", "n_days"]

_UA = {"User-Agent": "CSE4109-SPARK/ingest (WashU class project)"}


def _session() -> requests.Session:
    """A session that retries the transient failures both services throw."""
    retry = Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(_UA)
    return session


def _year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] by calendar year so one failed request costs one year."""
    return [(max(start, date(y, 1, 1)), min(end, date(y, 12, 31)))
            for y in range(start.year, end.year + 1)]


# -- NCEI daily summaries ------------------------------------------------------


def _parse_ncei(payload: list[dict]) -> pd.DataFrame:
    """NCEI JSON rows -> one row per (station, date), every field numeric.

    Values arrive as space-padded strings, and a field the station lacks is
    simply missing from the row, so each column is built explicitly.
    """
    raw = pd.DataFrame(payload)
    out = pd.DataFrame({
        "station": raw.get("STATION", pd.Series(dtype=str)).astype(str),
        "date": pd.to_datetime(raw.get("DATE", pd.Series(dtype=str))),
    })
    for element, column in NCEI_FIELDS.items():
        values = raw[element] if element in raw else pd.Series(np.nan, index=raw.index)
        out[column] = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
    return out


def fetch_ncei(station: str, start: date, end: date,
               session: requests.Session | None = None) -> pd.DataFrame:
    """Daily summaries for one station, in degrees F and inches."""
    session = session or _session()
    frames = []
    for lo, hi in _year_chunks(start, end):
        params = {"dataset": "daily-summaries", "stations": station,
                  "startDate": lo.isoformat(), "endDate": hi.isoformat(),
                  "dataTypes": ",".join(NCEI_FIELDS), "units": "standard",
                  "format": "json"}
        response = session.get(NCEI_API, params=params, timeout=120)
        response.raise_for_status()
        frames.append(_parse_ncei(response.json()))
    return pd.concat(frames, ignore_index=True) if frames else _parse_ncei([])


# -- IEM ASOS humidity ---------------------------------------------------------


def _parse_iem(text: str) -> pd.DataFrame:
    """Hourly METAR CSV -> daily mean dew point and RH per local day."""
    hourly = pd.read_csv(io.StringIO(text))
    hourly["date"] = pd.to_datetime(hourly["valid"]).dt.normalize()
    for column in ("dwpf", "relh"):
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce")
    daily = hourly.groupby("date").agg(
        iem_dewpoint_f=("dwpf", "mean"), iem_rh_pct=("relh", "mean"),
        iem_n_obs=("dwpf", "count")).reset_index()
    return daily


def fetch_iem_humidity(station: str, start: date, end: date,
                       session: requests.Session | None = None) -> pd.DataFrame:
    """Daily mean dew point (F) and RH (%) from routine hourly METARs.

    ``report_type=3`` keeps only routine hourly reports. Special reports
    cluster around bad weather and would tilt the daily mean toward it.
    """
    asos_id, tz = ASOS[station]
    session = session or _session()
    frames = []
    for lo, hi in _year_chunks(start, end):
        upper = pd.Timestamp(hi) + pd.Timedelta(days=1)   # year2/month2/day2 is exclusive
        params = {"station": asos_id, "data": ["dwpf", "relh"],
                  "year1": lo.year, "month1": lo.month, "day1": lo.day,
                  "year2": upper.year, "month2": upper.month, "day2": upper.day,
                  "tz": tz, "format": "onlycomma", "latlon": "no",
                  "missing": "empty", "trace": "empty", "report_type": 3}
        response = session.get(IEM_ASOS, params=params, timeout=300)
        response.raise_for_status()
        frames.append(_parse_iem(response.text))
    daily = pd.concat(frames, ignore_index=True)
    daily.insert(0, "station", station)
    return daily


# -- storage -------------------------------------------------------------------


def _merge_write(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge a new pull into the station file. Overlapping dates take the new pull.

    Rerunning the same pull is idempotent apart from ``pulled_at``. A narrow
    pull (``--start 2026-09-01``) refreshes those dates without dropping the
    rest of the record.
    """
    if path.exists():
        old = pd.read_parquet(path)
        fresh = pd.concat([old[~old["date"].isin(fresh["date"])], fresh],
                          ignore_index=True)
    fresh = fresh.sort_values("date").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(path, index=False)
    return fresh


def ingest(root="raw/noaa", start: date = DEFAULT_START, end: date | None = None,
           stations=None, session: requests.Session | None = None) -> dict:
    """Pull NCEI + IEM for each station into ``root``. Returns rows per station."""
    # Yesterday, not today: today's hourly average is incomplete and would
    # sit in the file until the next pull.
    end = end or date.today() - timedelta(days=1)
    stations = stations or [m.noaa_station for m in METROS.values()]
    session = session or _session()
    root = Path(root)
    pulled_at = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)

    written = {}
    for station in stations:
        ncei = fetch_ncei(station, start, end, session)
        humid = fetch_iem_humidity(station, start, end, session)
        daily = ncei.merge(humid, on=["station", "date"], how="outer")
        daily = daily[(daily["date"] >= pd.Timestamp(start))
                      & (daily["date"] <= pd.Timestamp(end))]
        daily["pulled_at"] = pulled_at
        written[station] = len(_merge_write(root / f"{station}.parquet", daily))
    return written


def load_daily(root="raw/noaa", stations=None) -> pd.DataFrame:
    """Every stored day for the given stations (all study stations by default)."""
    stations = stations or [m.noaa_station for m in METROS.values()]
    frames = [pd.read_parquet(p) for s in stations
              if (p := Path(root) / f"{s}.parquet").exists()]
    if not frames:
        raise FileNotFoundError(f"no NOAA station files under {root}; run "
                                f"`python -m src.ingest_noaa` first")
    return pd.concat(frames, ignore_index=True)


# -- weekly features -----------------------------------------------------------


def _daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Pick one value per field per day, with the fallbacks documented above."""
    out = pd.DataFrame({"station": daily["station"], "date": daily["date"]})
    out["temp_max_f"] = daily["tmax_f"]
    out["temp_min_f"] = daily["tmin_f"]
    out["temp_mean_f"] = daily["tavg_f"].fillna((daily["tmax_f"] + daily["tmin_f"]) / 2)
    out["precip_in"] = daily["prcp_in"]
    # IEM first so 2024 and 2025+ come from one source, NCEI only as a fallback.
    out["dewpoint_f"] = daily.get("iem_dewpoint_f", np.nan)
    out["dewpoint_f"] = out["dewpoint_f"].fillna(daily["adpt_f"])
    out["rh_mean"] = daily.get("iem_rh_pct", np.nan)
    out["rh_mean"] = out["rh_mean"].fillna(daily["rhav_pct"])
    return out


def weekly_features(as_of, fips=None, root="raw/noaa",
                    lag_days: int = AVAILABILITY_LAG_DAYS,
                    daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weekly weather per study county, using only days public by ``as_of``.

    A day counts only if ``date + lag_days <= as_of``, so the most recent
    week is usually partial or absent. Partial weeks are kept, with
    ``n_days`` giving the number of days that had a temperature reading.
    ``precip_in`` is a sum and so runs low in a partial week; normalize with
    ``n_days`` if that matters. Temperatures, dew point and RH are means of
    daily values: ``temp_min_f`` is the mean daily low, not the week's
    lowest reading, because the extreme is noisier for the same signal.

    ``daily`` lets tests pass rows in directly instead of reading ``root``.
    """
    if fips is None:
        fips = list(STUDY_FIPS)
    elif isinstance(fips, str):
        fips = [fips]
    station_of = {f: m.noaa_station for m in METROS.values() for f in m.counties}
    unknown = [f for f in fips if f not in station_of]
    if unknown:
        raise KeyError(f"not study counties: {unknown}")

    stations = sorted({station_of[f] for f in fips})
    if daily is None:
        daily = load_daily(root, stations)
    days = _daily_features(daily[daily["station"].isin(stations)])
    cutoff = pd.Timestamp(as_of).normalize() - pd.Timedelta(days=lag_days)
    days = days[days["date"] <= cutoff]
    if days.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)

    days = days.assign(week_end=days["date"].map(week_end_of))
    weekly = days.groupby(["station", "week_end"]).agg(
        temp_mean_f=("temp_mean_f", "mean"), temp_min_f=("temp_min_f", "mean"),
        temp_max_f=("temp_max_f", "mean"), dewpoint_f=("dewpoint_f", "mean"),
        rh_mean=("rh_mean", "mean"),
        precip_in=("precip_in", lambda s: s.sum(min_count=1)),
        n_days=("temp_mean_f", "count"),
    ).reset_index()

    counties = pd.DataFrame({"fips": fips, "station": [station_of[f] for f in fips]})
    out = counties.merge(weekly, on="station").drop(columns="station")
    out["n_days"] = out["n_days"].astype("int64")
    return out[WEEKLY_COLUMNS].sort_values(["fips", "week_end"]).reset_index(drop=True)


# -- CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull daily airport weather.")
    parser.add_argument("--start", default=DEFAULT_START.isoformat())
    parser.add_argument("--end", default=None, help="default: yesterday")
    parser.add_argument("--root", default="raw/noaa")
    args = parser.parse_args()

    end = date.fromisoformat(args.end) if args.end else None
    written = ingest(args.root, date.fromisoformat(args.start), end)
    daily = load_daily(args.root, list(written))
    for station, count in written.items():
        rows = daily[daily["station"] == station]
        last_temp = rows.loc[rows["tmax_f"].notna(), "date"].max()
        print(f"  {station}  {count:>6,} days  {rows['date'].min():%Y-%m-%d}"
              f" .. {rows['date'].max():%Y-%m-%d}  (last TMAX {last_temp:%Y-%m-%d})")


if __name__ == "__main__":
    main()
