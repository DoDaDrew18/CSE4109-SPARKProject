"""Offline tests for the Open-Meteo / CAMS air quality ingester. No network.

The fixture copies the shape the live service returns: parallel hourly
arrays, local times without offset, nulls before the archive starts.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ingest_airquality import (AVAILABILITY_LAG_DAYS, COUNTY_POINTS, WEEKLY_COLUMNS,
                                   _parse, fetch_county, ingest, load_daily, weekly_features)
from src.study import STUDY_FIPS


def _payload(day: str, pm=1.0, oz=50.0, null_hours=0):
    times = [f"{day}T{h:02d}:00" for h in range(24)]
    pm25 = [None] * null_hours + [pm] * (24 - null_hours)
    ozone = [None] * null_hours + [oz] * (24 - null_hours)
    return {"latitude": 38.6, "longitude": -90.2,
            "hourly_units": {"time": "iso8601", "pm2_5": "μg/m³", "ozone": "μg/m³"},
            "hourly": {"time": times, "pm2_5": pm25, "ozone": ozone}}


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        day = params["start_date"]
        payload = _payload(day, pm=float(day[:4]) - 2000)
        return type("R", (), {"raise_for_status": lambda self: None,
                              "json": lambda self: payload})()


def test_every_study_county_has_a_point():
    assert set(COUNTY_POINTS) == set(STUDY_FIPS)


def test_parse_daily_means_and_short_days():
    full = _parse(_payload("2025-01-02", pm=4.0, oz=60.0), "29510")
    assert full[["pm25", "ozone", "aq_hours"]].iloc[0].tolist() == [4.0, 60.0, 24]
    short = _parse(_payload("2022-08-03", null_hours=18), "29510")   # archive starts 18:00
    assert np.isnan(short["pm25"].iloc[0]) and short["aq_hours"].iloc[0] == 6


def test_fetch_one_request_per_year_in_local_time():
    session = FakeSession()
    out = fetch_county("21111", date(2024, 12, 30), date(2025, 1, 2), session)
    assert [c["start_date"] for c in session.calls] == ["2024-12-30", "2025-01-01"]
    assert all(c["timezone"] == "America/New_York" for c in session.calls)
    assert set(out["fips"]) == {"21111"}


def test_ingest_merges_and_keeps_older_days(tmp_path):
    ingest(tmp_path, date(2024, 1, 1), date(2024, 1, 1), fips=["17031"], session=FakeSession())
    ingest(tmp_path, date(2025, 1, 1), date(2025, 1, 1), fips=["17031"], session=FakeSession())
    daily = load_daily(tmp_path, ["17031"])
    assert sorted(daily["date"].dt.strftime("%Y")) == ["2024", "2025"]
    assert "pulled_at" in daily


def test_load_daily_without_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_daily(tmp_path)


def _daily():
    days = pd.date_range("2025-01-05", "2025-01-25")      # Sun Jan 5 .. Sat Jan 25
    return pd.DataFrame({"fips": "29510", "date": days, "pm25": np.arange(len(days), dtype=float),
                         "ozone": 50.0, "aq_hours": 24})


def test_weekly_features_respects_lag():
    out = weekly_features("2025-01-25", fips="29510", daily=_daily())
    assert list(out.columns) == WEEKLY_COLUMNS
    # days through Jan 18 are public on Jan 25 (7-day lag); Jan 18 ends week 2
    assert out["week_end"].max() == pd.Timestamp("2025-01-18")
    assert out["aq_n_days"].tolist() == [7, 7]
    assert out["pm25"].iloc[0] == pytest.approx(3.0)        # mean of 0..6
    later = weekly_features("2025-01-29", fips="29510", daily=_daily())
    assert later.iloc[-1]["aq_n_days"] == 4                 # partial week kept
    assert AVAILABILITY_LAG_DAYS == 7


def test_weekly_features_rejects_non_study_county():
    with pytest.raises(KeyError):
        weekly_features("2025-01-25", fips="06075", daily=_daily())


def test_weekly_features_empty_before_any_public_day():
    assert weekly_features("2025-01-06", fips="29510", daily=_daily()).empty
