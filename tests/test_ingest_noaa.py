"""Offline tests for NOAA weather ingestion and the weekly feature contract. No network.

The fixtures copy the shapes the live services really return: NCEI values
as space-padded strings with absent fields left out of the row, and IEM
hourly METARs as CSV.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ingest_noaa import (
    AVAILABILITY_LAG_DAYS,
    WEEKLY_COLUMNS,
    _merge_write,
    _parse_iem,
    _parse_ncei,
    _year_chunks,
    fetch_ncei,
    weekly_features,
)

STL = "USW00013994"
STL_CITY, STL_COUNTY, COOK = "29510", "29189", "17031"

# Shape of a real units=standard response. TAVG was requested but the station
# lacks it, so the key is simply absent. RHAV comes space-padded.
NCEI_PAYLOAD = [
    {"DATE": "2024-12-31", "STATION": STL, "TMAX": "40", "TMIN": "30",
     "PRCP": "0.12", "ADPT": "28", "RHAV": "   76", "AWND": "6.71"},
    {"DATE": "2025-01-01", "STATION": STL, "TMAX": "36", "TMIN": "26",
     "PRCP": "0.00", "AWND": "4.25"},
]

IEM_CSV = """station,valid,dwpf,relh
STL,2026-01-01 00:51,24.00,61.37
STL,2026-01-01 01:51,26.00,69.39
STL,2026-01-01 02:51,,
STL,2026-01-02 00:51,30.00,80.00
"""


def daily_rows(start, n, station=STL, tmax=50.0, tmin=30.0, dew=25.0, rh=60.0,
               prcp=0.1):
    """Synthetic stored rows, one per day, in the raw-file schema."""
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "station": station, "date": dates, "tmax_f": tmax, "tmin_f": tmin,
        "tavg_f": np.nan, "prcp_in": prcp, "adpt_f": np.nan, "rhav_pct": np.nan,
        "awnd_mph": 5.0, "iem_dewpoint_f": dew, "iem_rh_pct": rh, "iem_n_obs": 24,
    })


# -- parsing -------------------------------------------------------------------


def test_parse_ncei_strips_padding_and_fills_absent_fields():
    parsed = _parse_ncei(NCEI_PAYLOAD)
    assert list(parsed["date"]) == [pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01")]
    assert parsed["rhav_pct"].iloc[0] == 76.0          # "   76" -> 76
    assert parsed["prcp_in"].tolist() == [0.12, 0.0]   # inches, not tenths of mm
    assert parsed["tmax_f"].tolist() == [40.0, 36.0]   # degrees F as returned
    # TAVG is never in any row; ADPT drops out after 2024. Both become NaN.
    assert parsed["tavg_f"].isna().all()
    assert np.isnan(parsed["adpt_f"].iloc[1])


def test_fetch_ncei_requests_standard_units_per_year(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return NCEI_PAYLOAD

    class FakeSession:
        def get(self, url, params, timeout):
            calls.append(params)
            return FakeResponse()

    fetch_ncei(STL, date(2024, 6, 1), date(2025, 2, 1), session=FakeSession())
    assert [(c["startDate"], c["endDate"]) for c in calls] == [
        ("2024-06-01", "2024-12-31"), ("2025-01-01", "2025-02-01")]
    assert all(c["units"] == "standard" for c in calls)
    assert "TAVG" in calls[0]["dataTypes"] and "ADPT" in calls[0]["dataTypes"]


def test_parse_iem_averages_hourly_by_local_day():
    daily = _parse_iem(IEM_CSV).set_index("date")
    jan1 = daily.loc[pd.Timestamp("2026-01-01")]
    assert jan1["iem_dewpoint_f"] == pytest.approx(25.0)   # missing hour skipped
    assert jan1["iem_n_obs"] == 2
    assert daily.loc[pd.Timestamp("2026-01-02"), "iem_rh_pct"] == 80.0


def test_year_chunks_split_on_calendar_years():
    assert _year_chunks(date(2024, 3, 1), date(2026, 2, 1)) == [
        (date(2024, 3, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 2, 1)),
    ]


def test_merge_write_keeps_old_days_and_takes_new_values(tmp_path):
    path = tmp_path / f"{STL}.parquet"
    _merge_write(path, daily_rows("2026-01-01", 5, tmax=40.0))
    merged = _merge_write(path, daily_rows("2026-01-04", 4, tmax=45.0))
    assert len(merged) == 7
    assert merged.set_index("date").loc["2026-01-03", "tmax_f"] == 40.0
    assert merged.set_index("date").loc["2026-01-04", "tmax_f"] == 45.0


# -- availability --------------------------------------------------------------


def test_as_of_excludes_days_inside_the_lag():
    """A day is usable only once date + lag <= as_of: the day itself, not the week."""
    rows = daily_rows("2026-01-04", 7)                        # Sun Jan 4 .. Sat Jan 10
    as_of = pd.Timestamp("2026-01-10") + pd.Timedelta(days=AVAILABILITY_LAG_DAYS)
    full = weekly_features(as_of, fips=STL_CITY, daily=rows)
    assert full["n_days"].tolist() == [7]

    one_day_early = weekly_features(as_of - pd.Timedelta(days=1), fips=STL_CITY, daily=rows)
    assert one_day_early["n_days"].tolist() == [6]            # Jan 10 just outside

    too_early = weekly_features(pd.Timestamp("2026-01-04") + pd.Timedelta(
        days=AVAILABILITY_LAG_DAYS - 1), fips=STL_CITY, daily=rows)
    assert too_early.empty and list(too_early.columns) == WEEKLY_COLUMNS


def test_future_values_cannot_leak_into_a_partial_week():
    rows = daily_rows("2026-01-04", 7, tmax=40.0)
    rows.loc[rows["date"] >= "2026-01-08", "tmax_f"] = 99.0   # later, hotter days
    as_of = pd.Timestamp("2026-01-07") + pd.Timedelta(days=AVAILABILITY_LAG_DAYS)
    week = weekly_features(as_of, fips=STL_CITY, daily=rows).iloc[0]
    assert week["n_days"] == 4 and week["temp_max_f"] == 40.0


# -- weekly aggregation --------------------------------------------------------


def test_weeks_run_sunday_through_saturday():
    rows = daily_rows("2026-01-03", 9)                        # Sat Jan 3 .. Sun Jan 11
    rows["prcp_in"] = np.arange(9) * 1.0                      # 0, 1, ..., 8
    weekly = weekly_features("2026-03-01", fips=STL_CITY, daily=rows)
    assert weekly["week_end"].tolist() == [pd.Timestamp(d) for d in
                                           ("2026-01-03", "2026-01-10", "2026-01-17")]
    assert weekly["n_days"].tolist() == [1, 7, 1]
    assert weekly["precip_in"].tolist() == [0.0, 1 + 2 + 3 + 4 + 5 + 6 + 7, 8.0]
    assert (weekly["week_end"].dt.dayofweek == 5).all()      # every week_end a Saturday


def test_temp_mean_falls_back_to_midpoint_and_prefers_tavg():
    rows = daily_rows("2026-01-04", 2, tmax=50.0, tmin=30.0)
    rows.loc[1, "tavg_f"] = 44.0
    week = weekly_features("2026-03-01", fips=STL_CITY, daily=rows).iloc[0]
    assert week["temp_mean_f"] == pytest.approx((40.0 + 44.0) / 2)
    assert week["temp_min_f"] == 30.0 and week["temp_max_f"] == 50.0


def test_missing_humidity_gives_nan_columns_not_missing_columns():
    rows = daily_rows("2026-01-04", 7).drop(columns=["iem_dewpoint_f", "iem_rh_pct"])
    weekly = weekly_features("2026-03-01", fips=STL_CITY, daily=rows)
    assert list(weekly.columns) == WEEKLY_COLUMNS
    assert weekly["dewpoint_f"].isna().all() and weekly["rh_mean"].isna().all()
    assert weekly["temp_mean_f"].notna().all()


def test_ncei_dewpoint_fills_in_when_iem_is_missing():
    rows = daily_rows("2026-01-04", 2, dew=np.nan)
    rows["adpt_f"] = 20.0
    assert weekly_features("2026-03-01", fips=STL_CITY, daily=rows)["dewpoint_f"].iloc[0] == 20.0


def test_each_county_gets_its_metros_station():
    rows = pd.concat([daily_rows("2026-01-04", 7, tmax=40.0),
                      daily_rows("2026-01-04", 7, station="USW00094846", tmax=20.0)])
    weekly = weekly_features("2026-03-01", fips=[STL_CITY, STL_COUNTY, COOK], daily=rows)
    by_fips = weekly.set_index("fips")["temp_max_f"]
    assert by_fips[STL_CITY] == by_fips[STL_COUNTY] == 40.0
    assert by_fips[COOK] == 20.0


def test_unknown_county_is_rejected():
    with pytest.raises(KeyError, match="study"):
        weekly_features("2026-03-01", fips="06037", daily=daily_rows("2026-01-04", 1))
