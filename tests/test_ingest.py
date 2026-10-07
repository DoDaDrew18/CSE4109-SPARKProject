"""Offline tests for the NSSP ingester's date handling. No network."""

from datetime import date

import pytest

from src.ingest_nssp import _check_epiweek, _year_chunks, mmwr_week_start


def test_mmwr_week_start_matches_cdc_calendar():
    assert mmwr_week_start(202601) == date(2026, 1, 4)    # Jan 4 is a Sunday
    assert mmwr_week_start(202501) == date(2024, 12, 29)  # week 1 starts in Dec
    assert mmwr_week_start(202240) == date(2022, 10, 2)


def test_as_of_rejects_calendar_dates():
    """The leak: Delphi reads 20260215 as an epiweek and returns latest data."""
    assert _check_epiweek("202607") == 202607
    for bad in ("20260215", "2026-02-15", "202654", "202600"):
        with pytest.raises(ValueError, match="epiweek"):
            _check_epiweek(bad)


def test_year_chunks_cover_range_without_overlap():
    assert _year_chunks("202240-202410") == ["202240-202253", "202301-202353", "202401-202410"]
    assert _year_chunks("202601-202610") == ["202601-202610"]
