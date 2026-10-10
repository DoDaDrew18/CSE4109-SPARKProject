"""Offline tests for the CDC NSSP table transform. No network."""

import pandas as pd

from src.ingest_cdc_nssp import to_snapshots
from src.snapshot import SnapshotStore


def cdc_rows():
    return pd.DataFrame([
        # week_end, geography, county, fips, hsa, covid, flu, rsv
        ["2026-10-03T00:00:00.000", "Illinois", "Cook", "17031", "287", "0.21", "0.34", "0.02"],
        ["2026-10-03T00:00:00.000", "Missouri", "St. Louis City", "29510", "541", "0.3", None, "0.01"],
        ["2026-10-03T00:00:00.000", "Alabama", "Autauga", "1001", "12", "0.5", "0.1", "0.0"],
        ["2026-10-03T00:00:00.000", "Illinois", "All", None, None, "0.2", "0.3", "0.02"],
    ], columns=["week_end", "geography", "county", "fips", "hsa_nci_id",
                "percent_visits_covid", "percent_visits_influenza", "percent_visits_rsv"])


def test_to_snapshots_drops_rollups_and_blanks_and_aligns_weeks():
    frames = to_snapshots(cdc_rows(), "2026-10-10")

    flu = frames["cdc_nssp_influenza"]
    assert set(flu["geo_value"]) == {"17031", "01001"}      # state row and blank flu dropped, FIPS padded
    assert len(frames["cdc_nssp_covid"]) == 3
    # CDC's Saturday week_end becomes the Sunday week start used by every snapshot.
    assert set(flu["time_value"]) == {pd.Timestamp("2026-09-27")}
    assert flu.loc[flu.geo_value == "17031", "value"].item() == 0.34


def test_to_snapshots_output_passes_store_validation(tmp_path):
    store = SnapshotStore(tmp_path)
    for source, frame in to_snapshots(cdc_rows(), "2026-10-10").items():
        store.write_snapshot(source, "2026-10-10", frame)
    vintage = store.load_vintage("cdc_nssp_rsv", "2026-10-10")
    assert len(vintage) == 3 and "hsa_nci_id" in vintage.columns
