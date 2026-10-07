"""Offline tests for the spatial layer: hot spots, FIPS joins, SaTScan I/O, map helpers."""

from datetime import timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.ingest_nssp import mmwr_week_start
from src.snapshot import SnapshotStore
from src.spatial import satscan
from src.spatial.hotspots import gi_star, local_moran, respiratory_share, week_values
from src.spatial.map import _bins, epiweek_of, first_print_share
from src.spatial.shapes import _read_pep, boundary_url, fips_join_report, lower48


# -- fixtures ----------------------------------------------------------------

def _lattice(n=8, hot=3, island=False, seed=0):
    """n x n unit squares (in degrees, lower-48 FIPS) with a hot hot x hot corner."""
    rng = np.random.default_rng(seed)
    cells, values = [], []
    for r in range(n):
        for c in range(n):
            fips = f"29{r * n + c:03d}"
            cells.append({"fips": fips, "geometry": box(-95 + c, 35 + r, -94 + c, 36 + r)})
            values.append({"fips": fips,
                           "resp_pct": (10.0 if r < hot and c < hot else 1.0)
                           + rng.normal(0, 0.1)})
    if island:
        cells.append({"fips": "29999", "geometry": box(-70, 35, -69, 36)})
        values.append({"fips": "29999", "resp_pct": 1.0})
    shapes = gpd.GeoDataFrame(cells, crs="EPSG:4326")
    return shapes, pd.DataFrame(values)


def _write_signal(store, source, issue, rows):
    frame = pd.DataFrame(rows, columns=["geo_value", "time_value", "value"])
    frame["issue"] = pd.Timestamp(issue)
    store.write_snapshot(source, issue, frame)


# -- hot spots -----------------------------------------------------------------

def test_gi_star_finds_the_planted_cluster():
    shapes, values = _lattice()
    out = gi_star(shapes, values, permutations=999, seed=1)
    cls = out.set_index("fips")["class"]
    assert cls["29009"].startswith("Hot spot")          # (1,1): inside the 3x3 block
    assert cls["29000"].startswith("Hot spot")
    assert not cls["29063"].startswith("Hot")           # far corner
    assert set(out.columns) >= {"fips", "value", "z", "p", "p_fdr", "class"}
    assert (out["p_fdr"] >= out["p"] - 1e-12).all()     # FDR never lowers p
    assert out.set_index("fips").loc["29009", "z"] > 2


def test_gi_star_attaches_islands_instead_of_failing():
    shapes, values = _lattice(island=True)
    out = gi_star(shapes, values, permutations=199, seed=1)
    assert out.set_index("fips").loc["29999", "island"]
    assert out["z"].notna().all()


def test_local_moran_labels_cluster_high_high():
    shapes, values = _lattice()
    out = local_moran(shapes, values, permutations=999, seed=1).set_index("fips")
    assert out.loc["29009", "class"] == "High-High"


def test_hotspots_drop_non_conus_counties():
    shapes, values = _lattice()
    shapes.loc[0, "fips"] = values.loc[0, "fips"] = "02013"   # an Alaska FIPS
    out = gi_star(shapes, values, permutations=99)
    assert "02013" not in set(out["fips"])
    assert len(lower48(shapes)) == len(shapes) - 1


# -- respiratory share & FIPS joins --------------------------------------------

def test_respiratory_share_sums_signals_and_honors_as_of(tmp_path):
    store = SnapshotStore(tmp_path)
    week = "2025-12-21"
    for source, v in (("nssp_covid", 1.0), ("nssp_influenza", 8.0), ("nssp_rsv", 2.0)):
        _write_signal(store, source, "2025-12-27",
                      [("29510", week, v), ("29189", week, v)])
        later = [("29510", week, v * 2)]
        if source == "nssp_covid":
            # Flu and RSV missing for this county -> dropped, not summed.
            later.append(("17031", week, 1.0))
        _write_signal(store, source, "2026-01-10", later)

    first = respiratory_share(store, as_of="2025-12-31").set_index("fips")
    assert first.loc["29510", "resp_pct"] == pytest.approx(11.0)
    assert first.loc["29510", "week_end"] == pd.Timestamp("2025-12-27")

    latest = respiratory_share(store).set_index("fips")
    assert latest.loc["29510", "resp_pct"] == pytest.approx(22.0)
    assert "17031" not in latest.index

    assert len(week_values(respiratory_share(store), "2025-12-27")) == 2
    with pytest.raises(ValueError, match="Saturday"):
        week_values(latest.reset_index(), "2025-12-24")


def test_fips_join_report_flags_unmatched():
    shapes = pd.DataFrame({"fips": ["29510", "29189", "17031"]})
    values = pd.DataFrame({"fips": ["29510", "29189", "51515"]})   # 51515: retired VA city
    report = fips_join_report(shapes, values)
    assert report["matched"] == 2
    assert report["values_without_shape"] == ["51515"]
    assert report["shapes_without_value"] == 1


def test_population_file_parsing_pads_fips_and_skips_state_rows(tmp_path):
    path = tmp_path / "pep.csv"
    path.write_text(
        "SUMLEV,STATE,COUNTY,STNAME,CTYNAME,POPESTIMATE2025\n"
        "040,29,000,Missouri,Missouri,6200000\n"
        "050,29,510,Missouri,St. Louis city,278144\n"
        "050,1,1,Alabama,Autauga County,60000\n",
        encoding="latin-1")
    pop = _read_pep(path, "POPESTIMATE2025")
    assert pop.to_dict() == {"29510": 278144, "01001": 60000}


def test_boundary_url_shape():
    assert boundary_url(2021, "20m").endswith("GENZ2021/shp/cb_2021_us_county_20m.zip")
    with pytest.raises(ValueError):
        boundary_url(2021, "1m")


# -- SaTScan -------------------------------------------------------------------

def _counties():
    return pd.DataFrame({"fips": ["29510", "29189"], "lat": [38.6358, 38.6406],
                         "lon": [-90.2451, -90.4436], "population": [278144, 990911]})


def test_estimate_counts_scales_percent_by_population():
    share = pd.DataFrame({"fips": ["29510", "29189", "99999"],
                          "week_end": pd.to_datetime(["2025-12-27"] * 3),
                          "resp_pct": [10.0, 5.0, 1.0]})
    pop = pd.Series({"29510": 52_000, "29189": 104_000})
    out = satscan.estimate_counts(share, pop, visits_per_person_year=0.5).set_index("fips")
    assert out.loc["29510", "ed_visits_est"] == pytest.approx(500)
    assert out.loc["29510", "cases"] == 50
    assert out.loc["29189", "cases"] == 50
    assert "99999" not in out.index      # no population, no denominator


def test_satscan_writers_produce_expected_lines(tmp_path):
    share = pd.DataFrame({"fips": ["29510", "29189"] * 2,
                          "week_start": pd.to_datetime(["2025-12-14"] * 2 + ["2025-12-21"] * 2),
                          "week_end": pd.to_datetime(["2025-12-20"] * 2 + ["2025-12-27"] * 2),
                          "resp_pct": [5.0, 4.0, 11.0, 0.0]})
    paths = satscan.write_inputs(tmp_path, share, _counties(), "2025-12-20",
                                 "2025-12-27", model="permutation")
    cases = paths["case"].read_text().splitlines()
    assert cases[0].split()[0] == "29189" or cases[0].split()[0] == "29510"
    assert all(len(line.split()) == 3 for line in cases)
    assert not any(line.endswith("0 2025/12/27") and line.startswith("29189") for line in cases)
    assert "29510 38.635800 -90.245100" in paths["coordinates"].read_text()
    assert "29189 2025 990911" in paths["population"].read_text()

    prm = paths["parameters"].read_text()
    for key in ("AnalysisType=3", "ModelType=2", "CoordinatesType=1",
                "StartDate=2025/12/14", "EndDate=2025/12/27", "PrecisionCaseTimes=3",
                "TimeAggregationLength=7"):
        assert key in prm
    assert f"CaseFile={paths['case'].resolve()}" in prm


def test_parameter_file_rejects_poisson_without_population(tmp_path):
    with pytest.raises(ValueError, match="population"):
        satscan.write_parameter_file(tmp_path / "a.prm", case_file="c", coordinates_file="g",
                                     start="2025-12-01", end="2025-12-27",
                                     results_file="r", model="poisson")


SAMPLE_RESULTS = """\
SaTScan v10.1
_____________________________

CLUSTERS DETECTED

1.Location IDs included.: 29510, 29189, 17163,
                          17119
  Coordinates / radius..: (38.635800 N, 90.245100 W) / 25.30 km
  Time frame............: 2025/12/14 to 2025/12/27
  Number of cases.......: 1234
  Expected cases........: 800.12
  Observed / expected...: 1.54
  Test statistic........: 120.33
  P-value...............: < 0.00000000000000001

2.Location IDs included.: 29095
  Time frame............: 2025/12/21 to 2025/12/27
  Number of cases.......: 300
  Expected cases........: 250.00
  Relative risk.........: 1.21
  Log likelihood ratio..: 4.5
  P-value...............: 0.23

_____________________________
"""


def test_parse_results_text(tmp_path):
    path = tmp_path / "results.txt"
    path.write_text(SAMPLE_RESULTS)
    out = satscan.parse_results_text(path)
    assert list(out["cluster"]) == [1, 2]
    assert out.loc[0, "location_ids"] == ["29510", "29189", "17163", "17119"]
    assert out.loc[0, "observed"] == 1234
    assert out.loc[0, "start"] == pd.Timestamp("2025-12-14")
    assert out.loc[0, "p_value"] < 1e-10
    assert out.loc[1, "relative_risk"] == pytest.approx(1.21)
    assert out.loc[1, "llr"] == pytest.approx(4.5)


def test_parse_col_file_with_and_without_header(tmp_path):
    with_header = tmp_path / "a.col.txt"
    with_header.write_text("CLUSTER LOC_ID LATITUDE LONGITUDE P_VALUE\n"
                           "1 29510 38.63 -90.24 0.001\n")
    out = satscan.parse_col_file(with_header)
    assert out.loc[0, "LOC_ID"] == 29510 and out.loc[0, "P_VALUE"] == 0.001

    bare = tmp_path / "b.col.txt"
    bare.write_text("1 29510 38.63 -90.24 25.3\n")
    out = satscan.parse_col_file(bare)
    assert list(out.columns) == ["CLUSTER", "LOC_ID", "LATITUDE", "LONGITUDE", "RADIUS"]


# -- map helpers ---------------------------------------------------------------

def test_epiweek_of_round_trips_mmwr_calendar():
    for week in (202416, 202452, 202501, 202552, 202553, 202601, 202610):
        start = mmwr_week_start(week)
        assert epiweek_of(start) == week
        assert epiweek_of(start + timedelta(days=6)) == week


def test_first_print_takes_each_countys_earliest_issue(tmp_path):
    store = SnapshotStore(tmp_path)
    week = "2025-12-21"
    for source in ("nssp_covid", "nssp_influenza", "nssp_rsv"):
        # 29510 first appears at lag 2, 29189 at lag 1 and is revised at lag 2.
        _write_signal(store, source, "2026-01-03", [("29189", week, 1.0)])
        _write_signal(store, source, "2026-01-10", [("29510", week, 3.0), ("29189", week, 2.0)])
    out, known = first_print_share("2025-12-27", root=tmp_path)
    by = out.set_index("fips")["resp_pct"]
    assert by["29189"] == pytest.approx(3.0)       # 3 signals x first print 1.0
    assert by["29510"] == pytest.approx(9.0)
    assert known == pd.Timestamp("2026-01-10")


def test_first_print_is_empty_offline_without_data(tmp_path):
    out, known = first_print_share("2025-12-27", root=tmp_path)
    assert out.empty and known is None


def test_bins_are_increasing_and_cover_range():
    values = pd.Series(np.linspace(0.5, 30, 500))
    bins = _bins(values)
    assert bins[0] == 0 and bins[-1] >= 30
    assert all(a < b for a, b in zip(bins, bins[1:]))
