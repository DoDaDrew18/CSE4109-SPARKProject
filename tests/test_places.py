"""Place search: ranking, state filters, ZIPs and the geometry handed to /api/region."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box, shape

from src.app.places import Index, _split_state, clean_place_name, search
from src.app.region import resolve


@pytest.fixture
def idx() -> Index:
    # two side-by-side counties sharing a border, plus a third far away
    counties = gpd.GeoDataFrame({
        "fips": ["47157", "47155", "17031"],
        "name": ["Shelby County", "Fayette County", "Cook County"],
        "state": ["TN", "TN", "IL"],
        "population": [900_000, 40_000, 5_000_000],
    }, geometry=[box(-90.3, 35.0, -89.6, 35.4), box(-89.6, 35.0, -89.2, 35.4),
                 box(-88.3, 41.4, -87.5, 42.2)], crs="EPSG:4326")
    counties["key"] = counties["name"].str.lower()
    counties["base"] = counties["name"].str.replace(" County", "").str.lower()
    cities = pd.DataFrame({
        "name": ["Memphis", "Memphis", "Nashville-Davidson", "Nashville", "Cookeville", "Germantown"],
        "state": ["TN", "TX", "TN", "NC", "TN", "TN"],
        "lat": [35.11, 34.72, 36.17, 35.97, 36.16, 35.08],
        "lon": [-89.97, -100.54, -86.78, -77.96, -85.50, -89.79],
        "aland": [7.6e8, 3e6, 1.2e9, 8e6, 8e7, 4.9e7],
        "population": [610_000, 2_000, 700_000, 5_800, 37_000, np.nan],
        "county": ["Shelby County", "Hall County", "Davidson County", "Nash County",
                   "Putnam County", "Shelby County"],
    })
    cities["key"] = cities["name"].str.lower().str.replace("-", " ")
    zips = pd.DataFrame({"zip": ["38103", "38104", "60601"], "lat": [35.15, 35.13, 41.88],
                         "lon": [-90.05, -90.0, -87.62], "aland": [1e7, 1.2e7, 1e6],
                         "county": ["Shelby County", "Shelby County", "Cook County"],
                         "state": ["TN", "TN", "IL"]})
    return Index(counties, cities, zips)


def test_clean_place_name():
    assert clean_place_name("Memphis city") == "Memphis"
    assert clean_place_name("Indianapolis city (balance)") == "Indianapolis"
    assert clean_place_name("Nashville-Davidson metropolitan government (balance)") == "Nashville-Davidson"
    assert clean_place_name("Boise City city") == "Boise City"
    assert clean_place_name("Truth or Consequences city") == "Truth or Consequences"


def test_split_state():
    assert _split_state("Springfield, MO") == ("springfield", "MO")
    assert _split_state("springfield missouri") == ("springfield", "MO")
    assert _split_state("new york") == ("new york", None)
    assert _split_state("Memphis") == ("memphis", None)


def test_populous_city_first_and_state_filter(idx):
    labels = [r["label"] for r in search("Memphis", idx=idx)]
    assert labels[:2] == ["Memphis, TN", "Memphis, TX"]
    assert [r["label"] for r in search("memphis tx", idx=idx)] == ["Memphis, TX"]


def test_big_prefix_match_beats_tiny_exact_match(idx):
    assert search("Nashville", idx=idx)[0]["label"] == "Nashville-Davidson, TN"


def test_county_by_base_name(idx):
    first = search("cook", idx=idx)[0]
    assert (first["kind"], first["label"]) == ("county", "Cook County, IL")
    assert first["outline"] is True


def test_zip_prefix(idx):
    hits = search("381", idx=idx)
    assert [h["label"] for h in hits] == ["ZIP 38103", "ZIP 38104"]
    assert hits[0]["detail"] == "Shelby County, TN"


def test_short_or_unknown_queries(idx):
    assert search("m", idx=idx) == []
    assert search("xyzzy", idx=idx) == []


def test_county_geometry_selects_only_that_county(idx):
    hit = next(r for r in search("shelby", idx=idx) if r["kind"] == "county")
    region = resolve(hit["geometry"], frame=idx.counties)
    assert region.fips == ["47157"]                 # Fayette shares a border but is not pulled in


def test_city_box_contains_city_and_bounds_match(idx):
    hit = search("Germantown", idx=idx)[0]
    g = shape(hit["geometry"])
    assert g.contains(shape({"type": "Point", "coordinates": [-89.79, 35.08]}))
    (s, w), (n, e) = hit["bounds"]
    assert np.allclose([w, s, e, n], g.bounds, atol=1e-4)
    assert "pop" not in hit["detail"]               # no estimate for this place: none shown
