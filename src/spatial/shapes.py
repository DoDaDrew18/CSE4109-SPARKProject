"""County boundaries, populations and centroids, keyed by 5-digit FIPS.

Sources (all verified reachable, no key, October 2026):

  Boundaries  Census cartographic boundary files, e.g.
              https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_500k.zip
              (``20m`` is the coarse national version, ~0.9 MB zipped.)
  Population  Census Population Estimates Program county totals,
              https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv

Why the **2021** boundary vintage and not the newest one
--------------------------------------------------------
In 2022 Census replaced Connecticut's 8 counties (09001-09015) with 9
planning regions (09110-09190). NSSP still reports the *old* CT counties, so
any boundary file from 2022 on silently drops Connecticut from the join.
2021 is the newest vintage whose FIPS set matches NSSP. The 2025 population
file follows the new CT geography, so for the old CT counties we fall back to
the 2021 population estimates (``co-est2021-alldata.csv``), which still use
them. ``fips_join_report`` is the check that this stays true.

Centroids are computed in an equal-area projection (EPSG:5070, CONUS Albers)
and converted back to lat/long. Taking centroids of raw lat/long polygons
treats degrees as planar and drifts for large western counties.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

__all__ = [
    "REFERENCE_DIR", "BOUNDARY_YEAR", "boundary_url", "population_url",
    "download", "load_boundaries", "load_population", "add_centroids",
    "load_counties", "lower48", "fips_join_report", "NON_CONUS_STATES",
]

REFERENCE_DIR = Path("raw/reference")
BOUNDARY_YEAR = 2021
POPULATION_YEAR = 2025

# Alaska, Hawaii and the territories. They have no land neighbors in the
# lower 48, so contiguity-based statistics treat them as islands; maps inset
# or drop them.
NON_CONUS_STATES = ("02", "15", "60", "66", "69", "72", "78")

EQUAL_AREA_CRS = "EPSG:5070"   # NAD83 / Conus Albers
LATLON_CRS = "EPSG:4326"

_UA = {"User-Agent": "CSE4109-SPARK/spatial"}


def boundary_url(year: int = BOUNDARY_YEAR, resolution: str = "500k") -> str:
    """Census cartographic boundary county zip; resolution is 500k, 5m or 20m."""
    if resolution not in ("500k", "5m", "20m"):
        raise ValueError(f"resolution must be 500k, 5m or 20m, got {resolution!r}")
    return (f"https://www2.census.gov/geo/tiger/GENZ{year}/shp/"
            f"cb_{year}_us_county_{resolution}.zip")


def population_url(year: int = POPULATION_YEAR) -> str:
    """PEP county totals for the decade vintage ending in ``year``."""
    return (f"https://www2.census.gov/programs-surveys/popest/datasets/"
            f"2020-{year}/counties/totals/co-est{year}-alldata.csv")


def download(url: str, directory: Path = REFERENCE_DIR) -> Path:
    """Fetch ``url`` into ``directory`` once; later calls reuse the file.

    Written to a temporary name and renamed, so an interrupted download never
    leaves a truncated file that the cache would then trust forever.
    """
    directory = Path(directory)
    target = directory / url.rsplit("/", 1)[-1]
    if target.exists() and target.stat().st_size > 0:
        return target
    directory.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=300, headers=_UA)
    response.raise_for_status()
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(response.content)
    partial.rename(target)
    return target


def load_boundaries(resolution: str = "500k", year: int = BOUNDARY_YEAR,
                    directory: Path = REFERENCE_DIR) -> gpd.GeoDataFrame:
    """County polygons in lat/long with ``fips``, ``name``, ``state`` columns."""
    path = download(boundary_url(year, resolution), directory)
    raw = gpd.read_file(f"zip://{path}")
    out = gpd.GeoDataFrame({
        "fips": raw["GEOID"].astype(str).str.zfill(5),
        "name": raw["NAMELSAD"] if "NAMELSAD" in raw else raw["NAME"],
        "state": raw["STUSPS"],
        "state_fips": raw["STATEFP"].astype(str).str.zfill(2),
    }, geometry=raw.geometry.values, crs=raw.crs)
    return out.to_crs(LATLON_CRS).sort_values("fips").reset_index(drop=True)


def _read_pep(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, dtype={"STATE": str, "COUNTY": str},
                        encoding="latin-1")
    frame = frame[frame["SUMLEV"].astype(int) == 50]   # 40 = state totals
    fips = frame["STATE"].str.zfill(2) + frame["COUNTY"].str.zfill(3)
    return pd.Series(frame[column].astype("int64").values, index=fips.values,
                     name="population")


def load_population(year: int = POPULATION_YEAR,
                    directory: Path = REFERENCE_DIR) -> pd.Series:
    """Resident population by FIPS, newest PEP estimate.

    Counties missing from the newest file (old Connecticut counties, see the
    module docstring) are filled from the 2021 vintage, which still has them.
    """
    newest = _read_pep(download(population_url(year), directory),
                       f"POPESTIMATE{year}")
    legacy = _read_pep(download(population_url(2021), directory),
                       "POPESTIMATE2021")
    filled = pd.concat([newest, legacy[~legacy.index.isin(newest.index)]])
    filled.index.name = "fips"
    return filled.sort_index()


def add_centroids(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ``lat``/``lon`` centroid columns computed in an equal-area CRS."""
    out = frame.copy()
    centers = frame.geometry.to_crs(EQUAL_AREA_CRS).centroid.to_crs(LATLON_CRS)
    out["lat"] = centers.y.round(6).values
    out["lon"] = centers.x.round(6).values
    return out


def lower48(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop Alaska, Hawaii and territories (plus DC stays: it is contiguous)."""
    keep = ~frame["fips"].str[:2].isin(NON_CONUS_STATES)
    return frame.loc[keep].reset_index(drop=True)


def load_counties(resolution: str = "500k", year: int = BOUNDARY_YEAR,
                  directory: Path = REFERENCE_DIR) -> gpd.GeoDataFrame:
    """Boundaries + population + centroids: the one frame every caller wants."""
    shapes = load_boundaries(resolution, year, directory)
    population = load_population(directory=directory)
    shapes["population"] = shapes["fips"].map(population).astype("Int64")
    return add_centroids(shapes)


def fips_join_report(shapes: pd.DataFrame, values: pd.DataFrame,
                     key: str = "fips") -> dict:
    """How cleanly ``values`` joins onto ``shapes``; catches FIPS drift.

    Returns counts plus the unmatched FIPS on each side. A data county with
    no shape is invisible on the map and silently dropped from Gi*; that is
    the failure this report exists to surface.
    """
    shape_ids = set(shapes[key])
    value_ids = set(values[key])
    return {
        "shapes": len(shape_ids),
        "values": len(value_ids),
        "matched": len(shape_ids & value_ids),
        "values_without_shape": sorted(value_ids - shape_ids),
        "shapes_without_value": len(shape_ids - value_ids),
    }
