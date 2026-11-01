"""Turn a drawn shape into the US counties it covers.

The map sends GeoJSON (a box or polygon in lon/lat). A county counts as
inside when its polygon intersects the shape; counties that only graze it
are kept too, because ED and wastewater data come per county, not per
square mile. A selection is capped at ``MAX_COUNTIES`` so a continent-sized
box doesn't fan out into thousands of county requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from src.spatial.shapes import EQUAL_AREA_CRS, LATLON_CRS, load_counties

__all__ = ["Region", "RegionError", "MAX_COUNTIES", "counties", "resolve", "region_for_fips"]

MAX_COUNTIES = 150


class RegionError(ValueError):
    """The shape is malformed, outside the US, or too large."""


@dataclass
class Region:
    fips: list[str]                      # sorted by population, largest first
    names: dict[str, str]                # fips -> "St. Louis County, MO"
    population: int
    center: tuple[float, float]          # lon, lat of the shape's centroid
    label: str                           # "St. Louis County, MO and 4 nearby counties"
    states: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"fips": self.fips, "names": self.names, "population": self.population,
                "center": list(self.center), "label": self.label, "states": self.states}


@lru_cache(maxsize=1)
def counties() -> gpd.GeoDataFrame:
    """All US counties (Census 1:20m cartographic boundaries + population)."""
    return load_counties("20m")


def _label(top: pd.DataFrame) -> str:
    first = f"{top.iloc[0]['name']}, {top.iloc[0]['state']}"
    rest = len(top) - 1
    return first if rest == 0 else f"{first} and {rest} nearby count{'y' if rest == 1 else 'ies'}"


def _region(hits: pd.DataFrame, center) -> Region:
    hits = hits.assign(population=hits["population"].fillna(0).astype(int)) \
               .sort_values("population", ascending=False)
    return Region(
        fips=hits["fips"].tolist(),
        names={f: f"{n}, {s}" for f, n, s in zip(hits["fips"], hits["name"], hits["state"])},
        population=int(hits["population"].sum()),
        center=(round(float(center[0]), 4), round(float(center[1]), 4)),
        label=_label(hits),
        states=sorted(hits["state"].unique().tolist()),
    )


def resolve(geojson: dict, frame: gpd.GeoDataFrame | None = None,
            max_counties: int = MAX_COUNTIES) -> Region:
    """Counties intersecting a GeoJSON geometry, Feature or FeatureCollection."""
    if not isinstance(geojson, dict):
        raise RegionError("region must be a GeoJSON object")
    if geojson.get("type") == "FeatureCollection":
        feats = geojson.get("features") or []
        if not feats:
            raise RegionError("empty FeatureCollection")
        geojson = feats[0]
    if geojson.get("type") == "Feature":
        geojson = geojson.get("geometry") or {}
    if geojson.get("type") not in ("Polygon", "MultiPolygon"):
        raise RegionError("draw a box or polygon")
    try:
        geom = shape(geojson)
    except Exception as exc:  # shapely raises several types for bad rings
        raise RegionError(f"invalid shape: {exc}") from exc
    if geom.is_empty or not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        raise RegionError("the shape has no area")

    frame = counties() if frame is None else frame
    hits = frame[frame.intersects(geom)]
    if hits.empty:
        raise RegionError("no US county inside the selected area")
    if len(hits) > max_counties:
        raise RegionError(f"the area covers {len(hits)} counties; select at most "
                          f"{max_counties} (zoom in or draw a smaller shape)")
    center = gpd.GeoSeries([geom], crs=LATLON_CRS).to_crs(EQUAL_AREA_CRS) \
                .centroid.to_crs(LATLON_CRS).iloc[0]
    return _region(hits, (center.x, center.y))


def region_for_fips(fips, frame: gpd.GeoDataFrame | None = None) -> Region:
    """A region made of named counties (for tests and the CLI)."""
    frame = counties() if frame is None else frame
    wanted = [fips] if isinstance(fips, str) else list(fips)
    hits = frame[frame["fips"].isin(wanted)]
    if hits.empty:
        raise RegionError(f"unknown FIPS {wanted}")
    lon = float((hits["lon"] * hits["population"].fillna(1)).sum() / hits["population"].fillna(1).sum())
    lat = float((hits["lat"] * hits["population"].fillna(1)).sum() / hits["population"].fillna(1).sum())
    return _region(hits, (lon, lat))
