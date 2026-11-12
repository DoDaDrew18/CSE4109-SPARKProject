"""Place search for the map: counties, cities and ZIP codes, all local.

A public geocoder (Photon) answered in 30-50 s per query when tested, too
slow and too flaky for a live map, so the app searches Census files it keeps
on disk instead:

* counties: the 1:20m boundaries ``region`` already uses (with population),
* cities and towns: the Census Gazetteer places file, ranked by the Census
  population estimates (unincorporated places, which have none, rank by land
  area after the incorporated ones),
* ZIP codes: the Census Gazetteer ZIP Code Tabulation Areas.

The three files download once into ``raw/reference`` (public domain, ~9 MB).
Each hit carries a GeoJSON geometry ready for ``/api/region``: a county's own
outline pulled in slightly (a shared border would otherwise select every
neighbour too), or a box around a city or ZIP sized from its land area.
"""

from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box, mapping

from src.spatial.shapes import LATLON_CRS, REFERENCE_DIR, download

__all__ = ["Index", "index", "search", "clean_place_name"]

GAZETTEER = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/"
PLACES_URL = GAZETTEER + "2024_Gaz_place_national.zip"
ZCTA_URL = GAZETTEER + "2024_Gaz_zcta_national.zip"
POP_URL = ("https://www2.census.gov/programs-surveys/popest/datasets/"
           "2020-2024/cities/totals/sub-est2024.csv")

# Census writes the legal type after the name ("Memphis city", "Nashville-
# Davidson metropolitan government (balance)"); one of these is dropped.
SUFFIXES = ("metropolitan government", "metro government", "consolidated government",
            "unified government", "city and borough", "metro township", "urban county",
            "zona urbana", "comunidad", "municipality", "borough", "village", "town",
            "city", "CDP")
COUNTY_WORDS = re.compile(r"\s+(county|parish|borough|census area|city and borough|municipality|"
                          r"municipio|planning region|city)$", re.I)
SKIP_STATES = ("PR",)            # no ED data for Puerto Rico in this study
COUNTY_INSET = 0.01              # degrees a county outline is pulled in
STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
    "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}
ABBREVS = set(STATES.values())


def clean_place_name(name: str) -> str:
    """'Memphis city' -> 'Memphis'; 'Indianapolis city (balance)' -> 'Indianapolis'."""
    name = re.sub(r"\s*\(balance\)$", "", name.strip())
    for suffix in SUFFIXES:
        if name.endswith(" " + suffix):
            return name[: -len(suffix) - 1]
    return name


def _key(text: str) -> str:
    """Lower case, 'saint' -> 'st', punctuation dropped: 'St. Louis' == 'saint louis'."""
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("saint ", "st "))
    return re.sub(r"\s+", " ", text).strip()


def _read_gazetteer(path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            frame = pd.read_csv(f, sep="\t", dtype={"GEOID": str}, encoding="latin-1")
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def _county_of(points: pd.DataFrame, counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """Attach the county (fips, name, state) that contains each lat/lon point."""
    pts = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["lon"], points["lat"]),
                           crs=LATLON_CRS)
    hit = gpd.sjoin(pts, counties[["fips", "name", "state", "geometry"]].rename(
        columns={"name": "county", "state": "county_state"}), how="inner", predicate="within")
    return pd.DataFrame(hit.drop(columns=["geometry", "index_right"]))


@dataclass
class Index:
    counties: gpd.GeoDataFrame   # fips, name, state, population, geometry, key, base
    cities: pd.DataFrame         # name, state, lat, lon, aland, population, county, key
    zips: pd.DataFrame           # zip, lat, lon, aland, county, state


def build(counties: gpd.GeoDataFrame, directory=REFERENCE_DIR) -> Index:
    """Read (downloading once) the Census files and build the search index."""
    c = counties.copy()
    c["key"] = c["name"].map(_key)
    c["base"] = c["name"].str.replace(COUNTY_WORDS, "", regex=True).map(_key)

    raw = _read_gazetteer(download(PLACES_URL, directory))
    pop = pd.read_csv(download(POP_URL, directory), dtype=str, encoding="latin-1")
    pop = pop[pop["SUMLEV"] == "162"]
    pop = pd.Series(pd.to_numeric(pop["POPESTIMATE2024"]).values,
                    index=(pop["STATE"] + pop["PLACE"]).values)
    pop = pop[~pop.index.duplicated()]
    cities = pd.DataFrame({
        "geoid": raw["GEOID"].str.zfill(7), "name": raw["NAME"].map(clean_place_name),
        "state": raw["USPS"], "lat": raw["INTPTLAT"], "lon": raw["INTPTLONG"],
        "aland": raw["ALAND"].astype(float)})
    cities["population"] = cities["geoid"].map(pop)
    cities = cities[~cities["state"].isin(SKIP_STATES)]
    cities = _county_of(cities, c).drop(columns=["county_state"])
    cities["key"] = cities["name"].map(_key)

    z = _read_gazetteer(download(ZCTA_URL, directory))
    zips = pd.DataFrame({"zip": z["GEOID"].str.zfill(5), "lat": z["INTPTLAT"],
                         "lon": z["INTPTLONG"], "aland": z["ALAND"].astype(float)})
    zips = _county_of(zips, c).rename(columns={"county_state": "state"})
    zips = zips[~zips["state"].isin(SKIP_STATES)]
    return Index(c, cities.reset_index(drop=True), zips.reset_index(drop=True))


@lru_cache(maxsize=1)
def index() -> Index:
    from src.app.region import counties
    return build(counties())


def _split_state(q: str) -> tuple[str, str | None]:
    """'springfield, mo' / 'springfield missouri' -> ('springfield', 'MO')."""
    q = _key(q)
    for name in sorted(STATES, key=len, reverse=True):
        if q.endswith(" " + name) and len(q) > len(name) + 1:
            return q[: -len(name) - 1].strip(), STATES[name]
    head, _, tail = q.rpartition(" ")
    if head and tail.upper() in ABBREVS:
        return head, tail.upper()
    return q, None


def _tier(keys: pd.Series, q: str) -> pd.Series:
    """0 exact, 1 prefix, 2 a later word starts with q, 3 no match."""
    t = pd.Series(3, index=keys.index)
    t[keys.str.contains(r"\b" + re.escape(q), regex=True)] = 2
    t[keys.str.startswith(q)] = 1
    t[keys == q] = 0
    return t


def _box(lat: float, lon: float, aland: float, lo_km: float, hi_km: float):
    half = min(hi_km, max(lo_km, 0.6 * math.sqrt(max(aland, 0) / 1e6)))
    dlat = half / 111.0
    dlon = half / (111.0 * max(math.cos(math.radians(lat)), 0.2))
    return box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _result(kind: str, label: str, detail: str, geom, outline: bool) -> dict:
    w, s, e, n = geom.bounds
    return {"kind": kind, "label": label, "detail": detail, "outline": outline,
            "bounds": [[round(s, 5), round(w, 5)], [round(n, 5), round(e, 5)]],
            "geometry": mapping(geom)}


def _county_geometry(row):
    g = row.geometry.buffer(-COUNTY_INSET)
    if g.is_empty or g.area < 1e-5:          # tiny independent cities: a small box inside
        p = row.geometry.representative_point()
        g = box(p.x - 0.004, p.y - 0.004, p.x + 0.004, p.y + 0.004)
    return g.simplify(0.003, preserve_topology=True)


def search(q: str, limit: int = 8, idx: Index | None = None) -> list[dict]:
    """Best matches for ``q``: cities, counties and ZIP codes, most populous first."""
    q = (q or "").strip()
    if len(q) < 2:
        return []
    idx = idx or index()
    digits = re.sub(r"\D", "", q)
    if digits and len(digits) == len(q.replace(" ", "")):          # a ZIP (or its start)
        z = idx.zips[idx.zips["zip"].str.startswith(digits)].sort_values("zip").head(limit)
        return [_result("ZIP", f"ZIP {r.zip}", f"{r.county}, {r.state}",
                        _box(r.lat, r.lon, r.aland, 1.5, 12), False) for r in z.itertuples()]

    text, state = _split_state(q)
    if not text:
        return []
    c = idx.counties if state is None else idx.counties[idx.counties["state"] == state]
    ci = idx.cities if state is None else idx.cities[idx.cities["state"] == state]
    ct = np.minimum(_tier(c["key"], text), _tier(c["base"], text))
    pt = _tier(ci["key"], text)
    hits = pd.concat([
        pd.DataFrame({"kind": "county", "i": c.index[ct < 3], "tier": ct[ct < 3].values,
                      "pop": c.loc[ct < 3, "population"].fillna(0).astype(float).values}),
        pd.DataFrame({"kind": "city", "i": ci.index[pt < 3], "tier": pt[pt < 3].values,
                      # unincorporated places (no estimate) rank after towns, by land area
                      "pop": ci.loc[pt < 3, "population"].fillna(-1 / (1 + ci.loc[pt < 3, "aland"])).values}),
    ], ignore_index=True)
    # exact and prefix matches compete on population (an exact name counts 3x), so
    # "Nashville" finds Nashville-Davidson (TN) before Nashville, NC; word matches follow
    hits["rank"] = np.where(hits["tier"] == 0, 3.0, 1.0) * hits["pop"]
    hits["group"] = (hits["tier"] >= 2).astype(int)
    hits = hits.sort_values(["group", "rank"], ascending=[True, False]).head(limit)
    out = []
    for h in hits.itertuples():
        if h.kind == "county":
            r = idx.counties.loc[h.i]
            out.append(_result("county", f"{r['name']}, {r['state']}",
                               f"pop {int(r['population'] or 0):,}", _county_geometry(r), True))
        else:
            r = idx.cities.loc[h.i]
            pop = f" · pop {int(r['population']):,}" if pd.notna(r["population"]) else ""
            out.append(_result("city", f"{r['name']}, {r['state']}", f"{r['county']}{pop}",
                               _box(r["lat"], r["lon"], r["aland"], 4, 30), False))
    return out
