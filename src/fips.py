"""County name -> 5-digit FIPS lookup against the Census county reference file.

Several sources (CDC NWSS wastewater, news text) name counties instead of
giving FIPS codes, and they spell the same county several ways:

    "Saint Louis" / "St. Louis County" / "st louis"     -> 29189
    "Saint Louis City" / "St. Louis city"               -> 29510
    "Juneau" (Census: "Juneau City and Borough")        -> 02110
    "DeKalb" / "De Kalb", "Doña Ana" / "Dona Ana"

Everything joins on FIPS downstream, so a name that maps to the wrong county
would silently put one county's data under another. The lookup therefore
refuses to guess: an unknown or ambiguous name raises ``KeyError`` rather
than returning a near miss.

Matching rules (all verified against the 2020 file, Oct 2026):

* Case, periods, apostrophes, hyphens, spaces and accents are ignored, and
  "Saint"/"Sainte" are folded to "St"/"Ste".
* County-type suffixes are optional: County, Parish, Borough, Census Area,
  City and Borough, Municipality, Municipio. ``"Cook"`` == ``"Cook County"``.
* "city" is NOT a droppable suffix, because it is what tells an independent
  city from the county of the same name (St. Louis city vs St. Louis County,
  Baltimore, Richmond VA, Fairfax VA ...). A bare name means the county,
  matching Census convention.
* Two fallbacks, each used only when the exact form is absent from the state:
  "X City" maps to county X (Bedford City VA merged into Bedford County in
  2013 and is gone from the 2020 file, but NWSS still names it), and bare
  "X" maps to "X city" (NWSS writes Virginia's Radford and Salem bare, and
  Virginia has no county of either name).

Reference file: https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt
(pipe-delimited, 3,235 rows incl. territories; columns STATE|STATEFP|COUNTYFP|
COUNTYNS|COUNTYNAME|CLASSFP|FUNCSTAT). There is no newer ``codes202x``
directory, so Connecticut uses its eight legacy counties, which is also what
NWSS reports. Cached under ``raw/reference/``.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

__all__ = ["REFERENCE_URL", "FipsLookup", "county_fips", "load_reference",
           "state_postal", "normalize_county"]

REFERENCE_URL = (
    "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"
)
DEFAULT_PATH = Path("raw/reference/national_county2020.txt")

_UA = {"User-Agent": "CSE4109-SPARK/fips"}

# Census suffixes that callers routinely omit. Longest first so "city and
# borough" is stripped whole rather than leaving "city and".
_SUFFIXES = ("city and borough", "census area", "municipality", "municipio",
             "borough", "county", "parish")

STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN",
    "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR", "guam": "GU",
    "american samoa": "AS", "northern mariana islands": "MP",
    "us virgin islands": "VI", "united states virgin islands": "VI",
    "virgin islands": "VI",
}


def _fold(text: str) -> str:
    """Lowercase, strip accents, turn punctuation into spaces, squeeze."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[.'’]", "", text)          # "St." -> "st", "O'Brien"
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_county(name: str) -> str:
    """Canonical comparison key for a county name; see module rules.

    Spaces are removed at the end so "De Kalb"/"DeKalb" and "La Salle"/
    "LaSalle" agree. No two counties in one state collide under this.
    """
    words = _fold(name)
    words = re.sub(r"\bsainte\b", "ste", words)
    words = re.sub(r"\bsaint\b", "st", words)
    for suffix in _SUFFIXES:
        if words.endswith(" " + suffix):
            words = words[: -len(suffix) - 1]
            break
    return words.replace(" ", "")


def state_postal(state: str) -> str:
    """Two-letter postal code from a postal code or full name ("Missouri")."""
    folded = _fold(state)
    if len(folded) == 2 and folded.upper() in set(STATE_NAMES.values()):
        return folded.upper()
    try:
        return STATE_NAMES[folded]
    except KeyError:
        raise KeyError(f"unknown state {state!r}") from None


def load_reference(path=DEFAULT_PATH, download: bool = True) -> pd.DataFrame:
    """The Census county file as ``state, fips, name``; downloads once if absent."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} missing and download=False")
        response = requests.get(REFERENCE_URL, timeout=60, headers=_UA)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    raw = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8")
    return pd.DataFrame({
        "state": raw["STATE"],
        "fips": raw["STATEFP"].str.zfill(2) + raw["COUNTYFP"].str.zfill(3),
        "name": raw["COUNTYNAME"],
    })


class FipsLookup:
    """Name -> FIPS index over one reference table."""

    def __init__(self, reference: pd.DataFrame) -> None:
        index: dict[tuple[str, str], set[str]] = {}
        for state, fips, name in reference[["state", "fips", "name"]].itertuples(index=False):
            index.setdefault((state, normalize_county(name)), set()).add(fips)
        self._index = index

    @classmethod
    def from_file(cls, path=DEFAULT_PATH, download: bool = True) -> "FipsLookup":
        return cls(load_reference(path, download))

    def county_fips(self, state: str, county_name: str) -> str:
        postal = state_postal(state)
        key = normalize_county(county_name)
        hits = self._index.get((postal, key))
        if not hits and key.endswith("city"):
            hits = self._index.get((postal, key[: -len("city")]))   # Bedford City VA
        if not hits:
            hits = self._index.get((postal, key + "city"))           # Radford VA
        if not hits:
            raise KeyError(f"no county {county_name!r} in {postal}")
        if len(hits) > 1:
            raise KeyError(f"{county_name!r} in {postal} is ambiguous: {sorted(hits)}")
        return next(iter(hits))

    def states_with(self, county_name: str) -> list[str]:
        """Postal codes of every state that has a county by this name.

        Sewersheds cross state lines (a Kansas City, MO plant serves Wyandotte
        County, KS), so callers need a way to look outside the site's state.
        """
        key = normalize_county(county_name)
        return sorted(s for (s, k) in self._index if k == key)


@lru_cache(maxsize=4)
def _default_lookup(path: str) -> FipsLookup:
    return FipsLookup.from_file(path)


def county_fips(state: str, county_name: str, reference=DEFAULT_PATH) -> str:
    """5-char FIPS for a county, e.g. ``county_fips("Missouri", "Saint Louis")``.

    ``state`` may be a full name or postal code. Raises ``KeyError`` for an
    unknown or ambiguous name instead of guessing.
    """
    return _default_lookup(str(reference)).county_fips(state, county_name)
