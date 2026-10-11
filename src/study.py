"""Study design constants shared by every stage: which places, which weeks.

One home metro (St. Louis) where everything is built and tuned, and five
comparison metros that the finished pipeline is run on unchanged to test
whether the result generalizes.

Week convention, used everywhere downstream of ingestion
--------------------------------------------------------
A row's week is identified by ``week_end``: the **Saturday** that ends the
MMWR (CDC epi) week. NSSP stores the Sunday a week *starts* as
``time_value``; converting is ``week_end = time_value + 6 days``.

Availability convention
-----------------------
Every weekly feature loader takes an ``as_of`` date and returns only what
was public on or before it. That is the whole study design (see
``src/snapshot.py``); a loader that ignores ``as_of`` is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

__all__ = ["Metro", "METROS", "HOME_METRO", "STUDY_FIPS", "BACKTEST_START",
           "metro_of", "week_end_of"]


@dataclass(frozen=True)
class Metro:
    key: str                  # short id used in tables: "stl"
    name: str                 # display name
    counties: dict            # FIPS -> county name
    noaa_station: str         # GHCN-Daily id of the main airport station
    state: str                # two-letter postal code of the core county


METROS: dict[str, Metro] = {m.key: m for m in (
    Metro("stl", "St. Louis", {"29510": "St. Louis city", "29189": "St. Louis County"},
          "USW00013994", "MO"),   # Lambert-St. Louis Intl
    Metro("kc", "Kansas City", {"29095": "Jackson County"},
          "USW00003947", "MO"),   # Kansas City Intl
    Metro("chi", "Chicago", {"17031": "Cook County"},
          "USW00094846", "IL"),   # O'Hare
    Metro("ind", "Indianapolis", {"18097": "Marion County"},
          "USW00093819", "IN"),   # Indianapolis Intl
    Metro("mem", "Memphis", {"47157": "Shelby County"},
          "USW00013893", "TN"),   # Memphis Intl
    Metro("lou", "Louisville", {"21111": "Jefferson County"},
          "USW00093821", "KY"),   # Louisville Muhammad Ali Intl
)}

HOME_METRO = "stl"

STUDY_FIPS: tuple[str, ...] = tuple(f for m in METROS.values() for f in m.counties)

# First NSSP week with an archived first print (epiweek 202416 starts Sun
# 2024-04-14). Weeks before this only exist as already-revised values, so no
# honest backtest can score them.
BACKTEST_START = date(2024, 4, 14)


def metro_of(fips: str) -> str:
    """Metro key for a study county FIPS, e.g. '29510' -> 'stl'."""
    for metro in METROS.values():
        if fips in metro.counties:
            return metro.key
    raise KeyError(f"{fips!r} is not a study county")


def week_end_of(day) -> pd.Timestamp:
    """The Saturday ending the MMWR week that contains ``day``."""
    d = pd.Timestamp(day).normalize()
    return d + timedelta(days=(5 - d.weekday()) % 7)
