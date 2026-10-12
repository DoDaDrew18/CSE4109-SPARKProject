"""Ask the map a question in plain English: "where is flu rising around St. Louis?"

    python -m src.spatial.query "where is flu rising around St. Louis?"
    python -m src.spatial.query "RSV hot spots in the Midwest at the peak of the 2024-25 season"
    python -m src.spatial.query --parser ollama "..."    # or anthropic; default auto (Ollama if running, else rules)

The pipeline chart's box S2 ("Map + NL search", Demo II): a question in; a
county map, a short written answer, the study counties' nowcasts and the
local news that was public at the time out. A question flows through four
steps, each guarding against a different way the demo could mislead:

1. **Read the question** into a small, closed ``MapQuery``: illness, level or
   week-over-week change, week, place, how many places to list. A rule
   parser covers the phrasings a demo needs and runs offline; an LLM parser
   (local Ollama or Claude, the clients the news labeler uses) handles
   paraphrase. The LLM only fills that schema, with an enum for every closed
   field, so it cannot invent a place or a number.
2. **Check it against the data** (``resolve``): weeks snap to the MMWR
   Saturday and are clamped to the weeks that exist, places must exist in
   the county shapes, and every adjustment is written into the answer ("no
   data after Oct 3, 2026; showing that week"), never made silently.
3. **Compute from the data only.** Every number in the answer comes from the
   CDC table, the county_week vintages or the backtest predictions, never
   from the LLM. Clusters are computed on the whole lower 48 and then
   cropped, so zooming in does not change what counts as a cluster:
   Getis-Ord Gi* for levels, and local Moran's I for change, because Gi*
   assumes non-negative values and a change can be negative.
4. **Show only the news that was public then.** For study counties in view:
   articles labeled relevant (human labels first, else grounded LLM labels),
   published in the two weeks ending that Saturday -- the window the
   nowcast's news features use -- with the quote that grounds each label.

Values are Health Service Area (HSA) values copied to every county in the
HSA (README, data guide), so lists name one row per HSA ("Cook (Chicago), IL
area"), not five counties with the same number.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.fips import STATE_NAMES, normalize_county
from src.study import METROS, STUDY_FIPS, week_end_of

__all__ = ["SIGNALS", "REGIONS", "MapQuery", "Gazetteer", "load_values", "load_shapes",
           "parse_rules", "parse_llm", "resolve", "answer", "Answer", "summary_lines", "ask"]

CDC_CSV = Path("raw/cdc_nssp/rdmq-nq56_2026-10-10.csv")
SNAPSHOTS = Path("raw/snapshots")
COUNTY_WEEK = Path("data/county_week.parquet")
PREDICTIONS = Path("data/results/predictions.csv")

SIGNALS = {"resp": "respiratory illness (COVID + flu + RSV)", "covid": "COVID-19",
           "flu": "flu", "rsv": "RSV"}
MEASURES = ("level", "change")
NEWS_DAYS = 14            # the nowcast's news features: target week and the week before
MAX_TOP = 25
LOW_VOLUME = ("Sparse",)     # CDC's own trend flag for too few visits

# Census regions, lower 48 + DC (Alaska and Hawaii are not on the map).
REGIONS = {
    "northeast": ("CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"),
    "midwest": ("IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"),
    "south": ("DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN",
              "AR", "LA", "OK", "TX"),
    "west": ("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "CA", "OR", "WA"),
}
METRO_ALIASES = {"stl": ("st. louis", "st louis", "saint louis", "stl"),
                 "kc": ("kansas city", "kc"), "chi": ("chicago", "chicagoland"),
                 "ind": ("indianapolis", "indy"), "mem": ("memphis",), "lou": ("louisville",)}
# Metro views: a box around the metro's study counties, in degrees (lon, lat).
METRO_HALF_SPAN = (2.0, 1.3)
COUNTY_HALF_SPAN = (1.0, 0.7)


# -- the query ------------------------------------------------------------------

@dataclass
class MapQuery:
    """What the map should show. Closed fields only, so it can be validated."""
    signal: str = "resp"             # resp | covid | flu | rsv
    measure: str = "level"           # level | change (week over week)
    direction: str = "up"            # up | down: which way a change question looks
    when: str = "latest"             # latest | peak | YYYY-MM-DD (any day in the week)
    season: str = ""                 # for peak: "2025-26"; blank = season of the latest data
    place: str = "us"                # us | region:<name> | state:<XX> | metro:<key> | county:<fips>
    top: int = 5
    question: str = ""
    parser: str = "rules"
    notes: list[str] = field(default_factory=list)
    week_end: pd.Timestamp | None = None    # set by resolve()
    prev_week: pd.Timestamp | None = None

    def describe(self) -> dict:
        return {"signal": self.signal, "measure": self.measure, "direction": self.direction,
                "when": self.when, "season": self.season, "place": self.place, "top": self.top,
                "parser": self.parser}


class Gazetteer:
    """County names -> FIPS from the shapes, plus place labels and view boxes."""

    def __init__(self, shapes: pd.DataFrame):
        self.shapes = shapes
        self.by_name: dict[str, list[str]] = {}
        for fips, name in zip(shapes["fips"], shapes["name"]):
            self.by_name.setdefault(normalize_county(name), []).append(fips)
        self.state_of = dict(zip(shapes["fips"], shapes["state"]))
        self.name_of = dict(zip(shapes["fips"], shapes["name"]))
        self.states = set(shapes["state"])

    def counties_named(self, name: str, state: str | None = None) -> list[str]:
        hits = self.by_name.get(normalize_county(name), [])
        return [f for f in hits if state is None or self.state_of[f] == state]

    def label(self, place: str) -> str:
        kind, _, value = place.partition(":")
        if kind == "us":
            return "the lower 48 states"
        if kind == "region":
            return f"the {value.title()}"
        if kind == "state":
            return next((n.title() for n, p in STATE_NAMES.items() if p == value), value)
        if kind == "metro":
            return f"the {METROS[value].name} area"
        return f"{self.name_of.get(value, value)}, {self.state_of.get(value, '')}"

    def counties_in(self, place: str) -> list[str]:
        """FIPS shown for a place: a state's or region's counties, or every
        county touching the view box around a metro or a county."""
        kind, _, value = place.partition(":")
        s = self.shapes
        if kind == "us":
            return list(s["fips"])
        if kind == "region":
            return list(s.loc[s["state"].isin(REGIONS[value]), "fips"])
        if kind == "state":
            return list(s.loc[s["state"] == value, "fips"])
        core = list(METROS[value].counties) if kind == "metro" else [value]
        lon, lat = (s.loc[s["fips"].isin(core), c].mean() for c in ("lon", "lat"))
        dx, dy = METRO_HALF_SPAN if kind == "metro" else COUNTY_HALF_SPAN
        return list(s.cx[lon - dx:lon + dx, lat - dy:lat + dy]["fips"])

    def core(self, place: str) -> list[str]:
        """The counties a place is *about* (a metro's study counties, the one
        county); for peak-finding and the headline number."""
        kind, _, value = place.partition(":")
        if kind == "metro":
            return list(METROS[value].counties)
        if kind == "county":
            return [value]
        return self.counties_in(place)


# -- data -------------------------------------------------------------------------

_CDC = {"percent_visits_covid": "covid", "percent_visits_influenza": "flu", "percent_visits_rsv": "rsv",
        "ed_trends_covid": "trend_covid", "ed_trends_influenza": "trend_flu", "ed_trends_rsv": "trend_rsv"}


def load_values(csv=CDC_CSV, store_root=SNAPSHOTS) -> tuple[pd.DataFrame, str]:
    """County-week ED shares: ``fips, week_end, covid, flu, rsv, resp`` (+ CDC
    trend labels and HSA ids when they exist), and a one-line source credit.

    The CDC table when present (freshest, and it carries CDC's own trend call
    per area); otherwise the latest Delphi snapshot. Both are a single,
    already-revised vintage: right for a map of what happened, not for
    scoring (that is what the vintage store is for).
    """
    csv = Path(csv)
    if csv.exists():
        raw = pd.read_csv(csv, usecols=["week_end", "county", "fips", "hsa", "hsa_nci_id", *_CDC],
                          dtype={"fips": str, "hsa_nci_id": str})
        raw = raw[(raw["county"] != "All") & raw["fips"].notna()]
        out = raw.rename(columns=_CDC).drop(columns="county")
        out["fips"] = out["fips"].str.split(".").str[0].str.zfill(5)
        out["week_end"] = pd.to_datetime(out["week_end"])
        source = f"CDC NSSP (data.cdc.gov rdmq-nq56, file {csv.name})"
    else:
        from src.snapshot import SnapshotStore
        from src.spatial.hotspots import respiratory_share
        share = respiratory_share(SnapshotStore(store_root))
        if share.empty:
            raise FileNotFoundError(f"no CDC table at {csv} and no NSSP snapshots under {store_root}")
        out = share.rename(columns={"influenza": "flu"})[["fips", "week_end", "covid", "flu", "rsv"]]
        source = "CDC NSSP via Delphi Epidata (latest snapshot)"
    # Respiratory = sum of the three, only when all three exist (as everywhere in the repo).
    out["resp"] = out[["covid", "flu", "rsv"]].sum(axis=1, min_count=3)
    return out.sort_values(["fips", "week_end"]).reset_index(drop=True), source


def load_shapes() -> gpd.GeoDataFrame:
    """Lower-48 counties with ``fips, name, state, lat, lon`` (no population download)."""
    from src.spatial.shapes import add_centroids, load_boundaries, lower48
    return lower48(add_centroids(load_boundaries("500k")))


# -- reading the question: rules -------------------------------------------------------

_MONTHS = {m: i for i, names in enumerate(
    (("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",),
     ("jun", "june"), ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"),
     ("oct", "october"), ("nov", "november"), ("dec", "december")), start=1) for m in names}
_MONTH = r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|" \
         r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?"
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}
_ILLNESS = (("covid", r"\b(covid(?:-?19)?|coronavirus|sars-?cov-?2)\b"),
            ("flu", r"\b(flu|influenza)\b"), ("rsv", r"\b(rsv|syncytial)\b"))
_UP = (r"\b(ris(?:e|es|ing)|rose|increas\w*|grow\w*|grew|climb\w*|jump\w*|spik\w*|surg(?:e|es|ed|ing)|"
       r"accelerat\w*|spread\w*|worsen\w*|going up|getting worse|picking up)\b")
_DOWN = (r"\b(falling|fell|fallen|declin\w*|decreas\w*|drop\w*|eas(?:e|ed|es|ing)|subsid\w*|"
         r"improv\w*|going down|getting better|reced\w*|cooling)\b")
_CHANGE = r"\b(chang\w*|trend\w*|week[- ]over[- ]week|since (?:the )?(?:last|previous|prior) week)\b"


def _holiday(name: str, year: int) -> date:
    if name == "christmas":
        return date(year, 12, 25)
    if name == "new year's eve":
        return date(year, 12, 31)
    if name == "new year":
        return date(year, 1, 1)
    if name == "halloween":
        return date(year, 10, 31)
    if name == "independence day":
        return date(year, 7, 4)
    first = date(year, 11, 1)                     # thanksgiving: 4th Thursday of November
    return first + timedelta(days=(3 - first.weekday()) % 7 + 21)


_HOLIDAYS = ((r"christmas|xmas", "christmas"), (r"new year'?s eve", "new year's eve"),
             (r"new year'?s?(?: day)?", "new year"), (r"thanksgiving", "thanksgiving"),
             (r"halloween", "halloween"), (r"(?:fourth of july|july 4(?:th)?|independence day)",
                                           "independence day"))


def _year(text: str | None, fallback: int) -> int:
    if not text:
        return fallback
    y = int(text)
    return y + 2000 if y < 100 else y


def _parse_when(q: str, latest: pd.Timestamp, notes: list[str]) -> tuple[str, str]:
    """(when, season) from the lowercased question. ``latest`` anchors anything relative."""
    season = ""
    m = re.search(r"\b(20\d\d)\s*[-–/]\s*(\d\d(?:\d\d)?)\b(?:\s*(?:flu |respiratory )?season)?", q)
    if m and int(m.group(2)[-2:]) == (int(m.group(1)) + 1) % 100:
        season = f"{m.group(1)}-{m.group(2)[-2:]}"
    elif re.search(r"\blast (?:flu |respiratory )?season\b", q):
        season = _season_of(latest - pd.DateOffset(years=1))
    if re.search(r"\bpeak|\bworst week\b|\bhighest week\b", q):
        return "peak", season

    if m := re.search(r"\b(20\d\d)-(\d{1,2})-(\d{1,2})\b", q):
        return _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)), notes), season
    if m := re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", q):
        return _iso(_year(m.group(3), _recent_year(int(m.group(1)), int(m.group(2)), latest)),
                    int(m.group(1)), int(m.group(2)), notes), season
    if m := re.search(_MONTH + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s*(20\d\d))?", q):
        month, day = _MONTHS[_month_key(m.group(1))], int(m.group(2))
        return _iso(_year(m.group(3), _recent_year(month, day, latest)), month, day, notes), season
    if m := re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH + r"(?:,?\s*(20\d\d))?", q):
        month, day = _MONTHS[_month_key(m.group(2))], int(m.group(1))
        return _iso(_year(m.group(3), _recent_year(month, day, latest)), month, day, notes), season
    for pattern, name in _HOLIDAYS:
        if m := re.search(rf"\b(?:{pattern})\b(?:\s*(?:of\s*)?(20\d\d))?", q):
            year = int(m.group(1)) if m.group(1) else None
            if year is None:                     # the most recent one the data reaches
                year = latest.year
                while pd.Timestamp(_holiday(name, year)) > latest + timedelta(days=6):
                    year -= 1
            return _holiday(name, year).isoformat(), season
    if m := re.search(_MONTH + r"\s+(20\d\d)\b", q):
        return _month_end(_MONTHS[_month_key(m.group(1))], int(m.group(2)), latest, notes), season
    if m := re.search(r"\bin\s+" + _MONTH + r"(?!\s*\d)", q):
        if not m.group(1).startswith("may"):     # "may" alone is a verb more often than a month
            month = _MONTHS[_month_key(m.group(1))]
            year = latest.year if month <= latest.month else latest.year - 1
            return _month_end(month, year, latest, notes), season
    if m := re.search(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+"
                      r"(week|month|year)s?\s+ago\b", q):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMBERS[m.group(1)]
        weeks = {"week": 1, "month": 4, "year": 52}[m.group(2)] * n
        return (latest - timedelta(weeks=weeks)).date().isoformat(), season
    if re.search(r"\blast month\b", q):
        first = latest.replace(day=1)
        return _month_end((first - timedelta(days=1)).month, (first - timedelta(days=1)).year,
                          latest, notes), season
    if re.search(r"\b(last|a) year ago\b|\bthis time last year\b", q):
        return (latest - timedelta(weeks=52)).date().isoformat(), season
    if re.search(r"\blast week\b", q):
        notes.append("“last week” = the most recent week with data")
    return "latest", season


def _month_key(token: str) -> str:
    t = token.rstrip(".")
    return "sep" if t.startswith("sep") else t if t in _MONTHS else t[:3]


def _recent_year(month: int, day: int, latest: pd.Timestamp) -> int:
    """No year given: the most recent such date the data reaches."""
    try:
        this = pd.Timestamp(latest.year, month, day)
    except ValueError:
        return latest.year
    return latest.year if this <= latest + timedelta(days=6) else latest.year - 1


def _iso(year: int, month: int, day: int, notes: list[str]) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        notes.append(f"{year}-{month:02d}-{day:02d} is not a date; showing the latest week")
        return "latest"


def _month_end(month: int, year: int, latest: pd.Timestamp, notes: list[str]) -> str:
    """A month means its last week: the Saturday on or before the month's end."""
    end = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    saturday = end - timedelta(days=(end.weekday() - 5) % 7)
    notes.append(f"{end:%B %Y} = its last full week, ending {saturday:%b} {saturday.day}")
    return saturday.date().isoformat()


def _season_of(day) -> str:
    """Respiratory seasons run October to September: 2025-12-27 -> "2025-26"."""
    d = pd.Timestamp(day)
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _parse_place(q: str, original: str, gaz: Gazetteer, notes: list[str]) -> str:
    """Most specific place named wins: county > metro > state > region > US."""
    found = []                                         # (position, place)
    for m in re.finditer(r"\b(?:city of st\.? louis|st\.? louis city|saint louis city)\b", q):
        found.append((m.start(), "county:29510", m.span()))
    for m in re.finditer(r"((?:[a-z.'’-]+\s+){0,2}[a-z.'’-]+)\s+(?:county|parish)\b", q):
        words = m.group(1).split()
        for n in (3, 2, 1):
            if len(words) < n:
                continue
            name = " ".join(words[-n:])
            state = _state_near(original[m.end():m.end() + 24], gaz.states)
            hits = gaz.counties_named(name, state) or gaz.counties_named(name)
            if hits:
                found.append((m.start(), "county:" + _pick_county(hits, name, state, gaz, notes),
                              m.span()))
                break
    spans = [s for *_, s in found]
    masked = _mask(q, spans)
    for key, aliases in METRO_ALIASES.items():
        for alias in aliases:
            for m in re.finditer(rf"\b{re.escape(alias)}\b", masked):
                found.append((m.start(), f"metro:{key}", m.span()))
    masked = _mask(q, [s for *_, s in found])
    for name, postal in sorted(STATE_NAMES.items(), key=lambda kv: -len(kv[0])):
        if postal in gaz.states:
            part = r"(?:(?:north|south|east|west|central)(?:ern)?\s+|(?:up|down)state\s+)?"
            for m in re.finditer(rf"\b{part}{re.escape(name)}\b", masked):
                found.append((m.start(), f"state:{postal}", m.span()))
                masked = _mask(masked, [m.span()])
    for region in REGIONS:
        for m in re.finditer(rf"\b(?:the\s+)?{region}(?:ern)?\b(?!\s+(?:dakota|carolina|virginia))",
                             masked):
            found.append((m.start(), f"region:{region}", m.span()))
    if not found:
        return "us"
    rank = {"county": 0, "metro": 1, "state": 2, "region": 3}
    places = sorted(found, key=lambda f: (rank[f[1].split(":")[0]], f[0]))
    distinct = list(dict.fromkeys(p for _, p, _ in sorted(found)))
    chosen = places[0][1]
    others = [p for p in distinct if p != chosen and not _contains(p, chosen, gaz)]
    if others:
        notes.append(f"several places named; showing {gaz.label(chosen)} "
                     f"(ask again for {', '.join(gaz.label(p) for p in others)})")
    return chosen


def _contains(outer: str, inner: str, gaz: Gazetteer) -> bool:
    """True if ``outer`` merely contains ``inner`` (Missouri around St. Louis County)."""
    kind, _, value = outer.partition(":")
    if kind not in ("state", "region"):
        return outer.startswith("metro:") and inner.startswith("county:") and \
            inner.split(":")[1] in METROS[value].counties
    states = (value,) if kind == "state" else REGIONS[value]
    core = gaz.core(inner)
    return bool(core) and all(gaz.state_of.get(f) in states for f in core)


def _mask(q: str, spans) -> str:
    out = list(q)
    for a, b in spans:
        out[a:b] = "#" * (b - a)
    return "".join(out)


def _state_near(after: str, valid) -> str | None:
    """A state written right after a county name: ", KY", " Missouri"."""
    if (m := re.match(r"\s*,?\s*([A-Z]{2})\b", after)) and m.group(1) in valid:
        return m.group(1)
    tail = after.lower()
    for name, postal in sorted(STATE_NAMES.items(), key=lambda kv: -len(kv[0])):
        if re.match(rf"\s*,?\s*{re.escape(name)}\b", tail):
            return postal
    return None


def _pick_county(hits: list[str], name: str, state: str | None, gaz: Gazetteer,
                 notes: list[str]) -> str:
    if len(hits) == 1:
        return hits[0]
    study = [h for h in hits if h in STUDY_FIPS]
    pick = study[0] if study else sorted(hits)[0]
    notes.append(f"{gaz.name_of[pick]} exists in {len(hits)} states; showing "
                 f"{gaz.label('county:' + pick)} (add the state for another)")
    return pick


def parse_rules(question: str, gaz: Gazetteer, latest) -> MapQuery:
    """The offline parser: regular expressions over a closed vocabulary."""
    latest = pd.Timestamp(latest)
    norm = " " + re.sub(r"\s+", " ", question.replace("’", "'")) + " "
    q = norm.lower()
    notes: list[str] = []
    named = [s for s, pattern in _ILLNESS if re.search(pattern, q)]
    if len(named) == 1:
        signal = named[0]
    else:
        signal = "resp"
        if len(named) > 1:
            notes.append(f"{' and '.join(SIGNALS[s] for s in named)} asked together; "
                         "showing all three combined")
    down = bool(re.search(_DOWN, q))
    measure = "change" if (re.search(_UP, q) or down or re.search(_CHANGE, q)) else "level"
    top = 5
    if m := re.search(r"\btop\s+(\d+)\b|\b(\d+)\s+(?:highest|worst|most|biggest|counties|places|areas)\b", q):
        top = int(m.group(1) or m.group(2))
        q = _mask(q, [m.span()])                       # "top 10 Dec 2025" is not Dec 10
    when, season = _parse_when(q, latest, notes)
    place = _parse_place(q, norm, gaz, notes)
    return MapQuery(signal=signal, measure=measure, direction="down" if down and not re.search(_UP, q)
                    else "up", when=when, season=season, place=place, top=max(1, min(top, MAX_TOP)),
                    question=question.strip(), parser="rules", notes=notes)


# -- reading the question: an LLM ------------------------------------------------------

LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "signal": {"type": "string", "enum": list(SIGNALS)},
        "measure": {"type": "string", "enum": list(MEASURES)},
        "direction": {"type": "string", "enum": ["up", "down"]},
        "when": {"type": "string"},
        "season": {"type": "string"},
        "place_type": {"type": "string", "enum": ["us", "region", "state", "metro", "county"]},
        "region": {"type": "string", "enum": ["", *REGIONS]},
        "state": {"type": "string"},
        "metro": {"type": "string", "enum": ["", *METROS]},
        "county": {"type": "string"},
        "top": {"type": "integer"},
    },
    "required": ["signal", "measure", "direction", "when", "season", "place_type", "region",
                 "state", "metro", "county", "top"],
    "additionalProperties": False,
}

LLM_SYSTEM = """You turn a question about a map of US emergency department (ED) visits for \
respiratory illness into JSON for a mapping program. You only choose settings; the program \
computes every number from CDC data. The data covers weeks ending {first} to {latest}.

Fields:
- signal: "flu", "covid", "rsv", or "resp" (all three combined; use it when the question names \
none of them, or several).
- measure: "change" if the question asks where something is rising, falling, spreading or how \
it changed from the week before; otherwise "level".
- direction: "down" if it asks where activity is falling, declining or improving; else "up".
- when: "latest" (also for now, this week, last week, currently), "peak" (the highest week of a \
season), or a date YYYY-MM-DD inside the week asked about. Resolve relative dates ("two weeks \
ago", "last month", "Christmas") against {latest}, not today's date. A month means its last week.
- season: for "peak", the season as "2025-26" (seasons run October to September); "" means the \
season of {latest}.
- place_type plus exactly one of region / state / metro / county (leave the others ""):
  us = the whole country; region = one of {regions}; state = two-letter postal code;
  metro = {metros};
  county = "Name County, ST" for any other county (St. Louis city is "St. Louis city, MO").
- top: how many places to list (default 5, at most {max_top}).

If the question is not about this map, use the defaults: resp, level, up, latest, us, top 5."""

LLM_USER = "<question>{question}</question>"


def llm_request(question: str, first, latest, model: str) -> dict:
    """Keyword arguments for ``client.messages.create``; the same request
    serves Claude and the Ollama adapter (``src.news.ollama_client``)."""
    metros = "; ".join(f"{k} = {m.name}" for k, m in METROS.items())
    system = LLM_SYSTEM.format(first=pd.Timestamp(first).date(), latest=pd.Timestamp(latest).date(),
                               regions=", ".join(REGIONS), metros=metros, max_top=MAX_TOP)
    return {"model": model, "max_tokens": 16000, "system": system,
            "messages": [{"role": "user", "content": LLM_USER.format(question=question.strip())}],
            # A short, closed parse: low effort keeps an interactive demo responsive.
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": LLM_SCHEMA}}}


def _from_llm(data: dict, question: str, gaz: Gazetteer, parser: str) -> MapQuery:
    """Validate the LLM's JSON field by field; anything invalid raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    signal, measure, direction = data.get("signal"), data.get("measure"), data.get("direction")
    if signal not in SIGNALS or measure not in MEASURES or direction not in ("up", "down"):
        raise ValueError(f"bad signal/measure/direction: {signal}/{measure}/{direction}")
    when = str(data.get("when") or "latest").strip().lower()
    if when not in ("latest", "peak"):
        when = pd.Timestamp(when).date().isoformat()          # raises on nonsense
    season = str(data.get("season") or "").strip()
    if season and not re.fullmatch(r"20\d\d-\d\d", season):
        raise ValueError(f"bad season {season!r}")
    kind = data.get("place_type")
    if kind == "us":
        place = "us"
    elif kind == "region" and data.get("region") in REGIONS:
        place = f"region:{data['region']}"
    elif kind == "state" and str(data.get("state", "")).upper() in gaz.states:
        place = f"state:{data['state'].upper()}"
    elif kind == "metro" and data.get("metro") in METROS:
        place = f"metro:{data['metro']}"
    elif kind == "county":
        m = re.match(r"\s*(.+?)(?:\s+county)?\s*,\s*([A-Za-z]{2})\s*$", str(data.get("county", "")), re.I)
        if not m:
            raise ValueError(f"bad county {data.get('county')!r}")
        name = m.group(1)
        hits = gaz.counties_named(name, m.group(2).upper()) or gaz.counties_named(
            name + " city", m.group(2).upper())
        if not hits:
            raise ValueError(f"no county {data.get('county')!r}")
        place = f"county:{hits[0]}"
    else:
        raise ValueError(f"bad place {kind!r}")
    top = int(data.get("top") or 5)
    return MapQuery(signal=signal, measure=measure, direction=direction, when=when, season=season,
                    place=place, top=max(1, min(top, MAX_TOP)), question=question.strip(),
                    parser=parser)


def parse_llm(question: str, gaz: Gazetteer, first, latest, client, model: str,
              parser: str) -> MapQuery:
    """Ask the LLM for the settings; fall back to the rule parser on any failure,
    and say so in the notes. The LLM never produces a number shown on the map."""
    request = llm_request(question, first, latest, model)
    try:
        if parser == "anthropic":
            # Claude: opt into the server-side refusal fallback, so a declined
            # parse is retried on another model instead of failing the demo.
            response = client.beta.messages.create(**request, betas=["server-side-fallback-2026-07-01"],
                                                   fallbacks="default")
        else:
            response = client.messages.create(**request)
        if getattr(response, "stop_reason", None) == "refusal":
            raise ValueError("the model declined")
        text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
        query = _from_llm(json.loads(text), question, gaz, f"{parser}:{getattr(response, 'model', model)}")
    except Exception as exc:   # noqa: BLE001 - any failure means "use the rules", never a crash
        query = parse_rules(question, gaz, latest)
        query.notes.insert(0, f"LLM parse failed ({type(exc).__name__}: {str(exc)[:80]}); used the rule parser")
    return query


# -- checking it against the data ----------------------------------------------------------

def resolve(query: MapQuery, values: pd.DataFrame, gaz: Gazetteer) -> MapQuery:
    """Pin the week (and the week before, for change) to weeks that have data."""
    weeks = pd.DatetimeIndex(sorted(values.loc[values[query.signal].notna(), "week_end"].unique()))
    if weeks.empty:
        raise ValueError(f"no data for {query.signal}")
    first, latest = weeks[0], weeks[-1]
    if query.when == "latest":
        week = latest
    elif query.when == "peak":
        season = query.season or _season_of(latest)
        start = pd.Timestamp(int(season[:4]), 10, 1)
        if not query.season and (latest - start).days < 16 * 7:
            # A season a few weeks old has no peak yet (it comes Dec-Feb): use the last one.
            weeks_in = (latest - start).days // 7 + 1
            start -= pd.DateOffset(years=1)
            season = _season_of(start)
            query.notes.append(f"the current season has {weeks_in} week{'s' if weeks_in > 1 else ''} "
                               f"of data so far; using {season}")
        core = values[values["fips"].isin(gaz.core(query.place))
                      & values["week_end"].between(start, start + pd.DateOffset(years=1) - timedelta(days=1))]
        series = core.groupby("week_end")[query.signal].mean().dropna()
        if series.empty:
            query.notes.append(f"no {season} data for that place; showing the latest week")
            week = latest
        else:
            week = series.idxmax()
            query.notes.append(f"peak of the {season} season in {gaz.label(query.place)}: "
                               f"week ending {week:%b} {week.day}, {week.year}")
    else:
        week = week_end_of(query.when)
        if week > latest:
            query.notes.append(f"no data after {latest:%b} {latest.day}, {latest.year}; showing that week")
            week = latest
        elif week < first:
            query.notes.append(f"data starts {first:%b} {first.day}, {first.year}; showing that week")
            week = first
        elif week not in weeks:
            week = weeks[weeks <= week][-1]
            query.notes.append(f"no {query.signal} data that week; showing the week ending {week:%b} {week.day}")
    query.week_end = pd.Timestamp(week)
    earlier = weeks[weeks < week]
    query.prev_week = earlier[-1] if len(earlier) else None
    if query.measure == "change" and query.prev_week is not None \
            and query.prev_week != week - timedelta(weeks=1):
        query.notes.append(f"change measured against the week ending {query.prev_week:%b} "
                           f"{query.prev_week.day} (the nearest earlier week with data)")
    if query.measure == "change" and query.prev_week is None:
        query.notes.append("no earlier week to compare with; showing levels")
        query.measure = "level"
    return query


# -- computing the answer ------------------------------------------------------------------

@dataclass
class Answer:
    query: MapQuery
    frame: gpd.GeoDataFrame          # counties in view: value, prev, change, trend, cluster
    places: pd.DataFrame             # top list, one row per HSA
    study: list[dict]                # study metros in view: values, vintage, nowcast
    articles: pd.DataFrame           # news public in the two weeks to the Saturday
    source: str
    center: tuple[float, float]
    zoom: int
    regional: bool
    counts: dict = field(default_factory=dict)
    bounds: tuple[float, float, float, float] | None = None    # lon/lat box to fit beside the panel


def _week(values: pd.DataFrame, week, signal: str) -> pd.DataFrame:
    rows = values[values["week_end"] == week]
    keep = ["fips", signal] + [c for c in ("hsa", "hsa_nci_id", f"trend_{signal}") if c in rows]
    return rows[keep].rename(columns={signal: "value", f"trend_{signal}": "trend"})


def _clusters(shapes: gpd.GeoDataFrame, frame: pd.DataFrame, column: str, measure: str,
              permutations: int) -> pd.DataFrame:
    """National cluster classes: Gi* on levels, local Moran's I on change."""
    from src.spatial.hotspots import gi_star, local_moran
    data = frame[["fips", column]].dropna()
    if len(data) < 3:
        return pd.DataFrame(columns=["fips", "cluster"])
    stat = gi_star if measure == "level" else local_moran
    out = stat(shapes, data, column=column, permutations=permutations)
    return out[["fips", "class"]].rename(columns={"class": "cluster"})


def reliable_fips(values: pd.DataFrame, week, signal: str, weeks: int = 8) -> set[str]:
    """Counties whose numbers are worth ranking and testing for clusters.

    A tiny ED reports 0% most weeks and 50% the week one of its two visits is
    flu (Calloway, KY at the 2025-26 peak). Such a county would top every
    "highest" list, and in a permutation test its outlier widens the null
    for everyone: in the test lattice two 50% cells hide a 16-county cluster
    at 3x background that Gi* finds once they are left out. So a county
    counts only if its ED reported some COVID, flu or RSV in at least half of
    the last ``weeks`` weeks, and CDC does not flag its trend for this signal
    as "Sparse". ("Limited Data" areas are kept: their values look like
    everyone else's.) Excluded counties still appear on the map.
    """
    week = pd.Timestamp(week)
    recent = values[values["week_end"].between(week - timedelta(weeks=weeks - 1), week)]
    active = recent[["covid", "flu", "rsv"]].sum(axis=1, min_count=1).fillna(0) > 0
    share = active.groupby(recent["fips"]).mean()
    ok = set(share[share >= 0.5].index)
    if f"trend_{signal}" in values:
        now = values[values["week_end"] == week]
        ok -= set(now.loc[now[f"trend_{signal}"].isin(LOW_VOLUME), "fips"])
    return ok


def _top_places(view: pd.DataFrame, query: MapQuery, gaz: Gazetteer,
                sizes: pd.Series | None = None) -> pd.DataFrame:
    """The top ``query.top`` places in view, one row per HSA; ``sizes`` counts
    each HSA's counties nationally, so a view box cutting an HSA doesn't shrink it."""
    col = "value" if query.measure == "level" else "change"
    rows = view.dropna(subset=[col])
    if query.measure == "change" and query.direction == "down":
        rows = rows.sort_values(col)
    else:
        rows = rows.sort_values(col, ascending=False)
    if "hsa_nci_id" in rows and rows["hsa_nci_id"].notna().any():
        n = rows["hsa_nci_id"].map(sizes) if sizes is not None else \
            rows.groupby("hsa_nci_id")["fips"].transform("size")
        rows = rows.assign(n_counties=n.fillna(1).astype(int)).drop_duplicates("hsa_nci_id")
        rows["place"] = rows["hsa"].fillna("").str.split(" - ").str[0].str.strip() + " area"
    else:
        rows = rows.assign(n_counties=1,
                           place=[f"{gaz.name_of[f]}, {gaz.state_of[f]}" for f in rows["fips"]])
    cols = ["place", "fips", "value", "prev", "change", "n_counties"] + (["trend"] if "trend" in rows else [])
    return rows[cols].head(query.top).reset_index(drop=True)


def _study_details(query: MapQuery, view: pd.DataFrame, county_week=COUNTY_WEEK,
                   predictions=PREDICTIONS) -> list[dict]:
    """One entry per study metro in view: the map's values plus, for the
    respiratory share, what the nowcaster knew on that Saturday and what the
    models said (when the vintage table and backtest predictions exist)."""
    week = query.week_end
    table = pd.read_parquet(county_week) if Path(county_week).exists() else None
    preds = None
    if Path(predictions).exists():
        preds = pd.read_csv(predictions, dtype={"fips": str}, parse_dates=["as_of", "week_end"])
        preds = preds[preds["as_of"] == week]
    out = []
    for metro in METROS.values():
        fips = [f for f in metro.counties if f in set(view["fips"])]
        if not fips:
            continue
        row = view[view["fips"] == fips[0]].iloc[0]
        item = {"metro": metro.name, "fips": fips, "value": row["value"], "prev": row["prev"],
                "change": row["change"], "trend": row.get("trend")}
        if table is not None:
            t = table[table["fips"] == fips[0]]
            known = t[(t["as_of"] == week) & (t["lag"] >= 1)].dropna(subset=["ed_resp"])
            if len(known):
                newest = known.sort_values("lag").iloc[0]
                item["known_value"] = float(newest["ed_resp"])
                item["known_week"] = pd.Timestamp(newest["week_end"])
            elif (t["as_of"] == week).any():          # a prediction date with nothing published
                item["known_value"] = None
            labels = t[t["week_end"] == week][["ed_resp_first", "ed_resp_final"]].dropna(how="all")
            if len(labels):
                item["first_print"] = labels["ed_resp_first"].iloc[0]
                item["final"] = labels["ed_resp_final"].iloc[0]
                seen = t.loc[(t["week_end"] == week) & t["ed_resp"].notna(), "as_of"]
                item["first_seen"] = seen.min() if len(seen) else None
        if preds is not None and len(preds):
            p = preds[preds["fips"] == fips[0]].set_index("rung")["pred"]
            item["nowcast"] = {r: float(p[r]) for r in ("0_naive", "6_gbm_history", "12_gbm_news")
                               if r in p and pd.notna(p[r])}
        out.append(item)
    return out


def cited_articles(fips, week_end, days: int = NEWS_DAYS, catalog="labels/articles.csv",
                   llm_labels="data/labels/llm_labels_grounded.parquet") -> tuple[pd.DataFrame, str]:
    """Relevant articles about ``fips`` published in the ``days`` ending ``week_end``.

    Human labels win where they exist (they are the validated source); else
    grounded LLM labels. Publish date, never event week, decides visibility:
    an article from after the Saturday is never shown for it.
    """
    from src.news.features import article_table
    from src.news.schema import human_label_files, load_human_labels

    cols = ["article_id", "published", "fips", "concern", "quote", "title", "outlet", "url", "labeler"]
    source = "human"
    arts = article_table("human")
    if arts.empty:
        source = "llm"
        arts = article_table("llm", labels_path=llm_labels)
    if arts.empty:
        return pd.DataFrame(columns=cols), "none"
    cat = (pd.read_csv(catalog, dtype=str, keep_default_na=False).set_index("article_id")
           if Path(catalog).exists() else None)
    if cat is not None and arts["published"].isna().any():
        # Human labels carry no date and the fetch manifest may not be on this
        # machine; the committed catalog has the same page dates.
        fill = pd.to_datetime(arts["article_id"].map(cat["published"]), errors="coerce")
        arts["published"] = arts["published"].fillna(fill)
    end = pd.Timestamp(week_end).normalize()
    keep = (arts["relevant"].map(lambda v: v is True or v is np.True_) & arts["fips"].isin(list(fips))
            & arts["published"].between(end - timedelta(days=days - 1), end))
    arts = arts[keep].copy()
    if arts.empty:
        return pd.DataFrame(columns=cols), source
    if source == "human":
        humans = load_human_labels(human_label_files())
        quotes = humans[humans["quote"].fillna("") != ""].drop_duplicates("article_id")
        arts["quote"] = arts["article_id"].map(dict(zip(quotes["article_id"], quotes["quote"])))
    else:
        llm = pd.read_parquet(llm_labels)
        llm = llm[llm["run"] == llm["run"].min()].drop_duplicates("article_id") if "run" in llm else llm
        q = llm.set_index("article_id")
        arts["quote"] = arts["article_id"].map(q["concern_quote"].where(q["concern_quote"].notna(),
                                                                         q.get("fips_quote")))
    if cat is not None:
        for col in ("title", "outlet", "url"):
            arts[col] = arts["article_id"].map(cat[col])
    arts["labeler"] = source
    return arts.reindex(columns=cols).sort_values("published", ascending=False).reset_index(drop=True), source


def answer(query: MapQuery, values: pd.DataFrame, shapes: gpd.GeoDataFrame, gaz: Gazetteer,
           source: str = "", permutations: int = 9999, county_week=COUNTY_WEEK,
           predictions=PREDICTIONS, with_news: bool = True) -> Answer:
    """Everything the map and the written answer need, computed from the data."""
    sig, week, prev = query.signal, query.week_end, query.prev_week
    now = _week(values, week, sig)
    before = _week(values, prev, sig)[["fips", "value"]].rename(columns={"value": "prev"}) \
        if prev is not None else pd.DataFrame({"fips": pd.Series(dtype=str),
                                               "prev": pd.Series(dtype=float)})
    frame = now.merge(before, on="fips", how="left")
    frame["change"] = frame["value"] - frame["prev"]
    column = "value" if query.measure == "level" else "change"
    ok = reliable_fips(values, week, sig)
    frame = frame.merge(_clusters(shapes, frame[frame["fips"].isin(ok)], column, query.measure,
                                  permutations), on="fips", how="left")
    frame.loc[~frame["fips"].isin(ok) & frame[column].notna(), "cluster"] = "Low volume (not tested)"

    in_view = set(gaz.counties_in(query.place))
    shown = shapes[shapes["fips"].isin(in_view)].merge(frame, on="fips", how="left")
    shown["cluster"] = shown["cluster"].fillna("No data")
    data = shown.dropna(subset=[column])

    kind = query.place.split(":")[0]
    core = shapes[shapes["fips"].isin(gaz.core(query.place))]
    center = (float(core["lat"].mean()), float(core["lon"].mean()))
    if kind in ("metro", "county"):
        dx, dy = METRO_HALF_SPAN if kind == "metro" else COUNTY_HALF_SPAN
        bounds = (center[1] - dx, center[0] - dy, center[1] + dx, center[0] + dy)
    else:
        bounds = tuple(float(v) for v in shown.total_bounds)
    zoom = {"us": 4, "region": 5, "state": 6, "metro": 8, "county": 8}[kind]
    if kind == "us":
        center = (38.5, -96.5)

    cluster_word = "Hot spot" if query.measure == "level" else (
        "High-High" if query.direction == "up" else "Low-Low")
    counts = {"counties": int(len(shown)), "with_data": int(len(data)),
              "clusters": int(shown["cluster"].str.startswith(cluster_word).sum()),
              "rising": int((data["change"] > 0).sum()), "falling": int((data["change"] < 0).sum()),
              "median": float(data["value"].median()) if len(data) else float("nan")}
    study = _study_details(query, shown, county_week, predictions)
    if with_news and study:
        fips = [f for s in study for f in s["fips"]]
        articles, news_source = cited_articles(fips, week)
    else:
        articles, news_source = pd.DataFrame(), "none"
    counts["news_source"] = news_source
    sizes = frame.groupby("hsa_nci_id")["fips"].size() if "hsa_nci_id" in frame else None
    ranked = shown["fips"].isin(ok)
    unranked = shown.loc[~ranked].dropna(subset=[column])
    key = "hsa_nci_id" if "hsa_nci_id" in unranked else "fips"
    counts["unranked"] = int(unranked[key].nunique())
    return Answer(query=query, frame=shown, places=_top_places(shown[ranked], query, gaz, sizes),
                  study=study,
                  articles=articles, source=source, center=center, zoom=zoom,
                  regional=kind in ("metro", "county", "state"), counts=counts, bounds=bounds)


# -- the written answer ------------------------------------------------------------------------

def _pct(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.1f}%"


def _pts(x) -> str:
    if x is None or pd.isna(x):
        return "–"
    return "±0.0 pts" if round(x, 1) == 0 else f"{x:+.1f} pts"


def _date(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts:%b} {ts.day}, {ts.year}"


def summary_lines(ans: Answer, gaz: Gazetteer) -> dict:
    """The written answer as plain strings: headline, bullets, study, news."""
    q, c = ans.query, ans.counts
    what = SIGNALS[q.signal]
    where = gaz.label(q.place)
    week = _date(q.week_end)
    n, k = c["with_data"], c["clusters"]
    if q.measure == "level":
        head = (f"Week ending {week}, {where}: the median county had {_pct(c['median'])} of ED "
                f"visits for {what}; " + (f"{k} counties are Gi* hot spots (FDR < 0.10)." if k
                                          else "no county is a significant Gi* hot spot."))
        lead = "Highest"
    else:
        moving = "rising" if q.direction == "up" else "falling"
        head = (f"Week ending {week} vs the week before, {where}: {what} rose in {c['rising']} of "
                f"{n} counties and fell in {c['falling']}; "
                + (f"{k} form significant clusters of {moving} counties." if k
                   else f"no significant cluster of {moving} counties stands out from the national pattern."))
        lead = "Biggest rises" if q.direction == "up" else "Biggest falls"
    places, notes = [], list(q.notes)
    if c.get("unranked"):
        notes.append(f"{c['unranked']} low-volume area{'s' if c['unranked'] > 1 else ''} left out "
                     "of the list and the cluster test (mostly-zero weeks, or CDC flags it "
                     "sparse); still on the map")
    for r in ans.places.itertuples(index=False):
        more = f" ({r.n_counties} counties)" if r.n_counties > 1 else ""
        trend = f"; CDC trend: {r.trend}" if hasattr(r, "trend") and isinstance(r.trend, str) else ""
        places.append(f"{r.place}{more}: {_pct(r.value)} ({_pts(r.change)} vs prior week{trend})")
    study = []
    for s in ans.study:
        line = (f"{s['metro']}: {_pct(s['value'])} {what} ({_pts(s['change'])}"
                + (f"; CDC trend: {s['trend']}" if isinstance(s.get("trend"), str) else "") + ").")
        if "known_value" in s:
            sat = f"Sat {q.week_end:%b} {q.week_end.day}"
            if s["known_value"] is None:
                line += f" On {sat} no official respiratory number was public yet."
            else:
                line += (f" On {sat} the newest official respiratory number was "
                         f"{_pct(s['known_value'])} (week ending {s['known_week']:%b} {s['known_week'].day}).")
            if pd.notna(s.get("first_print")):
                when = (f" (public by {_date(s['first_seen'])})"
                        if s.get("first_seen") is not None and pd.notna(s["first_seen"]) else "")
                line += (f" This week first printed at {_pct(s['first_print'])}{when}, "
                         f"settled at {_pct(s.get('final'))}.")
        now = s.get("nowcast") or {}
        if "6_gbm_history" in now:
            line += (f" Nowcast that day: {_pct(now['6_gbm_history'])} (gradient boosting; "
                     f"carry-forward {_pct(now.get('0_naive'))}).")
        study.append(line)
    news = []
    for a in ans.articles.itertuples(index=False):
        quote = f" “{a.quote}”" if isinstance(a.quote, str) and a.quote else ""
        news.append(f"{a.outlet or 'news'}, {pd.Timestamp(a.published):%b} {pd.Timestamp(a.published).day}"
                    f" (concern {a.concern}): {a.title or a.article_id}.{quote}")
    if ans.study and not len(ans.articles):
        news_note = {"none": "No news labels on this machine (label with src.news.label_app, or "
                             "run src.news.label + src.news.ground).",
                     "human": "No hand-labeled local news about these counties in the two weeks "
                              "to that Saturday.",
                     "llm": "No grounded LLM-labeled local news about these counties in the two "
                            "weeks to that Saturday."}[ans.counts.get("news_source", "none")]
        news = [news_note]
    return {"headline": head, "lead": lead, "places": places, "study": study, "news": news,
            "notes": notes}


# -- one call for the CLI and the web app -------------------------------------------------------

def ask(question: str, values: pd.DataFrame, shapes: gpd.GeoDataFrame, gaz: Gazetteer,
        source: str = "", parser: str = "auto", client=None, model: str | None = None,
        permutations: int = 9999, **kwargs) -> Answer:
    weeks = values["week_end"]
    first, latest = weeks.min(), weeks.max()
    if parser == "auto":
        parser, client = _auto_client()
    if parser == "rules":
        query = parse_rules(question, gaz, latest)
    else:
        if client is None:
            client = _client(parser)
        from src.news.label import MODEL_ID, OLLAMA_MODEL
        model = model or (MODEL_ID if parser == "anthropic" else OLLAMA_MODEL)
        query = parse_llm(question, gaz, first, latest, client, model, parser)
    return answer(resolve(query, values, gaz), values, shapes, gaz, source, permutations, **kwargs)


def _auto_client():
    """Ollama if it is running locally (free); otherwise the offline rules.
    Claude is used only when asked for (--parser anthropic): it costs money."""
    from src.news.label import OLLAMA_MODEL
    from src.news.ollama_client import OllamaClient, OllamaError
    client = OllamaClient()
    try:
        client.check(OLLAMA_MODEL)
        return "ollama", client
    except OllamaError:
        return "rules", None


def _client(parser: str):
    if parser == "ollama":
        from src.news.ollama_client import OllamaClient
        return OllamaClient()
    import anthropic
    return anthropic.Anthropic()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Ask the county map a question in plain English.")
    p.add_argument("question", nargs="+")
    p.add_argument("--parser", choices=("auto", "rules", "ollama", "anthropic"), default="auto")
    p.add_argument("--model", default=None)
    p.add_argument("--out", default="demo/query.html")
    p.add_argument("--permutations", type=int, default=9999)
    args = p.parse_args(argv)

    from src.spatial.query_map import render
    values, source = load_values()
    shapes = load_shapes()
    gaz = Gazetteer(shapes)
    ans = ask(" ".join(args.question), values, shapes, gaz, source, args.parser, model=args.model,
              permutations=args.permutations)
    text = summary_lines(ans, gaz)
    print(f"read as: {json.dumps(ans.query.describe())}")
    for note in text["notes"]:
        print(f"  note: {note}")
    print(text["headline"])
    print(f"{text['lead']}:")
    for i, line in enumerate(text["places"], 1):
        print(f"  {i}. {line}")
    for line in text["study"] + text["news"]:
        print(f"  - {line}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(ans, gaz).save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
