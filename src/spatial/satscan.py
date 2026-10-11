"""Export county-week respiratory data for SaTScan and read its clusters back.

SaTScan (https://www.satscan.org) is a desktop program with no web API, but it
runs headless from a parameter file. This module writes everything that run
needs and parses the result, so SaTScan stays an optional step:

    python -m src.spatial.satscan --start 2025-11-02 --end 2026-01-31 \
        --model permutation --out data/satscan
    # then, if SaTScan is installed:
    satscan data/satscan/analysis.prm               # macOS/Linux CLI
    SaTScanBatch64.exe data\\satscan\\analysis.prm  # Windows
    # the GUI also opens the .prm (File > Open Session) and can run it.

On macOS the installer puts the app in /Applications/SaTScan.app; the
command-line binary lives inside the bundle (``find_satscan`` looks there).

Counts are an APPROXIMATION
---------------------------
NSSP publishes the *percent* of ED visits that were respiratory, never the
visit counts. SaTScan's Poisson and space-time permutation models need
counts. We reconstruct them as

    est. ED visits per week = population x (ED visits per person-year) / 52
    est. respiratory cases  = round(resp_pct / 100 x est. ED visits)

with a national rate of 0.45 visits per person-year (NHAMCS reports ~140-155
million ED visits a year for ~335 million people). This assumes every county
uses the ED at the national rate and that NSSP sees all of a county's EDs;
neither is true. The relative pattern across counties and weeks is what the
scan statistic uses, so it remains informative, but the "observed cases" in
SaTScan output are model numbers, not real visit counts. The space-time
permutation model is the safer choice: it conditions on each county's total
and each week's total, so a wrong per-county denominator largely cancels.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.snapshot import coerce_date

__all__ = [
    "ED_VISITS_PER_PERSON_YEAR", "MODELS", "estimate_counts",
    "write_case_file", "write_population_file", "write_coordinates_file",
    "write_parameter_file", "write_inputs", "find_satscan", "run_satscan",
    "parse_results_text", "parse_col_file",
]

ED_VISITS_PER_PERSON_YEAR = 0.45

# SaTScan ModelType codes for the two models that suit this data.
MODELS = {"poisson": 0, "permutation": 2}


def _date(value) -> str:
    return coerce_date(value).strftime("%Y/%m/%d")


def estimate_counts(share: pd.DataFrame, population: pd.Series,
                    visits_per_person_year: float = ED_VISITS_PER_PERSON_YEAR
                    ) -> pd.DataFrame:
    """Turn respiratory % into approximate weekly case counts (see module doc).

    ``share`` needs ``fips``, ``week_end``, ``resp_pct``; ``population`` is
    indexed by FIPS. Counties with no population are dropped, since without
    one there is no denominator to scale by.
    """
    out = share[["fips", "week_end", "resp_pct"]].copy()
    out["population"] = out["fips"].map(population)
    out = out.dropna(subset=["population", "resp_pct"])
    out["ed_visits_est"] = out["population"].astype(float) * visits_per_person_year / 52
    out["cases"] = (out["resp_pct"] / 100 * out["ed_visits_est"]).round().astype("int64")
    return out.reset_index(drop=True)


def write_case_file(counts: pd.DataFrame, path) -> Path:
    """``<fips> <cases> <YYYY/MM/DD>``, one line per county-week, dated to week end."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = counts[counts["cases"] > 0]
    with path.open("w") as fh:
        for fips, cases, week in zip(rows["fips"], rows["cases"], rows["week_end"]):
            fh.write(f"{fips} {int(cases)} {_date(week)}\n")
    return path


def write_population_file(population: pd.Series, path, year: int) -> Path:
    """``<fips> <year> <population>``; SaTScan interpolates between years."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for fips, pop in population.dropna().items():
            fh.write(f"{fips} {year} {int(pop)}\n")
    return path


def write_coordinates_file(counties: pd.DataFrame, path) -> Path:
    """``<fips> <latitude> <longitude>``: SaTScan's lat/long order (CoordinatesType=1)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for fips, lat, lon in zip(counties["fips"], counties["lat"], counties["lon"]):
            fh.write(f"{fips} {lat:.6f} {lon:.6f}\n")
    return path


def write_parameter_file(path, *, case_file, coordinates_file, start, end,
                         results_file, model: str = "permutation",
                         population_file=None, monte_carlo: int = 999,
                         max_spatial_pct: float = 10,
                         max_temporal_pct: float = 50) -> Path:
    """A retrospective space-time (.prm) session for SaTScan's batch mode.

    Only the keys this analysis depends on are written; SaTScan fills every
    other key with its default (and the GUI will re-save a complete file).
    ``max_spatial_pct`` defaults to 10% of the population at risk rather than
    SaTScan's 50%, because on a national map a 50% window can return "half
    the country" as one cluster, which says nothing about where.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
    if model == "poisson" and population_file is None:
        raise ValueError("the Poisson model needs a population file")

    lines = [
        "[Input]",
        f"CaseFile={case_file}",
        "PrecisionCaseTimes=3",                 # 3 = day
        f"StartDate={_date(start)}",
        f"EndDate={_date(end)}",
        f"PopulationFile={population_file or ''}",
        f"CoordinatesFile={coordinates_file}",
        "UseGridFile=n",
        "CoordinatesType=1",                    # 1 = latitude/longitude
        "",
        "[Analysis]",
        "AnalysisType=3",                       # 3 = retrospective space-time
        f"ModelType={MODELS[model]}",
        "ScanAreas=1",                          # 1 = high rates only
        "TimeAggregationUnits=3",               # days ...
        "TimeAggregationLength=7",              # ... in blocks of 7 = MMWR weeks
        "",
        "[Output]",
        f"ResultsFile={results_file}",
        "MostLikelyClusterEachCentroidASCII=y",  # -> .col.txt
        "CensusAreasReportedClustersASCII=y",    # -> .gis.txt
        "OutputShapefiles=n",
        "OutputGoogleEarthKML=n",
        "",
        "[Spatial Window]",
        f"MaxSpatialSizeInPopulationAtRisk={max_spatial_pct:g}",
        "SpatialWindowShapeType=0",             # 0 = circular
        "",
        "[Temporal Window]",
        "MinimumTemporalClusterSize=1",
        "MaxTemporalSizeInterpretation=0",      # 0 = percent of study period
        f"MaxTemporalSize={max_temporal_pct:g}",
        "",
        "[Inference]",
        f"MonteCarloReps={monte_carlo}",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def write_inputs(out_dir, share: pd.DataFrame, counties: pd.DataFrame,
                 start, end, model: str = "permutation",
                 population_year: int = 2025) -> dict:
    """Write case, population, coordinates and .prm files into ``out_dir``.

    ``share`` is :func:`src.spatial.hotspots.respiratory_share` output;
    ``counties`` is :func:`src.spatial.shapes.load_counties` output. Only
    counties present in both are exported, so every case has a coordinate.
    """
    out_dir = Path(out_dir)
    start, end = coerce_date(start), coerce_date(end)
    window = share[(share["week_end"] >= start) & (share["week_end"] <= end)]
    window = window[window["fips"].isin(counties["fips"])]
    population = counties.set_index("fips")["population"].dropna()

    counts = estimate_counts(window, population)
    used = counties[counties["fips"].isin(counts["fips"].unique())]

    paths = {
        "case": write_case_file(counts, out_dir / "cases.cas"),
        "coordinates": write_coordinates_file(used, out_dir / "counties.geo"),
        "population": write_population_file(
            population[population.index.isin(used["fips"])],
            out_dir / "population.pop", population_year),
    }
    # Paths inside the .prm are absolute so SaTScan finds them regardless of
    # the directory it is launched from.
    paths["parameters"] = write_parameter_file(
        out_dir / "analysis.prm",
        case_file=paths["case"].resolve(),
        coordinates_file=paths["coordinates"].resolve(),
        population_file=paths["population"].resolve(),
        start=start - pd.Timedelta(days=6), end=end,
        results_file=(out_dir / "results.txt").resolve(), model=model)
    return paths


_CANDIDATES = ("satscan", "satscan64", "SaTScanBatch64", "SaTScanBatch")
_MAC_BUNDLES = ("/Applications/SaTScan.app",)


def find_satscan() -> Path | None:
    """Path to a SaTScan command-line binary, or None if it is not installed."""
    for name in _CANDIDATES:
        found = shutil.which(name)
        if found:
            return Path(found)
    for bundle in _MAC_BUNDLES:
        macos = Path(bundle) / "Contents" / "MacOS"
        if macos.is_dir():
            for path in sorted(macos.iterdir()):
                if path.name.lower().startswith("satscan") and path.is_file():
                    return path
    return None


def run_satscan(parameter_file, binary: Path | None = None) -> Path:
    """Run SaTScan in batch mode on ``parameter_file``; returns the results path."""
    binary = binary or find_satscan()
    if binary is None:
        raise FileNotFoundError(
            "SaTScan is not installed. Download it from https://www.satscan.org "
            "and run the .prm file in batch mode or via the GUI.")
    subprocess.run([str(binary), str(parameter_file)], check=True)
    text = Path(parameter_file).read_text()
    return Path(re.search(r"^ResultsFile=(.*)$", text, re.M).group(1).strip())


# -- reading results -------------------------------------------------------

_CLUSTER_START = re.compile(r"^\s*(\d+)\.\s*Location IDs included\.*:\s*(.*)$")
_FIELD = re.compile(r"^\s*([A-Za-z][A-Za-z /()-]*?)\.{2,}:\s*(.*)$")
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

_NUMERIC_FIELDS = {
    "Number of cases": "observed",
    "Expected cases": "expected",
    "Observed / expected": "obs_over_exp",
    "Relative risk": "relative_risk",
    "Log likelihood ratio": "llr",
    "Test statistic": "test_statistic",
    "P-value": "p_value",
    "Population": "population",
}


def _number(text: str) -> float | None:
    match = _NUMBER.search(text.replace(",", ""))
    return float(match.group()) if match else None


def parse_results_text(path) -> pd.DataFrame:
    """Clusters from SaTScan's main text report, one row per cluster.

    The report is meant for people, but its cluster blocks are regular:
    ``N.Location IDs included.: a, b, ...`` (wrapping onto indented lines)
    followed by ``Label....: value`` lines. Reading this file needs no output
    options turned on, so it works on any SaTScan run.
    """
    clusters, current, last_key = [], None, None
    for line in Path(path).read_text(errors="replace").splitlines():
        start = _CLUSTER_START.match(line)
        if start:
            current = {"cluster": int(start.group(1)), "_ids": start.group(2)}
            clusters.append(current)
            last_key = "_ids"
            continue
        if current is None:
            continue
        field = _FIELD.match(line)
        if field:
            last_key = field.group(1).strip()
            current[last_key] = field.group(2).strip()
        elif line.strip() and line.startswith(" ") and last_key == "_ids":
            current["_ids"] += " " + line.strip()
        elif not line.strip():
            last_key = None

    rows = []
    for raw in clusters:
        row = {"cluster": raw["cluster"],
               "location_ids": [i for i in re.split(r"[,\s]+", raw["_ids"]) if i]}
        frame = raw.get("Time frame", "")
        dates = re.findall(r"\d{4}/\d{1,2}/\d{1,2}", frame)
        row["start"] = pd.Timestamp(dates[0]) if dates else pd.NaT
        row["end"] = pd.Timestamp(dates[-1]) if dates else pd.NaT
        for label, column in _NUMERIC_FIELDS.items():
            if label in raw:
                row[column] = _number(raw[label])
        rows.append(row)
    return pd.DataFrame(rows)


_COL_DEFAULT = ["CLUSTER", "LOC_ID", "LATITUDE", "LONGITUDE", "RADIUS",
                "START_DATE", "END_DATE", "NUMBER_LOC", "LLR", "P_VALUE",
                "OBSERVED", "EXPECTED", "ODE"]


def parse_col_file(path) -> pd.DataFrame:
    """SaTScan's ``.col.txt`` cluster table (one row per cluster).

    Recent versions write a header row; older ones do not, in which case the
    documented column order is assumed for as many columns as are present.
    """
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame(columns=_COL_DEFAULT)
    first = lines[0].split()
    has_header = not first[0].lstrip("-").replace(".", "").isdigit()
    header = first if has_header else None
    body = [l.split() for l in (lines[1:] if has_header else lines)]
    width = max(len(r) for r in body) if body else len(header or [])
    columns = header or (_COL_DEFAULT + [f"COL{i}" for i in range(len(_COL_DEFAULT), width)])[:width]
    frame = pd.DataFrame([r[:len(columns)] for r in body], columns=columns)
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            frame[column] = converted
    return frame


def main() -> None:
    from src.snapshot import SnapshotStore
    from src.spatial.hotspots import respiratory_share
    from src.spatial.shapes import load_counties, lower48

    parser = argparse.ArgumentParser(description="Write SaTScan input files.")
    parser.add_argument("--store", default="raw/snapshots")
    parser.add_argument("--as-of", default=None,
                        help="vintage date; default latest (display only)")
    parser.add_argument("--start", required=True, help="first week_end (Saturday)")
    parser.add_argument("--end", required=True, help="last week_end (Saturday)")
    parser.add_argument("--model", default="permutation", choices=sorted(MODELS))
    parser.add_argument("--out", default="data/satscan")
    parser.add_argument("--run", action="store_true", help="run SaTScan if installed")
    args = parser.parse_args()

    share = respiratory_share(SnapshotStore(args.store), args.as_of)
    counties = lower48(load_counties("500k"))
    paths = write_inputs(args.out, share, counties, args.start, args.end, args.model)
    for kind, path in paths.items():
        print(f"  {kind:<12} {path}")
    binary = find_satscan()
    if args.run and binary:
        results = run_satscan(paths["parameters"], binary)
        print(parse_results_text(results).to_string())
    elif binary is None:
        print("SaTScan not found; run the .prm with SaTScan's batch binary or GUI.")


if __name__ == "__main__":
    main()
