"""Build the ``county_week`` table: every model reads this and nothing else.

One row per ``(fips, week_end, as_of)``. Usage, from the repo root:

    python -m src.align                      # every backfilled vintage
    python -m src.align --history 26         # shorter look-back per as_of

Which weeks appear for each ``as_of`` -- the timing contract
-------------------------------------------------------------
``as_of`` is a prediction date D. In this project D is always a vintage issue
date: the Saturday ending an MMWR week (``src.backfill_nssp`` stamps vintages
that way). For each D the table holds:

* the **target week**: ``week_end == week_end_of(D)``. When D is a Saturday this
  is the week ending *that day*. NSSP publishes a week roughly one week after
  it ends (Delphi's lag-1 first print), so on D the target week's ED values are
  normally **missing** -- estimating it is the nowcast. ``lag == 0``.
* ``history`` earlier weeks, ``lag == 1 .. history``, with ED values as known on
  D. Lag 1 is usually the first print (incomplete, revised upward later), lag
  2-3 is partially revised, older lags are close to settled. Archive holes
  show up as missing values at small lags, not as fabricated ones.

The history window doubles as the source of honest training labels: the value
of week D' *as known on D* is the row ``(week_end=D', as_of=D)``, so a model at
origin D can learn from past weeks without ever reading a later vintage (see
``src.nowcast``). That is why the default window is a year, not a few lags.

Labels -- answer key only
-------------------------
``ed_resp_final`` (sum of the three signals in the latest vintage) and
``ed_resp_first`` (sum of each signal's first published value) are attached for
scoring. They are not "known on as_of"; :data:`LABEL_COLUMNS` names them so
the model code can assert they never enter a feature matrix.

Optional feature sources (weather, wastewater, news) are loaded lazily through
a shared contract, ``weekly_features(as_of, fips) -> [fips, week_end, ...]``.
A missing module or missing data logs a warning and the columns are simply
absent; every loader is checked for rows dated after ``as_of``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.snapshot import SnapshotStore, coerce_date
from src.study import HOME_METRO, STUDY_FIPS, metro_of, week_end_of

log = logging.getLogger(__name__)

SIGNALS = {"ed_covid": "nssp_covid", "ed_flu": "nssp_influenza", "ed_rsv": "nssp_rsv"}
ED_COLUMNS = tuple(SIGNALS)
LABEL_COLUMNS = ("ed_resp_final", "ed_resp_first")
KEY = ["fips", "week_end", "as_of"]

# (module, expected columns). Order is the order columns appear in the table.
FEATURE_LOADERS = {
    "src.ingest_noaa": ("temp_mean_f", "temp_min_f", "temp_max_f", "dewpoint_f",
                        "rh_mean", "precip_in", "n_days"),
    "src.ingest_nwss": ("ww_covid", "ww_flu", "ww_rsv", "ww_n_sites"),
    "src.news.features": ("news_n", "news_concern"),
}

DEFAULT_HISTORY = 52
OUTPUT = Path("data/county_week.parquet")


class LeakError(AssertionError):
    """A feature describes a week after the date it is supposed to be known on."""


# -- NSSP -------------------------------------------------------------------

def _week_end(time_value: pd.Series) -> pd.Series:
    """NSSP ``time_value`` is the Sunday a week starts; rows key on its Saturday."""
    return pd.to_datetime(time_value) + timedelta(days=6)


def _sum_resp(frame: pd.DataFrame) -> pd.Series:
    """Respiratory share = covid + flu + rsv; missing if any part is missing.

    A partial sum would look like a sharp drop, which is worse than a gap.
    """
    return frame[list(ED_COLUMNS)].sum(axis=1, min_count=len(ED_COLUMNS))


def _wide(frames: dict[str, pd.DataFrame], fips) -> pd.DataFrame:
    """Join one ``[geo_value, time_value, value]`` frame per signal into columns."""
    out = None
    for column, frame in frames.items():
        part = frame.loc[frame["geo_value"].isin(fips), ["geo_value", "time_value", "value"]]
        part = part.rename(columns={"geo_value": "fips", "value": column})
        part["week_end"] = _week_end(part.pop("time_value"))
        out = part if out is None else out.merge(part, on=["fips", "week_end"], how="outer")
    return out


def nssp_vintage(store: SnapshotStore, as_of, fips=STUDY_FIPS) -> pd.DataFrame:
    """ED shares for every week as known on ``as_of``: [fips, week_end, ed_*]."""
    return _wide({c: store.load_vintage(s, as_of) for c, s in SIGNALS.items()}, fips)


def nssp_labels(store: SnapshotStore, fips=STUDY_FIPS) -> pd.DataFrame:
    """Answer key per (fips, week_end): final and first-print respiratory share.

    The first print of each signal is its value in the earliest issue that
    contains the week; for most weeks that is the lag-1 print, for archive
    holes it is whatever came first.
    """
    final, first = {}, {}
    for column, source in SIGNALS.items():
        final[column] = store.latest_vintage(source)
        issues = store.available_issues(source)
        if issues:
            history = pd.concat([store._read_file(store.path_for(source, i)) for i in issues])
            history = history.sort_values(["geo_value", "time_value", "issue"], kind="mergesort")
            first[column] = history.drop_duplicates(["geo_value", "time_value"], keep="first")
        else:
            first[column] = final[column]
    fin, fst = _wide(final, fips), _wide(first, fips)
    labels = fin[["fips", "week_end"]].assign(ed_resp_final=_sum_resp(fin))
    return labels.merge(fst[["fips", "week_end"]].assign(ed_resp_first=_sum_resp(fst)),
                        on=["fips", "week_end"], how="outer")


# -- optional features ---------------------------------------------------------

def _missing_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError:          # parent package absent
        return True


def check_no_future(frame: pd.DataFrame, as_of, name: str = "features") -> None:
    """Raise :class:`LeakError` if any row describes a week ending after ``as_of``."""
    as_of = coerce_date(as_of)
    if frame.empty:
        return
    late = pd.to_datetime(frame["week_end"]) > as_of
    if late.any():
        raise LeakError(f"{name}: {int(late.sum())} row(s) have week_end after "
                        f"as_of {as_of.date()}, e.g. {frame.loc[late, 'week_end'].max()}")


def _load_optional(module: str, columns, as_of, fips,
                   warned: set | None = None) -> pd.DataFrame | None:
    """Call ``module.weekly_features``; ``None`` (with a warning) if unavailable.

    Only "not there yet" failures are swallowed: an import error, missing data
    on disk, or a loader with no rows. A leak is never swallowed. ``warned``
    keeps a 130-date build from repeating the same warning 130 times.
    """
    warned = set() if warned is None else warned

    def skip(reason):
        if module not in warned:
            log.warning("skipping %s (first at as_of %s): %s", module,
                        pd.Timestamp(as_of).date(), reason)
            warned.add(module)

    try:
        loader = importlib.import_module(module).weekly_features
    except (ImportError, AttributeError) as exc:
        skip(exc)
        return None
    try:
        frame = loader(as_of, fips=list(fips))
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        skip(exc)
        return None
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    frame["fips"] = frame["fips"].astype(str).str.zfill(5)
    frame["week_end"] = pd.to_datetime(frame["week_end"]).dt.normalize()
    check_no_future(frame, as_of, module)
    keep = [c for c in columns if c in frame.columns]
    return frame[["fips", "week_end", *keep]]


# -- the table ---------------------------------------------------------------

def _slice(vintage: pd.DataFrame, as_of: pd.Timestamp, fips, history: int) -> pd.DataFrame:
    """The (fips x week) grid for one as_of, with ED values where published."""
    target = week_end_of(as_of)
    weeks = [target - timedelta(weeks=k) for k in range(history + 1)]
    grid = pd.MultiIndex.from_product([list(fips), weeks], names=["fips", "week_end"])
    out = grid.to_frame(index=False).merge(vintage, on=["fips", "week_end"], how="left")
    out["as_of"] = as_of
    out["lag"] = ((target - out["week_end"]).dt.days // 7).astype("int64")
    # The target week can end after a mid-week as_of; nothing about it can be
    # known yet, whatever the vintage says.
    out.loc[out["week_end"] > as_of, list(ED_COLUMNS)] = float("nan")
    return out


def build_county_week(as_of_dates, fips=STUDY_FIPS, store_root="raw/snapshots",
                      history: int = DEFAULT_HISTORY,
                      feature_modules=None) -> pd.DataFrame:
    """The aligned county x week x as_of table described in docs/OVERVIEW.md."""
    store = SnapshotStore(store_root)
    fips = tuple(fips)
    modules = FEATURE_LOADERS if feature_modules is None else feature_modules
    dead: set[str] = set()
    warned: set[str] = set()
    slices = []
    for as_of in sorted({coerce_date(d) for d in as_of_dates}):
        part = _slice(nssp_vintage(store, as_of, fips), as_of, fips, history)
        for module, columns in modules.items():
            if module in dead:
                continue
            feats = _load_optional(module, columns, as_of, fips, warned)
            if feats is None:
                if _missing_module(module):
                    dead.add(module)  # not installed: no point asking again
                continue
            part = part.merge(feats, on=["fips", "week_end"], how="left")
        slices.append(part)
    if not slices:
        raise ValueError("no as_of dates given")
    table = pd.concat(slices, ignore_index=True)

    table["ed_resp"] = _sum_resp(table)
    table["metro"] = table["fips"].map(metro_of)
    table["home"] = table["metro"] == HOME_METRO
    table = table.merge(nssp_labels(store, fips), on=["fips", "week_end"], how="left")

    features = [c for cols in modules.values() for c in cols if c in table.columns]
    front = ["fips", "metro", "home", "week_end", "as_of", "lag", *ED_COLUMNS, "ed_resp"]
    table = table[front + features + list(LABEL_COLUMNS)]
    assert_no_leak(table)
    return table.sort_values(KEY).reset_index(drop=True)


def assert_no_leak(table: pd.DataFrame) -> None:
    """The table's two guarantees, checked on every build.

    1. Nothing known on ``as_of`` describes a week ending after it.
    2. Rows are unique per (fips, week_end, as_of).
    """
    known = [c for c in table.columns
             if c not in LABEL_COLUMNS and c not in ("fips", "metro", "home",
                                                      "week_end", "as_of", "lag")]
    future = (table["week_end"] > table["as_of"]) & table[known].notna().any(axis=1)
    if future.any():
        raise LeakError(f"{int(future.sum())} row(s) carry values for weeks after as_of")
    if table.duplicated(KEY).any():
        raise LeakError("county_week has duplicate (fips, week_end, as_of) rows")


def vintage_dates(store_root="raw/snapshots") -> list[pd.Timestamp]:
    """Saturday issue dates on disk for all three NSSP signals: the as_of grid.

    Off-grid issues (a live ``ingest_nssp`` run on a Tuesday) still count for
    the labels via ``latest_vintage``, but they are not prediction dates: the
    backtest compares origins one week apart, each a Saturday.
    """
    store = SnapshotStore(store_root)
    common = None
    for source in SIGNALS.values():
        issues = set(store.available_issues(source))
        common = issues if common is None else common & issues
    return sorted(d for d in (common or []) if d.dayofweek == 5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/county_week.parquet.")
    parser.add_argument("--store", default="raw/snapshots")
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dates = vintage_dates(args.store)
    table = build_county_week(dates, store_root=args.store, history=args.history)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f"{len(table):,} rows, {len(dates)} as_of dates "
          f"({dates[0].date()} .. {dates[-1].date()}) -> {args.out}")
    print("columns:", ", ".join(table.columns))


if __name__ == "__main__":
    main()
