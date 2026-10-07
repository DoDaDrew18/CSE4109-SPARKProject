"""Append-only snapshot store and point-in-time vintage loader.

The project's central claim is that text improves county-level respiratory
nowcasts *once the model sees only the data that existed on the prediction
date*. Surveillance data breaks that guarantee in two ways:

  Reporting lag  a reference week's value is not published during that week
  Backfill       the first published value is incomplete and revised upward

So every observation is stored with two dates, following the Delphi Epidata
convention:

  ``time_value``  the reference week the number describes
  ``issue``       the date that number was published

A *vintage* is the answer to "what did we know on date D". It is built by
taking every row whose ``issue <= D`` and, for each
``(geo_value, time_value)`` pair, keeping the most recently issued one. Rows
whose earliest issue is after D are absent entirely, because on date D they
had not been published yet. Training and scoring against a vintage is what
makes the backtest honest; reading the finalized series instead is the leak
this module exists to prevent.

Snapshots are immutable. A snapshot file is written once and never edited,
because rewriting history would silently change what past models "knew".
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

__all__ = [
    "SnapshotStore",
    "SnapshotError",
    "SnapshotExistsError",
    "SchemaError",
    "REQUIRED_COLUMNS",
    "coerce_date",
    "empty_vintage",
]

# Canonical columns every snapshot must carry. Extra columns are passed
# through untouched so sources can keep their own metadata.
REQUIRED_COLUMNS = ("geo_value", "time_value", "value", "issue")

_KEY = ["geo_value", "time_value"]
_SOURCE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ISSUE_FILE_RE = re.compile(r"^issue=(\d{4}-\d{2}-\d{2})\.parquet$")


class SnapshotError(Exception):
    """Base class for snapshot store failures."""


class SnapshotExistsError(SnapshotError):
    """Raised when a write would alter an already-stored snapshot."""


class SchemaError(SnapshotError):
    """Raised when a frame does not satisfy the snapshot contract."""


def coerce_date(value) -> pd.Timestamp:
    """Normalize a date-like value to a midnight-aligned ``pd.Timestamp``.

    Accepts ``str`` ("2026-01-07"), ``datetime.date``, ``datetime.datetime``
    and ``pd.Timestamp``. Normalizing to midnight means an ``as_of`` of
    "2026-01-07" compares equal to an issue stamped at any time that day,
    which keeps vintage boundaries inclusive in the way callers expect.
    """
    if value is None:
        raise SchemaError("date value is required, got None")
    if isinstance(value, str):
        value = value.strip()
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"cannot interpret {value!r} as a date") from exc
    if pd.isna(stamp):
        raise SchemaError(f"cannot interpret {value!r} as a date")
    if stamp.tz is not None:
        stamp = stamp.tz_convert(None)
    return stamp.normalize()


def empty_vintage(columns=REQUIRED_COLUMNS) -> pd.DataFrame:
    """An empty vintage with the right columns and dtypes.

    Asking for a vintage from before the first snapshot is normal, not an
    error: at the start of a backtest nothing has been published yet. It has
    to come back typed rather than as a bare ``DataFrame()``, otherwise every
    downstream caller has to special-case the first few weeks.
    """
    frame = pd.DataFrame(
        {
            "geo_value": pd.Series(dtype="object"),
            "time_value": pd.Series(dtype="datetime64[ns]"),
            "value": pd.Series(dtype="float64"),
            "issue": pd.Series(dtype="datetime64[ns]"),
        }
    )
    for extra in columns:
        if extra not in frame.columns:
            frame[extra] = pd.Series(dtype="object")
    return frame[list(columns)]


def _validate_source(source: str) -> str:
    if not isinstance(source, str) or not _SOURCE_RE.match(source):
        raise SchemaError(
            f"source must be a lowercase slug (letters, digits, _, -), got {source!r}"
        )
    return source


def _normalize(frame: pd.DataFrame, issue: pd.Timestamp) -> pd.DataFrame:
    """Type-check and canonicalize a frame destined for ``issue``."""
    if not isinstance(frame, pd.DataFrame):
        raise SchemaError(f"expected a DataFrame, got {type(frame).__name__}")

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"snapshot is missing required column(s): {missing}")
    if frame.empty:
        raise SchemaError("refusing to store an empty snapshot")

    out = frame.copy()
    out["geo_value"] = out["geo_value"].astype("object").map(
        lambda g: g if isinstance(g, str) else str(g)
    )
    if out["geo_value"].map(lambda g: g.strip() == "").any():
        raise SchemaError("geo_value may not be blank")

    out["time_value"] = out["time_value"].map(coerce_date)
    out["issue"] = out["issue"].map(coerce_date)
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype("float64")

    stray = out.loc[out["issue"] != issue, "issue"].unique()
    if len(stray):
        raise SchemaError(
            f"snapshot declared issue {issue.date()} but rows carry "
            f"{[pd.Timestamp(s).date() for s in stray]}"
        )

    # A reference week cannot be published before it has happened. This catches
    # the common mistake of swapping the two date columns, which would quietly
    # invert the whole vintage logic instead of failing.
    early = out["issue"] < out["time_value"]
    if early.any():
        bad = out.loc[early].iloc[0]
        raise SchemaError(
            f"issue {bad['issue'].date()} precedes time_value "
            f"{bad['time_value'].date()}; the two date columns may be swapped"
        )

    dupes = out.duplicated(subset=_KEY, keep=False)
    if dupes.any():
        bad = out.loc[dupes, _KEY].iloc[0].to_dict()
        raise SchemaError(
            f"snapshot has more than one row for the same key: {bad}"
        )

    ordered = list(REQUIRED_COLUMNS) + [
        c for c in out.columns if c not in REQUIRED_COLUMNS
    ]
    return out[ordered].sort_values(_KEY).reset_index(drop=True)


def _same_content(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """True if two snapshot frames hold the same values.

    ``DataFrame.equals`` also compares dtypes, and under pandas 3 a fresh frame
    carries second-precision timestamps while the parquet round trip returns
    millisecond precision. Same instants, different dtype: ``equals`` would
    call an identical re-run a rewrite and refuse it.
    """
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    try:
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
    except AssertionError:
        return False
    return True


class SnapshotStore:
    """Immutable, on-disk store of dated source pulls.

    Layout::

        <root>/<source>/issue=YYYY-MM-DD.parquet

    One file per source per publication date. Each file holds the series as
    that source reported it on that date, which may span many reference weeks
    and revise earlier ones.
    """

    def __init__(self, root) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"SnapshotStore({str(self.root)!r})"

    # -- writing ---------------------------------------------------------

    def path_for(self, source: str, issue) -> Path:
        source = _validate_source(source)
        issue = coerce_date(issue)
        return self.root / source / f"issue={issue.date().isoformat()}.parquet"

    def write_snapshot(self, source: str, issue, frame: pd.DataFrame) -> Path:
        """Store ``frame`` as the snapshot ``source`` published on ``issue``.

        Writing the same content twice is a no-op, so a re-run of an ingestion
        script is safe. Writing *different* content for an issue already on
        disk raises :class:`SnapshotExistsError`: that would rewrite what a
        past model knew, and the honest fix is a new issue date, not an edit.
        """
        source = _validate_source(source)
        issue = coerce_date(issue)
        normalized = _normalize(frame, issue)
        target = self.path_for(source, issue)

        if target.exists():
            existing = self._read_file(target)
            if _same_content(existing, normalized):
                return target
            raise SnapshotExistsError(
                f"snapshot {source} issue={issue.date()} already exists with "
                f"different content; snapshots are immutable, so record the "
                f"correction under a later issue date instead"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_parquet(target, index=False)
        return target

    # -- reading ---------------------------------------------------------

    def sources(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and _SOURCE_RE.match(p.name)
        )

    def available_issues(self, source: str) -> list[pd.Timestamp]:
        """Publication dates on disk for ``source``, oldest first."""
        source = _validate_source(source)
        directory = self.root / source
        if not directory.exists():
            return []
        issues = []
        for path in directory.iterdir():
            match = _ISSUE_FILE_RE.match(path.name)
            if match:
                issues.append(pd.Timestamp(match.group(1)))
        return sorted(issues)

    def _read_file(self, path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        frame["time_value"] = pd.to_datetime(frame["time_value"])
        frame["issue"] = pd.to_datetime(frame["issue"])
        return frame.sort_values(_KEY).reset_index(drop=True)

    def load_vintage(self, source: str, as_of, *, drop_issue: bool = False):
        """Return ``source`` as it was known on ``as_of``.

        For each ``(geo_value, time_value)`` the most recently issued row at or
        before ``as_of`` wins. Reference weeks first published after ``as_of``
        do not appear at all. ``as_of`` is inclusive: a snapshot issued that
        same day is visible.
        """
        source = _validate_source(source)
        as_of = coerce_date(as_of)

        visible = [i for i in self.available_issues(source) if i <= as_of]
        if not visible:
            return empty_vintage()

        frames = [self._read_file(self.path_for(source, i)) for i in visible]
        combined = pd.concat(frames, ignore_index=True)

        # Sort by issue so `keep="last"` resolves each key to its newest
        # revision; this is the whole point of the loader.
        combined = combined.sort_values(_KEY + ["issue"], kind="mergesort")
        vintage = combined.drop_duplicates(subset=_KEY, keep="last")
        vintage = vintage.sort_values(_KEY).reset_index(drop=True)

        if drop_issue:
            vintage = vintage.drop(columns=["issue"])
        return vintage

    def latest_vintage(self, source: str):
        """The finalized series. Fine for descriptive reporting, never for scoring."""
        issues = self.available_issues(source)
        if not issues:
            return empty_vintage()
        return self.load_vintage(source, issues[-1])

    def revision_history(self, source: str, geo_value: str, time_value):
        """Every value ever published for one county-week, oldest first.

        This is the raw material for the backfill analysis: how far the first
        print sits below the settled value, and how long it takes to converge.
        """
        source = _validate_source(source)
        time_value = coerce_date(time_value)
        issues = self.available_issues(source)
        if not issues:
            return empty_vintage()

        frames = [self._read_file(self.path_for(source, i)) for i in issues]
        combined = pd.concat(frames, ignore_index=True)
        mask = (combined["geo_value"] == geo_value) & (
            combined["time_value"] == time_value
        )
        history = combined.loc[mask].sort_values("issue")
        return history.reset_index(drop=True)
