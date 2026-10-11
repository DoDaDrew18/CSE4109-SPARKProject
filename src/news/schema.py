"""The shape of one article label, shared by humans, the LLM and every consumer.

Labels arrive from two very different producers -- a CSV that people type by
hand and JSON that a model emits -- and both are wrong in predictable ways
("Y", " 17031", "Flu, RSV", "2.0"). Rather than let each consumer cope, every
row passes through ``validate_label`` once, which either normalizes it to the
canonical form below or says exactly what is wrong. No pydantic: the rules are
few and the error messages matter more than the type machinery.

Canonical row (see docs/LABELING_GUIDE.md for meaning):

    article_id   str
    labeler      str     human_a | human_b | adjudicated | llm
    relevant     bool | None
    fips         str | None    one of STUDY_FIPS, "other", "unclear"
    illness      str | None    "|"-joined sorted subset of ILLNESSES, or "none"
    concern      int | None    0..3
    event_week   Timestamp | None   Saturday ending the MMWR week described
    quote        str | None    (humans; the LLM has one quote per field)

``None`` means "no label", which is different from a negative label: a
grounding failure nulls a field, it never turns it into ``n`` or ``0``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.study import STUDY_FIPS, week_end_of

__all__ = ["ILLNESSES", "FIPS_CHOICES", "LABEL_FIELDS", "HUMAN_COLUMNS",
           "HUMAN_LABELERS", "LabelFormatError", "validate_label",
           "load_human_labels", "human_consensus"]

ILLNESSES: tuple[str, ...] = ("covid", "flu", "rsv", "respiratory_general")
FIPS_CHOICES: tuple[str, ...] = STUDY_FIPS + ("other", "unclear")
LABEL_FIELDS: tuple[str, ...] = ("relevant", "fips", "illness", "concern", "event_week")

HUMAN_COLUMNS: tuple[str, ...] = ("article_id", "labeler", "relevant", "fips", "illness",
                                  "concern", "event_week", "quote", "notes", "is_example")
HUMAN_LABELERS: tuple[str, ...] = ("human_a", "human_b", "adjudicated")

_TRUE = {"y", "yes", "true", "1", "t"}
_FALSE = {"n", "no", "false", "0", "f"}


class LabelFormatError(ValueError):
    """Raised when a labels file has rows that cannot be normalized."""


def _blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))       # NaN, NaT, pd.NA
    except (TypeError, ValueError):
        return False


def _relevant(value, errors):
    if _blank(value):
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    errors.append(f"relevant={value!r} is not y/n")
    return None


def _fips(value, errors):
    if _blank(value):
        return None
    s = str(value).strip().lower()
    if s.endswith(".0"):            # a FIPS read as a float from a spreadsheet
        s = s[:-2]
    if s.isdigit():
        s = s.zfill(5)
    if s not in FIPS_CHOICES:
        errors.append(f"fips={value!r} is not a study county, 'other' or 'unclear'")
        return None
    return s


def _illness(value, errors):
    if _blank(value):
        return None
    if isinstance(value, (list, tuple, set)):
        parts = [str(p) for p in value]
    else:
        parts = str(value).replace(",", "|").replace(";", "|").split("|")
    parts = {p.strip().lower() for p in parts if p.strip()}
    if not parts:
        return None
    if parts == {"none"}:
        return "none"
    if "none" in parts:
        errors.append("illness 'none' cannot be combined with an illness")
        return None
    bad = parts - set(ILLNESSES)
    if bad:
        errors.append(f"illness has unknown values {sorted(bad)}")
        return None
    return "|".join(sorted(parts))


def _concern(value, errors):
    if _blank(value):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        errors.append(f"concern={value!r} is not a number")
        return None
    if f != int(f) or not 0 <= f <= 3:
        errors.append(f"concern={value!r} is not an integer 0-3")
        return None
    return int(f)


def _event_week(value, errors):
    if _blank(value):
        return None
    try:
        return week_end_of(value)
    except (ValueError, TypeError):
        errors.append(f"event_week={value!r} is not a date")
        return None


def validate_label(row: dict) -> tuple[dict, list[str]]:
    """Normalize one label row; return (clean_row, errors).

    Cross-field rules enforce the guide's definition of ``relevant``: a
    relevant article must name a study county and an illness, and a
    not-relevant article cannot carry concern above 0 -- otherwise the weekly
    features would count articles the labeler themselves said don't apply.
    Fields that fail are nulled; other fields are kept.
    """
    errors: list[str] = []
    clean = dict(row)
    clean["article_id"] = "" if _blank(row.get("article_id")) else str(row["article_id"]).strip()
    if not clean["article_id"]:
        errors.append("article_id is blank")
    clean["relevant"] = _relevant(row.get("relevant"), errors)
    clean["fips"] = _fips(row.get("fips"), errors)
    clean["illness"] = _illness(row.get("illness"), errors)
    clean["concern"] = _concern(row.get("concern"), errors)
    clean["event_week"] = _event_week(row.get("event_week"), errors)

    if clean["relevant"] is True:
        if clean["fips"] in ("other", "unclear"):
            errors.append(f"relevant=y but fips={clean['fips']!r}; relevant needs a study county")
        if clean["illness"] == "none":
            errors.append("relevant=y but illness=none")
    elif clean["relevant"] is False:
        if clean["concern"] not in (None, 0):
            errors.append(f"relevant=n but concern={clean['concern']}; must be 0")
        if clean["illness"] not in (None, "none"):
            errors.append(f"relevant=n but illness={clean['illness']!r}; must be none")
    return clean, errors


def load_human_labels(path) -> pd.DataFrame:
    """Read and validate the human labels CSV; example rows are dropped.

    Raises ``LabelFormatError`` listing *every* bad row (with its line number
    in the file) so a labeler fixes the sheet in one pass instead of one error
    at a time.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in HUMAN_COLUMNS if c not in raw.columns and c not in ("notes", "is_example")]
    if missing:
        raise LabelFormatError(f"{path}: missing columns {missing}")
    if "is_example" in raw.columns:
        raw = raw[~raw["is_example"].str.strip().str.lower().isin(_TRUE)]

    rows, problems = [], []
    for line, rec in zip(raw.index + 2, raw.to_dict("records")):   # +2: header, 1-based
        clean, errors = validate_label(rec)
        labeler = str(rec.get("labeler", "")).strip()
        if labeler not in HUMAN_LABELERS:
            errors.append(f"labeler={labeler!r} not in {HUMAN_LABELERS}")
        clean["labeler"] = labeler
        if errors:
            problems.append(f"line {line} ({clean['article_id'] or '?'}): " + "; ".join(errors))
        rows.append(clean)
    if problems:
        raise LabelFormatError(f"{path}: {len(problems)} bad rows\n  " + "\n  ".join(problems))

    out = pd.DataFrame(rows, columns=[c for c in HUMAN_COLUMNS if c != "is_example"])
    dupes = out.duplicated(["article_id", "labeler"], keep=False)
    if dupes.any():
        raise LabelFormatError(f"{path}: duplicate (article_id, labeler): "
                               f"{out.loc[dupes, 'article_id'].unique().tolist()}")
    out["concern"] = out["concern"].astype("Int64")
    out["event_week"] = pd.to_datetime(out["event_week"])
    return out


def human_consensus(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per article: the human answer the LLM is scored against.

    Precedence: an ``adjudicated`` row wins outright. Otherwise a field takes
    the value the humans agree on, and becomes ``None`` where they disagree --
    an unresolved disagreement is not ground truth, so it is excluded from
    scoring rather than silently resolved by picking a labeler.
    """
    humans = labels[labels["labeler"].isin(HUMAN_LABELERS)]
    out = []
    for article_id, group in humans.groupby("article_id", sort=True):
        adj = group[group["labeler"] == "adjudicated"]
        if len(adj):
            row = adj.iloc[0][list(LABEL_FIELDS)].to_dict()
        else:
            row = {}
            for field in LABEL_FIELDS:
                values = {None if _blank(v) else v for v in group[field]}
                row[field] = values.pop() if len(values) == 1 else None
        row["article_id"] = article_id
        row["n_labelers"] = int(group["labeler"].nunique())
        out.append(row)
    return pd.DataFrame(out, columns=["article_id", *LABEL_FIELDS, "n_labelers"])
