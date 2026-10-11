"""Weekly news features per county, as they could have been known on ``as_of``.

The contract other stages build on:

    weekly_features(as_of, fips=None, source="human")
        -> DataFrame[fips: str, week_end: datetime64, news_n: int64, news_concern: float64]

* ``news_n``       number of relevant articles about the county published that week
* ``news_concern`` mean concern (0-3) over those articles with a concern label

Availability is by **publish date**, never ``event_week``: an article
describing last week's surge could only be read once it was published, so an
``as_of`` loader must not see it earlier. Likewise ``week_end`` is the MMWR
week of publication (Saturday, ``src.study.week_end_of``); the news feature
for a week is "what the news said that week", which is what a nowcaster on
that date had.

Rows are sparse: a county-week with no relevant articles has no row. Join and
fill ``news_n = 0``; leave ``news_concern`` NaN there (no articles is not the
same as calm articles). With 60-200 articles this is a signal around peaks,
not a weekly series (docs/OVERVIEW.md).

Human labels are the default because the agreement report decides whether
LLM labels are good enough (src/news/agreement.py); pass ``source="llm"`` to
use grounded LLM labels once they have earned it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.study import STUDY_FIPS, week_end_of

__all__ = ["FEATURE_COLUMNS", "HUMAN_LABELS", "LLM_LABELS", "MANIFEST", "weekly_features",
           "article_table"]

FEATURE_COLUMNS = ("fips", "week_end", "news_n", "news_concern")
HUMAN_LABELS = Path("labels/human_labels.csv")
LLM_LABELS = Path("data/labels/llm_labels_grounded.parquet")
MANIFEST = Path("raw/news/articles.parquet")
LOCAL_TZ = "America/Chicago"   # all six metros are Central or Eastern


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"fips": pd.Series(dtype="object"),
                         "week_end": pd.Series(dtype="datetime64[ns]"),
                         "news_n": pd.Series(dtype="int64"),
                         "news_concern": pd.Series(dtype="float64")})


def _publish_dates(values) -> pd.Series:
    """Publish timestamps -> local calendar dates (naive midnight)."""
    ts = pd.to_datetime(pd.Series(values), utc=False, errors="coerce", format="mixed")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return ts.dt.normalize()


def article_table(source: str = "human", labels_path=None, manifest_path=MANIFEST) -> pd.DataFrame:
    """One row per labeled article: article_id, published, relevant, fips, concern.

    Human labels are collapsed to the consensus (``schema.human_consensus``);
    LLM labels use the first run, after grounding has nulled unsupported fields.
    Publish dates come from the label file if it has them, else the manifest.
    """
    from src.news.schema import human_consensus, load_human_labels

    if source not in ("human", "llm"):
        raise ValueError(f"source must be 'human' or 'llm', not {source!r}")
    path = Path(labels_path) if labels_path is not None else (HUMAN_LABELS if source == "human" else LLM_LABELS)
    if not path.exists():
        return pd.DataFrame(columns=["article_id", "published", "relevant", "fips", "concern"])

    if source == "human":
        labels = human_consensus(load_human_labels(path))
    else:
        labels = pd.read_parquet(path)
        if "run" in labels and len(labels):
            labels = labels[labels["run"] == labels["run"].min()]
        labels = labels.drop_duplicates("article_id")
    labels = labels.assign(article_id=labels["article_id"].astype(str))

    if "published" not in labels or labels["published"].isna().any():
        if Path(manifest_path).exists():
            man = pd.read_parquet(manifest_path)[["article_id", "published"]]
            man = man.assign(article_id=man["article_id"].astype(str))
            labels = labels.drop(columns=["published"], errors="ignore").merge(man, on="article_id", how="left")
        elif "published" not in labels:
            labels = labels.assign(published=pd.NaT)
    labels = labels.assign(published=_publish_dates(labels["published"].to_numpy()).to_numpy())
    return labels[["article_id", "published", "relevant", "fips", "concern"]].reset_index(drop=True)


def weekly_features(as_of, fips=None, labels_path=None, source: str = "human",
                    manifest_path=MANIFEST) -> pd.DataFrame:
    """County-week news features using only articles published on or before ``as_of``.

    ``fips`` may be one FIPS string or a list; None means all study counties.
    Returns an empty, correctly typed frame when there are no labels yet, so
    downstream joins work before the first article is labeled.
    """
    arts = article_table(source, labels_path, manifest_path)
    if arts.empty:
        return _empty()
    cutoff = pd.Timestamp(as_of).normalize()
    keep = (arts["relevant"].map(lambda v: isinstance(v, (bool, np.bool_)) and bool(v))
            & arts["fips"].isin(STUDY_FIPS)
            & arts["published"].notna()
            & (arts["published"] <= cutoff))
    if fips is not None:
        keep &= arts["fips"].isin([fips] if isinstance(fips, str) else list(fips))
    arts = arts[keep]
    if arts.empty:
        return _empty()

    arts = arts.assign(week_end=arts["published"].map(week_end_of),
                       concern=pd.to_numeric(arts["concern"], errors="coerce"))
    out = (arts.groupby(["fips", "week_end"], as_index=False)
           .agg(news_n=("article_id", "size"), news_concern=("concern", "mean")))
    return out.astype({"fips": "object", "week_end": "datetime64[ns]", "news_n": "int64",
                       "news_concern": "float64"})[list(FEATURE_COLUMNS)] \
        .sort_values(["fips", "week_end"]).reset_index(drop=True)
