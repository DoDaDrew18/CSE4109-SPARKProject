"""Weekly news features: the as_of availability rule and the output contract.

The leak to guard against is the same one the snapshot store guards against
for NSSP: a feature for date D must not use an article published after D,
even if the article describes an earlier week (``event_week``).
"""

import pandas as pd
import pytest

from src.news.features import FEATURE_COLUMNS, weekly_features

HEADER = "article_id,labeler,relevant,fips,illness,concern,event_week,quote,notes,is_example\n"


@pytest.fixture
def corpus(tmp_path):
    """Four articles in St. Louis County / Cook County around Jan 2026.

    Week ending Sat 2026-01-10: a1 (Mon 01-05, concern 2), a2 (Fri 01-09, concern 3)
    Week ending Sat 2026-01-17: a3 (Sun 01-11, concern 1; describes the week before)
    a4 is not relevant; a5 is Cook County.
    """
    labels = tmp_path / "human_labels.csv"
    labels.write_text(HEADER +
                      "a1,human_a,y,29189,flu,2,,q,,\n"
                      "a2,human_a,y,29189,flu|rsv,3,,q,,\n"
                      "a3,human_a,y,29189,flu,1,2026-01-10,q,,\n"
                      "a4,human_a,n,other,none,0,,,,\n"
                      "a5,human_a,y,17031,covid,2,,q,,\n"
                      "EX,human_a,y,29189,flu,3,,q,,1\n")
    manifest = tmp_path / "articles.parquet"
    pd.DataFrame({
        "article_id": ["a1", "a2", "a3", "a4", "a5"],
        "url": ["u"] * 5, "outlet": ["o"] * 5, "metro": ["stl"] * 4 + ["chi"],
        "fips_guess": [None] * 5, "title": ["t"] * 5,
        "published": pd.to_datetime(["2026-01-05", "2026-01-09 18:30", "2026-01-11",
                                     "2026-01-06", "2026-01-07"], format="mixed"),
    }).to_parquet(manifest)
    return labels, manifest


def test_as_of_excludes_articles_published_later(corpus):
    labels, manifest = corpus
    f = weekly_features("2026-01-09", labels_path=labels, manifest_path=manifest)
    stl = f[f["fips"] == "29189"]
    assert list(stl["week_end"]) == [pd.Timestamp("2026-01-10")]
    assert stl["news_n"].iloc[0] == 2 and stl["news_concern"].iloc[0] == pytest.approx(2.5)

    earlier = weekly_features("2026-01-08", labels_path=labels, manifest_path=manifest)
    assert earlier.loc[earlier["fips"] == "29189", "news_n"].tolist() == [1]   # a2 not yet published


def test_publish_week_not_event_week(corpus):
    """a3 describes the week ending 01-10 but was published 01-11: it belongs
    to week 01-17 and is invisible to as_of 01-10."""
    labels, manifest = corpus
    f = weekly_features("2026-01-31", fips="29189", labels_path=labels, manifest_path=manifest)
    assert list(f["week_end"]) == [pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-17")]
    assert list(f["news_n"]) == [2, 1]
    assert weekly_features("2026-01-10", fips="29189", labels_path=labels,
                           manifest_path=manifest)["news_n"].sum() == 2


def test_relevant_only_examples_dropped_and_fips_filter(corpus):
    labels, manifest = corpus
    f = weekly_features("2026-12-31", labels_path=labels, manifest_path=manifest)
    assert f["news_n"].sum() == 4                       # a4 irrelevant, EX example row
    assert set(f["fips"]) == {"29189", "17031"}
    only = weekly_features("2026-12-31", fips=["17031"], labels_path=labels, manifest_path=manifest)
    assert list(only["fips"]) == ["17031"]


def test_output_contract_and_empty_frame(tmp_path, corpus):
    labels, manifest = corpus
    f = weekly_features("2026-12-31", labels_path=labels, manifest_path=manifest)
    assert tuple(f.columns) == FEATURE_COLUMNS
    assert str(f["week_end"].dtype).startswith("datetime64") and f["news_n"].dtype == "int64"
    assert (f["week_end"].dt.dayofweek == 5).all()      # Saturdays

    empty = weekly_features("2026-12-31", labels_path=tmp_path / "nope.csv")
    assert tuple(empty.columns) == FEATURE_COLUMNS and empty.empty
    assert empty["news_n"].dtype == "int64" and empty["news_concern"].dtype == "float64"
    assert weekly_features("2025-01-01", labels_path=labels, manifest_path=manifest).empty


def test_llm_source_uses_grounded_first_run(tmp_path):
    path = tmp_path / "llm.parquet"
    pd.DataFrame({
        "article_id": ["a1", "a1", "a2"], "run": [1, 2, 1],
        "published": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-06"]),
        "relevant": pd.array([True, True, True], dtype="boolean"),
        "fips": ["29189", "29189", None],                 # a2's county failed grounding
        "concern": pd.array([2, 3, 1], dtype="Int64"),
    }).to_parquet(path)
    f = weekly_features("2026-01-31", labels_path=path, source="llm")
    assert f["news_n"].tolist() == [1] and f["news_concern"].tolist() == [2.0]
