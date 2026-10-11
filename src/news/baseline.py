"""Word-list sentiment baseline (VADER): is generic "negativity" informative?

docs/OVERVIEW.md argues for a 0-3 *concern* scale instead of open-ended
sentiment, because outbreak news reads negative almost every time. This module
tests that claim instead of assuming it: score every article with VADER's
compound sentiment (-1 very negative .. +1 very positive) and measure how well
it separates the human concern levels. If VADER already tracks concern, the
LLM is adding cost without adding signal; if it doesn't, that is the
justification for the labeling effort.

Usage:  python -m src.news.baseline   # -> data/labels/vader_scores.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["score_texts", "spearman", "separation"]


def score_texts(texts: dict[str, str]) -> pd.DataFrame:
    """VADER compound score per article (title + body)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    rows = [{"article_id": aid, "vader_compound": analyzer.polarity_scores(text)["compound"]}
            for aid, text in texts.items()]
    return pd.DataFrame(rows, columns=["article_id", "vader_compound"])


def spearman(x, y) -> float:
    """Spearman rank correlation (average ranks for ties), numpy/pandas only."""
    x, y = pd.Series(list(x), dtype=float), pd.Series(list(y), dtype=float)
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return float("nan")
    rx, ry = x[ok].rank(), y[ok].rank()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def separation(vader: pd.DataFrame, consensus: pd.DataFrame) -> dict:
    """Spearman of compound vs human concern, plus compound by concern level.

    Expect a *negative* rho if sentiment tracks concern (worse news, lower
    compound). A rho near zero means sentiment says little about severity.
    """
    m = vader.merge(consensus[["article_id", "concern"]], on="article_id").dropna(subset=["concern"])
    m["concern"] = m["concern"].astype(int)
    by_level = (m.groupby("concern")["vader_compound"].agg(["count", "mean", "std"])
                .reset_index().rename(columns={"mean": "mean_compound", "std": "sd"}))
    return {"n": len(m), "spearman": spearman(m["vader_compound"], m["concern"]),
            "by_level": by_level}


def main(argv=None):
    from src.news.ground import load_article_texts
    from src.news.schema import human_consensus, load_human_labels

    p = argparse.ArgumentParser(description="VADER sentiment baseline.")
    p.add_argument("--manifest", default="raw/news/articles.parquet")
    p.add_argument("--text-dir", default="raw/news/text")
    p.add_argument("--humans", default="labels/human_labels.csv")
    p.add_argument("--out", default="data/labels/vader_scores.parquet")
    args = p.parse_args(argv)

    ids = pd.read_parquet(args.manifest)["article_id"].astype(str)
    texts = {k: v for k, v in load_article_texts(ids, args.text_dir, args.manifest).items()
             if v.strip()}
    scores = score_texts(texts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.out, index=False)
    print(f"scored {len(scores)} articles -> {args.out}")
    if Path(args.humans).exists():
        sep = separation(scores, human_consensus(load_human_labels(args.humans)))
        print(f"Spearman(compound, human concern) = {sep['spearman']:.3f} (n={sep['n']})")
        print(sep["by_level"].to_string(index=False))


if __name__ == "__main__":
    main()
