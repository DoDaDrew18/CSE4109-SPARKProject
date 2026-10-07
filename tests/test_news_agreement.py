"""Agreement statistics against hand-computed and textbook values.

The decision "use human labels or LLM labels" rests on these numbers, so the
kappa implementation is checked against values worked out on paper, not
against another library.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.news.agreement import (NULL, bootstrap_ci, build_report, cohen_kappa, compare_field,
                                confusion_matrix, decide, percent_agreement, precision_recall)
from src.news.baseline import separation, spearman


def textbook_pairs():
    """Wikipedia's Cohen's kappa example: 50 items, 20 yes/yes, 5 yes/no,
    10 no/yes, 15 no/no. po = 0.7, pe = 0.5, kappa = 0.4."""
    pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
    return [a for a, _ in pairs], [b for _, b in pairs]


def test_kappa_textbook_example():
    a, b = textbook_pairs()
    assert percent_agreement(a, b) == pytest.approx(0.7)
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_unweighted_and_quadratic_kappa_hand_computed():
    # O: (0,0)=1 (0,1)=1 (1,1)=1 (2,2)=1; rows a = [2,1,1], cols b = [1,2,1]
    # unweighted: po = 3/4, pe = (2*1 + 1*2 + 1*1)/16 = 5/16 -> (12/16-5/16)/(11/16) = 7/11
    # quadratic: w01 = .25, w02 = 1; sum wO = .25; sum wE = 1.25 -> 1 - .25/1.25 = 0.8
    a, b = [0, 0, 1, 2], [0, 1, 1, 2]
    assert cohen_kappa(a, b, [0, 1, 2]) == pytest.approx(7 / 11)
    assert cohen_kappa(a, b, [0, 1, 2], "quadratic") == pytest.approx(0.8)
    # linear: w01 = .5, w02 = 1; sum wO = .5; sum wE = .5 + .5 + .125 + .125 + .25 + .25 = 1.75
    assert cohen_kappa(a, b, [0, 1, 2], "linear") == pytest.approx(1 - 0.5 / 1.75)


def test_quadratic_penalizes_far_misses_more():
    truth = [0, 1, 2, 3, 0, 1, 2, 3]
    near = [1, 1, 2, 3, 0, 1, 2, 2]
    far = [3, 1, 2, 3, 0, 1, 2, 0]
    levels = [0, 1, 2, 3]
    assert cohen_kappa(near, truth, levels) == pytest.approx(cohen_kappa(far, truth, levels))
    assert cohen_kappa(near, truth, levels, "quadratic") > cohen_kappa(far, truth, levels, "quadratic")


def test_kappa_edge_cases():
    assert cohen_kappa(["y", "n", "y"], ["y", "n", "y"]) == pytest.approx(1.0)
    assert math.isnan(cohen_kappa(["y", "y"], ["y", "y"]))   # chance agreement is total
    assert math.isnan(cohen_kappa([], []))
    # systematic disagreement is below zero
    assert cohen_kappa(["y", "n", "y", "n"], ["n", "y", "n", "y"]) == pytest.approx(-1.0)


def test_confusion_matrix_orientation():
    cm = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert cm.loc["a", "b"] == 1 and cm.loc["a", "a"] == 1 and cm.loc["b", "b"] == 1
    assert cm.to_numpy().sum() == 3


def test_precision_recall_hand_computed():
    pred = ["29189", "29189", "29510", NULL]
    truth = ["29189", "29510", "29510", "29189"]
    pr = precision_recall(pred, truth).set_index("class")
    assert NULL not in pr.index
    assert pr.loc["29189", "precision"] == pytest.approx(0.5)
    assert pr.loc["29189", "recall"] == pytest.approx(0.5)
    assert pr.loc["29510", "precision"] == pytest.approx(1.0)
    assert pr.loc["29510", "recall"] == pytest.approx(0.5)


def test_bootstrap_ci_brackets_estimate_and_is_reproducible():
    a, b = textbook_pairs()
    lo, hi = bootstrap_ci(cohen_kappa, a, b, n_boot=500, seed=1)
    assert lo < 0.4 < hi and -1 <= lo and hi <= 1
    assert (lo, hi) == bootstrap_ci(cohen_kappa, a, b, n_boot=500, seed=1)


def frame(rows):
    return pd.DataFrame(rows, columns=["article_id", "labeler", "relevant", "fips", "illness",
                                       "concern", "event_week"])


def test_compare_field_scopes_county_to_relevant_and_counts_nulls():
    ref = frame([["a", "x", True, "29189", "flu", 2, None],
                 ["b", "x", False, None, "none", 0, None],
                 ["c", "x", True, "29510", "flu", 3, None]])
    llm = frame([["a", "llm", True, "29189", "flu", 2, None],
                 ["b", "llm", False, None, "none", 0, None],
                 ["c", "llm", True, None, "flu", None, None]])   # grounding nulled fips + concern
    fips = compare_field(llm, ref, "fips", n_boot=50)
    assert fips["n"] == 2 and fips["pct_agree"] == 0.5 and fips["coverage"] == 0.5
    conc = compare_field(llm, ref, "concern", n_boot=50)
    assert conc["n"] == 3 and conc["pct_agree"] == pytest.approx(2 / 3)


def test_decision_rule():
    human = {"field": "fips", "n": 30, "kappa": 0.8}
    assert decide({"field": "fips", "n": 60, "kappa": 0.7}, human).startswith("use human labels")
    assert decide({"field": "fips", "n": 60, "kappa": 0.85}, human).startswith("LLM labels acceptable")
    assert decide({"field": "fips", "n": 60, "kappa": 0.85}, None).startswith("undecided")
    # everyone said "flu": kappa undefined, fall back to percent agreement
    flat = {"field": "illness", "n": 30, "kappa": float("nan"), "pct_agree": 1.0}
    assert "κ undefined" in decide({"field": "illness", "n": 30, "kappa": 0.0, "pct_agree": 0.9}, flat)
    assert decide({"field": "illness", "n": 30, "kappa": 0.0, "pct_agree": 0.9}, flat).startswith("use human")
    hc = {"field": "concern", "n": 30, "kappa": 0.2, "kappa_quadratic": 0.9}
    assert decide({"field": "concern", "n": 30, "kappa": 0.5, "kappa_quadratic": 0.8}, hc) \
        .startswith("use human labels")    # quadratic, not unweighted, decides concern


def test_build_report_end_to_end():
    rng = np.random.default_rng(0)
    rows, llm = [], []
    for i in range(30):
        rel = bool(i % 3)
        f = "29189" if i % 2 else "17031"
        c = int(rng.integers(1, 4)) if rel else 0
        for lab in ("human_a", "human_b"):
            cc = c if (lab == "human_a" or i % 7) else max(c - 1, 0)
            rows.append([f"a{i}", lab, rel, f if rel else None, "flu" if rel else "none", cc, None])
        for run in (1, 2):
            llm.append({"article_id": f"a{i}", "run": run, "model": "m", "prompt_hash": "h",
                        "relevant": rel, "fips": f if rel and i % 5 else None,
                        "illness": "flu" if rel else "none", "concern": c, "event_week": None})
    humans = frame(rows)
    vader = pd.DataFrame({"article_id": [f"a{i}" for i in range(30)],
                          "vader_compound": rng.uniform(-1, 1, 30)})
    report, decisions = build_report(humans, pd.DataFrame(llm), vader, n_boot=100)
    assert set(decisions) == {"relevant", "fips", "illness", "concern"}
    for heading in ("## Human vs human", "## LLM vs human consensus", "## Decision",
                    "## Confusion matrices", "## LLM self-consistency", "## Baseline: VADER"):
        assert heading in report
    assert decisions["fips"].startswith("use human labels")   # LLM dropped 1 in 5 counties


def test_spearman_and_vader_separation():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(np.corrcoef([1, 2.5, 2.5, 4], [1, 2, 3, 4])[0, 1])
    vader = pd.DataFrame({"article_id": list("abcd"), "vader_compound": [0.5, 0.1, -0.3, -0.8]})
    cons = pd.DataFrame({"article_id": list("abcd"), "concern": [0, 1, 2, 3]})
    sep = separation(vader, cons)
    assert sep["n"] == 4 and sep["spearman"] == pytest.approx(-1.0)
    assert list(sep["by_level"]["concern"]) == [0, 1, 2, 3]


def test_vader_scores_offline():
    from src.news.baseline import score_texts
    s = score_texts({"bad": "The deadly outbreak is a terrible crisis for hospitals.",
                     "good": "Flu cases fell and the health department is pleased."}).set_index("article_id")
    assert s.loc["bad", "vader_compound"] < 0 < s.loc["good", "vader_compound"]
