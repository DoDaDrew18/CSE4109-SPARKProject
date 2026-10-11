"""How far can we trust LLM labels? Agreement with humans, field by field.

The rule we committed to in docs/OVERVIEW.md (Layer 4): two of us label the
same ~30 articles, which measures how well *humans* agree. If the LLM agrees
with the human consensus **less** than humans agree with each other on a
field, we use human labels for that field and say so in the report. Humans
are not ground truth -- their own disagreement is the ceiling the LLM is held
to, which is why both numbers sit side by side here.

Usage, from the repo root:

    python -m src.news.agreement      # -> data/labels/agreement_report.md

Statistics are implemented with numpy so every number in the report can be
traced to a few lines here (no sklearn):

* Cohen's kappa, unweighted and weighted (linear/quadratic). Quadratic
  weights are used for ``concern`` because it is ordinal: calling a surge
  "rising" is a smaller miss than calling it "none".
* Percent agreement, confusion matrices, per-class precision/recall.
* Percentile bootstrap 95% CIs, resampling articles. With 30 double-labeled
  articles the intervals are wide; reporting them keeps us from
  over-reading a 0.05 gap in kappa.

Scoring conventions:

* A field the LLM left null (or that grounding nulled) counts as its own
  category ``∅`` -- a disagreement, never silently dropped. The exception is
  weighted kappa, which needs numbers; there nulls are excluded and the
  coverage column shows how many were.
* ``fips`` and ``illness`` are scored only on articles where at least one
  side says relevant: a county label on an irrelevant article is moot. The
  same scope applies to the human-human and LLM-human comparisons.
* Human disagreements without adjudication are excluded from the consensus
  (``schema.human_consensus``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.news.schema import ILLNESSES, load_human_labels, human_consensus

__all__ = ["NULL", "FIELDS", "confusion_matrix", "cohen_kappa", "percent_agreement",
           "bootstrap_ci", "precision_recall", "paired_values", "compare_field",
           "decide", "build_report"]

NULL = "∅"
FIELDS = ("relevant", "fips", "illness", "concern")
CONCERN_LEVELS = (0, 1, 2, 3)


# ---------------------------------------------------------------- statistics

def confusion_matrix(a, b, labels=None) -> pd.DataFrame:
    """Counts with rows = ``a`` (rater), columns = ``b`` (reference)."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("a and b must be the same length")
    labels = list(labels) if labels is not None else sorted(set(a) | set(b), key=str)
    index = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for x, y in zip(a, b):
        m[index[x], index[y]] += 1
    return pd.DataFrame(m, index=labels, columns=labels)


def cohen_kappa(a, b, labels=None, weights=None) -> float:
    """Cohen's kappa; ``weights`` None (unweighted), "linear" or "quadratic".

    kappa = 1 - sum(W * O) / sum(W * E), with O the observed proportions, E
    the proportions expected if the two raters labeled independently with
    their own marginals, and W the disagreement weights. Unweighted kappa is
    W = 1 off the diagonal, which reduces to the familiar (po - pe)/(1 - pe).
    For weighted kappa ``labels`` must be in their natural order.

    Returns NaN when chance agreement is total (both raters used a single
    category): kappa is undefined there, and 1.0 would overstate it.
    """
    m = confusion_matrix(a, b, labels).to_numpy(dtype=float)
    n = m.sum()
    if n == 0:
        return float("nan")
    k = m.shape[0]
    observed = m / n
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    elif weights == "quadratic":
        w = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    denom = (w * expected).sum()
    if denom == 0:
        return float("nan")
    return float(1 - (w * observed).sum() / denom)


def percent_agreement(a, b) -> float:
    a, b = list(a), list(b)
    return float(np.mean([x == y for x, y in zip(a, b)])) if a else float("nan")


def bootstrap_ci(stat, a, b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of ``stat(a, b)`` resampling paired items with replacement.

    Resamples where the statistic is undefined (NaN) are dropped; if most are,
    the CI is reported as NaN rather than built from a biased few.
    """
    a, b = np.asarray(list(a), dtype=object), np.asarray(list(b), dtype=object)
    if len(a) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        stats.append(stat(a[idx], b[idx]))
    stats = np.asarray(stats, dtype=float)
    stats = stats[~np.isnan(stats)]
    if len(stats) < n_boot / 2:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def precision_recall(pred, truth, labels=None) -> pd.DataFrame:
    """Per-class precision/recall/F1 of ``pred`` against ``truth``.

    ``support`` is the number of true items in the class; the ``∅`` class
    (null predictions) is excluded from the rows, since "precision of
    abstaining" means nothing, but its effect shows up as lost recall.
    """
    pred, truth = list(pred), list(truth)
    labels = labels if labels is not None else sorted((set(truth) | set(pred)) - {NULL}, key=str)
    rows = []
    for lab in labels:
        tp = sum(p == lab and t == lab for p, t in zip(pred, truth))
        n_pred = sum(p == lab for p in pred)
        n_true = sum(t == lab for t in truth)
        prec = tp / n_pred if n_pred else float("nan")
        rec = tp / n_true if n_true else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if n_pred and n_true and (prec + rec) else float("nan")
        rows.append({"class": lab, "precision": prec, "recall": rec, "f1": f1,
                     "support": n_true, "predicted": n_pred})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pairing

def _as_category(value):
    if value is None:
        return NULL
    try:
        if pd.isna(value):
            return NULL
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "y" if value else "n"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value)
    return str(value)


def paired_values(rater: pd.DataFrame, ref: pd.DataFrame, field: str):
    """Aligned (rater, reference) category lists for ``field``.

    Both frames are one-row-per-article. The reference must be non-null; for
    fips/illness at least one side must say relevant.
    """
    m = rater.merge(ref, on="article_id", suffixes=("_r", "_ref"))
    m = m[m[f"{field}_ref"].map(_as_category) != NULL]
    if field in ("fips", "illness"):
        rel = (m["relevant_r"].map(_as_category) == "y") | (m["relevant_ref"].map(_as_category) == "y")
        m = m[rel]
    return [_as_category(v) for v in m[f"{field}_r"]], [_as_category(v) for v in m[f"{field}_ref"]]


def compare_field(rater: pd.DataFrame, ref: pd.DataFrame, field: str,
                  n_boot: int = 2000, seed: int = 0) -> dict:
    """Agreement summary for one field; concern also gets quadratic kappa."""
    a, b = paired_values(rater, ref, field)
    labels = None
    if field == "concern":
        labels = [*CONCERN_LEVELS, NULL]
    out = {"field": field, "n": len(a),
           "coverage": float(np.mean([x != NULL for x in a])) if a else float("nan"),
           "pct_agree": percent_agreement(a, b),
           "kappa": cohen_kappa(a, b, labels) if a else float("nan")}
    out["kappa_lo"], out["kappa_hi"] = bootstrap_ci(lambda x, y: cohen_kappa(x, y, labels),
                                                    a, b, n_boot, seed)
    if field == "concern":
        keep = [(x, y) for x, y in zip(a, b) if x != NULL and y != NULL]
        aw, bw = [x for x, _ in keep], [y for _, y in keep]
        qk = lambda x, y: cohen_kappa(x, y, CONCERN_LEVELS, "quadratic")   # noqa: E731
        out["kappa_quadratic"] = qk(aw, bw) if keep else float("nan")
        out["kq_lo"], out["kq_hi"] = bootstrap_ci(qk, aw, bw, n_boot, seed)
    return out


def headline_kappa(summary: dict) -> float:
    """The number the decision rule uses: quadratic for concern, else unweighted."""
    return summary["kappa_quadratic"] if summary["field"] == "concern" else summary["kappa"]


def decide(llm: dict, human: dict | None) -> str:
    """The OVERVIEW decision rule for one field.

    Kappa is undefined when a rater uses a single category (e.g. every
    relevant article is flu); there we fall back to percent agreement and say
    so, rather than leave the field undecided.
    """
    if human is None or human["n"] == 0:
        return "undecided: no double-labeled articles for this field yet"
    lk, hk = headline_kappa(llm), headline_kappa(human)
    if np.isnan(lk) or np.isnan(hk):
        la, ha = llm.get("pct_agree", float("nan")), human.get("pct_agree", float("nan"))
        if np.isnan(la) or np.isnan(ha):
            return "undecided: agreement undefined"
        if la < ha:
            return f"use human labels (κ undefined; LLM agreement {la:.0%} < human-human {ha:.0%})"
        return f"LLM labels acceptable (κ undefined; LLM agreement {la:.0%} ≥ human-human {ha:.0%})"
    if lk < hk:
        return f"use human labels (LLM κ {lk:.2f} < human-human κ {hk:.2f})"
    return f"LLM labels acceptable (LLM κ {lk:.2f} ≥ human-human κ {hk:.2f})"


# ---------------------------------------------------------------- report

def _one_per_article(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("article_id", keep="first").reset_index(drop=True)


def _fmt(x, digits=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _kappa_cell(s: dict) -> str:
    k = f"{_fmt(s['kappa'])} [{_fmt(s['kappa_lo'])}, {_fmt(s['kappa_hi'])}]"
    if s["field"] == "concern":
        k += f"; quadratic {_fmt(s['kappa_quadratic'])} [{_fmt(s['kq_lo'])}, {_fmt(s['kq_hi'])}]"
    return k


def _table(summaries) -> list[str]:
    lines = ["| Field | n | % agree | κ [95% CI] | coverage |", "| --- | --- | --- | --- | --- |"]
    for s in summaries:
        lines.append(f"| {s['field']} | {s['n']} | {_fmt(100 * s['pct_agree'], 0)}% | "
                     f"{_kappa_cell(s)} | {_fmt(100 * s['coverage'], 0)}% |")
    return lines


def _md_frame(df: pd.DataFrame, index=True) -> list[str]:
    df = df.reset_index() if index else df
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) if isinstance(v, float) else str(v) for v in row) + " |")
    return lines


def _illness_binary(rater, ref) -> pd.DataFrame:
    a, b = paired_values(rater, ref, "illness")
    rows = []
    for ill in ILLNESSES:
        pa = ["y" if x != NULL and ill in str(x).split("|") else "n" for x in a]
        pb = ["y" if ill in str(y).split("|") else "n" for y in b]
        pr = precision_recall(pa, pb, ["y"]).iloc[0]
        rows.append({"illness": ill, "precision": pr["precision"], "recall": pr["recall"],
                     "support": int(pr["support"]),
                     "kappa": cohen_kappa(pa, pb, ["n", "y"]) if pa else float("nan")})
    return pd.DataFrame(rows)


def build_report(humans: pd.DataFrame, llm: pd.DataFrame | None, vader: pd.DataFrame | None = None,
                 n_boot: int = 2000, seed: int = 0) -> tuple[str, dict]:
    """Markdown report plus a {field: decision} dict."""
    lines = ["# LLM vs human labeling agreement", "",
             "Generated by `python -m src.news.agreement`. Decision rule (docs/OVERVIEW.md, "
             "Layer 4): if the LLM agrees with the human consensus less than humans agree "
             "with each other on a field, use human labels for that field. κ for `concern` "
             "in the decision is quadratic-weighted. `∅` = no label (abstained or failed "
             "grounding); counted as a disagreement.", ""]

    # human-human
    a = humans[humans["labeler"] == "human_a"]
    b = humans[humans["labeler"] == "human_b"]
    double = sorted(set(a["article_id"]) & set(b["article_id"]))
    hh = {}
    lines += ["## Human vs human", "",
              f"{len(double)} articles labeled by both human_a and human_b; "
              f"{humans['article_id'].nunique()} articles have any human label.", ""]
    if double:
        hh = {f: compare_field(_one_per_article(a), _one_per_article(b), f, n_boot, seed)
              for f in FIELDS}
        lines += _table(hh.values()) + [""]

    decisions = {}
    if llm is None or not len(llm):
        lines += ["## LLM vs human consensus", "", "No LLM labels found.", ""]
        return "\n".join(lines), decisions

    consensus = human_consensus(humans)
    run1 = _one_per_article(llm[llm["run"] == llm["run"].min()]) if "run" in llm else _one_per_article(llm)
    meta = []
    for col in ("model", "served_model", "prompt_hash"):
        if col in llm:
            meta.append(f"{col}: {', '.join(sorted(map(str, llm[col].dropna().unique())))}")
    lc = {f: compare_field(run1, consensus, f, n_boot, seed) for f in FIELDS}
    lines += ["## LLM vs human consensus", "", "; ".join(meta), "",
              f"{consensus['article_id'].nunique()} articles with a human consensus.", ""]
    lines += _table(lc.values()) + [""]

    lines += ["## Decision", "", "| Field | Decision |", "| --- | --- |"]
    for f in FIELDS:
        decisions[f] = decide(lc[f], hh.get(f))
        lines.append(f"| {f} | {decisions[f]} |")
    lines.append("")

    lines += ["## LLM precision / recall by class (vs consensus)", ""]
    for f in ("relevant", "fips", "concern"):
        pred, truth = paired_values(run1, consensus, f)
        if pred:
            lines += [f"**{f}**", ""] + _md_frame(precision_recall(pred, truth), index=False) + [""]
    if len(paired_values(run1, consensus, "illness")[0]):
        lines += ["**illness** (each illness scored as present/absent)", ""]
        lines += _md_frame(_illness_binary(run1, consensus), index=False) + [""]

    lines += ["## Confusion matrices (rows = LLM, columns = human consensus)", ""]
    for f in FIELDS:
        pred, truth = paired_values(run1, consensus, f)
        if pred:
            cm = confusion_matrix(pred, truth)
            cm.index.name = f"LLM \\ human"
            lines += [f"**{f}**", ""] + _md_frame(cm) + [""]

    if "run" in llm and llm["run"].nunique() > 1:
        r1, r2 = sorted(llm["run"].unique())[:2]
        x = _one_per_article(llm[llm["run"] == r1])
        y = _one_per_article(llm[llm["run"] == r2])
        sc = [compare_field(x, y, f, n_boot, seed) for f in FIELDS]
        lines += [f"## LLM self-consistency (run {r1} vs run {r2})", "",
                  "Same model, same prompt, twice. The model's sampling cannot be fixed at "
                  "temperature 0, so this is the measured noise floor of a single LLM run.", ""]
        lines += _table(sc) + [""]

    ground_cols = [c for c in ("fips_match", "illness_match", "concern_match", "event_week_match")
                   if c in llm]
    if ground_cols:
        from src.news.ground import summarize
        lines += ["## Quote grounding (all runs)", ""] + _md_frame(summarize(llm), index=False) + [""]

    if vader is not None and len(vader):
        from src.news.baseline import separation
        sep = separation(vader, consensus)
        lines += ["## Baseline: VADER sentiment vs human concern", "",
                  f"Spearman ρ between VADER compound and human concern: "
                  f"{_fmt(sep['spearman'])} (n={sep['n']}). Mean compound by concern level:", ""]
        lines += _md_frame(sep["by_level"], index=False) + [""]
    return "\n".join(lines), decisions


def main(argv=None):
    p = argparse.ArgumentParser(description="LLM vs human agreement report.")
    p.add_argument("--humans", default="labels/human_labels.csv")
    p.add_argument("--llm", default="data/labels/llm_labels_grounded.parquet",
                   help="grounded LLM labels (falls back to ungrounded with a warning)")
    p.add_argument("--vader", default="data/labels/vader_scores.parquet")
    p.add_argument("--out", default="data/labels/agreement_report.md")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)

    humans = load_human_labels(args.humans)
    llm = None
    if Path(args.llm).exists():
        llm = pd.read_parquet(args.llm)
    elif Path("data/labels/llm_labels.parquet").exists():
        print("warning: no grounded labels; scoring raw LLM labels. Run python -m src.news.ground")
        llm = pd.read_parquet("data/labels/llm_labels.parquet")
    vader = pd.read_parquet(args.vader) if Path(args.vader).exists() else None
    report, decisions = build_report(humans, llm, vader, n_boot=args.n_boot)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    for f, d in decisions.items():
        print(f"{f:9s} {d}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
