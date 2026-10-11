"""The hallucination check: every LLM label must be backed by a quote that is
really in the article.

Why this exists: the LLM can produce a fluent, plausible label for an article
it misread -- or one that says nothing about our counties at all. We can't
check its *reasoning*, but we can check its *evidence*. The labeler must copy
the sentence supporting each field (docs/LABELING_GUIDE.md); this module looks
for that sentence in the article text and nulls any field whose quote is not
found. A null is honest ("no label"); a made-up label is not.

Matching, strictest first (the method is recorded per field so a reviewer can
audit anything that was not an exact match):

1. ``exact``      -- substring after normalizing case, whitespace, curly
                     quotes, dashes and unicode forms.
2. ``nopunct``    -- substring after also dropping punctuation (the model
                     often drops or adds a comma).
3. ``ellipsis``   -- the quote has "..." and every part is found, in order,
                     within one paragraph-sized span.
4. ``fuzzy``      -- a word window with difflib ratio >= ``FUZZY_MIN`` (0.95),
                     AND at most one quote word missing from the window, AND
                     that word is not a number or a direction word ("rising",
                     "not"...). Numbers and direction words are exactly where
                     a near-miss quote flips the meaning, so they must match.

Quotes under ``MIN_WORDS`` words are rejected: "flu cases" is in every article.

County claims get a second check: the ``fips_quote`` must name the county, its
city, or a place in it (``PLACE_ALIASES``). A quote that proves "flu is rising"
but never says where cannot support a county label.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.study import STUDY_FIPS

__all__ = ["FUZZY_MIN", "MIN_WORDS", "PLACE_ALIASES", "QUOTE_FIELDS", "Match",
           "normalize", "find_quote", "names_county", "ground_row", "ground_labels",
           "load_article_texts"]

FUZZY_MIN = 0.95
MIN_WORDS = 4
MAX_ELLIPSIS_SPAN = 800     # chars; roughly "same paragraph"

# field -> the quote column that must support it
QUOTE_FIELDS = {"fips": "fips_quote", "illness": "illness_quote",
                "concern": "concern_quote", "event_week": "event_quote"}

# Places that, appearing in a quote, tie it to one study county. Deliberately
# short and specific: county name, core city, a few large municipalities and
# the hospitals/school districts local news names most. Ambiguity is accepted
# where the guide resolves it ("St. Louis" alone -> the city, 29510).
PLACE_ALIASES: dict[str, tuple[str, ...]] = {
    "29510": ("st. louis", "saint louis", "city of st. louis", "barnes-jewish",
              "st. louis children's", "central west end", "soulard", "tower grove",
              "north city", "south city", "slps", "st. louis public schools"),
    "29189": ("st. louis county", "saint louis county", "clayton", "florissant",
              "kirkwood", "chesterfield", "ferguson", "webster groves", "university city",
              "maryland heights", "creve coeur", "ballwin", "hazelwood", "ladue",
              "mehlville", "affton", "brentwood", "sunset hills", "town and country",
              "parkway school", "rockwood school", "missouri baptist"),
    "29095": ("jackson county", "kansas city", "independence", "lee's summit",
              "blue springs", "raytown", "grandview", "children's mercy",
              "university health", "truman medical", "kcps"),
    "17031": ("cook county", "chicago", "evanston", "oak park", "skokie", "cicero",
              "schaumburg", "arlington heights", "lurie children's", "rush university",
              "northwestern memorial", "stroger", "cps"),
    "18097": ("marion county", "indianapolis", "indy", "speedway", "beech grove",
              "lawrence township", "riley hospital", "eskenazi", "ips"),
    "47157": ("shelby county", "memphis", "germantown", "collierville", "bartlett",
              "millington", "le bonheur", "regional one", "mscs"),
    "21111": ("jefferson county", "louisville", "jeffersontown", "st. matthews",
              "shively", "norton", "uofl health", "jcps"),
}

# Words whose substitution changes what a quote claims. A fuzzy match may not
# drop or alter any of them.
_CRITICAL = {"no", "not", "never", "none", "rising", "rise", "rose", "rises", "falling", "fall",
             "fell", "falls", "increase", "increased", "increasing", "decrease", "decreased",
             "decreasing", "decline", "declined", "declining", "up", "down", "high", "higher",
             "highest", "low", "lower", "lowest", "doubled", "tripled", "surge", "surging",
             "more", "fewer", "less", "covid", "flu", "influenza", "rsv"}

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-", "­": None,
    "…": "...",
})


def normalize(text: str) -> str:
    """Lowercase, unify quotes/dashes/ellipses/unicode forms, collapse spaces.

    Everything a copy-paste or a model's tokenizer may change without
    changing the words.
    """
    text = unicodedata.normalize("NFKC", str(text)).translate(_TRANSLATE)
    text = re.sub(r"\.\s*\.\s*\.", "...", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _nopunct(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text)).strip()


def _words(text: str) -> list[str]:
    return _nopunct(text).split()


@dataclass(frozen=True)
class Match:
    found: bool
    method: str        # exact | nopunct | ellipsis | fuzzy | too_short | missing | not_found
    score: float


def _ellipsis_match(nq: str, nt: str) -> bool:
    parts = [p.strip(" ,;:.\"'") for p in nq.split("...")]
    parts = [p for p in parts if p]
    if len(parts) < 2 or any(len(_words(p)) < 2 for p in parts):
        return False
    hay = _nopunct(nt)
    start = 0
    first = None
    for part in parts:
        idx = hay.find(_nopunct(part), start)
        if idx < 0:
            return False
        first = idx if first is None else first
        start = idx + len(_nopunct(part))
    return start - first <= MAX_ELLIPSIS_SPAN


def _fuzzy_match(qw: list[str], tw: list[str], min_ratio: float) -> float:
    """Best ratio over word windows near the quote's length, or 0 if the best
    window fails the missing-word guard."""
    q = " ".join(qw)
    n = len(qw)
    best, best_win = 0.0, None
    for size in (n - 1, n, n + 1):
        if size < 1 or size > len(tw):
            continue
        for s in range(len(tw) - size + 1):
            win = tw[s:s + size]
            sm = SequenceMatcher(None, " ".join(win), q, autojunk=False)
            if sm.real_quick_ratio() < min_ratio or sm.quick_ratio() < min_ratio:
                continue
            r = sm.ratio()
            if r > best:
                best, best_win = r, win
    if best_win is None or best < min_ratio:
        return best
    missing = [w for w in qw if w not in set(best_win)]
    if len(missing) > 1 or any(any(c.isdigit() for c in w) or w in _CRITICAL for w in missing):
        return 0.0
    return best


def find_quote(quote, text: str, min_ratio: float = FUZZY_MIN) -> Match:
    """Is ``quote`` in ``text``? See the module docstring for the tiers."""
    if quote is None or (isinstance(quote, float) and pd.isna(quote)) or not str(quote).strip():
        return Match(False, "missing", 0.0)
    nq = normalize(quote).strip(" \"'")
    nt = normalize(text)
    if len(_words(nq.replace("...", " "))) < MIN_WORDS:
        return Match(False, "too_short", 0.0)
    if nq.rstrip(".") in nt:
        return Match(True, "exact", 1.0)
    if _nopunct(nq) in _nopunct(nt):
        return Match(True, "nopunct", 1.0)
    if "..." in nq:
        return Match(True, "ellipsis", 1.0) if _ellipsis_match(nq, nt) else Match(False, "not_found", 0.0)
    score = _fuzzy_match(_words(nq), _words(nt), min_ratio)
    if score >= min_ratio:
        return Match(True, "fuzzy", round(score, 4))
    return Match(False, "not_found", round(score, 4))


def names_county(quote, fips: str) -> bool:
    """Does the quote name ``fips``'s county, city, or a place in it?"""
    if quote is None or fips not in PLACE_ALIASES:
        return False
    q = " " + _nopunct(normalize(quote)) + " "
    return any(" " + _nopunct(alias) + " " in q for alias in PLACE_ALIASES[fips])


def _needs_quote(field: str, value, relevant) -> bool:
    """Which values are claims that need evidence. Negatives ("not relevant",
    "no illness", "unclear county") can't be quoted, so they don't need one."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    if field == "fips":
        return value in STUDY_FIPS
    if field == "illness":
        return value != "none"
    if field == "concern":
        return relevant is True or int(value) > 0
    return True                                         # event_week


def ground_row(row: dict, text: str, strict_place: bool = True,
               min_ratio: float = FUZZY_MIN) -> dict:
    """Ground one LLM label row against its article text.

    Adds per field: ``grounded_<f>`` (False only when a needed quote failed),
    ``<f>_match`` (method) and ``<f>_score``; keeps the model's original value
    as ``<f>_raw``. A county quote that does not name the county sets
    ``fips_place_ok=False`` and, under ``strict_place``, nulls ``fips`` too.
    ``relevant=True`` survives only if at least one field quote grounded.
    """
    out = dict(row)
    relevant = row.get("relevant")
    relevant = None if relevant is None or pd.isna(relevant) else bool(relevant)
    any_grounded = False
    for field, qcol in QUOTE_FIELDS.items():
        value = row.get(field)
        value = None if value is None or (not isinstance(value, str) and pd.isna(value)) else value
        out[f"{field}_raw"] = value
        if not _needs_quote(field, value, relevant):
            out[f"grounded_{field}"], out[f"{field}_match"], out[f"{field}_score"] = True, "not_needed", None
            continue
        m = find_quote(row.get(qcol), text, min_ratio)
        out[f"grounded_{field}"], out[f"{field}_match"], out[f"{field}_score"] = m.found, m.method, m.score
        if m.found:
            any_grounded = True
        else:
            out[field] = None

    out["fips_place_ok"] = None
    if out["fips_raw"] in STUDY_FIPS and out["grounded_fips"]:
        ok = names_county(row.get("fips_quote"), out["fips_raw"])
        out["fips_place_ok"] = ok
        if not ok and strict_place:
            out["fips"], out["grounded_fips"] = None, False

    out["relevant_raw"] = relevant
    out["grounded_relevant"] = True
    if relevant is True and not any_grounded:
        out["relevant"], out["grounded_relevant"] = None, False
    return out


def load_article_texts(article_ids, text_dir="raw/news/text", manifest=None) -> dict[str, str]:
    """Article id -> title + body. The title is included because it is part
    of what was published and quotes from headlines are legitimate."""
    titles = {}
    if manifest is not None and Path(manifest).exists():
        m = pd.read_parquet(manifest)
        titles = dict(zip(m["article_id"].astype(str), m["title"].fillna("")))
    texts = {}
    for aid in article_ids:
        path = Path(text_dir) / f"{aid}.txt"
        body = path.read_text(encoding="utf-8") if path.exists() else ""
        texts[str(aid)] = f"{titles.get(str(aid), '')}\n{body}"
    return texts


def ground_labels(labels: pd.DataFrame, texts: dict[str, str], strict_place: bool = True,
                  min_ratio: float = FUZZY_MIN) -> pd.DataFrame:
    """Ground every row; an article with no text grounds nothing."""
    rows = [ground_row(r, texts.get(str(r["article_id"]), ""), strict_place, min_ratio)
            for r in labels.to_dict("records")]
    return pd.DataFrame(rows)


def summarize(grounded: pd.DataFrame) -> pd.DataFrame:
    """Per field: how many claims needed a quote, how many grounded, by method."""
    out = []
    for field in ("relevant", *QUOTE_FIELDS):
        if field == "relevant":
            needed = grounded["relevant_raw"] == True   # noqa: E712
            ok = grounded.loc[needed, "grounded_relevant"]
            methods = {}
        else:
            needed = grounded[f"{field}_match"] != "not_needed"
            ok = grounded.loc[needed, f"grounded_{field}"]
            methods = grounded.loc[needed, f"{field}_match"].value_counts().to_dict()
        out.append({"field": field, "claims": int(needed.sum()), "grounded": int(ok.sum()),
                    "rate": float(ok.mean()) if len(ok) else float("nan"), **methods})
    if "fips_place_ok" in grounded:
        checked = grounded["fips_place_ok"].notna()
        out.append({"field": "fips (names place)", "claims": int(checked.sum()),
                    "grounded": int(grounded.loc[checked, "fips_place_ok"].astype(bool).sum()),
                    "rate": float(grounded.loc[checked, "fips_place_ok"].astype(bool).mean())
                    if checked.any() else float("nan")})
    df = pd.DataFrame(out)
    methods = [c for c in df.columns if c not in ("field", "claims", "grounded", "rate")]
    df[methods] = df[methods].fillna(0).astype(int)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="Check LLM label quotes against article text.")
    p.add_argument("--labels", default="data/labels/llm_labels.parquet")
    p.add_argument("--out", default="data/labels/llm_labels_grounded.parquet")
    p.add_argument("--text-dir", default="raw/news/text")
    p.add_argument("--manifest", default="raw/news/articles.parquet")
    p.add_argument("--lenient-place", action="store_true",
                   help="flag, but keep, county labels whose quote names no place")
    args = p.parse_args(argv)

    labels = pd.read_parquet(args.labels)
    texts = load_article_texts(labels["article_id"].unique(), args.text_dir, args.manifest)
    grounded = ground_labels(labels, texts, strict_place=not args.lenient_place)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grounded.to_parquet(args.out, index=False)
    print(summarize(grounded).to_string(index=False))
    print(f"wrote {len(grounded)} rows -> {args.out}")


if __name__ == "__main__":
    main()
