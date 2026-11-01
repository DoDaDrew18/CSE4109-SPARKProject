"""The one gate between workers and the answer (after CalHacks 13's ``orchestrator/validate.ts``).

Deterministic, no LLM calls. Three jobs:

1. **Vintage rule.** A finding whose ``known_by`` is after the question's
   date is dropped. This is the study's leak check (``align`` raises
   ``LeakError``) applied to the app: an answer "as of" a date may only use
   what was public then.
2. **Prompt-injection hygiene.** Headlines and article quotes are data
   scraped from the web. A finding whose text carries instruction-like
   markers never reaches the LLM. The patterns are the news labeler's
   (``src.news.injection``, ported from CalHacks) plus two for this app's
   own finding schema and tool names.
3. **Citations.** The answer must cite findings as ``[F3]``; citations of
   findings that don't exist are reported, so a made-up source is visible.
"""

from __future__ import annotations

import re

import pandas as pd

from src.app.workers import Finding
from src.news.injection import INJECTION_PATTERNS

__all__ = ["INJECTION", "injection_marker", "check_finding", "check_question", "check_answer"]

INJECTION: tuple[re.Pattern, ...] = INJECTION_PATTERNS + tuple(re.compile(p, re.I) for p in (
    r"\b(tool_use|function_call)\b",
    r"\"(claim|source_url|known_by|confidence|severity|findings|worker)\"\s*:",
))

MAX_CLAIM = 700
MAX_QUESTION = 500
CITE = re.compile(r"\[F(\d+)\]")


def injection_marker(text: str) -> str | None:
    """The first injection marker in ``text``, or None."""
    for pattern in INJECTION:
        m = pattern.search(text or "")
        if m:
            return m.group(0).strip()[:60]
    return None


def check_finding(f: Finding, as_of) -> tuple[bool, str]:
    """``(ok, reason)``. Fails closed on the vintage rule, open on everything else."""
    try:
        known = pd.Timestamp(f.known_by)
    except Exception:
        return False, f"unparseable known_by {f.known_by!r}"
    if known.normalize() > pd.Timestamp(as_of).normalize():
        return False, f"future data: known {known.date()} > as of {pd.Timestamp(as_of).date()}"
    if not f.claim or len(f.claim) > MAX_CLAIM:
        return False, "empty or oversized claim"
    marker = injection_marker(f.claim)
    if marker:
        return False, f"injection text: {marker!r}"
    if f.source_url and not f.source_url.startswith(("https://", "http://")):
        return False, "bad source url"
    return True, ""


def check_question(question: str) -> str:
    """Normalize the user's question; raise ValueError when unusable."""
    q = " ".join((question or "").split())
    if not q:
        raise ValueError("ask a question about the selected area")
    if len(q) > MAX_QUESTION:
        raise ValueError(f"keep the question under {MAX_QUESTION} characters")
    return q


def check_answer(answer: str, findings: list[Finding]) -> dict:
    """Which findings the answer cites, and which citations point nowhere."""
    ids = {f.id for f in findings}
    cited = [f"F{n}" for n in CITE.findall(answer or "")]
    return {"cited": sorted(set(cited) & ids, key=lambda s: int(s[1:])),
            "unknown": sorted(set(cited) - ids, key=lambda s: int(s[1:])),
            "uncited_answer": bool(findings) and not cited}
