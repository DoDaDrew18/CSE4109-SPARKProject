"""LLM labeler: one structured label per article, with a quote for every field.

Usage, from the repo root (needs ANTHROPIC_API_KEY in the environment):

    python -m src.news.label                 # label every article in the manifest
    python -m src.news.label --runs 2        # twice, to measure self-consistency
    python -m src.news.label --limit 5       # smoke test on a few articles

then ``python -m src.news.ground`` to drop any label whose quote is not in the
article.

Design choices, each aimed at the team's main worry (made-up labels):

* **The prompt is the labeling guide.** docs/LABELING_GUIDE.md is sent
  verbatim as the system prompt, so the LLM and the humans follow literally
  the same rules and agreement numbers compare like with like. Editing the
  guide changes ``prompt_hash``, which is stored on every row, so labels from
  different prompt versions can never be silently mixed.
* **Structured output, not free text.** The response is constrained to a JSON
  schema (``output_config.format``) whose enums are the guide's categories, so
  the model cannot invent a county or an illness. Forced tool calls are not
  available on this model; a JSON schema is the supported way to get the same
  guarantee. The parsed JSON still goes through ``schema.validate_label``,
  because a schema cannot express cross-field rules (relevant => study county).
* **Determinism.** Claude Opus 5.5 rejects ``temperature`` (sampling is
  fixed server-side), so we cannot force temperature 0. Instead the effort
  level is pinned, and ``--runs 2`` measures how often the model disagrees
  with itself -- an honest number instead of an assumed one.
* **Caching.** The guide is a stable prefix marked ``cache_control``, so after
  the first article only the article text is billed at the full input rate.
* **Provenance.** Each row stores the requested model, the model that actually
  served it (checked against the request), the prompt hash and a UTC
  timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.news.schema import FIPS_CHOICES, ILLNESSES, validate_label

__all__ = ["MODEL_ID", "EFFORT", "OUTPUT_SCHEMA", "LABEL_COLUMNS", "MissingAPIKeyError",
           "system_prompt", "prompt_hash", "build_request", "parse_response",
           "label_article", "label_corpus", "make_client"]

# Change the model here and nowhere else. claude-opus-5-5 is the current
# default Opus: the strongest generally available model short of the much
# pricier Fable tier, and at 60-200 articles cost is a few dollars at most.
MODEL_ID = "claude-opus-5-5"
# Opus 5.5 defaults to "medium"; pin it so a server-side default change can't
# silently change our labels. Thinking cannot be disabled on this model.
EFFORT = "medium"
MAX_TOKENS = 16000          # includes thinking; a short JSON answer needs far less
# No server-side refusal fallback, on purpose: a fallback would silently label
# some articles with a different model, and agreement numbers must describe
# one model. A refusal is recorded as an error row and reported, not patched.

GUIDE_PATH = Path(__file__).resolve().parents[2] / "docs" / "LABELING_GUIDE.md"
OUT_PATH = Path("data/labels/llm_labels.parquet")

OUTPUT_INSTRUCTIONS = """

---

# Your task

You are labeling one news article at a time using the guide above. Read the
article, decide each field exactly as a careful human labeler following this
guide would, and answer with the JSON object only.

- Use "" (empty string) for any blank field or quote.
- `illness` is a list: [] or ["none"] when not relevant.
- Each `*_quote` must be copied verbatim from the article (title or body) and
  must support that specific field; `fips_quote` must name the place.
- If you cannot quote support for a field, leave the field blank rather than
  guess. Unsupported labels are discarded automatically.
- `event_week` is a YYYY-MM-DD Saturday, or "".
"""

USER_TEMPLATE = """<article id="{article_id}">
<outlet>{outlet}</outlet>
<published>{published}</published>
<title>{title}</title>
<body>
{text}
</body>
</article>"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {"type": "boolean"},
        "fips": {"type": "string", "enum": [*FIPS_CHOICES, ""]},
        "fips_quote": {"type": "string"},
        "illness": {"type": "array", "items": {"type": "string", "enum": [*ILLNESSES, "none"]}},
        "illness_quote": {"type": "string"},
        "concern": {"type": "integer", "enum": [0, 1, 2, 3]},
        "concern_quote": {"type": "string"},
        "event_week": {"type": "string"},
        "event_quote": {"type": "string"},
    },
    "required": ["relevant", "fips", "fips_quote", "illness", "illness_quote",
                 "concern", "concern_quote", "event_week", "event_quote"],
    "additionalProperties": False,
}

QUOTE_COLUMNS = ("fips_quote", "illness_quote", "concern_quote", "event_quote")
LABEL_COLUMNS = ("article_id", "labeler", "run", "model", "served_model", "prompt_hash",
                 "labeled_at", "published", "relevant", "fips", "illness", "concern",
                 "event_week", *QUOTE_COLUMNS, "stop_reason", "error",
                 "input_tokens", "cache_read_tokens", "output_tokens")


class MissingAPIKeyError(RuntimeError):
    """No Anthropic credentials in the environment."""


def system_prompt(guide_path=GUIDE_PATH) -> str:
    return Path(guide_path).read_text(encoding="utf-8") + OUTPUT_INSTRUCTIONS


def prompt_hash(system: str) -> str:
    """Hash of everything that shapes the answer except the article itself."""
    blob = json.dumps({"system": system, "user": USER_TEMPLATE, "schema": OUTPUT_SCHEMA,
                       "effort": EFFORT}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_request(article: dict, text: str, system: str, model: str = MODEL_ID) -> dict:
    """Keyword arguments for ``client.messages.create``.

    The system block carries the cache breakpoint; the article (the only part
    that varies) goes after it in the user turn.
    """
    user = USER_TEMPLATE.format(
        article_id=article.get("article_id", ""), outlet=article.get("outlet", "") or "",
        published=str(article.get("published", "") or "")[:10],
        title=article.get("title", "") or "", text=text)
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": EFFORT,
                          "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        "messages": [{"role": "user", "content": user}],
    }


def _empty_label() -> dict:
    return {"relevant": None, "fips": None, "illness": None, "concern": None,
            "event_week": None, **{q: None for q in QUOTE_COLUMNS}}


def parse_response(response) -> tuple[dict, str | None]:
    """Turn an API response into a label dict; never raises.

    Returns (label, error). On any failure -- refusal, truncation, no text,
    invalid JSON, wrong types -- the affected fields are None and ``error``
    says why, so one bad article costs one row, not the whole run.
    """
    label = _empty_label()
    stop = getattr(response, "stop_reason", None)
    if stop == "refusal":
        return label, "refusal"
    text = "".join(getattr(b, "text", "") for b in getattr(response, "content", []) or []
                   if getattr(b, "type", None) == "text")
    if not text.strip():
        return label, f"no text in response (stop_reason={stop})"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        why = "truncated at max_tokens" if stop == "max_tokens" else "invalid JSON"
        return label, f"{why}: {exc.msg}"
    if not isinstance(data, dict):
        return label, f"expected a JSON object, got {type(data).__name__}"

    for q in QUOTE_COLUMNS:
        v = data.get(q)
        label[q] = v.strip() if isinstance(v, str) and v.strip() else None
    raw = {k: (None if data.get(k) in ("", [], None) else data.get(k))
           for k in ("relevant", "fips", "illness", "concern", "event_week")}
    if raw["relevant"] is not None and not isinstance(raw["relevant"], bool):
        raw["relevant"] = str(raw["relevant"])
    clean, errors = validate_label({"article_id": "x", **raw})
    for k in ("relevant", "fips", "illness", "concern", "event_week"):
        label[k] = clean[k]
    return label, ("; ".join(errors) or None)


def label_article(client, article: dict, text: str, system: str, run: int = 1,
                  model: str = MODEL_ID, phash: str | None = None) -> dict:
    """Label one article; API errors are recorded on the row, not raised,
    except authentication errors, which would fail every article the same way."""
    import anthropic

    row = {"article_id": str(article["article_id"]), "labeler": "llm", "run": run,
           "model": model, "served_model": None, "prompt_hash": phash or prompt_hash(system),
           "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "published": article.get("published"), "stop_reason": None,
           "input_tokens": None, "cache_read_tokens": None, "output_tokens": None}
    try:
        response = client.messages.create(**build_request(article, text, system, model))
    except anthropic.AuthenticationError:
        raise
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        return {**row, **_empty_label(), "error": f"api: {type(exc).__name__}: {exc}"}

    label, error = parse_response(response)
    usage = getattr(response, "usage", None)
    row.update(label)
    row.update({"served_model": getattr(response, "model", None),
                "stop_reason": getattr(response, "stop_reason", None), "error": error,
                "input_tokens": getattr(usage, "input_tokens", None),
                "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None)})
    return row


def make_client():
    """An Anthropic client, or a clear error explaining how to provide a key."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Export it in your shell "
            "(`export ANTHROPIC_API_KEY=sk-ant-...`; never commit it) and re-run. Nothing was labeled.")
    import anthropic
    return anthropic.Anthropic()


def _to_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    df["concern"] = df["concern"].astype("Int64")
    df["relevant"] = df["relevant"].astype("boolean")
    df["event_week"] = pd.to_datetime(df["event_week"])
    df["published"] = pd.to_datetime(df["published"])
    return df


def label_corpus(manifest: pd.DataFrame, texts: dict[str, str], client, runs: int = 1,
                 out=OUT_PATH, model: str = MODEL_ID, guide_path=GUIDE_PATH,
                 progress=print) -> pd.DataFrame:
    """Label every article ``runs`` times and append to ``out``.

    Resumable: (article_id, run, prompt_hash, model) rows already in ``out``
    without an error are skipped, so a crash or a rate limit doesn't re-bill
    finished articles, while a guide edit (new hash) re-labels everything.
    """
    system = system_prompt(guide_path)
    phash = prompt_hash(system)
    out = Path(out)
    existing = pd.read_parquet(out) if out.exists() else _to_frame([])
    done = set()
    if len(existing):
        ok = existing[existing["error"].isna()]
        done = set(zip(ok["article_id"], ok["run"], ok["prompt_hash"], ok["model"]))
        # drop failed rows for this prompt so their retries replace them
        retry = existing["error"].notna() & (existing["prompt_hash"] == phash)
        existing = existing[~retry]

    rows = []
    for run in range(1, runs + 1):
        for article in manifest.to_dict("records"):
            aid = str(article["article_id"])
            if (aid, run, phash, model) in done:
                continue
            row = label_article(client, article, texts.get(aid, ""), system, run, model, phash)
            rows.append(row)
            progress(f"run {run} {aid}: relevant={row['relevant']} fips={row['fips']} "
                     f"concern={row['concern']}" + (f" ERROR {row['error']}" if row["error"] else ""))

    result = pd.concat([d for d in (existing, _to_frame(rows)) if len(d)], ignore_index=True) \
        if rows or len(existing) else _to_frame([])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Label news articles with the LLM.")
    p.add_argument("--manifest", default="raw/news/articles.parquet")
    p.add_argument("--text-dir", default="raw/news/text")
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--runs", type=int, default=1, help="label each article N times (self-consistency)")
    p.add_argument("--limit", type=int, default=None, help="only the first N articles")
    p.add_argument("--model", default=MODEL_ID)
    args = p.parse_args(argv)

    try:
        client = make_client()
    except MissingAPIKeyError as exc:
        raise SystemExit(f"error: {exc}")
    manifest = pd.read_parquet(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    # Body only: the title is sent in its own tag. Articles without fetched
    # text are skipped -- a label from a headline alone is not comparable.
    texts = {aid: Path(args.text_dir, f"{aid}.txt").read_text(encoding="utf-8")
             for aid in manifest["article_id"].astype(str)
             if Path(args.text_dir, f"{aid}.txt").exists()}
    missing = (~manifest["article_id"].astype(str).isin(texts)).sum()
    if missing:
        print(f"skipping {missing} articles with no text in {args.text_dir}")
    manifest = manifest[manifest["article_id"].astype(str).isin(texts)]
    df = label_corpus(manifest, texts, client, runs=args.runs, out=args.out, model=args.model)
    errors = df["error"].notna().sum()
    print(f"{len(df)} label rows in {args.out} ({errors} with errors); "
          f"next: python -m src.news.ground")


if __name__ == "__main__":
    main()
