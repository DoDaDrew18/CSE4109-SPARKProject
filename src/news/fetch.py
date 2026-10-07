"""Download catalog articles politely; extract main text and the real publish date.

Usage, from the repo root::

    python -m src.news.fetch                 # every catalog row not yet fetched
    python -m src.news.fetch --refetch       # everything again
    python -m src.news.fetch --limit 5       # smoke test

Writes (all gitignored -- article text is copyrighted):

    raw/news/text/{article_id}.txt     extracted main text
    raw/news/html/{article_id}.html    the page as served, for re-extraction
    raw/news/articles.parquet          manifest: one row per fetch attempt outcome

If the live page answers 403 (bot wall) or now redirects to the homepage
(e.g. WDRB after its merge into wdrbwave.com), the Wayback Machine's copy
closest to the publish date is used instead and the manifest says
``via=wayback``. A robots.txt disallow is never routed around.

The script then updates the metadata-only catalog ``labels/articles.csv`` in place
(``fetched_ok``, ``n_words``, ``published``, ``date_source``, ``dup_of``).

Why the publish date gets so much care
--------------------------------------
News enters the nowcast as "what was public by date D", so an article's date
is a leakage boundary exactly like an NSSP issue date. GDELT's ``seendate``
is when GDELT crawled the page, which can trail publication by days (or
precede an edit), and a search snippet's date can be the *updated* date. The
page's own ``datePublished`` metadata is the best evidence we have, so it
wins, and the manifest records how far the other dates were from it.

Politeness
----------
A descriptive User-Agent, ``robots.txt`` honored via ``urllib.robotparser``
(a disallowed URL is recorded as such, never fetched), and at most one
request per host every ``PER_HOST_INTERVAL`` seconds.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib import robotparser
from urllib.parse import urlsplit

import pandas as pd
import requests

from src.news.collect import (CATALOG_PATH, RAW_DIR, USER_AGENT, RateLimiter, mark_duplicates,
                              article_id, read_catalog, write_catalog)

__all__ = ["extract_published", "extract_text", "word_count", "is_paywalled", "RobotsCache",
           "fetch_one", "guess_stl_fips", "mark_text_duplicates", "run"]

TEXT_DIR = RAW_DIR / "text"
HTML_DIR = RAW_DIR / "html"
MANIFEST_PATH = RAW_DIR / "articles.parquet"
PER_HOST_INTERVAL = 4.0
TIMEOUT = 30
# All six study metros sit in US Central or Eastern time. A timestamp like
# 2025-01-30T03:00:00Z is an evening story on Jan 29 locally; dating it Jan 30
# would shift it into the next day and occasionally the next MMWR week.
LOCAL_TZ = "America/Chicago"

# A browser-ish Accept keeps some CMSs from serving a bot stub; the UA itself
# stays honest about who we are.
_HEADERS = {"User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8"}

# Meta tags that carry the *original* publication time, in the order we
# trust them. Modified/updated tags are deliberately absent.
_META_DATE_KEYS = (
    "article:published_time", "og:article:published_time", "datepublished",
    "parsely-pub-date", "sailthru.date", "publish-date", "publish_date",
    "pubdate", "publishdate", "dc.date.issued", "dcterms.created", "date",
    "cxenseparse:recs:publishtime", "article.published",
)


# --------------------------------------------------------------------------
# Publish date
# --------------------------------------------------------------------------

def _to_local_date(value: str) -> str | None:
    """Parse an ISO-ish timestamp; return its local calendar date (YYYY-MM-DD)."""
    text = str(value).strip()
    if not text:
        return None
    try:
        stamp = pd.Timestamp(text)
    except (ValueError, TypeError):
        m = re.search(r"(20\d\d)-(\d\d)-(\d\d)", text)
        if not m:
            return None
        stamp = pd.Timestamp(m.group(0))
    if pd.isna(stamp) or not 2000 <= stamp.year <= 2100:
        return None
    if stamp.tz is not None:
        stamp = stamp.tz_convert(LOCAL_TZ)
    return stamp.strftime("%Y-%m-%d")


def _jsonld_dates(soup) -> list[str]:
    """Every ``datePublished`` in JSON-LD blocks, walking @graph and lists."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "datePublished" in node and isinstance(node["datePublished"], str):
                found.append(node["datePublished"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for tag in soup.find_all("script", type=re.compile("ld\\+json", re.I)):
        try:
            walk(json.loads(tag.string or tag.get_text() or "", strict=False))
        except (json.JSONDecodeError, TypeError):
            continue
    return found


def extract_published(html: str) -> tuple[str | None, str]:
    """(YYYY-MM-DD or None, source) for the page's original publish date.

    Precedence: JSON-LD ``datePublished`` > publish-time ``<meta>`` >
    ``itemprop=datePublished`` > first ``<time datetime>``. Structured data
    first because CMSs generate it from the database field, while visible
    text and ``<time>`` tags are often "updated" stamps.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for value in _jsonld_dates(soup):
        day = _to_local_date(value)
        if day:
            return day, "jsonld"

    metas = {}
    for tag in soup.find_all("meta"):
        key = (tag.get("property") or tag.get("name") or tag.get("itemprop") or "").strip().lower()
        if key and tag.get("content") and key not in metas:
            metas[key] = tag["content"]
    for key in _META_DATE_KEYS:
        if key in metas:
            day = _to_local_date(metas[key])
            if day:
                return day, f"meta:{key}"

    tag = soup.find(attrs={"itemprop": "datePublished"})
    if tag is not None:
        day = _to_local_date(tag.get("datetime") or tag.get("content") or tag.get_text())
        if day:
            return day, "itemprop"

    tag = soup.find("time", attrs={"datetime": True})
    if tag is not None:
        day = _to_local_date(tag["datetime"])
        if day:
            return day, "time_tag"
    return None, "none"


def extract_title(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()
    return soup.title.get_text(strip=True) if soup.title else ""


# --------------------------------------------------------------------------
# Main text
# --------------------------------------------------------------------------

def _fallback_text(html: str) -> str:
    """Readability-lite: paragraphs of the densest container.

    Used only when trafilatura is missing or returns nothing. Takes the
    ``<article>`` if there is one, else the element whose direct ``<p>``
    children hold the most text, which is where a story body lives on
    nearly every news template.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    root = soup.find("article")
    if root is None:
        best, best_len = soup.body or soup, 0
        for parent in {p.parent for p in soup.find_all("p") if p.parent is not None}:
            n = sum(len(p.get_text(" ", strip=True)) for p in parent.find_all("p", recursive=False))
            if n > best_len:
                best, best_len = parent, n
        root = best
    paras = [p.get_text(" ", strip=True) for p in root.find_all("p")]
    return "\n\n".join(p for p in paras if len(p) > 1)


def extract_text(html: str) -> str:
    """Main article text, trafilatura first, BeautifulSoup fallback."""
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=False,
                                   favor_precision=True) or ""
    except ImportError:
        text = ""
    if len(text.split()) < 40:
        alt = _fallback_text(html)
        if len(alt.split()) > len(text.split()):
            text = alt
    return text.strip()


# Phrases that mean we got the lede plus a subscription prompt, not the story.
# Seen on stltoday.com (Lee Enterprises), labortribune.com, dailymemphian.com.
_PAYWALL = re.compile(
    r"to continue reading (this story|this article)|subscribe to continue|"
    r"already a subscriber\?|your gift purchase was successful|no promotional rates found|"
    r"this article is for subscribers|subscriber[- ]only (content|story)",
    re.IGNORECASE)


def is_paywalled(text: str) -> bool:
    """True if the extracted text is a teaser cut off by a subscription wall.

    A teaser is worse than nothing for labeling: it names the topic but not
    the county or the severity, so the label would rest on a headline.
    """
    return bool(_PAYWALL.search(text or ""))


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


# --------------------------------------------------------------------------
# County hint (St. Louis is two counties)
# --------------------------------------------------------------------------

def guess_stl_fips(text: str) -> str:
    """'29189' if only St. Louis County is named, '29510' if only the city, else ''.

    The metro has two study counties and stories usually say "St. Louis"
    meaning both. Only an unambiguous mention moves the guess off blank;
    the labelers make the real call.
    """
    t = text.lower()
    county = len(re.findall(r"st\.? louis county", t))
    city = len(re.findall(r"(city of st\.? louis|st\.? louis city|st\.? louis city's)", t))
    if county and not city:
        return "29189"
    if city and not county:
        return "29510"
    return ""


# --------------------------------------------------------------------------
# Near-duplicate text
# --------------------------------------------------------------------------

def _shingles(text: str, k: int = 6) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def mark_text_duplicates(catalog: pd.DataFrame, texts: dict[str, str],
                         threshold: float = 0.6) -> pd.DataFrame:
    """Point ``dup_of`` at the earliest copy when two texts share most 6-word shingles.

    Catches wire stories (AP, a sister station's script) re-run under a new
    headline, which the title check in ``collect.mark_duplicates`` misses.
    """
    out = catalog.copy()
    ids = [a for a in out["article_id"] if texts.get(a)]
    sh = {a: _shingles(texts[a]) for a in ids}
    pub = dict(zip(out["article_id"], out["published"].replace("", "9999")))
    ids.sort(key=lambda a: (pub.get(a, "9999"), a))
    for i, a in enumerate(ids):
        if not sh[a]:
            continue
        for b in ids[:i]:
            if not sh[b]:
                continue
            jac = len(sh[a] & sh[b]) / len(sh[a] | sh[b])
            if jac >= threshold:
                root = out.loc[out["article_id"] == b, "dup_of"].iloc[0] or b
                out.loc[out["article_id"] == a, "dup_of"] = root
                break
    return out


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

class RobotsCache:
    """One parsed robots.txt per host, fetched with our own UA.

    ``RobotFileParser.read()`` uses urllib's default UA, which some CDNs
    block outright, so we fetch the file ourselves and hand it to ``parse``.
    Following the robotparser convention: 401/403 on robots.txt means
    disallow everything, any other failure means no restrictions.
    """

    def __init__(self, session: requests.Session):
        self.session = session
        self._cache: dict[str, robotparser.RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._cache:
            rp = robotparser.RobotFileParser()
            try:
                resp = self.session.get(base + "/robots.txt", headers=_HEADERS, timeout=TIMEOUT)
                if resp.status_code in (401, 403):
                    rp.disallow_all = True
                elif resp.ok:
                    rp.parse(resp.text.splitlines())
                else:
                    rp.allow_all = True
            except requests.RequestException:
                rp.allow_all = True
            self._cache[base] = rp
        return self._cache[base].can_fetch(USER_AGENT, url)


# CDX rather than the /wayback/available endpoint: in testing "available"
# answered with HTML (rate limiting) or an empty result for URLs CDX lists.
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"


def closest_snapshot(stamps: list[str], near: str = "") -> str | None:
    """The 14-digit Wayback timestamp closest to (preferably after) ``near``.

    A capture taken after publication is the article as published; one far
    later may carry edits, so the earliest capture on/after ``near`` wins.
    """
    stamps = sorted(s for s in stamps if re.fullmatch(r"\d{14}", s))
    if not stamps:
        return None
    if near:
        key = near.replace("-", "")
        after = [s for s in stamps if s[:8] >= key]
        if after:
            return after[0]
    return stamps[-1]


def fetch_wayback(url: str, session: requests.Session, robots: RobotsCache,
                  limiters: dict[str, RateLimiter], near: str = "",
                  make_limiter: Callable[[], RateLimiter] = lambda: RateLimiter(PER_HOST_INTERVAL)) -> dict:
    """Fallback for pages the outlet now blocks (bot 403) or has deleted.

    Only used when the live site *answered* but refused or redirected; a
    robots.txt disallow is respected and never routed around.
    """
    row = {"url": url, "final_url": "", "http_status": None, "robots_ok": None,
           "error": "", "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None),
           "_html": "", "via": "wayback"}
    limiters.setdefault("archive.org", make_limiter()).wait()
    try:
        snap = session.get(WAYBACK_CDX, params={"url": url, "output": "json", "limit": 50,
                                                "filter": "statuscode:200"},
                           headers=_HEADERS, timeout=60).json()
    except (requests.RequestException, ValueError) as exc:
        row["error"] = f"wayback lookup {type(exc).__name__}"
        return row
    stamp = closest_snapshot([r[1] for r in snap[1:]], near)
    if stamp is None:
        row["error"] = "no wayback snapshot"
        return row
    raw = f"https://web.archive.org/web/{stamp}id_/{url}"
    if not robots.allowed(raw):
        row.update(robots_ok=False, error="robots.txt disallows (wayback)")
        return row
    row["robots_ok"] = True
    limiters.setdefault("web.archive.org", make_limiter()).wait()
    try:
        resp = session.get(raw, headers=_HEADERS, timeout=60)
    except requests.RequestException as exc:
        row["error"] = f"wayback {type(exc).__name__}"
        return row
    row.update(http_status=resp.status_code, final_url=raw)
    if not resp.ok or _is_soft_404(url, resp.url.split("id_/", 1)[-1]):
        row["error"] = f"wayback HTTP {resp.status_code}"
        return row
    resp.encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" \
        else resp.apparent_encoding
    row["_html"] = resp.text
    return row


def fetch_one(url: str, session: requests.Session, robots: RobotsCache,
              limiters: dict[str, RateLimiter],
              make_limiter: Callable[[], RateLimiter] = lambda: RateLimiter(PER_HOST_INTERVAL)) -> dict:
    """GET one article; return a manifest row (html under key ``_html``)."""
    row = {"url": url, "final_url": "", "http_status": None, "robots_ok": None,
           "error": "", "fetched_at": datetime.now(timezone.utc).replace(tzinfo=None),
           "_html": "", "via": "live"}
    if not robots.allowed(url):
        row.update(robots_ok=False, error="robots.txt disallows")
        return row
    row["robots_ok"] = True
    host = urlsplit(url).netloc
    limiters.setdefault(host, make_limiter()).wait()
    try:
        resp = session.get(url, headers=_HEADERS, timeout=TIMEOUT, allow_redirects=True)
    except requests.RequestException as exc:
        row["error"] = type(exc).__name__
        return row
    row.update(http_status=resp.status_code, final_url=resp.url)
    if not resp.ok:
        row["error"] = f"HTTP {resp.status_code}"
        return row
    if _is_soft_404(url, resp.url):
        # Dead article URLs on merged/redesigned station sites redirect to the
        # homepage with a 200; extracting that would label the wrong text.
        row["error"] = "redirected to homepage"
        return row
    resp.encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" \
        else resp.apparent_encoding
    row["_html"] = resp.text
    return row


def _is_soft_404(requested: str, final: str) -> bool:
    """True when an article URL landed on a site's front page or a bare section."""
    req, fin = urlsplit(requested), urlsplit(final)
    final_path = fin.path.strip("/")
    return bool(req.path.strip("/")) and (final_path == "" or final_path.count("/") == 0
                                          and len(final_path) < 20 and fin.netloc != req.netloc)


def _days_between(a: str, b: str):
    if not a or not b:
        return None
    return (pd.Timestamp(a) - pd.Timestamp(b)).days


def run(catalog_path=CATALOG_PATH, refetch: bool = False, limit: int | None = None,
        min_words: int = 80, wayback: bool = True, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Fetch catalog rows, write text + manifest, update the catalog."""
    catalog = read_catalog(catalog_path)
    for d in (TEXT_DIR, HTML_DIR):
        d.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()
    done = set()
    if len(manifest) and not refetch:
        # Throttles (429), server errors and timeouts are worth another try;
        # 403/404/robots are not.
        transient = manifest["error"].str.match(r"HTTP (429|5\d\d)|[A-Z]\w*(Error|Timeout)|wayback lookup")
        if wayback and "via" not in manifest.columns:      # manifests from before the fallback
            transient |= manifest["error"].isin(["HTTP 403", "HTTP 404", "redirected to homepage"])
        done = set(manifest.loc[~transient, "article_id"])

    session = requests.Session()
    robots, limiters = RobotsCache(session), {}
    todo = catalog[~catalog["article_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    new_rows = []
    for i, rec in enumerate(todo.to_dict("records"), 1):
        aid = rec["article_id"]
        assert aid == article_id(rec["url"]), f"catalog id mismatch for {rec['url']}"
        row = fetch_one(rec["url"], session, robots, limiters)
        if wayback and not row["_html"] and row["robots_ok"] and (
                row["error"] in ("HTTP 403", "HTTP 404", "HTTP 410", "redirected to homepage")):
            alt = fetch_wayback(rec["url"], session, robots, limiters, near=rec["published"])
            alt["live_error"] = row["error"]
            row = alt
        html = row.pop("_html")
        text, page_date, date_src, page_title = "", None, "none", ""
        if html:
            (HTML_DIR / f"{aid}.html").write_text(html, encoding="utf-8")
            text = extract_text(html)
            page_date, date_src = extract_published(html)
            page_title = extract_title(html)
            (TEXT_DIR / f"{aid}.txt").write_text(text, encoding="utf-8")
        n = word_count(text)
        row.update(article_id=aid, n_words=n, page_published=page_date or "",
                   date_source=date_src, page_title=page_title,
                   catalog_published=rec["published"], gdelt_seendate=rec["gdelt_seendate"],
                   seen_minus_published_days=_days_between(rec["gdelt_seendate"], page_date),
                   catalog_minus_page_days=_days_between(rec["published"], page_date),
                   paywalled=is_paywalled(text),
                   fetched_ok=bool(html) and n >= min_words and not is_paywalled(text))
        new_rows.append(row)
        log(f"[{i}/{len(todo)}] {aid} {row['http_status']} {n:>5}w {page_date or '?':10} "
            f"{date_src:22} {rec['url'][:70]}")

    if new_rows:
        fresh = pd.DataFrame(new_rows)
        manifest = (pd.concat([manifest[~manifest["article_id"].isin(fresh["article_id"])], fresh])
                    if len(manifest) else fresh)
        manifest.to_parquet(MANIFEST_PATH, index=False)

    catalog = apply_manifest(catalog, manifest)
    write_catalog(catalog, catalog_path)
    return catalog


def apply_manifest(catalog: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Copy fetch outcomes into the catalog; page metadata date wins over others."""
    if not len(manifest):
        return catalog
    out = catalog.copy()
    m = manifest.set_index("article_id")
    texts = {}
    for idx, rec in out.iterrows():
        aid = rec["article_id"]
        if aid not in m.index:
            continue
        r = m.loc[aid]
        out.at[idx, "fetched_ok"] = str(bool(r["fetched_ok"]))
        out.at[idx, "n_words"] = str(int(r["n_words"]))
        if r["page_published"]:
            out.at[idx, "published"] = r["page_published"]
            out.at[idx, "date_source"] = r["date_source"]
        elif not rec["published"] and rec["gdelt_seendate"]:
            out.at[idx, "published"] = rec["gdelt_seendate"]
            out.at[idx, "date_source"] = "gdelt_seendate"
        if not rec["title"] and r.get("page_title"):
            out.at[idx, "title"] = r["page_title"]
        path = TEXT_DIR / f"{aid}.txt"
        if path.exists() and bool(r["fetched_ok"]):
            # Teasers are excluded: paywall boilerplate is identical across a
            # site and would make every stltoday.com story a "duplicate".
            texts[aid] = path.read_text(encoding="utf-8")
            if rec["metro"] == "stl" and not rec["fips_guess"]:
                out.at[idx, "fips_guess"] = guess_stl_fips(texts[aid])
    out["dup_of"] = ""                      # recomputed from scratch every time
    return mark_text_duplicates(mark_duplicates(out), texts)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--refetch", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-wayback", action="store_true",
                        help="don't fall back to archive.org for 403/dead pages")
    args = parser.parse_args(argv)
    cat = run(args.catalog, refetch=args.refetch, limit=args.limit, wayback=not args.no_wayback)
    ok = cat["fetched_ok"] == "True"
    print(f"{ok.sum()}/{len(cat)} fetched ok -> {args.catalog}")


if __name__ == "__main__":
    main()
