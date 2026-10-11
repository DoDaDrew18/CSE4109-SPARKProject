"""Find candidate local news articles: GDELT DOC API, web search, hand-found URLs.

This stage only finds *URLs and metadata*. ``src/news/fetch.py`` downloads
the text. The output is the catalog ``labels/articles.csv``, which is
committed (metadata only -- article text is copyrighted and stays in
``raw/news/``).

Usage, from the repo root::

    python -m src.news.collect gdelt                 # query GDELT, all metros/seasons
    python -m src.news.collect gdelt --metro stl --window 2025-01
    python -m src.news.collect gkg --start 2025-01-20 --end 2025-01-27 --stride 4
    python -m src.news.collect add-url URL --metro stl [--title ...]
    python -m src.news.collect build                 # merge sources -> labels/articles.csv

What we verified against the live GDELT DOC 2.0 API (Oct 2026)
---------------------------------------------------------------
* **Lookback.** ``timespan`` is documented as capped at 3 months, BUT an
  explicit ``startdatetime``/``enddatetime`` pair reaches much further back:
  a Jan 2025 window returned ksdk.com and kctv5.com articles seen in Jan 2025.
  So both past flu peaks (2024-25, 2025-26) are reachable through the DOC API
  itself; we query them month by month. Coverage of small local outlets is
  patchy (GDELT crawls what it crawls), which is why web search and hand-found
  URLs supplement it.
* **Rate limit.** One request per 5 seconds, enforced by the server with a
  plain-text "Please limit requests to one every 5 seconds" body (HTTP 200 or
  429, not JSON). Breaking it earns a *penalty box*: after a burst, every
  request -- even one 30 s later, even at 12 s spacing -- got the throttle
  body for several minutes; after a 5-minute pause it answered normally. So
  we space requests ``MIN_INTERVAL`` (10 s) apart and back off
  exponentially from 300 s, rather than retrying at 5 s and extending the ban.
  On a shared campus network (WashU NAT) the budget is shared with every
  other GDELT user behind the same public IP, so expect long stalls; run
  collection in the background and accept partial windows (failures are
  logged per window and can be re-run with ``--window``).
* **Query length.** An OR of 6 illness terms and 9 ``domain:`` filters is
  rejected with the plain-text "Your query was too short or too long", so
  ``collect_gdelt`` splits each metro's outlets into groups of
  ``DOMAINS_PER_QUERY``.
* **Max records.** ``maxrecords`` caps at 250 per request; there is no
  paging, so a busy query must be split into smaller date windows.
* **Filters are query operators, not URL parameters.** ``sourcecountry:US``,
  ``sourcelang:english`` and ``domain:ksdk.com`` go *inside* ``query=``
  (``domain:`` matches the domain and its subdomains; ``domainis:`` is
  exact). ORed terms must be wrapped in parentheses, and a bare
  ``&sourcecountry=US`` URL parameter is silently ignored.
* **Only metadata.** ``mode=artlist`` returns url, title, seendate (UTC,
  ``YYYYMMDDTHHMMSSZ``), domain, language, sourcecountry -- never text, and
  ``seendate`` is when GDELT crawled it, not the publish date. ``fetch.py``
  reads the real publish date from the page and records the gap.
* **GKG raw files are the unthrottled route.** The 15-minute GDELT 2.0 GKG
  files (``data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.gkg.csv.zip``)
  download without any request budget; Jan 2025 and Jan 2026 files were
  fetched fine while the DOC API was refusing us. Each lists ~1,400 URLs with
  themes (``TAX_DISEASE_FLU``, ``TAX_DISEASE_COVID_19``) and page titles,
  so ``collect_gkg`` filters them by local domain + respiratory theme. Cost:
  ~0.5 GB per day of files, hence the ``--stride`` sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests

from src.study import METROS

__all__ = [
    "GDELT_API", "MIN_INTERVAL", "MAX_RECORDS", "LOCAL_DOMAINS", "ILLNESS_TERMS",
    "SEASON_WINDOWS", "CATALOG_COLUMNS", "normalize_url", "article_id",
    "outlet_of", "RateLimiter", "GdeltRateLimited", "parse_seendate",
    "parse_artlist", "GdeltClient", "build_query", "collect_gdelt",
    "candidate_row", "load_search_tsv", "RESPIRATORY_RE", "gkg_file_urls", "parse_gkg", "collect_gkg", "build_catalog", "mark_duplicates",
]

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
MIN_INTERVAL = 10.0         # seconds between GDELT calls; 5 is the documented floor
DOMAINS_PER_QUERY = 3       # longer ORs trip "Your query was too short or too long"
MAX_RECORDS = 250           # hard server cap per request
USER_AGENT = "CSE4109-SPARK/news (WashU class project; research use)"

RAW_DIR = Path("raw/news")
CANDIDATES_DIR = RAW_DIR / "candidates"      # one parquet per source, gitignored
CATALOG_PATH = Path("labels/articles.csv")

# Respiratory terms. Kept short because GDELT rejects very long queries, and
# broad on purpose: precision is measured later by the labelers, so a missed
# article costs more here than an off-topic one.
ILLNESS_TERMS: tuple[str, ...] = (
    "flu", "influenza", "covid", "rsv", '"respiratory illness"', '"respiratory virus"',
)

# Local outlets per metro. "Local" is defined by domain, not by GDELT's
# geography, because a national story that merely mentions Chicago is exactly
# the false positive we want to avoid.
LOCAL_DOMAINS: dict[str, tuple[str, ...]] = {
    "stl": ("stlpr.org", "ksdk.com", "kmov.com", "firstalert4.com", "fox2now.com", "stltoday.com",
            "stlamerican.com", "missouriindependent.com", "bnd.com", "riverfronttimes.com"),
    "kc": ("kcur.org", "kshb.com", "kctv5.com", "fox4kc.com", "kansascity.com", "kmbc.com"),
    "chi": ("wbez.org", "suntimes.com", "chicagotribune.com", "wgntv.com",
            "nbcchicago.com", "abc7chicago.com", "blockclubchicago.org"),
    "ind": ("wfyi.org", "indystar.com", "wthr.com", "fox59.com", "wishtv.com", "mirrorindy.org"),
    "mem": ("dailymemphian.com", "commercialappeal.com", "wreg.com", "actionnews5.com",
            "localmemphis.com", "wknofm.org"),
    "lou": ("lpm.org", "courier-journal.com", "wlky.com", "wdrb.com", "wave3.com", "whas11.com"),
}

# Month windows around both winter peaks plus quiet-period months, so the
# labeled set is not all "surge". GDELT returns <=250 rows per call, and a
# month per metro stays under that for local-domain queries.
SEASON_WINDOWS: tuple[tuple[str, str], ...] = (
    ("2024-10-01", "2024-11-01"),
    ("2024-12-01", "2025-01-01"), ("2025-01-01", "2025-02-01"), ("2025-02-01", "2025-03-01"),
    ("2025-06-01", "2025-07-01"),
    ("2025-09-01", "2025-10-01"),
    ("2025-11-15", "2025-12-15"), ("2025-12-15", "2026-01-15"), ("2026-01-15", "2026-02-15"),
    ("2026-05-01", "2026-06-01"),
)

CATALOG_COLUMNS: tuple[str, ...] = (
    "article_id", "url", "outlet", "published", "metro", "fips_guess", "title",
    "found_via", "fetched_ok", "n_words", "dup_of", "date_source", "gdelt_seendate",
    "stratum",
)

# Query parameters that identify a click, not a document. Dropping them is
# what makes the same article shared from two places hash to one id.
_TRACKING = re.compile(r"^(utm_.*|fbclid|gclid|mc_[ce]id|ocid|cmpid|taid|cid|"
                       r"_ga|ref|refsrc|outputtype|output_type|amp|__twitter_impression)$",
                       re.IGNORECASE)


# --------------------------------------------------------------------------
# URL identity
# --------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Canonical form of an article URL, used only to derive a stable id.

    Two links to the same article differ in scheme, ``www.``, case of the
    host, a trailing slash, an ``#anchor``, ``/amp`` suffixes or tracking
    parameters. None of those change the document, so none may change the id;
    otherwise one article would be labeled twice and counted twice in the
    weekly news features.
    """
    text = url.strip()
    if "://" not in text:
        text = "https://" + text
    parts = urlsplit(text)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("amp."):
        host = host[4:]
    port = f":{parts.port}" if parts.port and parts.port not in (80, 443) else ""
    path = re.sub(r"/+", "/", parts.path or "/")
    path = re.sub(r"/amp/?$", "/", path)
    path = path.rstrip("/") or "/"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                             if not _TRACKING.match(k)))
    return urlunsplit(("https", host + port, path, query, ""))


def article_id(url: str) -> str:
    """First 12 hex of sha1(normalized URL): short, stable, collision-safe at our scale."""
    return hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()[:12]


def outlet_of(url: str) -> str:
    """Registered-ish domain of a URL (``www.ksdk.com`` -> ``ksdk.com``).

    Good enough for US news hosts; ``chicago.suntimes.com`` keeps its
    subdomain because that subdomain *is* the outlet.
    """
    host = (urlsplit(normalize_url(url)).hostname or "")
    return host


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------

class RateLimiter:
    """Guarantee at least ``min_interval`` seconds between calls.

    ``clock`` and ``sleep`` are injectable so tests can check the timing
    without actually waiting.
    """

    def __init__(self, min_interval: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval = float(min_interval)
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> float:
        """Block until the next call is allowed; return seconds slept."""
        now = self._clock()
        slept = 0.0
        if self._last is not None:
            due = self._last + self.min_interval
            if now < due:
                slept = due - now
                self._sleep(slept)
                now = self._clock()
        self._last = now
        return slept


# --------------------------------------------------------------------------
# GDELT DOC API
# --------------------------------------------------------------------------

class GdeltRateLimited(RuntimeError):
    """GDELT kept answering 'Please limit requests' after every retry."""


def parse_seendate(text: str) -> pd.Timestamp:
    """GDELT ``seendate`` (``20250130T063000Z``, UTC) -> naive UTC Timestamp."""
    return pd.Timestamp(datetime.strptime(text, "%Y%m%dT%H%M%SZ"))


def parse_artlist(body: str) -> list[dict]:
    """Parse a ``mode=artlist&format=json`` body into plain dicts.

    GDELT returns ``{}`` (no ``articles`` key) for zero hits and sometimes
    emits raw control characters inside titles, which strict JSON rejects;
    both are normal and must not crash a long collection run.
    """
    body = body.strip()
    if not body:
        return []
    if body.startswith("Please limit requests"):
        raise GdeltRateLimited(body[:80])
    try:
        payload = json.loads(body, strict=False)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GDELT returned non-JSON: {body[:120]!r}") from exc
    rows = []
    for art in payload.get("articles", []) or []:
        url = art.get("url")
        if not url:
            continue
        title = re.sub(r"\s+([.,'!?:;])", r"\1", (art.get("title") or "").strip())
        rows.append({
            "url": url,
            "title": re.sub(r"\s+", " ", title),
            "seendate": parse_seendate(art["seendate"]) if art.get("seendate") else pd.NaT,
            "domain": (art.get("domain") or outlet_of(url)).lower(),
            "language": art.get("language"),
            "sourcecountry": art.get("sourcecountry"),
        })
    return rows


def _gdelt_stamp(day) -> str:
    return pd.Timestamp(day).strftime("%Y%m%d%H%M%S")


@dataclass
class GdeltClient:
    """Thin DOC API client that is polite by construction.

    Every request goes through one ``RateLimiter``; a throttle response is
    retried with exponential backoff (``backoff`` * 2**attempt seconds, capped
    at ``max_backoff``). The 300 s starting backoff is not timidity: any
    request made while throttled seems to extend the penalty, so short
    retries never succeed.
    """

    limiter: RateLimiter = field(default_factory=lambda: RateLimiter(MIN_INTERVAL))
    session: requests.Session = field(default_factory=requests.Session)
    retries: int = 3
    backoff: float = 300.0
    max_backoff: float = 900.0
    sleep: Callable[[float], None] = time.sleep
    timeout: float = 60.0

    def _get(self, params: dict) -> str:
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                resp = self.session.get(GDELT_API, params=params, timeout=self.timeout,
                                        headers={"User-Agent": USER_AGENT})
                body = resp.text
                throttled = resp.status_code == 429 or body.lstrip().startswith("Please limit")
            except requests.RequestException:
                throttled, body = True, ""
            if not throttled and resp.status_code < 500:
                return body
            if attempt < self.retries:
                self.sleep(min(self.backoff * 2 ** attempt, self.max_backoff))
        raise GdeltRateLimited(f"gave up after {self.retries + 1} attempts: {params.get('query')}")

    def search(self, query: str, start=None, end=None, timespan: str | None = None,
               max_records: int = MAX_RECORDS, sort: str = "datedesc") -> list[dict]:
        """One ``artlist`` call. Pass ``start``/``end`` for history (see module notes)."""
        params = {"query": query, "mode": "artlist", "format": "json",
                  "maxrecords": min(int(max_records), MAX_RECORDS), "sort": sort}
        if start is not None:
            params["startdatetime"] = _gdelt_stamp(start)
        if end is not None:
            params["enddatetime"] = _gdelt_stamp(end)
        if timespan:
            params["timespan"] = timespan
        return parse_artlist(self._get(params))


def build_query(metro: str, terms: Iterable[str] = ILLNESS_TERMS,
                domains: Iterable[str] | None = None) -> str:
    """GDELT query: any illness term, restricted to the metro's local outlets.

    Restricting by ``domain:`` rather than by place name is the point: a
    local station's flu story often never names the city ("area hospitals"),
    while national stories that do name it are not local signal.
    """
    domains = tuple(domains if domains is not None else LOCAL_DOMAINS[metro])
    term_part = "(" + " OR ".join(terms) + ")"
    dom_part = ("(" + " OR ".join(f"domain:{d}" for d in domains) + ")"
                if len(domains) > 1 else f"domain:{domains[0]}")
    return f"{term_part} {dom_part} sourcelang:english"


def collect_gdelt(client: GdeltClient, metros: Iterable[str] = tuple(METROS),
                  windows: Iterable[tuple[str, str]] = SEASON_WINDOWS,
                  log: Callable[[str], None] = print) -> pd.DataFrame:
    """Run ``build_query`` for every metro x window; return candidate rows."""
    rows = []
    windows = tuple(windows)
    for metro in metros:
        doms = LOCAL_DOMAINS[metro]
        groups = [doms[i:i + DOMAINS_PER_QUERY] for i in range(0, len(doms), DOMAINS_PER_QUERY)]
        for group, (start, end) in ((g, w) for g in groups for w in windows):
            query = build_query(metro, domains=group)
            try:
                hits = client.search(query, start=start, end=end)
            except (GdeltRateLimited, ValueError) as exc:
                log(f"  {metro} {group[0]}.. {start}: FAILED {exc}")
                continue
            log(f"  {metro} {'+'.join(group)} {start}..{end}: {len(hits)} hits")
            for hit in hits:
                rows.append(candidate_row(hit["url"], metro=metro, title=hit["title"],
                                          found_via="gdelt", gdelt_seendate=hit["seendate"]))
    return pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))


# --------------------------------------------------------------------------
# GDELT 2.0 GKG raw files (no API, no rate limit)
# --------------------------------------------------------------------------

GKG_BASE = "https://data.gdeltproject.org/gdeltv2/"
# GKG themes that mark respiratory coverage. TAX_DISEASE_FLU and
# TAX_DISEASE_COVID_19 are the common ones (checked on a Jan 2025 file).
# COVID_19 and HEALTH_PANDEMIC are deliberately absent: in 2025-26 they fire
# on every Fauci/RFK Jr./vaccine-policy politics story, and a first scan
# with them kept ~4 off-topic hits per on-topic one.
_GKG_THEMES = re.compile(r"TAX_DISEASE_(FLU|INFLUENZA|RESPIRATORY|PNEUMONIA|RSV|"
                         r"RESPIRATORY_SYNCYTIAL|WHOOPING_COUGH|PERTUSSIS)")

# Respiratory mention in a title or text; used to drop machine-found
# candidates (GDELT/GKG) that matched on a theme but never discuss illness.
RESPIRATORY_RE = re.compile(r"\b(flu|influenza|covid|coronavirus|rsv|respiratory|pneumonia|"
                            r"whooping cough|pertussis|h3n2|h5n1|subclade|virus(es)?)\b", re.I)
_GKG_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>")


def gkg_file_urls(start, end) -> list[str]:
    """URLs of the 15-minute GKG 2.1 files in ``[start, end)``.

    The fallback when the DOC API is throttling us: these are static files
    (``YYYYMMDDHHMMSS.gkg.csv.zip``, ~5-6 MB zipped, ~1,400 documents each)
    with no request budget, reachable back to Feb 2015. The cost is volume --
    a full day is 96 files, ~0.5 GB -- so sample with a stride.
    """
    stamps = pd.date_range(pd.Timestamp(start).floor("15min"), pd.Timestamp(end),
                           freq="15min", inclusive="left")
    return [f"{GKG_BASE}{t:%Y%m%d%H%M%S}.gkg.csv.zip" for t in stamps]


def parse_gkg(text: str, domains: Iterable[str]) -> list[dict]:
    """Rows of one GKG 2.1 file from local ``domains`` that carry a respiratory theme.

    Columns used: 1 DATE, 3 SourceCommonName, 4 DocumentIdentifier (URL),
    7/8 Themes/V2Themes, 26 Extras (holds ``<PAGE_TITLE>``).
    """
    wanted = {d.lower() for d in domains}
    rows = []
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) < 27:
            continue
        host = outlet_of(cols[4]) if cols[4].startswith("http") else ""
        if not any(host == d or host.endswith("." + d) for d in wanted):
            continue
        if not _GKG_THEMES.search(cols[7] + ";" + cols[8]):
            continue
        title = _GKG_TITLE.search(cols[26])
        rows.append({"url": cols[4], "title": html.unescape(title.group(1).strip()) if title else "",
                     "seendate": pd.Timestamp(datetime.strptime(cols[1], "%Y%m%d%H%M%S")),
                     "domain": host})
    return rows


def collect_gkg(start, end, metros: Iterable[str] = tuple(METROS), stride: int = 1,
                session: requests.Session | None = None,
                log: Callable[[str], None] = print) -> pd.DataFrame:
    """Scan every ``stride``-th GKG file in ``[start, end)`` for local respiratory URLs."""
    import io
    import zipfile

    session = session or requests.Session()
    metros = tuple(metros)
    dom_metro = {d: m for m in metros for d in LOCAL_DOMAINS[m]}
    out = []
    for url in gkg_file_urls(start, end)[::stride]:
        try:
            resp = session.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
        except (requests.RequestException, zipfile.BadZipFile, IndexError) as exc:
            log(f"  {url[-27:]}: {type(exc).__name__}")
            continue
        hits = parse_gkg(text, dom_metro)
        for h in hits:
            metro = next(m for d, m in dom_metro.items() if h["domain"] == d or h["domain"].endswith("." + d))
            out.append(candidate_row(h["url"], metro=metro, title=h["title"], found_via="gdelt",
                                     gdelt_seendate=h["seendate"]))
        log(f"  {url[-27:]}: {len(hits)} local respiratory")
    return pd.DataFrame(out, columns=list(CATALOG_COLUMNS))


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------

def _fips_guess(metro: str) -> str:
    """The metro's county if it has exactly one; St. Louis (city vs county) stays blank.

    Only a starting guess -- labelers and ``fetch.py`` refine it from the text.
    """
    counties = METROS[metro].counties
    return next(iter(counties)) if len(counties) == 1 else ""


def candidate_row(url: str, metro: str, title: str = "", found_via: str = "manual",
                  published=None, gdelt_seendate=None, stratum: str = "") -> dict:
    """One catalog row with every column present; unknowns blank."""
    if metro not in METROS:
        raise ValueError(f"unknown metro {metro!r}; expected one of {sorted(METROS)}")
    if found_via not in ("gdelt", "search", "manual"):
        raise ValueError(f"found_via must be gdelt|search|manual, got {found_via!r}")
    seen = pd.Timestamp(gdelt_seendate) if gdelt_seendate is not None else pd.NaT
    return {
        "article_id": article_id(url),
        "url": url.strip(),
        "outlet": outlet_of(url),
        "published": pd.Timestamp(published).strftime("%Y-%m-%d") if published else "",
        "metro": metro,
        "fips_guess": _fips_guess(metro),
        "title": title or "",
        "found_via": found_via,
        "fetched_ok": "",
        "n_words": "",
        "dup_of": "",
        "date_source": "search" if published else "",
        "gdelt_seendate": "" if pd.isna(seen) else seen.strftime("%Y-%m-%d"),
        "stratum": stratum,
    }


def load_search_tsv(path) -> pd.DataFrame:
    """Rows found by web search: url, outlet, published, metro, title, category, verified."""
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    raw.columns = [c.split("(")[0].strip() for c in raw.columns]
    rows = [candidate_row(r["url"], metro=r["metro"].strip(), title=r.get("title", ""),
                          found_via="search", published=r.get("published") or None,
                          stratum=r.get("category", "").strip())
            for r in raw.to_dict("records") if r.get("url", "").startswith("http")]
    return pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))


def _title_key(title: str) -> str:
    """Title reduced to lowercase words, minus station boilerplate after | or -."""
    t = re.split(r"\s[|]\s|\s[-–]\s(?=[A-Z0-9 ]{2,20}$)", str(title))[0]
    return " ".join(re.findall(r"[a-z0-9]+", t.lower()))


def mark_duplicates(catalog: pd.DataFrame) -> pd.DataFrame:
    """Fill ``dup_of`` for syndicated copies: same normalized title, different URL.

    Copies are kept, not dropped -- the same story on three stations is itself
    a (weak) signal of how loud the coverage was -- but ``dup_of`` points at
    the earliest copy so labelers label it once and features can dedupe.
    Text-level near-duplicates are caught later in ``fetch.py``.
    """
    out = catalog.copy()
    out["dup_of"] = out["dup_of"].fillna("")
    keys = out["title"].map(_title_key)
    order = out.assign(_k=keys, _p=out["published"].replace("", "9999"))
    for key, group in order[order["_k"].str.len() >= 25].groupby("_k"):
        if len(group) < 2:
            continue
        first = group.sort_values(["_p", "article_id"]).iloc[0]["article_id"]
        idx = group.index[group["article_id"] != first]
        out.loc[idx, "dup_of"] = first
    return out


def build_catalog(frames: Iterable[pd.DataFrame], existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge candidate sources into one catalog, one row per article_id.

    Precedence for the same article: ``manual`` > ``search`` > ``gdelt``,
    because a person looked at it; fields filled earlier by ``fetch.py``
    (``fetched_ok``, ``n_words``, page date) are preserved from ``existing``.
    """
    rank = {"manual": 0, "search": 1, "gdelt": 2}
    parts = [f for f in frames if f is not None and len(f)]
    if existing is not None and len(existing):
        parts.insert(0, existing)
    if not parts:
        return pd.DataFrame(columns=list(CATALOG_COLUMNS))
    allrows = pd.concat(parts, ignore_index=True).astype(str).replace({"nan": "", "NaT": ""})
    allrows["_r"] = allrows["found_via"].map(rank).fillna(3)
    merged = []
    for aid, group in allrows.groupby("article_id", sort=False):
        group = group.sort_values("_r", kind="stable")
        row = group.iloc[0].to_dict()
        for col in CATALOG_COLUMNS:        # fill blanks from lower-precedence copies
            if not row.get(col):
                vals = [v for v in group[col] if v]
                row[col] = vals[0] if vals else ""
        merged.append(row)
    out = pd.DataFrame(merged)[list(CATALOG_COLUMNS)]
    return mark_duplicates(out).sort_values(["metro", "published", "outlet"]).reset_index(drop=True)


def read_catalog(path=CATALOG_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(CATALOG_COLUMNS))
    cat = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in CATALOG_COLUMNS:
        if col not in cat.columns:
            cat[col] = ""
    return cat[list(CATALOG_COLUMNS)]


def write_catalog(catalog: pd.DataFrame, path=CATALOG_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    catalog[list(CATALOG_COLUMNS)].to_csv(path, index=False)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cmd_gdelt(args) -> None:
    metros = args.metro or list(METROS)
    windows = SEASON_WINDOWS
    if args.window:
        start = pd.Timestamp(args.window + "-01")
        windows = ((start.strftime("%Y-%m-%d"), (start + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")),)
    frame = collect_gdelt(GdeltClient(), metros, windows)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.window or "seasons"
    out = CANDIDATES_DIR / f"gdelt_{'-'.join(metros)}_{tag}.parquet"
    frame.to_parquet(out, index=False)
    print(f"{len(frame)} rows ({frame['article_id'].nunique()} unique) -> {out}")


def _cmd_gkg(args) -> None:
    metros = args.metro or list(METROS)
    frame = collect_gkg(args.start, args.end, metros, stride=args.stride)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    out = CANDIDATES_DIR / f"gkg_{'-'.join(metros)}_{args.start}_{args.end}.parquet"
    frame.drop_duplicates("article_id").to_parquet(out, index=False)
    print(f"{frame['article_id'].nunique()} unique -> {out}")


def _cmd_add_url(args) -> None:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    path = CANDIDATES_DIR / "manual.parquet"
    row = candidate_row(args.url, metro=args.metro, title=args.title or "", found_via="manual",
                        published=args.published, stratum=args.stratum or "")
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=list(CATALOG_COLUMNS))
    new = pd.concat([old[old["article_id"] != row["article_id"]], pd.DataFrame([row])])
    new.to_parquet(path, index=False)
    print(f"{row['article_id']}  {row['url']}")


def _cmd_build(args) -> None:
    frames = [pd.read_parquet(p).astype(str) for p in sorted(CANDIDATES_DIR.glob("*.parquet"))]
    frames += [load_search_tsv(p) for p in args.search_tsv]
    catalog = build_catalog(frames, existing=read_catalog(args.out))
    write_catalog(catalog, args.out)
    print(f"{len(catalog)} articles -> {args.out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gdelt", help="query the GDELT DOC API")
    g.add_argument("--metro", action="append", choices=sorted(METROS))
    g.add_argument("--window", help="single month YYYY-MM instead of SEASON_WINDOWS")
    g.set_defaults(func=_cmd_gdelt)
    k = sub.add_parser("gkg", help="scan GDELT GKG raw files (no rate limit, heavy download)")
    k.add_argument("--start", required=True, help="YYYY-MM-DD")
    k.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive)")
    k.add_argument("--stride", type=int, default=4, help="use every Nth 15-min file (4 = hourly)")
    k.add_argument("--metro", action="append", choices=sorted(METROS))
    k.set_defaults(func=_cmd_gkg)
    a = sub.add_parser("add-url", help="add a hand-found URL")
    a.add_argument("url")
    a.add_argument("--metro", required=True, choices=sorted(METROS))
    a.add_argument("--title")
    a.add_argument("--published", help="YYYY-MM-DD if known; fetch.py checks it")
    a.add_argument("--stratum", choices=("respiratory", "quiet", "hard_negative"))
    a.set_defaults(func=_cmd_add_url)
    b = sub.add_parser("build", help="merge all candidates into labels/articles.csv")
    b.add_argument("--search-tsv", action="append", default=[], type=Path)
    b.add_argument("--out", type=Path, default=CATALOG_PATH)
    b.set_defaults(func=_cmd_build)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
