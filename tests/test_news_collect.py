"""Offline tests for news collection: URL identity, GDELT parsing, rate limiting, dates."""

import hashlib
import json

import pandas as pd
import pytest

from src.news.collect import (GdeltClient, GdeltRateLimited, RateLimiter, article_id,
                              build_catalog, build_query, candidate_row, normalize_url,
                              parse_artlist)
from src.news.fetch import (extract_published, extract_text, guess_stl_fips,
                            mark_text_duplicates)


# --- URL identity ---------------------------------------------------------

def test_normalize_url_collapses_cosmetic_differences():
    canon = "https://ksdk.com/article/news/health/flu-surge/63-abc"
    for variant in (
        "http://www.ksdk.com/article/news/health/flu-surge/63-abc/",
        "https://WWW.KSDK.com/article/news/health/flu-surge/63-abc#comments",
        "https://ksdk.com/article/news/health/flu-surge/63-abc?utm_source=fb&fbclid=x1",
        "https://ksdk.com//article/news/health/flu-surge/63-abc/amp/",
    ):
        assert normalize_url(variant) == canon


def test_normalize_url_keeps_meaningful_query_sorted():
    a = normalize_url("https://example.com/story?id=5&page=2&utm_medium=x")
    b = normalize_url("https://example.com/story?page=2&id=5")
    assert a == b == "https://example.com/story?id=5&page=2"


def test_article_id_is_sha1_prefix_of_normalized_url():
    url = "https://www.stlpr.org/health-science/2025-01-30/flu"
    expected = hashlib.sha1(normalize_url(url).encode()).hexdigest()[:12]
    assert article_id(url) == expected
    assert len(article_id(url)) == 12
    assert article_id(url + "/") == article_id(url.replace("https", "http"))


# --- GDELT ----------------------------------------------------------------

GDELT_FIXTURE = json.dumps({"articles": [
    {"url": "https://www.ksdk.com/article/news/health/flu/63-d200", "url_mobile": "",
     "title": "If it seems like so many people are sick right now , well .... ",
     "seendate": "20250101T040000Z", "socialimage": "", "domain": "ksdk.com",
     "language": "English", "sourcecountry": "United States"},
    {"url": "https://www.kctv5.com/2025/01/29/missouri-bird-flu/", "url_mobile": "",
     "title": "Missouri issues \x01emergency rule", "seendate": "20250129T193000Z",
     "socialimage": "", "domain": "kctv5.com", "language": "English",
     "sourcecountry": "United States"},
]})


def test_parse_artlist_fixture():
    rows = parse_artlist(GDELT_FIXTURE)
    assert len(rows) == 2
    assert rows[0]["seendate"] == pd.Timestamp("2025-01-01 04:00:00")
    assert rows[0]["title"] == "If it seems like so many people are sick right now, well...."
    assert rows[1]["domain"] == "kctv5.com"


def test_parse_artlist_empty_and_throttle():
    assert parse_artlist("{}") == []
    assert parse_artlist("") == []
    with pytest.raises(GdeltRateLimited):
        parse_artlist("Please limit requests to one every 5 seconds or contact ...")
    with pytest.raises(ValueError, match="non-JSON"):
        parse_artlist("Your query was too short or too long.")


def test_build_query_uses_operators_inside_query():
    q = build_query("stl", terms=("flu", '"respiratory illness"'), domains=("ksdk.com", "stlpr.org"))
    assert q == '(flu OR "respiratory illness") (domain:ksdk.com OR domain:stlpr.org) sourcelang:english'
    assert build_query("kc", terms=("flu", "rsv"), domains=("kcur.org",)).count("(") == 1


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_calls():
    clock = FakeClock()
    lim = RateLimiter(5.0, clock=clock, sleep=clock.sleep)
    assert lim.wait() == 0              # first call never waits
    clock.now += 2.0
    assert lim.wait() == pytest.approx(3.0)
    clock.now += 7.0                    # already past the interval
    assert lim.wait() == 0
    assert lim.wait() == pytest.approx(5.0)
    assert clock.sleeps == [pytest.approx(3.0), pytest.approx(5.0)]


class FakeResponse:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status


class FakeSession:
    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.calls = []

    def get(self, url, params=None, **kw):
        self.calls.append(params)
        return FakeResponse(self.bodies.pop(0))


def test_gdelt_client_backs_off_on_throttle():
    clock = FakeClock()
    session = FakeSession(["Please limit requests to one every 5 seconds", GDELT_FIXTURE])
    client = GdeltClient(limiter=RateLimiter(6, clock=clock, sleep=clock.sleep),
                         session=session, backoff=10, sleep=clock.sleep)
    rows = client.search("flu", start="2025-01-01", end="2025-02-01")
    assert len(rows) == 2
    assert session.calls[0]["startdatetime"] == "20250101000000"
    assert session.calls[0]["maxrecords"] == 250
    assert 10 in clock.sleeps           # backoff happened


def test_gdelt_client_gives_up():
    clock = FakeClock()
    session = FakeSession(["Please limit requests"] * 3)
    client = GdeltClient(limiter=RateLimiter(6, clock=clock, sleep=clock.sleep),
                         session=session, retries=2, backoff=1, sleep=clock.sleep)
    with pytest.raises(GdeltRateLimited):
        client.search("flu")


# --- catalog --------------------------------------------------------------

def test_build_catalog_dedupes_and_marks_syndication():
    gd = pd.DataFrame([candidate_row("https://www.ksdk.com/a/flu-1", "stl", "Flu cases surge in St. Louis area hospitals", "gdelt",
                                     gdelt_seendate="2025-01-30")])
    se = pd.DataFrame([
        candidate_row("http://ksdk.com/a/flu-1/", "stl", "Flu cases surge in St. Louis area hospitals", "search",
                      published="2025-01-29"),
        candidate_row("https://www.kmov.com/b/flu", "stl", "Flu cases surge in St. Louis area hospitals | KMOV",
                      "search", published="2025-01-30"),
    ])
    cat = build_catalog([gd, se])
    assert len(cat) == 2                                  # same URL merged
    first = cat[cat["outlet"] == "ksdk.com"].iloc[0]
    assert first["found_via"] == "search" and first["gdelt_seendate"] == "2025-01-30"
    copy = cat[cat["outlet"] == "kmov.com"].iloc[0]
    assert copy["dup_of"] == first["article_id"]
    assert first["dup_of"] == ""


def test_candidate_row_rejects_bad_metro_and_source():
    with pytest.raises(ValueError):
        candidate_row("https://x.com/a", "nyc")
    with pytest.raises(ValueError):
        candidate_row("https://x.com/a", "stl", found_via="twitter")
    assert candidate_row("https://x.com/a", "chi")["fips_guess"] == "17031"
    assert candidate_row("https://x.com/a", "stl")["fips_guess"] == ""


# --- page parsing ---------------------------------------------------------

JSONLD_PAGE = """<html><head>
<meta property="article:modified_time" content="2025-02-10T12:00:00Z">
<meta property="article:published_time" content="2025-01-28T15:00:00Z">
<script type="application/ld+json">{"@context":"https://schema.org","@graph":[
 {"@type":"WebPage","name":"x"},
 {"@type":"NewsArticle","datePublished":"2025-01-30T02:30:00Z","dateModified":"2025-02-01T00:00:00Z"}]}
</script></head><body></body></html>"""


def test_extract_published_prefers_jsonld_and_local_date():
    # 02:30 UTC on Jan 30 is the evening of Jan 29 in St. Louis.
    assert extract_published(JSONLD_PAGE) == ("2025-01-29", "jsonld")


def test_extract_published_meta_fallbacks_ignore_modified():
    page = ('<html><head><meta property="article:modified_time" content="2026-03-01">'
            '<meta name="parsely-pub-date" content="2025-12-30T10:00:00-06:00"></head></html>')
    assert extract_published(page) == ("2025-12-30", "meta:parsely-pub-date")
    page = '<html><body><time datetime="2026-01-05">Jan 5</time></body></html>'
    assert extract_published(page) == ("2026-01-05", "time_tag")
    assert extract_published("<html><body>no date</body></html>") == (None, "none")
    broken = '<script type="application/ld+json">{not json</script><meta name="pubdate" content="2025-01-02">'
    assert extract_published(broken) == ("2025-01-02", "meta:pubdate")


def test_extract_text_finds_article_body():
    body = " ".join(["Flu cases are rising across St. Louis County hospitals this week."] * 8)
    page = (f"<html><body><nav>Home News Weather Sports</nav><article><h1>Flu</h1>"
            f"<p>{body}</p><p>{body}</p></article><footer>Copyright</footer></body></html>")
    text = extract_text(page)
    assert "Flu cases are rising" in text
    assert "Copyright" not in text and "Weather Sports" not in text


def test_guess_stl_fips():
    assert guess_stl_fips("St. Louis County health officials said") == "29189"
    assert guess_stl_fips("the City of St. Louis health department") == "29510"
    assert guess_stl_fips("St. Louis County and the city of St. Louis") == ""
    assert guess_stl_fips("St. Louis-area hospitals") == ""


def test_mark_text_duplicates_points_to_earliest():
    story = " ".join(f"word{i}" for i in range(200))
    cat = pd.DataFrame({"article_id": ["a", "b", "c"], "published": ["2025-01-02", "2025-01-01", "2025-01-03"],
                        "dup_of": ["", "", ""]})
    texts = {"a": story + " extra tail", "b": story, "c": "totally different text " * 30}
    out = mark_text_duplicates(cat, texts).set_index("article_id")["dup_of"]
    assert out["a"] == "b" and out["b"] == "" and out["c"] == ""


def test_paywall_teaser_detected():
    from src.news.fetch import is_paywalled
    assert is_paywalled("Missouri flu is high.\nTo continue reading this story, login or sign up.")
    assert not is_paywalled("Flu cases rose 40% in St. Louis County this week, officials said.")


def test_soft_404_and_wayback_helpers():
    from src.news.fetch import _is_soft_404, closest_snapshot
    assert _is_soft_404("https://www.wdrb.com/news/flu/article_1.html", "https://www.wdrbwave.com:443/")
    assert not _is_soft_404("https://ksdk.com/a/b", "https://www.ksdk.com/a/b/")
    stamps = ["20250105000000", "20250201120000", "20251201000000"]
    assert closest_snapshot(stamps, "2025-01-30") == "20250201120000"   # first capture after publish
    assert closest_snapshot(stamps, "2026-05-01") == "20251201000000"   # none after: latest
    assert closest_snapshot([], "2025-01-30") is None


def test_gkg_file_urls_and_parse():
    from src.news.collect import gkg_file_urls, parse_gkg
    urls = gkg_file_urls("2025-01-15 12:00", "2025-01-15 12:45")
    assert [u.rsplit("/", 1)[-1] for u in urls] == [f"20250115{t}00.gkg.csv.zip" for t in ("1200", "1215", "1230")]

    def line(url, themes, title):
        cols = [""] * 27
        cols[1], cols[4], cols[8] = "20250115120000", url, themes
        cols[26] = f"<PAGE_TITLE>{title}</PAGE_TITLE>"
        return "\t".join(cols)

    text = "\n".join([
        line("https://www.ksdk.com/flu-story", "TAX_DISEASE_FLU,12;GENERAL_HEALTH,3", "Flu surges &#x2013; KSDK"),
        line("https://www.ksdk.com/sports", "SPORTS,1", "Cardinals win"),
        line("https://national.example.com/flu", "TAX_DISEASE_FLU,1", "Flu nationally"),
    ])
    rows = parse_gkg(text, ["ksdk.com"])
    assert len(rows) == 1 and rows[0]["title"] == "Flu surges \u2013 KSDK"
    assert rows[0]["seendate"] == pd.Timestamp("2025-01-15 12:00")
