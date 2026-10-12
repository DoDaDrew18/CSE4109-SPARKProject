"""Offline tests for the natural-language map (Demo II): parsing, the checks
against the data, the computed answer, the LLM fallback, rendering and the app.

The geography is a synthetic lattice in the Midwest with the seven study
counties at their real FIPS, a planted flu cluster in one corner, a tiny ED
that reports 0% except one 50% week (the Calloway, KY artifact) and a CDC
"Sparse" area, so the properties that keep the demo honest can be checked:
the LLM never decides a number, lists skip low-volume artifacts, and news is
shown only once it was published.
"""

import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.spatial import query as Q
from src.spatial.query_map import DIVERGING, change_bins, render

LATEST = pd.Timestamp("2026-10-03")
PEAK = pd.Timestamp("2025-12-27")
STUDY_CELLS = {(5, 5): ("29189", "St. Louis County", "MO"), (5, 6): ("29510", "St. Louis city", "MO"),
               (6, 1): ("29095", "Jackson County", "MO"), (9, 8): ("17031", "Cook County", "IL"),
               (7, 9): ("18097", "Marion County", "IN"), (0, 6): ("47157", "Shelby County", "TN"),
               (4, 10): ("21111", "Jefferson County", "KY"), (4, 5): ("29099", "Jefferson County", "MO"),
               (4, 6): ("17077", "Jackson County", "IL")}
STATE_FIPS = {"MO": "29", "IL": "17", "IN": "18", "TN": "47", "KY": "21"}


def lattice():
    rows = []
    for r in range(10):
        for c in range(12):
            lon, lat = -96 + c, 34.5 + 0.75 * r
            if (r, c) in STUDY_CELLS:
                fips, name, st = STUDY_CELLS[(r, c)]
            else:
                st = "TN" if r < 2 else "MO" if c < 6 else "IL" if c < 9 else "IN"
                fips, name = f"{STATE_FIPS[st]}{600 + r * 12 + c:03d}", f"Cell{r}x{c} County"
            rows.append({"fips": fips, "name": name, "state": st, "state_fips": STATE_FIPS[st],
                         "row": r, "col": c, "geometry": box(lon, lat, lon + 1, lat + 0.75),
                         "lat": lat + 0.375, "lon": lon + 0.5})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


@pytest.fixture(scope="module")
def world():
    return build_world()


def build_world():
    shapes = lattice()
    weeks = pd.date_range("2025-10-04", LATEST, freq="7D")
    season = np.exp(-(((weeks - PEAK).days / 7) / 4.0) ** 2)            # flu peaks Dec 27
    records = []
    for cell in shapes.itertuples():
        hsa = f"h{cell.row}_{cell.col // 2}"
        lc = cell.col // 2 * 2                                            # HSA = a pair of cells
        hot = 3.0 if (3 <= cell.row <= 6 and lc in (6, 8)) else 1.0        # planted 4x4 cluster
        left = shapes[(shapes["row"] == cell.row) & (shapes["col"] == lc)].iloc[0]
        hsa_name = f"{left['name'].removesuffix(' County')}, {left['state']} - pair"
        for i, week in enumerate(weeks):
            flu = round(0.2 + 6.0 * hot * season[i] + 0.1 * (cell.row % 3), 2)
            covid, rsv = 1.0, 0.5
            trend = "Increasing" if i and season[i] > season[i - 1] else "Decreasing"
            if hsa == "h8_0":                       # a tiny ED: 0% except one 50% week
                flu, covid, rsv = (50.0 if week == PEAK else 0.0), 0.0, 0.0
                trend = "No Change"
            if hsa == "h9_0":
                trend = "Sparse"
            records.append({"fips": cell.fips, "week_end": week, "covid": covid, "flu": flu, "rsv": rsv,
                            "hsa": hsa_name, "hsa_nci_id": hsa, "trend_flu": trend,
                            "trend_covid": "No Change", "trend_rsv": "No Change"})
    values = pd.DataFrame(records)
    values["resp"] = values[["covid", "flu", "rsv"]].sum(axis=1, min_count=3)
    return values, shapes, Q.Gazetteer(shapes)


# ---------------------------------------------------------------- the rule parser

@pytest.mark.parametrize("question,expect", [
    ("where is flu rising around St. Louis?",
     dict(signal="flu", measure="change", direction="up", when="latest", place="metro:stl")),
    ("Where was flu rising around St. Louis at Christmas 2025?",
     dict(signal="flu", measure="change", when="2025-12-25", place="metro:stl")),
    ("RSV hot spots in the Midwest at the peak of the 2024-25 season",
     dict(signal="rsv", measure="level", when="peak", season="2024-25", place="region:midwest")),
    ("COVID in Cook County, IL two weeks ago", dict(signal="covid", when="2026-09-19", place="county:17031")),
    ("flu in Jefferson County, MO", dict(place="county:29099")),
    ("Where is respiratory illness falling in Missouri?",
     dict(signal="resp", measure="change", direction="down", place="state:MO")),
    ("top 10 places for flu Dec 2025", dict(top=10, when="2025-12-27", place="us")),
    ("flu and RSV in Memphis on 12/27/2025", dict(signal="resp", when="2025-12-27", place="metro:mem")),
    ("how bad was flu this fall in Chicago", dict(measure="level", place="metro:chi")),
    ("St. Louis city flu last week", dict(place="county:29510", when="latest")),
    ("flu in southern Illinois in January 2026", dict(place="state:IL", when="2026-01-31")),
    ("Kansas City flu", dict(place="metro:kc")),
    ("flu in West Virginia", dict(place="state:WV")),
    ("May flu be rising in Indiana?", dict(when="latest", measure="change", place="state:IN")),
    ("flu trends in marketing", dict(when="latest")),
    ("covid 3 weeks ago", dict(signal="covid", when="2026-09-12")),
    ("flu on Thanksgiving", dict(when="2025-11-27")),
    ("flu in December", dict(when="2025-12-27")),
])
def test_rule_parser(world, question, expect):
    _, _, gaz = world
    gaz.states.update({"WV"})                     # the lattice has no WV county; the parser only needs the code
    q = Q.parse_rules(question, gaz, LATEST)
    got = {k: getattr(q, k) for k in expect}
    assert got == expect, (question, q.notes)


def test_rule_parser_notes_explain_its_choices(world):
    _, _, gaz = world
    q = Q.parse_rules("flu in Jefferson County", gaz, LATEST)
    assert q.place == "county:21111" and "exists in 2 states" in q.notes[0]     # study county preferred
    q = Q.parse_rules("flu and RSV in Memphis", gaz, LATEST)
    assert "asked together" in q.notes[0]
    q = Q.parse_rules("flu in Chicago and Memphis", gaz, LATEST)
    assert q.place == "metro:chi" and "ask again for the Memphis area" in q.notes[0]
    q = Q.parse_rules("flu in St. Louis, Missouri", gaz, LATEST)
    assert q.place == "metro:stl" and q.notes == []                         # Missouri contains it


# ---------------------------------------------------------------- resolve

def test_resolve_clamps_snaps_and_says_so(world):
    values, _, gaz = world
    q = Q.resolve(Q.MapQuery(when="2030-01-01"), values, gaz)
    assert q.week_end == LATEST and "no data after" in q.notes[0]
    q = Q.resolve(Q.MapQuery(when="2020-01-01"), values, gaz)
    assert q.week_end == pd.Timestamp("2025-10-04") and "data starts" in q.notes[0]
    q = Q.resolve(Q.MapQuery(when="2025-12-24"), values, gaz)             # a Wednesday
    assert q.week_end == PEAK and q.prev_week == PEAK - pd.Timedelta(weeks=1)
    gap = values[values["week_end"] != pd.Timestamp("2025-12-20")]
    q = Q.resolve(Q.MapQuery(when="2025-12-20", measure="change"), gap, gaz)
    assert q.week_end == pd.Timestamp("2025-12-13")
    q = Q.resolve(Q.MapQuery(when="2025-12-27", measure="change"), gap, gaz)
    assert q.prev_week == pd.Timestamp("2025-12-13") and "nearest earlier week" in q.notes[-1]


def test_peak_uses_the_places_own_series_and_skips_a_young_season(world):
    values, _, gaz = world
    q = Q.resolve(Q.MapQuery(signal="flu", when="peak", place="metro:chi"), values, gaz)
    assert q.week_end == PEAK
    assert "has 1 week of data so far; using 2025-26" in q.notes[0]
    q = Q.resolve(Q.MapQuery(signal="flu", when="peak", season="2023-24"), values, gaz)
    assert q.week_end == LATEST and "no 2023-24 data" in q.notes[0]


# ---------------------------------------------------------------- the answer

def test_answer_computes_from_data_and_ranks_only_reliable_places(world):
    values, shapes, gaz = world
    q = Q.resolve(Q.MapQuery(signal="flu", when="2025-12-27", top=3), values, gaz)
    # 999 permutations: with 99 the smallest p is 0.01, which no county can
    # carry through the FDR correction across 120 tests.
    ans = Q.answer(q, values, shapes, gaz, "test", permutations=999, with_news=False,
                   county_week="nope.parquet", predictions="nope.csv")
    assert len(ans.frame) == len(shapes) and ans.counts["with_data"] == len(shapes)
    top = ans.places
    assert len(top) == 3 and top["place"].str.endswith(" area").all()
    assert top["value"].is_monotonic_decreasing
    assert not top["fips"].isin(values.loc[values["hsa_nci_id"].isin(["h8_0", "h9_0"]), "fips"]).any()
    assert ans.counts["unranked"] == 2                    # the 50% tiny ED and the Sparse area
    hot = ans.frame.loc[ans.frame["cluster"].str.startswith("Hot"), "fips"]
    block = shapes.loc[shapes["row"].between(3, 6) & shapes["col"].between(6, 9), "fips"]
    assert len(set(hot) & set(block)) >= 8 and set(hot) <= set(block)    # the planted cluster, only
    assert (top["n_counties"] == 2).all()                 # one row per HSA, both counties counted


def test_reliable_fips_drops_mostly_zero_and_sparse(world):
    values, _, _ = world
    ok = Q.reliable_fips(values, PEAK, "flu")
    tiny = set(values.loc[values["hsa_nci_id"] == "h8_0", "fips"])     # row 8, cols 0-1
    sparse = set(values.loc[values["hsa_nci_id"] == "h9_0", "fips"])
    assert not (ok & tiny) and not (ok & sparse) and len(ok) == values["fips"].nunique() - 4


def test_change_answer_and_metro_view(world):
    values, shapes, gaz = world
    q = Q.resolve(Q.parse_rules("where was flu rising around Chicago on Dec 20 2025?", gaz, LATEST),
                  values, gaz)
    ans = Q.answer(q, values, shapes, gaz, permutations=99, with_news=False,
                   county_week="nope.parquet", predictions="nope.csv")
    assert q.measure == "change" and ans.counts["rising"] > 0 and ans.counts["falling"] == 0
    assert "17031" in set(ans.frame["fips"]) and len(ans.frame) < len(shapes)     # cropped view
    assert ans.places["change"].is_monotonic_decreasing
    assert set(ans.frame["cluster"]) <= {"High-High", "Low-Low", "High-Low", "Low-High",
                                         "Not significant", "No data"}
    text = Q.summary_lines(ans, gaz)
    assert text["lead"] == "Biggest rises" and "rose in" in text["headline"]


def test_study_detail_walks_through_one_nowcast_date(world, tmp_path):
    values, shapes, gaz = world
    week = PEAK
    table = pd.DataFrame([
        {"fips": "17031", "week_end": week - pd.Timedelta(weeks=1), "as_of": week, "lag": 1,
         "ed_resp": 3.2, "ed_resp_first": 3.2, "ed_resp_final": 9.0},
        {"fips": "17031", "week_end": week, "as_of": week, "lag": 0, "ed_resp": np.nan,
         "ed_resp_first": 11.7, "ed_resp_final": 11.9},
        {"fips": "17031", "week_end": week, "as_of": week + pd.Timedelta(weeks=2), "lag": 2,
         "ed_resp": 11.7, "ed_resp_first": 11.7, "ed_resp_final": 11.9},
        {"fips": "29510", "week_end": week, "as_of": week, "lag": 0, "ed_resp": np.nan,
         "ed_resp_first": 11.0, "ed_resp_final": 11.0},
    ])
    table.to_parquet(tmp_path / "cw.parquet")
    pd.DataFrame([{"rung": "6_gbm_history", "fips": "17031", "as_of": week, "week_end": week, "pred": 9.5},
                  {"rung": "0_naive", "fips": "17031", "as_of": week, "week_end": week, "pred": 3.2}]
                 ).to_csv(tmp_path / "preds.csv", index=False)
    q = Q.resolve(Q.MapQuery(signal="resp", when="2025-12-27", place="metro:chi"), values, gaz)
    ans = Q.answer(q, values, shapes, gaz, permutations=99, with_news=False,
                   county_week=tmp_path / "cw.parquet", predictions=tmp_path / "preds.csv")
    chi = next(s for s in ans.study if s["metro"] == "Chicago")
    line = " ".join(Q.summary_lines(ans, gaz)["study"])
    assert chi["known_value"] == 3.2 and chi["known_week"] == week - pd.Timedelta(weeks=1)
    assert "newest official respiratory number was 3.2% (week ending Dec 20)" in line
    assert "first printed at 11.7% (public by Jan 10, 2026), settled at 11.9%" in line
    assert "Nowcast that day: 9.5% (gradient boosting; carry-forward 3.2%)" in line

    q = Q.resolve(Q.MapQuery(signal="resp", when="2025-12-27", place="metro:stl"), values, gaz)
    ans = Q.answer(q, values, shapes, gaz, permutations=99, with_news=False,
                   county_week=tmp_path / "cw.parquet", predictions=tmp_path / "preds.csv")
    stl = " ".join(Q.summary_lines(ans, gaz)["study"])
    assert "no official respiratory number was public yet" in stl     # Missouri, Dec 2025


def test_cited_articles_only_once_published(world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "labels/human").mkdir(parents=True)
    rows = [("a00000000001", "y", "17031", "flu", "3", "Several Chicago hospitals went on diversion"),
            ("a00000000002", "y", "17031", "flu", "2", "Cook County flu cases doubled this week"),
            ("a00000000003", "y", "47157", "flu", "2", "Memphis flu cases doubled this week"),
            ("a00000000004", "n", "other", "none", "0", "")]
    pd.DataFrame([{"article_id": a, "labeler": "human_a", "relevant": r, "fips": f, "illness": i,
                   "concern": c, "event_week": "", "quote": q, "notes": ""} for a, r, f, i, c, q in rows]
                 ).to_csv(tmp_path / "labels/human/human_a.csv", index=False)
    pd.DataFrame({"article_id": [r[0] for r in rows],
                  "published": ["2025-12-23", "2025-12-30", "2025-12-23", "2025-12-23"],
                  "title": ["Hospitals divert", "Later story", "Memphis story", "Allergy"],
                  "outlet": ["wbez.org"] * 4, "url": ["https://example.org/1"] * 4}
                 ).to_csv(tmp_path / "labels/articles.csv", index=False)
    arts, source = Q.cited_articles(["17031"], PEAK)
    assert source == "human" and list(arts["article_id"]) == ["a00000000001"]   # a2 came out later
    assert arts.loc[0, "quote"].startswith("Several Chicago hospitals")
    assert Q.cited_articles(["17031"], "2026-01-03")[0]["article_id"].tolist() == ["a00000000002",
                                                                                 "a00000000001"]
    empty, source = Q.cited_articles(["21111"], PEAK)
    assert empty.empty and source == "human"


# ---------------------------------------------------------------- the LLM parser

def fake_client(text, stop="end_turn", model="fake-model"):
    calls = []
    resp = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)],
                           stop_reason=stop, model=model)

    def create(kind):
        def _create(**kwargs):
            calls.append((kind, kwargs))
            return resp
        return _create
    return SimpleNamespace(messages=SimpleNamespace(create=create("plain")),
                           beta=SimpleNamespace(messages=SimpleNamespace(create=create("beta"))),
                           calls=calls)


GOOD = {"signal": "flu", "measure": "change", "direction": "up", "when": "2025-12-27", "season": "",
        "place_type": "metro", "region": "", "state": "", "metro": "chi", "county": "", "top": 7}


def test_llm_answer_is_validated_and_used(world):
    _, _, gaz = world
    client = fake_client(json.dumps(GOOD))
    q = Q.parse_llm("how's the flu spreading in chicagoland", gaz, "2025-10-04", LATEST, client,
                    "qwen3:8b", "ollama")
    assert (q.signal, q.measure, q.when, q.place, q.top) == ("flu", "change", "2025-12-27", "metro:chi", 7)
    assert q.parser == "ollama:fake-model" and q.notes == []
    kind, request = client.calls[0]
    assert kind == "plain"
    assert request["output_config"]["format"]["schema"] == Q.LLM_SCHEMA
    assert request["output_config"]["effort"] == "low"
    assert "<question>how's the flu spreading in chicagoland</question>" == request["messages"][0]["content"]
    assert "chicagoland" not in request["system"]                       # the question stays out of the prompt
    city = fake_client(json.dumps(GOOD | {"place_type": "county", "metro": "", "county": "St. Louis city, MO"}))
    assert Q.parse_llm("q", gaz, "2025-10-04", LATEST, city, "m", "ollama").place == "county:29510"


@pytest.mark.parametrize("text,stop", [
    ("not json at all", "end_turn"),
    (json.dumps(GOOD | {"place_type": "state", "metro": "", "state": "ZZ"}), "end_turn"),
    (json.dumps(GOOD | {"signal": "measles"}), "end_turn"),
    (json.dumps(GOOD | {"when": "the day after tomorrow"}), "end_turn"),
    (json.dumps(GOOD | {"place_type": "county", "county": "Atlantis County, MO"}), "end_turn"),
    ("", "refusal"),
])
def test_bad_llm_answers_fall_back_to_the_rules(world, text, stop):
    _, _, gaz = world
    q = Q.parse_llm("where is flu rising around St. Louis?", gaz, "2025-10-04", LATEST,
                    fake_client(text, stop), "m", "ollama")
    assert q.parser == "rules" and "LLM parse failed" in q.notes[0]
    assert (q.signal, q.measure, q.place) == ("flu", "change", "metro:stl")


def test_claude_requests_opt_into_the_refusal_fallback(world):
    _, _, gaz = world
    client = fake_client(json.dumps(GOOD), model="claude-opus-5-5")
    q = Q.parse_llm("flu in chicago", gaz, "2025-10-04", LATEST, client, "claude-opus-5-5", "anthropic")
    kind, request = client.calls[0]
    assert kind == "beta" and request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["fallbacks"] == "default" and request["model"] == "claude-opus-5-5"
    assert "temperature" not in request and "thinking" not in request
    assert q.parser == "anthropic:claude-opus-5-5"


# ---------------------------------------------------------------- rendering and the app

def test_change_bins_are_symmetric_round_numbers():
    bins = change_bins(pd.Series(np.linspace(-6, 6, 101)))
    inner = bins[1:-1]
    assert inner == [-x for x in inner[::-1]] and inner[3] > 0
    assert all(float(f"{b:g}") == b for b in inner) and len(DIVERGING) == len(bins) - 1


def test_render_escapes_everything_typed_or_scraped(world):
    values, shapes, gaz = world
    question = "<img src=x onerror=alert(1)> where was flu rising in Chicago on Dec 27 2025"
    q = Q.resolve(Q.parse_rules(question, gaz, LATEST), values, gaz)
    ans = Q.answer(q, values, shapes, gaz, permutations=99, with_news=False,
                   county_week="nope.parquet", predictions="nope.csv")
    ans.articles = pd.DataFrame([{"article_id": "a1", "published": PEAK, "fips": "17031", "concern": 3,
                                  "quote": "<b>ERs full</b>", "title": "<script>x()</script>",
                                  "outlet": "wbez.org", "url": "javascript:alert(1)", "labeler": "human"}])
    page = render(ans, gaz).get_root().render()
    assert "<img src=x" not in page and "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "<script>x()</script>" not in page and "&lt;script&gt;x()&lt;/script&gt;" in page
    assert "javascript:alert" not in page and "&lt;b&gt;ERs full&lt;/b&gt;" in page
    assert "cluster of rising counties" in page and 'class="tick"' in page


def test_map_app_answers_over_http_and_refuses_foreign_requests(world):
    from src.spatial.app import MapApp, make_server
    values, shapes, _ = world
    app = MapApp("rules", values=values, shapes=shapes, source="test", permutations=99,
                 county_week="nope.parquet", predictions="nope.csv", with_news=False)
    server = make_server(app, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None, headers=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        req = urllib.request.Request(base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    try:
        assert json.loads(call("/api/status")[1])["parser"] == "rules"
        status, body = call("/api/ask", {"question": "flu hot spots on Dec 27 2025"},
                            {"Content-Type": "application/json"})
        answer = json.loads(body)
        assert status == 200 and answer["week_end"] == "2025-12-27" and answer["read_as"]["parser"] == "rules"
        status, page = call(f"/map/{answer['id']}")
        assert status == 200 and b"spark-panel" in page
        assert call("/api/ask", b'{"question": "x"}', {"Content-Type": "text/plain"})[0] == 415
        assert call("/api/status", headers={"Host": "evil.example"})[0] == 403
        assert call("/map/999")[0] == 404
        assert "error" in json.loads(call("/api/ask", {"question": "  "},
                                          {"Content-Type": "application/json"})[1])
    finally:
        server.shutdown()
        server.server_close()
