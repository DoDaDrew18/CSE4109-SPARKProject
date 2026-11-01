"""Offline tests for the hand-labeling app: the labeling order, the checks on
save, labeler independence, adjudication, and the local HTTP layer.

The properties that matter most for the agreement report are tested
adversarially: a labeler must never see another labeler's answers, a quote
that is not in the article must not be saved, and the double-labeled set must
not move when the catalog grows.
"""

import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

from src.news import label_app as L
from src.news.collect import CATALOG_COLUMNS
from src.news.schema import LabelFormatError, human_label_files, load_human_labels

TEXT = ("Flu cases climb in St. Louis County\n"
        "Flu cases in St. Louis County have doubled in two weeks, the county health "
        "department said Monday.\n\n"
        "Emergency rooms at Mercy hospital in Creve Coeur are busy but not on diversion.\n\n"
        "Officials expect more cases before the holidays.")


def catalog(n_per=4, metros=("stl", "kc", "chi", "ind", "mem", "lou"),
            strata=("respiratory", "quiet", "hard_negative", "")):
    rows, i = [], 0
    for metro in metros:
        for stratum in strata:
            for _ in range(n_per):
                i += 1
                rows.append({c: "" for c in CATALOG_COLUMNS} | {
                    "article_id": f"{i:012x}", "url": f"https://example.com/{i}", "outlet": "ksdk.com",
                    "published": "2025-12-23", "metro": metro, "title": f"Story {i}",
                    "found_via": "gdelt" if not stratum else "search", "fetched_ok": "True",
                    "n_words": "300", "stratum": stratum})
    return pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))


# ---------------------------------------------------------------- assignments

def test_assignments_are_deterministic_stratified_and_order_independent():
    cat = catalog()
    cat.loc[0, "fetched_ok"] = "False"                  # paywall / 403: not labelable
    cat.loc[1, "dup_of"] = cat.loc[2, "article_id"]     # syndicated copy: label the original once
    a = L.make_assignments(cat, n_double=12, seed=1)
    b = L.make_assignments(cat.sample(frac=1, random_state=7), n_double=12, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == len(cat) - 2 and a["article_id"].is_unique
    assert cat.loc[0, "article_id"] not in set(a["article_id"])
    assert cat.loc[1, "article_id"] not in set(a["article_id"])
    assert list(a["position"]) == list(range(1, len(a) + 1))
    assert a["double"].sum() == 12 and a["double"].head(12).all()      # double set comes first

    strata = cat.set_index("article_id")["stratum"].replace("", "gdelt")
    picked = strata[a.loc[a["double"], "article_id"]]
    assert set(picked) == {"respiratory", "quiet", "hard_negative", "gdelt"}


def test_double_set_keeps_each_metros_share():
    """St. Louis has 3x the articles of each other metro (as in the real
    catalog); the double set must not dilute it to one-sixth."""
    cat = pd.concat([catalog(n_per=6, metros=("stl",)),
                     catalog(n_per=2, metros=("kc", "chi", "ind", "mem", "lou"))], ignore_index=True)
    cat["article_id"] = [f"{i:012x}" for i in range(1, len(cat) + 1)]
    for seed in range(10):
        a = L.make_assignments(cat, n_double=16, seed=seed)
        metros = cat.set_index("article_id")["metro"][a.loc[a["double"], "article_id"]]
        stl = int((metros == "stl").sum())
        assert 4 <= stl <= 8              # share 24/64 of 16 = 6, within one per stratum (4 strata)
        assert metros.value_counts().max() == stl


def test_a_new_seed_changes_the_order():
    cat = catalog()
    assert list(L.make_assignments(cat, seed=1)["article_id"]) != list(L.make_assignments(cat, seed=2)["article_id"])


def test_reassigning_appends_new_articles_and_never_moves_the_double_set():
    small = catalog(n_per=2)
    first = L.make_assignments(small, n_double=8, seed=3)
    grown = pd.concat([small, catalog(n_per=3).iloc[len(small):]], ignore_index=True)
    again = L.make_assignments(grown, n_double=8, seed=3, existing=first)
    pd.testing.assert_frame_equal(again.head(len(first)), first)
    tail = again.iloc[len(first):]
    assert len(tail) == len(grown) - len(small) and not tail["double"].any()
    assert list(again["position"]) == list(range(1, len(again) + 1))


def test_allocation_is_proportional_with_one_per_stratum():
    alloc = L._allocate({"respiratory": 89, "gdelt": 48, "hard_negative": 33, "quiet": 26}, 30)
    assert sum(alloc.values()) == 30 and min(alloc.values()) >= 1
    assert alloc["respiratory"] > alloc["gdelt"] > alloc["quiet"] - 1
    assert L._allocate({"a": 50, "b": 1}, 10) == {"a": 9, "b": 1}
    assert L._allocate({"a": 2, "b": 1}, 10) == {"a": 2, "b": 1}       # more asked than exist
    assert sum(L._allocate({"a": 5, "b": 5, "c": 5}, 2).values()) == 2  # fewer than strata


# ---------------------------------------------------------------- checks on save

def form(**kw):
    base = {"article_id": "000000000001", "relevant": "y", "fips": "29189", "illness": ["flu"],
            "concern": "2", "event_week": "", "notes": "",
            "quote": "Flu cases in St. Louis County have doubled in two weeks", "seconds": 42}
    return base | kw


def test_complete_relevant_label_is_normalized():
    row, errors, warnings = L.check_label(form(illness=["rsv", "flu"], event_week="2025-12-24"),
                                          TEXT, "human_a")
    assert errors == [] and warnings == []
    assert (row["relevant"], row["fips"], row["illness"], row["concern"]) == ("y", "29189", "flu|rsv", "2")
    assert row["event_week"] == "2025-12-27"             # snapped to the Saturday ending the week
    assert row["labeler"] == "human_a" and row["seconds"] == "42" and row["labeled_at"]


@pytest.mark.parametrize("change,message", [
    ({"quote": ""}, "needs a quote"),
    ({"fips": ""}, "study county"),
    ({"illness": []}, "illness"),
    ({"concern": ""}, "concern"),
    ({"relevant": ""}, "relevant"),
    ({"fips": "unclear"}, "relevant needs a study county"),
    ({"quote": "Hospitals in St. Louis County are on diversion tonight"}, "not found"),
    ({"quote": "flu cases"}, "4+ words"),
])
def test_incomplete_or_ungrounded_labels_are_refused(change, message):
    _, errors, _ = L.check_label(form(**change), TEXT, "human_a")
    assert any(message in e for e in errors), errors


def test_quote_that_does_not_name_the_county_is_only_a_warning():
    row, errors, warnings = L.check_label(
        form(quote="Officials expect more cases before the holidays"), TEXT, "human_a")
    assert errors == [] and "does not name" in warnings[0]


def test_not_relevant_gets_the_guides_defaults_and_needs_no_quote():
    row, errors, _ = L.check_label(form(relevant="n", fips="other", illness=[], concern="", quote=""),
                                   TEXT, "human_b")
    assert errors == []
    assert (row["relevant"], row["illness"], row["concern"]) == ("n", "none", "0")
    _, errors, _ = L.check_label(form(relevant="n", concern="2", quote=""), TEXT, "human_b")
    assert any("must be 0" in e for e in errors)


def test_save_upserts_and_schema_reads_the_directory(tmp_path):
    root = tmp_path / "human"
    for aid, concern in (("000000000001", "2"), ("000000000002", "1"), ("000000000001", "3")):
        row, errors, _ = L.check_label(form(article_id=aid, concern=concern), TEXT, "human_a")
        assert not errors
        L.save_label(row, root)
    saved = L.read_labels("human_a", root)
    assert list(saved["article_id"]) == ["000000000001", "000000000002"]
    assert saved.set_index("article_id").loc["000000000001", "concern"] == "3"   # edit replaced, not appended

    row, _, _ = L.check_label(form(article_id="000000000001", concern="2"), TEXT, "human_b")
    L.save_label(row, root)
    assert [f.name for f in human_label_files(root)] == ["human_a.csv", "human_b.csv"]
    labels = load_human_labels(root)
    assert len(labels) == 3 and set(labels["labeler"]) == {"human_a", "human_b"}
    assert labels["concern"].dtype == "Int64"


def test_load_human_labels_reports_bad_rows_with_their_file(tmp_path):
    good = tmp_path / "human_a.csv"
    pd.DataFrame([form() | {"labeler": "human_a", "illness": "flu"}]).to_csv(good, index=False)
    bad = tmp_path / "human_b.csv"
    pd.DataFrame([form() | {"labeler": "human_b", "illness": "flu", "concern": "9"}]).to_csv(bad, index=False)
    with pytest.raises(LabelFormatError) as exc:
        load_human_labels(tmp_path)
    assert "human_b.csv line 2" in str(exc.value)
    with pytest.raises(FileNotFoundError, match="label_app"):
        load_human_labels(tmp_path / "nothing-here")


def test_disagreements_count_only_what_the_report_scores():
    a = {"article_id": "x", "relevant": "n", "fips": "", "illness": "none", "concern": "0"}
    assert L.disagreements(a, a | {"fips": "other"}) == []          # county is moot when irrelevant
    b = {"article_id": "x", "relevant": "y", "fips": "29189", "illness": "flu", "concern": "2"}
    assert L.disagreements(b, b | {"concern": "3"}) == ["concern"]
    assert L.disagreements(b, b | {"illness": "rsv|flu"}) == ["illness"]
    assert L.disagreements(a, b)[0] == "relevant"
    assert L.disagreements(b, b | {"event_week": "2025-12-20"}) == ["event_week"]


# ---------------------------------------------------------------- the app

@pytest.fixture
def repo(tmp_path):
    """A repo-shaped folder: catalog, assignments, article text for 4 of 5 articles."""
    cat = catalog(n_per=1, metros=("stl",), strata=("respiratory", "quiet", "hard_negative", ""))
    extra = cat.iloc[[0]].assign(article_id="00000000000f", stratum="respiratory")
    cat = pd.concat([cat, extra], ignore_index=True)
    (tmp_path / "labels").mkdir()
    cat.to_csv(tmp_path / "labels/articles.csv", index=False)
    L.make_assignments(cat, n_double=3, seed=0).to_csv(tmp_path / "labels/assignments.csv", index=False)
    text_dir = tmp_path / "raw/news/text"
    text_dir.mkdir(parents=True)
    for aid in cat["article_id"][:-1]:
        (text_dir / f"{aid}.txt").write_text(TEXT, encoding="utf-8")
    return tmp_path


def with_text(app):
    return [i["article_id"] for i in app.state()["items"] if i["has_text"]]


def test_app_never_shows_one_labeler_another_labelers_answers(repo):
    alice, bob = L.LabelApp("human_a", repo), L.LabelApp("human_b", repo)
    aid = with_text(alice)[0]
    assert alice.save(form(article_id=aid, concern="3"))["ok"]
    view = bob.article(aid)
    assert view["label"] is None and "others" not in view
    assert alice.article(aid)["label"]["concern"] == "3"
    page = json.dumps(bob.state()) + json.dumps(view)
    for hidden in ("hard_negative", "quiet", "found_via", "fips_guess", "stratum"):
        assert hidden not in page                        # why an article was collected stays hidden


def test_app_queue_progress_next_and_missing_text(repo):
    app = L.LabelApp("human_a", repo)
    state = app.state()
    assert state["counts"] == {"done": 0, "total": 5, "double_done": 0, "double_total": 3, "with_text": 4}
    items = state["items"]
    no_text = next(i["article_id"] for i in items if not i["has_text"])
    assert app.article(no_text)["text"] is None
    assert "fetch" in app.save(form(article_id=no_text))["errors"][0]
    first = with_text(app)[0]
    result = app.save(form(article_id=first))
    assert result["ok"] and result["next"] not in (first, no_text, None)
    assert app.state()["counts"]["done"] == 1
    assert app.article("../../etc/passwd") is None and app.article("ffffffffffff") is None
    assert not app.save(form(article_id="ffffffffffff"))["ok"]


def test_adjudication_queue_holds_only_real_disagreements(repo):
    alice, bob = L.LabelApp("human_a", repo), L.LabelApp("human_b", repo)
    double = [i["article_id"] for i in alice.state()["items"] if i["double"] and i["has_text"]]
    agree, split = double[0], double[1]
    for app, concern in ((alice, "2"), (bob, "2")):
        assert app.save(form(article_id=agree, concern=concern))["ok"]
    for app, concern in ((alice, "2"), (bob, "3")):
        assert app.save(form(article_id=split, concern=concern))["ok"]
    judge = L.LabelApp("adjudicated", repo)
    assert [i["article_id"] for i in judge.state()["items"]] == [split]
    view = judge.article(split)
    assert view["others"]["differ"] == ["concern"]
    assert view["others"]["human_a"]["concern"] == "2" and view["others"]["human_b"]["concern"] == "3"
    assert judge.article(agree) is None
    assert judge.save(form(article_id=split, concern="3"))["ok"]
    assert [i["article_id"] for i in judge.state()["items"]] == [split]   # still reviewable
    lines = L.status(repo)
    assert any("disagree on 1, 0 not yet adjudicated" in line for line in lines)


def test_check_endpoint_reports_match_and_place(repo):
    app = L.LabelApp("human_a", repo)
    aid = with_text(app)[0]
    ok = app.check({"article_id": aid, "fips": "29189",
                    "quote": "flu cases in st. louis county have doubled in two weeks"})
    assert ok["found"] and ok["method"] == "exact" and ok["names_county"] is True
    assert app.check({"article_id": aid, "fips": "29510",
                      "quote": "Flu cases in St. Louis County have doubled"})["names_county"] is False
    assert app.check({"article_id": aid, "quote": "made up words about a surge"})["found"] is False


def test_http_layer_serves_saves_and_refuses_foreign_requests(repo):
    server = L.make_server(L.LabelApp("human_a", repo), port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def call(path, body=None, headers=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        req = urllib.request.Request(base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    try:
        status, page = call("/")
        assert status == 200 and b"SPARK" in page
        status, guide = call("/guide")
        assert status == 200 and b"Grounding rules" in guide
        state = json.loads(call("/api/state")[1])
        aid = next(i["article_id"] for i in state["items"] if i["has_text"])
        status, body = call("/api/save", form(article_id=aid), {"Content-Type": "application/json"})
        assert status == 200 and json.loads(body)["ok"]
        assert json.loads(call(f"/api/article?id={aid}")[1])["label"]["concern"] == "2"
        # a cross-site "simple" POST (no JSON content type) never reaches the save handler
        assert call("/api/save", json.dumps(form(article_id=aid)).encode(),
                    {"Content-Type": "text/plain"})[0] == 415
        assert call("/api/state", headers={"Host": "evil.example:80"})[0] == 403    # DNS rebinding
        assert call("/api/article?id=..%2F..%2Fsecret")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
