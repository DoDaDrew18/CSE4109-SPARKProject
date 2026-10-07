"""Offline tests for label validation, quote grounding, and LLM response parsing.

Grounding is the main defense against made-up labels, so most tests here try
to sneak a bad quote past it: reworded numbers, flipped directions, quotes
that never name the county, one-word "quotes". The LLM is never called; a
fake client returns canned responses, including malformed ones.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.news import label as L
from src.news.ground import find_quote, ground_row, names_county, normalize
from src.news.schema import (LabelFormatError, human_consensus, load_human_labels,
                             validate_label)
from src.study import STUDY_FIPS

ARTICLE = (
    "Flu cases climb in St. Louis County\n"
    "Flu cases in St. Louis County have doubled in two weeks, the county’s health "
    "department said Monday — and officials expect more.\n\n"
    "“We are seeing 412 positive tests this week,” said Dr. Faisal Khan. Emergency "
    "rooms at Mercy hospital in Creve Coeur are busy but not on diversion.\n\n"
    "Nationally, the CDC reported rising flu activity in 30 states."
)


# ---------------------------------------------------------------- grounding

def test_exact_quote_survives_curly_quotes_dashes_case_and_spacing():
    q = 'flu cases in st. louis county have doubled in two weeks, the county\'s health department said monday - and officials expect more.'
    m = find_quote(q, ARTICLE)
    assert m.found and m.method == "exact"
    m = find_quote('"We are seeing 412 positive tests\n   this week," said Dr. Faisal Khan.', ARTICLE)
    assert m.found


def test_dropped_comma_is_nopunct_match():
    m = find_quote("Flu cases in St. Louis County have doubled in two weeks the county's health department said",
                   ARTICLE)
    assert m.found and m.method == "nopunct"


def test_ellipsis_joins_parts_of_same_passage():
    m = find_quote("Flu cases in St. Louis County have doubled ... officials expect more", ARTICLE)
    assert m.found and m.method == "ellipsis"
    m = find_quote("Flu cases in St. Louis County have doubled … the CDC reported rising flu", ARTICLE)
    assert m.found  # both parts exist within the span
    m = find_quote("Flu cases in St. Louis County have doubled ... hospitals are on diversion", ARTICLE)
    assert not m.found


def test_fuzzy_allows_one_trivial_word_change():
    m = find_quote("Emergency rooms at Mercy hospital in Creve Coeur are very busy but not on diversion",
                   ARTICLE)
    assert m.found and m.method == "fuzzy" and m.score >= 0.95


def test_fuzzy_rejects_changed_numbers_and_flipped_direction():
    assert not find_quote("We are seeing 512 positive tests this week, said Dr. Faisal Khan.", ARTICLE).found
    assert not find_quote("Flu cases in St. Louis County have tripled in two weeks, the county's health "
                          "department said Monday", ARTICLE).found
    assert not find_quote("Emergency rooms at Mercy hospital in Creve Coeur are busy and now on diversion",
                          ARTICLE).found


def test_short_missing_and_invented_quotes_fail():
    assert find_quote("flu cases", ARTICLE).method == "too_short"
    assert find_quote("", ARTICLE).method == "missing"
    assert find_quote(None, ARTICLE).method == "missing"
    assert not find_quote("Hospitals across St. Louis County are overwhelmed with flu patients", ARTICLE).found


def test_normalize_unifies_unicode_punctuation():
    assert normalize("“Don’t” — wait…  OK") == "\"don't\" - wait... ok"


def test_every_study_county_has_aliases_and_place_check_works():
    for f in STUDY_FIPS:
        assert names_county(f"Officials in {'Chicago' if f == '17031' else 'X'} spoke", f) == (f == "17031")
    assert names_county("Flu cases in St. Louis County have doubled", "29189")
    assert names_county("ERs in Creve Coeur are busy", "29189")
    assert not names_county("Flu cases have doubled in two weeks", "29189")
    assert names_county("Evanston schools report absences", "17031")
    assert not names_county("Evanston schools report absences", "29189")


def llm_row(**kw):
    base = {"article_id": "a1", "relevant": True, "fips": "29189", "illness": "flu", "concern": 2,
            "event_week": None,
            "fips_quote": "Flu cases in St. Louis County have doubled in two weeks",
            "illness_quote": "Flu cases in St. Louis County have doubled in two weeks",
            "concern_quote": "Flu cases in St. Louis County have doubled in two weeks",
            "event_quote": None}
    base.update(kw)
    return base


def test_ground_row_keeps_supported_labels():
    out = ground_row(llm_row(), ARTICLE)
    assert (out["fips"], out["illness"], out["concern"], out["relevant"]) == ("29189", "flu", 2, True)
    assert out["grounded_fips"] and out["fips_place_ok"] and out["grounded_concern"]


def test_ground_row_nulls_fabricated_concern_and_keeps_raw():
    out = ground_row(llm_row(concern=3, concern_quote="Hospitals in St. Louis County are on diversion"), ARTICLE)
    assert out["concern"] is None and out["grounded_concern"] is False
    assert out["concern_raw"] == 3
    assert out["fips"] == "29189"          # other fields unaffected


def test_county_quote_that_names_no_place_is_nulled():
    out = ground_row(llm_row(fips="47157", fips_quote="the county's health department said Monday"),
                     ARTICLE)
    assert out["fips"] is None and out["fips_place_ok"] is False and out["grounded_fips"] is False
    lenient = ground_row(llm_row(fips="47157", fips_quote="the county's health department said Monday"),
                         ARTICLE, strict_place=False)
    assert lenient["fips"] == "47157" and lenient["fips_place_ok"] is False


def test_relevant_needs_some_grounded_quote_but_negatives_need_none():
    out = ground_row(llm_row(fips_quote="made up words about memphis", illness_quote="made up words here",
                             concern_quote="made up words here too"), ARTICLE)
    assert out["relevant"] is None and out["grounded_relevant"] is False
    neg = ground_row({"article_id": "a2", "relevant": False, "fips": "other", "illness": "none",
                      "concern": 0, "event_week": None}, ARTICLE)
    assert neg["relevant"] is False and neg["concern"] == 0 and neg["fips"] == "other"
    assert all(neg[f"grounded_{f}"] for f in ("fips", "illness", "concern", "event_week", "relevant"))


# ---------------------------------------------------------------- schema

def test_validate_label_normalizes_messy_input():
    clean, errors = validate_label({"article_id": " a1 ", "relevant": "Y", "fips": "17031.0",
                                    "illness": "RSV, Flu", "concern": "2.0", "event_week": "2026-01-07"})
    assert errors == []
    assert clean["relevant"] is True and clean["fips"] == "17031"
    assert clean["illness"] == "flu|rsv" and clean["concern"] == 2
    assert clean["event_week"] == pd.Timestamp("2026-01-10")   # Saturday ending that week


def test_validate_label_cross_field_rules():
    _, errors = validate_label({"article_id": "a", "relevant": "y", "fips": "other", "illness": "flu",
                                "concern": 1})
    assert any("study county" in e for e in errors)
    _, errors = validate_label({"article_id": "a", "relevant": "n", "fips": "", "illness": "none",
                                "concern": 2})
    assert any("must be 0" in e for e in errors)
    _, errors = validate_label({"article_id": "a", "relevant": "maybe", "fips": "99999", "illness": "flu|none",
                                "concern": 5})
    assert len(errors) == 4


def test_template_loads_with_examples_dropped():
    assert load_human_labels("labels/human_template.csv").empty


def test_human_csv_errors_list_every_bad_line(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("article_id,labeler,relevant,fips,illness,concern,event_week,quote\n"
                 "a1,human_a,y,29189,flu,2,,q\n"
                 "a2,human_c,y,29189,flu,2,,q\n"
                 "a3,human_a,y,29189,flu,9,,q\n")
    with pytest.raises(LabelFormatError) as exc:
        load_human_labels(p)
    assert "line 3" in str(exc.value) and "line 4" in str(exc.value)


def test_consensus_prefers_adjudication_and_nulls_disagreement():
    df = pd.DataFrame([
        {"article_id": "a", "labeler": "human_a", "relevant": True, "fips": "29189", "illness": "flu",
         "concern": 2, "event_week": None},
        {"article_id": "a", "labeler": "human_b", "relevant": True, "fips": "29189", "illness": "flu",
         "concern": 3, "event_week": None},
        {"article_id": "b", "labeler": "human_a", "relevant": True, "fips": "29510", "illness": "flu",
         "concern": 1, "event_week": None},
        {"article_id": "b", "labeler": "human_b", "relevant": True, "fips": "29189", "illness": "flu",
         "concern": 1, "event_week": None},
        {"article_id": "b", "labeler": "adjudicated", "relevant": True, "fips": "29510", "illness": "flu",
         "concern": 1, "event_week": None},
    ])
    c = human_consensus(df).set_index("article_id")
    assert c.loc["a", "fips"] == "29189" and pd.isna(c.loc["a", "concern"])
    assert c.loc["b", "fips"] == "29510"


def test_guide_covers_every_study_county():
    guide = L.GUIDE_PATH.read_text()
    assert all(f in guide for f in STUDY_FIPS)


# ---------------------------------------------------------------- LLM (mocked)

def response(text, stop="end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)],
                           stop_reason=stop, model=L.MODEL_ID,
                           usage=SimpleNamespace(input_tokens=100, cache_read_input_tokens=3000,
                                                 output_tokens=50))


GOOD = {"relevant": True, "fips": "29189", "fips_quote": "Flu cases in St. Louis County have doubled",
        "illness": ["flu"], "illness_quote": "Flu cases in St. Louis County have doubled",
        "concern": 2, "concern_quote": "Flu cases in St. Louis County have doubled",
        "event_week": "", "event_quote": ""}


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_parse_good_response():
    label, err = L.parse_response(response(json.dumps(GOOD)))
    assert err is None
    assert label["fips"] == "29189" and label["illness"] == "flu" and label["concern"] == 2
    assert label["event_week"] is None and label["event_quote"] is None


@pytest.mark.parametrize("text,stop,expect", [
    ('{"relevant": true, "fips": "291', "max_tokens", "truncated"),
    ("Sure! Here is the label: relevant", "end_turn", "invalid JSON"),
    ("[1, 2]", "end_turn", "expected a JSON object"),
    ("", "end_turn", "no text"),
    ("", "refusal", "refusal"),
])
def test_parse_malformed_responses_never_raise(text, stop, expect):
    label, err = L.parse_response(response(text, stop))
    assert expect in err
    assert all(label[k] is None for k in ("relevant", "fips", "illness", "concern"))


def test_parse_schema_violations_null_only_bad_fields():
    bad = {**GOOD, "fips": "12345", "concern": 7}
    label, err = L.parse_response(response(json.dumps(bad)))
    assert label["fips"] is None and label["concern"] is None
    assert label["illness"] == "flu" and "fips" in err and "concern" in err


def test_request_caches_guide_and_uses_structured_output():
    system = L.system_prompt()
    req = L.build_request({"article_id": "a1", "title": "T", "published": "2026-01-05"}, "ZQXBODY", system)
    assert req["model"] == L.MODEL_ID
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["output_config"]["format"]["schema"] == L.OUTPUT_SCHEMA
    assert "temperature" not in req and "tool_choice" not in req   # both rejected by Opus 5.5
    assert "ZQXBODY" in req["messages"][0]["content"] and "ZQXBODY" not in req["system"][0]["text"]
    assert L.prompt_hash(system) == L.prompt_hash(L.system_prompt())
    assert L.prompt_hash(system) != L.prompt_hash(system + " ")


def test_label_corpus_runs_twice_records_provenance_and_resumes(tmp_path):
    manifest = pd.DataFrame([{"article_id": "a1", "url": "u", "outlet": "KSDK",
                              "published": "2026-01-05", "title": "Flu cases climb"}])
    out = tmp_path / "llm.parquet"
    client = FakeClient([response(json.dumps(GOOD)), response("not json")])
    df = L.label_corpus(manifest, {"a1": ARTICLE}, client, runs=2, out=out, progress=lambda *_: None)
    assert list(df["run"]) == [1, 2]
    assert df.loc[0, "concern"] == 2 and pd.isna(df.loc[1, "concern"])
    assert df["prompt_hash"].nunique() == 1 and (df["model"] == L.MODEL_ID).all()
    assert df["labeled_at"].notna().all() and df.loc[0, "cache_read_tokens"] == 3000
    assert pd.read_parquet(out).shape[0] == 2

    # re-run: the good row is skipped, only the failed run-2 row is retried
    client2 = FakeClient([response(json.dumps(GOOD))])
    df2 = L.label_corpus(manifest, {"a1": ARTICLE}, client2, runs=2, out=out, progress=lambda *_: None)
    assert len(client2.calls) == 1
    assert sorted(df2["run"]) == [1, 2] and df2["error"].isna().all()


def test_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(L.MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        L.make_client()
