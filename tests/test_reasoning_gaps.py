"""
Offline checks for the question SHAPES the Ask parser used to get wrong
(2026-10-25 review: 17 of 40 realistic pharmacology questions misread) — a
question about two medicines taken together, about a drug CLASS, about a
CONCEPT with no medicine in it, a why-question about one topic, several
questions in one message. Each case names what it used to do.

The end-to-end answers for the same questions are checked in
test_answer_quality.py (network).
"""
import json
import os
import re

import pytest

import chat_router as cr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(q, context=None):
    return cr.parse_query(q, context=context)


# --------------------------------------------------------------- safety first
@pytest.mark.parametrize("q", [
    # REGRESSION: answered 'Yes — approved for Headache' (each drug's uses)
    "can i take ibuprofen for my headache if i'm on warfarin",
    "i'm on warfarin, can i take ibuprofen",
    "is it safe to take ibuprofen while taking warfarin",
    "can i take ibuprofen with warfarin for my back pain",
    "can i take ibuprofen and paracetamol for a headache",
])
def test_two_medicines_used_together_is_an_interaction_question(q):
    p = _p(q)
    assert p["mode"] == "interaction", (p["mode"], p["intents"])
    assert len(p["entities"]) == 2


def test_two_medicines_side_by_side_stay_separate_answers():
    assert _p("metformin and gliclazide side effects")["mode"] == "multi"


# --------------------------------------------------------------- drug classes
@pytest.mark.parametrize("q,wiki,target_word", [
    # REGRESSION: all four listed the class's members instead of answering
    ("how do ssris work", "Selective serotonin reuptake inhibitor", None),
    ("why do ace inhibitors cause a dry cough", "ACE inhibitor", "cough"),
    ("why do statins cause muscle pain", "Statin", "muscle"),
    ("which beta blockers are cardioselective", "Beta blocker", None),
    ("do statins interact with grapefruit", "Statin", None),
])
def test_questions_about_a_class_are_answered_about_the_class(q, wiki, target_word):
    p = _p(q)
    assert p["mode"] == "class_topic", p["mode"]
    assert p["class_mentions"][0]["wiki"] == wiki
    if target_word:
        assert target_word in p.get("relation_target", "")


def test_several_classes_named_are_all_classes_not_unknown_chemicals():
    # REGRESSION: 'diuretics' was looked up in PubChem as a chemical
    p = _p("why should you not take nsaids with ace inhibitors and diuretics")
    assert p["mode"] == "class_topic" and not p["entities"]
    assert [m["wiki"] for m in p["class_mentions"]] == [
        "Nonsteroidal anti-inflammatory drug", "ACE inhibitor", "Diuretic"]
    assert "diuretics" in p["relation_terms"]


@pytest.mark.parametrize("q", ["list some ace inhibitors", "what are some ssris",
                               "examples of benzodiazepines", "what's the strongest painkiller"])
def test_asking_for_members_still_lists_them(q):
    assert _p(q)["mode"] == "class_list"


def test_every_class_alias_names_an_article():
    for pat, subs, wiki in cr.CLASS_ALIASES:
        assert wiki and subs, pat


# ------------------------------------------------------------------- concepts
@pytest.mark.parametrize("q,concept", [
    # REGRESSION: 'Which medicine do you mean?' (no drug in the question)
    ("what is first pass metabolism", "first_pass"),
    ("what are the main cyp3a4 inhibitors", "cyp3a4"),
    # REGRESSION: 'g6pd deficiency' / 'serotonin syndrome' looked up as chemicals
    ("what drugs should be avoided in g6pd deficiency", "g6pd"),
    ("what is serotonin syndrome", "serotonin_syndrome"),
    ("which antibiotics are safe in penicillin allergy", "penicillin_allergy"),
    ("why does grapefruit juice affect so many drugs", "grapefruit"),
])
def test_concept_questions_without_a_medicine(q, concept):
    p = _p(q)
    assert p["mode"] == "concept", p["mode"]
    assert p["concepts"][0]["id"] == concept


def test_two_concepts_are_both_kept():
    p = _p("explain the difference between agonists and antagonists")
    assert [c["id"] for c in p["concepts"]] == ["agonist", "antagonist"]
    assert p["concept_terms"] == []          # just the definitions


def test_what_is_asked_about_a_concept_is_kept():
    assert set(_p("what drugs should be avoided in g6pd deficiency")["concept_terms"]) >= {
        "drugs", "avoided"}
    assert _p("what is first pass metabolism")["concept_terms"] == []


@pytest.mark.parametrize("q,concept", [
    # REGRESSION: 'therapeutic' made it a USES question → "couldn't find 'index'"
    ("what is the therapeutic index of digoxin", "therapeutic_index"),
    ("can you explain zero order kinetics using phenytoin", "zero_order"),
    ("is nitrofurantoin safe in g6pd deficiency", "g6pd"),
])
def test_concept_asked_about_a_named_drug(q, concept):
    p = _p(q)
    assert p["mode"] == "answer" and p["entities"]
    assert p["intents"][0] == "concept" and p["concepts"][0]["id"] == concept


@pytest.mark.parametrize("q,topic", [
    ("half-life of diazepam", "pk"),                  # covered by the PK answer
    ("is ibuprofen metabolised by cyp3a4", "pk"),     # defer_if: a metabolism question
    ("is codeine a prodrug", "pk"),
])
def test_concepts_a_topic_already_answers_stay_with_that_topic(q, topic):
    assert _p(q)["intents"][0] == topic


@pytest.mark.parametrize("q", ["what is hypertension", "what is hashimoto's thyroiditis"])
def test_unlisted_medical_terms_are_looked_up(q):
    p = _p(q)
    assert p["mode"] == "concept" and p["concept_query"]


@pytest.mark.parametrize("q", ["what's the weather like today", "what time is it"])
def test_off_topic_questions_are_not_looked_up(q):
    assert _p(q)["mode"] == "unknown"


def test_concepts_file_is_valid():
    with open(os.path.join(ROOT, "concepts.json"), encoding="utf-8") as fh:
        data = json.load(fh)["concepts"]
    ids = [c["id"] for c in data]
    assert len(ids) == len(set(ids)), "duplicate concept ids"
    topics = set(cr.INTENT_TITLES) | {"concept"}
    for c in data:
        assert c["wiki"] and c["patterns"] and c["label"], c["id"]
        for pat in c["patterns"]:
            re.compile(pat)
        assert c.get("with_drug", "concept") in topics, c["id"]
        if c.get("defer_if"):
            re.compile(c["defer_if"])


# -------------------------------------------------------------- why-questions
@pytest.mark.parametrize("q,target,term", [
    # REGRESSION: pasted the interactions section with no reason
    ("why does warfarin have so many interactions", "interactions", "CYP"),
    # REGRESSION: answered as a general record search
    ("why is amlodipine making my ankles swell", "ankles swell", "edema"),
])
def test_why_questions_about_one_drug_rank_explanations(q, target, term):
    p = _p(q)
    assert p["mode"] == "relation" and p["relation_derived"]
    assert p["relation_target"] == target and term in p["relation_terms"]


def test_why_with_an_effect_does_not_search_for_its_treatment():
    # 'pain' → 'analgesic' is right for 'what helps pain', wrong for 'why does X cause pain'
    terms = _p("why do statins cause muscle pain")["relation_terms"]
    assert not [t for t in terms if t.startswith(("anti", "analges"))]


def test_why_dangerous_stays_a_warnings_question():
    assert _p("why is warfarin dangerous")["mode"] != "relation"


def test_relation_target_stops_at_a_contrast():
    p = _p("why is vancomycin given orally for c diff but iv for other infections")
    assert p["mode"] == "relation" and p["relation_target"] == "c diff"
    assert p["relation_concepts"][0]["id"] == "c_diff"


# ------------------------------------------------------------ smaller fixes
def test_eating_after_a_dose_is_a_dosing_question():
    # REGRESSION: answered with 24 unrelated drug interactions
    assert _p("how long after taking omeprazole can i eat")["intents"][0] == "dosing"


def test_eating_a_food_on_a_drug_is_still_an_interaction():
    assert _p("can i eat grapefruit while on simvastatin")["intents"][0] == "interactions"


def test_long_term_side_effects_are_not_a_dosing_question():
    assert _p("what are the long term side effects of prednisolone")["intents"] == ["side_effects"]


def test_hay_fever_is_not_fever():
    # REGRESSION: listed antipyretics (paracetamol) for hay fever
    p = _p("what's the best antihistamine for hay fever")
    assert p["mode"] == "condition" and p["condition"]["label"] == "allergies & hay fever"


# ------------------------------------------------------- several questions
@pytest.mark.parametrize("q,parts", [
    ("what is serotonin syndrome and which drugs cause it",
     ["what is serotonin syndrome", "which drugs cause it"]),
    ("what is the mechanism of action of semaglutide and why does it cause weight loss",
     ["what is the mechanism of action of semaglutide", "why does it cause weight loss"]),
    ("is it safe? and can I drive?", ["is it safe", "can I drive?"]),
])
def test_several_questions_are_split(q, parts):
    assert cr.split_question(q) == parts


@pytest.mark.parametrize("q", ["how and why does aspirin work",
                               "is ibuprofen and paracetamol together safe",
                               "why do ace inhibitors cause cough but arbs don't"])
def test_one_question_with_and_is_not_split(q):
    assert cr.split_question(q) == [q]


def test_follow_up_about_a_concept_uses_it():
    ctx = {"concepts": [cr.concept_by_id("serotonin_syndrome")], "intents": []}
    p = _p("which drugs cause it", context=ctx)
    assert p["mode"] == "concept" and p["concepts"][0]["id"] == "serotonin_syndrome"
    assert "drugs" in p["concept_terms"]


def test_wikipedia_throttling_is_an_outage_not_a_missing_article(monkeypatch):
    # REGRESSION: a burst of questions got 429s; the empty result was cached as
    # 'no article on Serotonin syndrome' because the breaker never heard of it
    import chem_sources as cs

    class Throttled:
        status_code, headers = 429, {"retry-after": "29"}
    host = cs._host_of(cs.WIKI_API)
    cs._host_state.pop(host, None)
    calls = []
    monkeypatch.setattr(cs.get_session(), "get", lambda *a, **k: calls.append(1) or Throttled())
    try:
        assert cs.wikipedia_sections("Serotonin syndrome") == {}
        assert cs.host_failing(cs.WIKI_API)        # → the app won't cache it
        # the host asked for 29 s: honoured at once, no hammering meanwhile
        assert cs.host_degraded(cs.WIKI_API)
        assert cs.wikipedia_search("serotonin") == [] and len(calls) == 1
    finally:
        cs._host_state.pop(host, None)


def test_rate_limited_requests_are_not_retried():
    # REGRESSION: urllib3 retried 429s (sleeping the Retry-After) even when 429
    # was not in the retry list — 9.8 s per throttled Wikipedia request
    import chem_sources as cs
    retry = cs.get_session().adapters["https://"].max_retries
    assert 429 not in retry.status_forcelist and 503 in retry.status_forcelist
    assert not retry.respect_retry_after_header
    assert not retry.is_retry("GET", 429, has_retry_after=True)


def test_follow_up_about_a_drug_still_uses_the_drug():
    ctx = {"entity": {"kind": "library", "query": "diazepam", "display": "Diazepam"},
           "intents": ["mechanism"]}
    p = _p("what about its half-life?", context=ctx)
    assert p["entities"][0]["query"] == "diazepam" and p["intents"][0] == "pk"
