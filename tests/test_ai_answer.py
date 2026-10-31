"""The Claude layer, with no API calls: the number and citation checks, the shape of an answer,
the conversation loop against a scripted client, spend limits, and what the app shows or falls
back to. tests/eval_ai_questions.py is the live run that spends credit."""
import json
import os
import threading
import time
from types import SimpleNamespace as NS

import pytest
from streamlit.testing.v1 import AppTest

import ai_answer as ai
import chem_sources as cs
import chem_v4 as v
import sg_register as sgr

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chem_v4.py")
# parses as an unknown name, so the app fetches nothing before calling the model
NO_LOOKUP_Q = "what's the capital of france"
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november".split()


def _ev(*texts, question=""):
    ev = ai.Evidence(question)
    for t in texts:
        ev.add(t, "FDA label", "https://example.org")
    return ev


def _raw(headline="**No.**", sources=(), **extra):
    return dict({"kind": "answer", "shape": "", "headline": headline, "headline_sources": list(sources), "safety": [],
                 "sections": [], "chain": [], "chain_sources": [], "table_columns": [], "table_rows": [],
                 "followups": []}, **extra)


def _points(*texts, sources=("S1",), title="", line=0):
    return [{"title": title, "points": [{"text": t, "sources": list(sources), "line": line} for t in texts]}]


def _texts(out):
    return [p["text"] for sec in out["sections"] for p in sec["points"]]


# -------------------------------------------------------------- number checks
@pytest.mark.parametrize("text", ["Take 500 mg", "every 8 hours", "dose 500", "call 200 mg"])
def test_a_number_from_nowhere_is_rejected(text):
    assert ai.claim_ok(text, ["S1"], _ev("Take as directed by your doctor."))[0] is False


def test_a_number_in_the_cited_snippet_passes():
    assert ai.claim_ok("Take 500 mg", ["S1"], _ev("The dose is 500 mg twice daily."))[0] is True


def test_a_number_must_be_in_what_it_cites_not_just_any_snippet():
    ev = _ev("The dose is 500 mg.", "Half-life is 12 hours.")
    assert ai.claim_ok("Half-life is 500 hours", ["S2"], ev)[0] is False


def test_thousands_are_written_either_way():
    ev = _ev("Do not exceed 4000 mg in 24 hours.")
    assert ai.claim_ok("Maximum 4,000 mg in 24 hours", ["S1"], ev)[0] is True


@pytest.mark.parametrize("claim, snippet", [
    ("no more than 4 g in 24 hours", "Do not take more than 4,000 mg in 24 hours."),
    ("no more than 4,000 mg in 24 hours", "The limit is 4 g in 24 hours."),
    ("1 g per dose", "A single dose should not exceed 1000 mg."),
    ("1.5 g daily", "The dose is 1,500 mg daily."),
    ("500 mcg daily", "The dose is 0.5 mg daily."),
    ("0.5 mg daily", "The dose is 500 micrograms daily."),
])
def test_the_same_dose_in_other_units_is_the_same_dose(claim, snippet):
    assert ai.claim_ok(claim, ["S1"], _ev(snippet))[0] is True


def test_a_different_dose_in_other_units_still_fails():
    assert ai.claim_ok("no more than 2 g in 24 hours", ["S1"], _ev("Do not take more than 4,000 mg in 24 hours."))[0] is False


def test_a_number_a_source_writes_as_a_word_counts():
    ev = _ev("Doses should be taken no sooner than four hours apart, and no more than four doses (4000 mg) in 24 hours.")
    assert ai.claim_ok("4,000 mg in 24 hours: 4 hours apart, no more than 4 doses", ["S1"], ev)[0] is True
    assert ai.claim_ok("6 hours apart", ["S1"], ev)[0] is False


def test_being_asked_to_correct_a_number_says_which():
    client = FakeClient(_text(_raw("**Take 900 mg or 500 mg.**", ["S1"])), _text(_raw("**Take 500 mg.**", ["S1"])))
    res = _ask(client, prefetched=[{"text": "The dose is 500 mg.", "src": "FDA label"}])
    assert res["status"] == "ok" and "it states 900," in client.requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize("text", ["blocks H1 receptors", "inhibits CYP3A4", "type 2 diabetes", "a phase 3 trial",
                                  "5-HT reuptake", "D2 antagonist", "GLP-1 agonist", "5-HT2A antagonist", "β1 blocker",
                                  "omega-3 fatty acids", "a 2nd-line option", "CYP 2D6 inhibitor", "5-ASA drugs",
                                  "DPP-4 inhibitors", "COX-2 inhibitors", "beta-1 selective"])
def test_numbers_that_name_things_arent_measures(text):
    assert ai.claim_ok(text, [], _ev())[0] is True


def test_an_answer_with_no_source_may_not_state_a_measure():
    assert ai.claim_ok("The usual dose is 500 mg", [], _ev())[0] is False


def test_the_readers_own_number_and_the_emergency_number_need_no_source():
    ev = _ev("Quick medical attention is critical.", question="my toddler swallowed 10 panadol")
    assert ai.claim_ok("**Call 995 now**: 10 tablets is an emergency.", ["S1"], ev)[0] is True


def test_an_unknown_source_id_counts_as_no_source():
    assert ai.claim_ok("Take 500 mg", ["S9"], _ev("Half-life is 12 hours.")) == (False, [])


# ------------------------------------------------------------- citation checks
def test_a_citation_that_doesnt_say_the_claim_is_taken_away():
    ev = _ev("ACE inhibitors: captopril, enalapril, lisinopril.")
    assert ai.claim_ok("**ACE inhibitors** such as lisinopril: dry cough, raised potassium", ["S1"], ev) == (True, [])


def test_naming_the_drug_isnt_enough_for_a_claim_about_it():
    ev = _ev("Amlodipine, felodipine and nifedipine are calcium channel blockers.")
    assert ai.claim_ok("Calcium channel blockers such as amlodipine: ankle swelling, flushing", ["S1"], ev) == (True, [])


def test_a_citation_that_says_the_claim_stays():
    ev = _ev("Common adverse reactions are ankle swelling and flushing.")
    assert ai.claim_ok("**Amlodipine**: ankle swelling, flushing", ["S1"], ev) == (True, ["S1"])


def test_only_the_citations_that_help_are_shown():
    ev = _ev("Ankle swelling and flushing are common with amlodipine.", "The WHO lists essential medicines worldwide.")
    assert ai.claim_ok("Ankle swelling and flushing with amlodipine", ["S1", "S2"], ev) == (True, ["S1"])


def test_a_number_citing_the_wrong_snippet_finds_the_right_one():
    ev = _ev("Half-life is long.", "The half-life is about 26 hours in adults.")
    assert ai.claim_ok("Half-life about 26 hours", ["S1"], ev) == (True, ["S2"])


def test_wording_alone_never_finds_a_source():
    ev = _ev("Most people need more than one medicine.")
    assert ai.claim_ok("Most people need more than one medicine", [], ev) == (True, [])


def test_spellings_count_as_one_word():
    assert ai.backing("Paracetamol for oedema and tumours", ["acetaminophen for edema and tumors"]) == 1.0


# ---------------------------------------------------------------- verify
def test_a_headline_that_fails_fails_the_answer():
    assert ai.verify(_raw("**Take 900 mg**", ["S1"]), _ev("The dose is 500 mg.")) is None


@pytest.mark.parametrize("headline, kept", [
    ("**Adult maximum is 3200 mg daily** on the label; the usual pain dose is 400 mg every 4 to 6 hours.", "**Adult maximum is 3200 mg daily** on the label."),
    ("**Adult maximum is 3200 mg daily; usual pain dose 400 mg**.", "**Adult maximum is 3200 mg daily**."),
    ("Adult maximum is 3200 mg daily. The usual pain dose is 400 mg.", "Adult maximum is 3200 mg daily."),
])
def test_a_headline_keeps_the_part_that_checks_out(headline, kept):
    out = ai.verify(_raw(headline, ["S1"]), _ev("Do not exceed 3200 mg total daily."))
    assert out["headline"] == kept and out["headline_sources"] == ["S1"] and out["dropped"] == 1


def test_a_source_giving_a_range_in_words_backs_it_in_figures():
    source = "Diazepam has a biphasic half-life of about one to three days and two to seven days for the active metabolite."
    out = ai.verify(_raw("**About 1 to 3 days**; the metabolite lasts 2 to 7 days.", ["S1"]), _ev(source))
    assert out and out["headline_sources"] == ["S1"]


@pytest.mark.parametrize("text, says_one", [("one to three days", True), ("one or two tablets", True), ("one-to-3 hours", True), ("for one week", True),
                                            ("one of the most common effects", False), ("no one should take it", False), ("this one is safer", False)])
def test_one_is_a_number_only_where_it_counts_something(text, says_one):
    assert ("1" in ai._stated_numbers(text)) == says_one


def test_twice_is_a_number():
    assert "2" in ai._stated_numbers("Take it twice daily.")


def test_a_headline_failing_from_the_start_fails_however_the_rest_reads():
    assert ai.verify(_raw("**Take 900 mg**; the maximum is 3200 mg.", ["S1"]), _ev("Do not exceed 3200 mg total daily.")) is None


def test_a_side_point_that_fails_is_only_dropped():
    raw = _raw("**Yes.**", ["S1"], sections=_points("Dose is 900 mg", "Dose is 500 mg"))
    out = ai.verify(raw, _ev("The dose is 500 mg."))
    assert _texts(out) == ["Dose is 500 mg"] and out["dropped"] == 1


def test_a_chain_with_an_unsupported_number_is_dropped():
    out = ai.verify(_raw("**Yes.**", ["S1"], shape="explain", chain=["Blocks 90% of receptors", "Lowers pressure"],
                         chain_sources=["S1"]), _ev("It blocks receptors."))
    assert out["chain"] == [] and out["dropped"] == 1


def test_only_an_explanation_has_a_chain():
    fields = dict(chain=["COX inhibition", "↓ prostaglandins"], chain_sources=["S1"])
    assert ai.verify(_raw("**Yes.**", ["S1"], shape="explain", **fields), _ev("It works."))["chain"]
    assert ai.verify(_raw("**Yes.**", ["S1"], shape="compare", **fields), _ev("It works."))["chain"] == []


def test_written_citations_are_stripped():
    out = ai.verify(_raw("**Yes** [S1, S2].", ["S1"]), _ev("It works."))
    assert out["headline"] == "**Yes**."


@pytest.mark.parametrize("said", ["(general knowledge)", "(from general medical knowledge)"])
def test_a_note_about_general_knowledge_is_stripped(said):
    assert ai.verify(_raw(f"**Yes** {said}"), _ev())["headline"] == "**Yes**"


@pytest.mark.parametrize("headline", [
    "**A PPI plus two antibiotics.** Treatment duration is not stated in the sources.",
    "**A PPI plus two antibiotics.** The evidence doesn't give the duration."])
def test_talk_of_what_the_sources_lack_is_taken_off_an_answer(headline):
    assert ai.verify(_raw(headline), _ev())["headline"] == "**A PPI plus two antibiotics.**"


def test_it_stays_when_its_the_answer_to_what_was_asked():
    assert ai.verify(_raw("The sources don't state its half-life."), _ev())["headline"] == "The sources don't state its half-life."


@pytest.mark.parametrize("headline", ["**No.** Evidence does not support it in children.",
                                      "**No.** There is no evidence of benefit.",
                                      "**No.** The evidence does not support it.",
                                      "**No.** Food sources of vitamin K needn't be avoided."])
def test_a_finding_about_the_drug_isnt_talk_of_the_sources(headline):
    assert ai.verify(_raw(headline), _ev())["headline"] == headline


def test_a_different_number_isnt_a_repeat():
    raw = _raw("**Maximum 4,000 mg daily.**", ["S1"], shape="dose", sections=_points("Maximum 3,000 mg daily"))
    assert _texts(ai.verify(raw, _ev("Maximum 4000 mg daily, or 3000 mg daily in older people."))) == ["Maximum 3,000 mg daily"]


@pytest.mark.parametrize("shape", list(ai.SHAPES))
def test_points_and_warnings_are_capped_by_the_questions_shape(shape):
    raw = _raw("**Yes.**", ["S1"], shape=shape, sections=_points(*[f"Point {w}" for w in WORDS]),
               safety=[{"level": "amber", "text": f"Caution {w}", "sources": ["S1"]} for w in ("oscar", "papa", "quebec")],
               followups=[f"Next {w}?" for w in WORDS])
    out = ai.verify(raw, _ev("It works."))
    assert (len(_texts(out)), len(out["safety"])) == ai.SHAPES[shape] and len(out["followups"]) == ai.MAX_FOLLOWUPS
    assert out["cut"] == len(WORDS) + 3 - len(_texts(out)) - len(out["safety"])


def test_the_first_section_cant_use_up_the_room():
    assert ai._shares([6, 2, 1], 5) == [2, 2, 1]
    assert ai._shares([1, 1], 5) == [1, 1]


def test_a_danger_is_shown_even_when_the_shape_has_no_warnings():
    raw = _raw("**Acetylcysteine.**", ["S1"], shape="fact", safety=[
        {"level": "amber", "text": "A minor caution", "sources": []},
        {"level": "red", "text": "Possible overdose: call 995 now", "sources": []}])
    assert [w["level"] for w in ai.verify(raw, _ev("It works."))["safety"]] == ["red"]


def test_a_point_or_warning_repeating_the_headline_is_removed():
    headline = "**Metformin can rarely cause lactic acidosis.**"
    raw = _raw(headline, shape="side_effects", sections=_points("Lactic acidosis from metformin, rarely", "Vitamin B12 deficiency", sources=()),
               safety=[{"level": "amber", "text": "Lactic acidosis, rarely, from metformin", "sources": []}])
    out = ai.verify(raw, _ev())
    assert _texts(out) == ["Vitamin B12 deficiency"] and out["safety"] == []


def test_a_point_repeating_the_chain_or_table_is_removed():
    raw = _raw("**Yes.**", shape="explain", chain=["Blocks receptors", "Slows heart rate"],
               sections=_points("Blocks receptors, slowing heart rate", "Tiredness is common", sources=()))
    assert _texts(ai.verify(raw, _ev())) == ["Tiredness is common"]


def test_a_comparison_with_a_table_has_no_points():
    raw = _raw("**They differ.**", shape="compare", table_columns=["Feature", "A", "B"],
               table_rows=[{"cells": ["Half-life", "Short", "Long"], "sources": []}],
               sections=_points("Something else entirely", sources=()))
    out = ai.verify(raw, _ev())
    assert _texts(out) == [] and len(out["table_rows"]) == 1


def test_a_comparison_across_a_class_has_points_instead():
    raw = _raw("**Pravastatin.**", shape="compare", sections=_points(*[f"Point {w}" for w in WORDS[:5]], sources=()))
    assert len(_texts(ai.verify(raw, _ev()))) == 5


@pytest.mark.parametrize("empty", ["Not stated in sources", "None named", "n/a", "—"])
def test_a_row_only_one_drug_has_data_for_is_dropped(empty):
    raw = _raw("**They differ.**", ["S1"], table_columns=["Feature", "A", "B"],
               table_rows=[{"cells": ["Half-life", empty, "26 hours"], "sources": ["S1"]},
                           {"cells": ["Class", "SSRI", "SNRI"], "sources": ["S1"]}])
    out = ai.verify(raw, _ev("Half-life 26 hours."))
    assert [r["cells"][0] for r in out["table_rows"]] == ["Class"]


def test_a_row_saying_the_same_for_each_drug_is_dropped():
    raw = _raw("**They differ.**", table_columns=["Feature", "A", "B"],
               table_rows=[{"cells": ["Class", "NSAID", "NSAID."], "sources": []},
                           {"cells": ["Duration", "Short", "Long"], "sources": []}])
    assert [r["cells"][0] for r in ai.verify(raw, _ev())["table_rows"]] == ["Duration"]


def test_a_row_with_the_wrong_number_of_cells_is_dropped():
    raw = _raw("**They differ.**", ["S1"], table_columns=["Feature", "A", "B"],
               table_rows=[{"cells": ["Class", "SSRI"], "sources": ["S1"]}])
    out = ai.verify(raw, _ev("It works."))
    assert out["table_rows"] == [] and out["table_columns"] == []


def test_a_short_answer_has_no_headings_and_a_list_keeps_them():
    sections = _points("Point alpha", title="What happens") + _points("Point bravo", title="What to watch for")
    short = ai.verify(_raw("**Yes.**", ["S1"], shape="can_i", sections=sections), _ev("It works."))
    assert [s["title"] for s in short["sections"]] == [""] and _texts(short) == ["Point alpha", "Point bravo"]
    long = ai.verify(_raw("**Yes.**", ["S1"], shape="drugs_for", sections=sections), _ev("It works."))
    assert [s["title"] for s in long["sections"]] == ["What happens", "What to watch for"]


@pytest.mark.parametrize("kind", ["clarify", "off_topic", "crisis", "refuse"])
def test_other_kinds_carry_nothing_but_their_line(kind):
    out = ai.verify(_raw("A line.", kind=kind, sections=_points("x")), _ev())
    assert out["kind"] == kind and out["sections"] == [] and out["headline"] == "A line."


def test_general_means_the_answer_has_no_source():
    assert ai.verify(_raw("**Agonists activate receptors.**"), _ev())["general"] is True
    assert ai.verify(_raw("**Yes.**", ["S1"]), _ev("It works."))["general"] is False


def test_plain_text_is_what_a_follow_up_reads_back():
    out = ai.verify(_raw("**Yes.**", ["S1"], sections=_points("Point alpha")), _ev("It works."))
    assert ai.plain_text(out) == "**Yes.**\nPoint alpha"


def test_a_long_snippet_is_cut_to_the_sentence_with_the_claims_number():
    long = "Background about the drug. " * 20 + "Avoid alcohol for at least 3 days after therapy. " + "Other matters. " * 40
    cut = ai.excerpt(long, ["Avoid alcohol for 3 days"])
    assert "3 days" in cut and len(cut) < 100 and cut.startswith("…")
    assert ai.excerpt("A short one.", ["anything"]) == "A short one."


# ---------------------------------------------------------- written from an outline
_PLAN = {"shape": "drugs_for", "lines": ["The bottom line", "Line two", "Line three", "Line four"], "warning": ""}


def test_what_the_outline_asked_for_goes_before_what_the_model_added():
    points = [{"text": f"Point {w}", "sources": [], "line": n} for w, n in
              zip(WORDS, [0, 2, 0, 3, 0, 4])]
    out = ai.verify(_raw("**Yes.**", shape="drugs_for", sections=[{"title": "", "points": points}]), _ev(), _PLAN)
    # the outline's three lines, in their places, and room for the first addition
    assert _texts(out) == ["Point alpha", "Point bravo", "Point delta", "Point foxtrot"]


def _planned_answer(lines, *points, **fields):
    plan = dict(_PLAN, lines=["The bottom line"] + list(lines))
    sections = [{"title": "", "points": [{"text": t, "sources": [], "line": n} for t, n in points]}]
    return ai.verify(_raw("**Yes.**", shape="drugs_for", sections=sections, **fields), _ev(), plan)


def test_a_line_split_in_two_cant_push_out_a_line_of_the_outline():
    # as with diabetes: metformin three times over, then insulin cut
    lines = [f"Line {w}" for w in WORDS[:8]]
    points = [("Metformin adjunct to diet", 1), ("Metformin side effects", 8), ("Metformin recommended first", 1)] + \
             [(f"Point {w}", n) for w, n in zip(WORDS, (2, 3, 4, 5, 6, 7, 9))]
    out = _planned_answer(lines, *points)
    assert "Point golf" in _texts(out) and "Metformin side effects" in _texts(out)    # the last, and the one out of order
    assert "Metformin recommended first" not in _texts(out)
    assert len(_texts(out)) == 9


def test_a_split_line_keeps_its_pieces_with_a_little_more_room():
    lines = [f"Line {w}" for w in WORDS[:3]]
    points = [("First alpha", 2), ("Second bravo", 2), ("Third charlie", 2), ("Fourth delta", 2), ("Fifth echo", 2), ("Sixth foxtrot", 2),
              ("Point golf", 3), ("Point hotel", 4)]
    # three lines and a spare is four points, plus one for the pieces
    assert _texts(_planned_answer(lines, *points)) == ["First alpha", "Second bravo", "Third charlie", "Point golf", "Point hotel"]


def test_a_point_claiming_to_be_the_headlines_line_is_dropped_if_it_says_the_same():
    plan = dict(_PLAN, lines=["Metformin first-line alongside diet and exercise", "Line bravo"])
    points = [{"text": "Metformin is usual first-line, alongside diet, weight loss and exercise.", "sources": [], "line": 1},
              {"text": "Point bravo", "sources": [], "line": 2}]
    raw = _raw("**Metformin first-line**, with diet, weight loss and exercise.", shape="drugs_for", sections=[{"title": "", "points": points}])
    assert _texts(ai.verify(raw, _ev(), plan)) == ["Point bravo"]


def test_an_addition_takes_only_the_room_that_is_left():
    lines = [f"Line {w}" for w in WORDS[:2]]
    assert _texts(_planned_answer(lines, ("Added alpha", 0), ("Added bravo", 1), ("Point charlie", 2), ("Point delta", 3))) == \
        ["Added alpha", "Point charlie", "Point delta"]


def test_a_line_the_model_left_out_goes_back_in_where_it_belongs():
    lines = ["Triptans for severe attacks", "Avoid triptans with heart disease", "Propranolol prevents attacks"]
    assert _texts(_planned_answer(lines, ("**Triptans** when it's severe", 2), ("**Propranolol** to prevent them", 4))) == \
        ["**Triptans** when it's severe", "Avoid triptans with heart disease.", "**Propranolol** to prevent them"]
    assert _planned_answer(lines, ("**Triptans** when it's severe", 2), ("**Propranolol** to prevent them", 4))["sections"][0]["points"][1]["sources"] == []


def test_a_doses_line_left_out_stays_out_as_it_has_no_figure():
    plan = dict(_PLAN, shape="dose", lines=["The usual dose", "Children by weight", "Maximum daily dose capped for adults"])
    sections = [{"title": "", "points": [{"text": "Children: 25 mg/kg/day", "sources": ["S1"], "line": 2}]}]
    out = ai.verify(_raw("**500 mg**", ["S1"], shape="dose", sections=sections), _ev("500 mg, or 25 mg/kg/day in children."), plan)
    assert _texts(out) == ["Children: 25 mg/kg/day"]


@pytest.mark.parametrize("cited", ["[S15]", "(S15)", "[S1, S2]", "(S1, S2)", "(s3)"])
def test_a_snippet_number_in_the_text_is_taken_out(cited):
    out = ai.verify(_raw(f"**Yes** for most adults {cited}; children by weight.", ["S1"]), _ev("Yes for most adults."))
    assert out["headline"] == "**Yes** for most adults; children by weight."


def test_a_line_left_out_stays_out_if_it_has_a_number_or_was_said_elsewhere():
    assert _texts(_planned_answer(["Maximum 4 g a day", "Point bravo"], ("Point bravo", 3))) == ["Point bravo"]
    assert _texts(_planned_answer(["Monitor weight glucose lipids", "Point bravo"], ("Monitor weight, glucose and lipids", 3))) == \
        ["Monitor weight, glucose and lipids"]


def test_with_no_numbering_nothing_can_be_put_back():
    lines = [f"Line {w}" for w in WORDS[:2]]
    assert _texts(_planned_answer(lines, ("Point charlie", 0), ("Point delta", 0))) == ["Point charlie", "Point delta"]


def test_a_headline_saying_no_drug_is_needed_must_have_it_from_the_outline():
    plan = dict(_PLAN, lines=["First-line: SSRIs or SNRIs, plus CBT", "Line two"])
    raw = _raw("**No drug is needed first: SSRIs or SNRIs plus CBT are first-line.**", shape="drugs_for")
    assert ai.verify(raw, _ev(), plan)["headline"] == "**First-line: SSRIs or SNRIs, plus CBT**"
    plan = dict(_PLAN, lines=["No drug treats autism itself; therapy is the mainstay", "Line two"])
    raw = _raw("**No drug treats autism itself**, so therapy is the mainstay.", shape="drugs_for")
    assert ai.verify(raw, _ev(), plan)["headline"] == raw["headline"]


def test_the_sentences_that_say_a_line_are_found_for_each_line_in_turn():
    sentences = ["Agitation, tremor and sweating are its signs.", "Agitation and tremor can be mild.", "The syndrome was named in 1991.",
                 "Withdrawing the serotonergic drugs is the treatment."]
    lines = ["Signs: agitation, sweating, tremor, muscle jerks", "Treatment: withdraw the serotonergic drugs"]
    assert ai.sentences_saying(lines, sentences) == [0, 3, 1]


def test_one_shared_word_isnt_saying_the_same_thing():
    assert ai.sentences_saying(["Tremor is a sign of it"], ["The tremor was mild in most people."]) == []
    assert ai.sentences_saying(["Tremor sweating fever agitation rigidity clonus diarrhoea"], ["Tremor and sweating."]) == []


def _doc(text, src="FDA label"):
    return {"text": text, "src": src, "url": "https://e.org", "topic": "side_effects", "section": "Adverse reactions"}


def test_what_the_outline_says_is_looked_up_in_what_is_already_loaded():
    ctx = {"_corpus": [_doc("Common adverse reactions are nausea and diarrhoea."), _doc("Store below 30 degrees in a dry place."),
                       _doc("Nausea and diarrhoea " * 40)]}
    secs = {"title": "Metformin", "url": "https://w.org", "lead": "Metformin can lower vitamin B12 levels with long-term use.", "sections": []}
    en = {"kind": "library", "display": "Metformin"}
    plan = {"lines": ["Mostly gut upset", "Common: nausea, diarrhoea"], "warning": "Long-term use can lower vitamin B12"}
    out = v._ai_plan_snippets(plan, {"_ai_seen": [(en, ctx, "metformin"), (en, ctx, "metformin")], "_ai_articles": [secs]})
    assert [(s["text"], s["src"]) for s in out] == [("Common adverse reactions are nausea and diarrhoea.", "FDA label"),
                                                    ("Metformin can lower vitamin B12 levels with long-term use.", "Wikipedia: Metformin")]
    assert out[0]["section"] == "Metformin · Adverse reactions"


def test_a_sentence_opening_with_its_keeps_what_its_refers_to():
    ctx = {"_corpus": [_doc("Fluoxetine is metabolised to norfluoxetine."), _doc("Its mean terminal half-life was 8.6 days."),
                       _doc("Store below 30 degrees."), dict(_doc("It is a white powder."), section="Description")]}
    texts = [d["text"] for d in v._ai_corpus(ctx)]
    assert texts[1] == "Fluoxetine is metabolised to norfluoxetine. Its mean terminal half-life was 8.6 days."
    assert texts[3] == "It is a white powder."      # the sentence before is from another section


def test_there_is_a_limit_to_the_sentences_looked_up_by_the_outline(monkeypatch):
    monkeypatch.setattr(v, "_AI_PLAN_SNIPPETS", 2)
    ctx = {"_corpus": [_doc(f"Nausea and diarrhoea with case {w}.") for w in WORDS]}
    out = v._ai_plan_snippets({"lines": ["Nausea, diarrhoea"]}, {"_ai_seen": [({"kind": "library", "display": "X"}, ctx, "x")]})
    assert len(out) == 2


def test_a_drug_is_only_described_by_its_class_when_thats_what_is_asked(monkeypatch):
    monkeypatch.setattr(v, "_ai_load", lambda *a, **k: {"ctx": {"drug_hit": {"name": "Buspirone", "classes": ["x"]}, "_corpus": []}, "query": "buspirone"})
    monkeypatch.setattr(v, "_ctx_glance", lambda ctx: {"classes": ["Azaspirodecanedione derivatives"], "src": {"classes": "WHO ATC"}})
    opts = {"region": "US"}
    assert [s["section"] for s in v._ai_drug_tool({"drug": "Buspirone", "topics": ["overview"]}, opts)] == ["Class"]
    assert v._ai_drug_tool({"drug": "Buspirone", "topics": ["uses", "side_effects"]}, opts) == []


def test_a_fact_has_no_room_for_additions():
    raw = _raw("**Acetylcysteine.**", shape="fact", sections=_points("Point alpha", sources=()))
    assert _texts(ai.verify(raw, _ev(), dict(_PLAN, shape="fact", lines=["Acetylcysteine."]))) == []


def test_the_outline_picks_the_warning():
    plan = dict(_PLAN, warning="Both raise bleeding risk with anticoagulants")
    raw = _raw("**Yes.**", shape="drugs_for", safety=[
        {"level": "red", "text": "Contraindicated in heart bypass surgery", "sources": []},
        {"level": "amber", "text": "Bleeding risk rises with anticoagulants", "sources": []}])
    assert [w["text"] for w in ai.verify(raw, _ev(), plan)["safety"]] == ["Bleeding risk rises with anticoagulants"]


def test_the_outlines_warning_is_shown_if_the_answer_left_it_out():
    plan = dict(_PLAN, warning="Monitor weight and glucose")
    assert [w["text"] for w in ai.verify(_raw("**Yes.**", shape="drugs_for"), _ev(), plan)["safety"]] == ["Monitor weight and glucose"]


def test_the_outlines_warning_still_needs_a_source_for_a_number():
    plan = dict(_PLAN, warning="Never exceed 4 g a day")
    assert ai.verify(_raw("**Yes.**", shape="drugs_for"), _ev(), plan)["safety"] == []


def test_no_warning_in_the_outline_means_none_is_added_from_a_label():
    raw = _raw("**Yes.**", shape="drugs_for", safety=[{"level": "red", "text": "Contraindicated in heart bypass surgery", "sources": []}])
    assert ai.verify(raw, _ev(), _PLAN)["safety"] == []


class OutlineClient:
    """The scripted client, with the options call `outline` makes first."""

    def __init__(self, *responses):
        self.fake = FakeClient(*responses)
        self.options = {}

    def with_options(self, **kw):
        self.options = kw
        return self.fake


def _planned(lines, shape="drugs_for", warning="", stop="end_turn"):
    return _text({"shape": shape, "lines": lines, "warning": warning}, stop=stop)


def test_an_outline_keeps_to_what_its_shape_has_room_for():
    many = [f"Line {w}" for w in WORDS]
    assert len(ai.outline("q", client=OutlineClient(_planned(many)))["lines"]) == ai.SHAPES["drugs_for"][0]
    assert len(ai.outline("q", client=OutlineClient(_planned(many, "fact")))["lines"]) == 2
    assert len(ai.outline("q", client=OutlineClient(_planned(many, "compare")))["lines"]) == 6


def test_an_outline_uses_the_settings_its_model_needs():
    client = OutlineClient(_planned(["x"]))
    ai.outline("q", client=client)
    req = client.fake.requests[0]
    assert req["model"] == ai.OUTLINE_MODEL and req["thinking"] == {"type": "between_tools"}
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["output_config"]["format"]["schema"] is ai.OUTLINE_SCHEMA and client.options["max_retries"] == 0
    client = OutlineClient(_planned(["x"]))
    ai.outline("q", client=client, model=ai.MODEL)
    assert client.fake.requests[0]["thinking"] == {"type": "disabled"}


def test_an_outline_says_where_the_reader_is():
    client = OutlineClient(_planned(["x"]))
    ai.outline("q", client=client, region="SG")
    assert "Singapore" in client.fake.requests[0]["system"][0]["text"]


def test_an_outline_counts_towards_the_cost_at_its_own_price():
    spent = {}
    ai.outline("q", client=OutlineClient(_planned(["x"])), usage=spent)
    assert ai.cost_usd(spent) == pytest.approx((1000 * 2.0 + 100 * 10.0) / 1e6) and spent["calls"] == 1


def test_trouble_with_the_outline_means_no_outline():
    class Down:
        def with_options(self, **kw):
            raise ConnectionError("down")
    assert ai.outline("q", client=Down()) is None
    assert ai.outline("q", client=OutlineClient(_planned(["x"], stop="max_tokens"))) is None
    bad = NS(stop_reason="end_turn", content=[NS(type="text", text="{not json")], usage=None)
    assert ai.outline("q", client=OutlineClient(bad)) is None


def test_an_outline_with_no_lines_says_the_question_was_turned_away():
    assert ai.outline("q", client=OutlineClient(_planned([])))["lines"] == []


def test_the_answer_is_written_from_the_numbered_lines():
    client = FakeClient(_text(_raw()))
    _ask(client, plan=dict(_PLAN, warning="Check kidney function"), prefetched=[{"text": "A label line.", "src": "FDA label"}])
    content = client.requests[0]["messages"][-1]["content"]
    assert "1. The bottom line\n2. Line two" in content and "Warning: Check kidney function" in content
    assert content.index("1. The bottom line") < content.index("[S1]") < content.index("Line 1 becomes the headline")
    assert client.requests[0]["model"] == ai.MODEL


def test_a_comparison_is_told_to_write_table_rows():
    client = FakeClient(_text(_raw()))
    _ask(client, plan=dict(_PLAN, shape="compare"))
    assert "table row" in client.requests[0]["messages"][-1]["content"]
    client = FakeClient(_text(_raw()))
    _ask(client, plan=_PLAN)
    assert "table row" not in client.requests[0]["messages"][-1]["content"]


def test_a_doses_figure_is_shown_to_the_model_only_if_a_source_states_it():
    plan = dict(_PLAN, shape="dose", lines=["Maximum 4 g in 24 hours for adults", "Line two"])
    client = FakeClient(_text(_raw()))
    _ask(client, plan=plan, prefetched=[{"text": "No more than 4,000 mg in 24 hours.", "src": "FDA label"}])
    assert "1. Maximum 4 g in 24 hours" in client.requests[0]["messages"][-1]["content"]
    client = FakeClient(_text(_raw()))
    _ask(client, plan=plan, prefetched=[{"text": "No more than 10 tablets in 24 hours.", "src": "FDA label"}])
    assert "Maximum 4 g" not in client.requests[0]["messages"][-1]["content"]


def test_a_question_about_brands_goes_by_the_register_not_the_outline():
    client = FakeClient(_text(_raw()))
    _ask(client, plan=dict(_PLAN, shape="fact"))
    assert "up to 8" in client.requests[0]["messages"][-1]["content"]


def test_with_evidence_or_a_question_turned_away_there_is_no_thinking():
    client = FakeClient(_text(_raw()))
    _ask(client, prefetched=[{"text": "A label line.", "src": "FDA label"}])
    assert client.requests[0]["thinking"] == {"type": "disabled"}
    client = FakeClient(_text(_raw("Only medicines.", kind="off_topic")))
    _ask(client, plan=dict(_PLAN, lines=[]))
    assert client.requests[0]["thinking"] == {"type": "disabled"}


def test_with_nothing_to_go_on_the_model_thinks_and_can_look_things_up():
    client = FakeClient(_text(_raw()))
    _ask(client)
    assert "thinking" not in client.requests[0] and "tool_choice" not in client.requests[0]


def test_out_of_time_means_no_more_looking_things_up():
    client = FakeClient(_text(_raw()))
    _ask(client, tools_until=time.monotonic() - 1)
    assert client.requests[0]["tool_choice"] == {"type": "none"}
    client = FakeClient(_text(_raw()))
    _ask(client, tools_until=time.monotonic() + 100)
    assert "tool_choice" not in client.requests[0]


# --------------------------------------------------------------- Evidence
def test_evidence_gives_one_id_per_text_and_skips_blanks():
    ev = ai.Evidence()
    shown = ev.format([{"text": "Dose  is\n500 mg.", "src": "FDA label", "section": "Dosage"},
                       {"text": "dose is 500 mg.", "src": "Wikipedia"}, {"text": "  ", "src": "X"}])
    assert list(ev.items) == ["S1"] and shown.count("[S1]") == 2
    assert "(FDA label · Dosage) Dose is 500 mg." in shown


# ---------------------------------------------------------- the conversation
def _text(raw, stop="end_turn"):
    return NS(stop_reason=stop, content=[NS(type="text", text=json.dumps(raw))],
              usage=NS(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0))


def _tool_use(name, inp, id_="t1"):
    return NS(stop_reason="tool_use", content=[NS(type="tool_use", name=name, input=inp, id=id_)],
              usage=NS(input_tokens=500, output_tokens=50, cache_read_input_tokens=0))


class FakeClient:
    """Plays back scripted responses and records each request."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(dict(kw, messages=list(kw["messages"])))
        return self.responses.pop(0)


def _ask(client, question="q", **kw):
    kw.setdefault("tools", {})
    return ai.answer(question, client=client, today="2026-10-30", **kw)


def test_one_call_when_the_evidence_is_already_there():
    client = FakeClient(_text(_raw("**500 mg.**", ["S1"])))
    res = _ask(client, prefetched=[{"text": "The dose is 500 mg.", "src": "FDA label"}])
    assert res["status"] == "ok" and res["usage"]["calls"] == 1
    assert "[S1]" in client.requests[0]["messages"][-1]["content"]


def test_request_uses_the_settings_this_model_needs():
    client = FakeClient(_text(_raw()))
    _ask(client)
    req = client.requests[0]
    assert req["model"] == ai.MODEL and req["output_config"]["effort"] == ai.EFFORT
    assert req["output_config"]["format"]["schema"] is ai.ANSWER_SCHEMA
    for not_accepted in ("temperature", "top_p", "top_k", "fallbacks"):
        assert not_accepted not in req


def test_tool_results_reach_the_model_and_can_be_cited():
    client = FakeClient(_tool_use("get_drug_evidence", {"drug": "x", "topics": [], "keywords": []}),
                        _text(_raw("**500 mg.**", ["S1"])))
    res = _ask(client, tools={"get_drug_evidence": lambda inp: [{"text": "Dose 500 mg.", "src": "FDA label"}]})
    assert res["status"] == "ok" and res["answer"]["headline_sources"] == ["S1"]
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and "[S1]" in result["content"]


def test_a_failing_source_is_reported_to_the_model_not_raised():
    def broken(inp):
        raise ConnectionError("down")
    client = FakeClient(_tool_use("get_drug_evidence", {"drug": "x"}), _text(_raw()))
    assert _ask(client, tools={"get_drug_evidence": broken})["status"] == "ok"
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_nothing_found_is_said_plainly():
    client = FakeClient(_tool_use("search_encyclopedia", {"subject": "x"}), _text(_raw()))
    _ask(client, tools={"search_encyclopedia": lambda inp: []})
    assert "Nothing found" in client.requests[1]["messages"][-1]["content"][0]["content"]


def test_the_last_round_cannot_ask_for_more_tools():
    looping = [_tool_use("search_encyclopedia", {"subject": "x"}, f"t{i}") for i in range(ai.MAX_ROUNDS - 1)]
    client = FakeClient(*looping, _text(_raw()))
    assert _ask(client, tools={"search_encyclopedia": lambda inp: []})["status"] == "ok"
    assert client.requests[-1]["tool_choice"] == {"type": "none"}
    assert "tool_choice" not in client.requests[0]


def test_a_refusal_is_not_shown_as_an_answer():
    res = _ask(FakeClient(NS(stop_reason="refusal", content=[], usage=None)))
    assert res["status"] == "refusal" and res["answer"] is None


def test_a_cut_off_answer_is_not_shown():
    res = _ask(FakeClient(_text(_raw(), stop="max_tokens")))
    assert res["status"] == "failed" and res["answer"] is None


def test_unparseable_output_is_not_shown():
    bad = NS(stop_reason="end_turn", content=[NS(type="text", text="{not json")], usage=None)
    assert _ask(FakeClient(bad))["status"] == "failed"


def test_thinking_blocks_are_skipped_when_reading_the_answer():
    resp = _text(_raw())
    resp.content.insert(0, NS(type="thinking", thinking="…"))
    assert _ask(FakeClient(resp))["status"] == "ok"


def test_a_wrong_number_gets_one_chance_to_be_corrected():
    client = FakeClient(_text(_raw("**900 mg.**", ["S1"])), _text(_raw("**500 mg.**", ["S1"])))
    res = _ask(client, prefetched=[{"text": "The dose is 500 mg.", "src": "FDA label"}])
    assert res["status"] == "ok" and res["answer"]["headline"] == "**500 mg.**"
    assert "could not verify" in client.requests[1]["messages"][-1]["content"]


def test_a_number_still_wrong_after_correction_is_never_shown():
    client = FakeClient(_text(_raw("**900 mg.**", ["S1"])), _text(_raw("**800 mg.**", ["S1"])))
    res = _ask(client, prefetched=[{"text": "The dose is 500 mg.", "src": "FDA label"}])
    assert res["status"] == "unverified" and res["answer"] is None and res["usage"]["calls"] == 2


def test_history_comes_before_the_question():
    client = FakeClient(_text(_raw()))
    _ask(client, "and in pregnancy?", history=[("half life of diazepam", "**30-56 hours**")])
    roles = [(m["role"], m["content"]) for m in client.requests[0]["messages"]]
    assert roles == [("user", "half life of diazepam"), ("assistant", "**30-56 hours**"),
                     ("user", "and in pregnancy?")]


def test_cost_is_estimated_at_each_models_list_price():
    spent = {}
    ai._add_usage(spent, NS(usage=NS(input_tokens=1_000_000, output_tokens=1_000_000)))
    assert ai.cost_usd(spent) == pytest.approx(0.60)
    ai._add_usage(spent, NS(usage=NS(input_tokens=1_000_000, output_tokens=1_000_000)), ai.OUTLINE_MODEL)
    assert ai.cost_usd(spent) == pytest.approx(12.60)
    assert ai.cost_sgd(spent) == pytest.approx(12.60 * ai.SGD_PER_USD)


def test_the_prompt_cache_is_priced_as_it_is_charged():
    spent = {}
    ai._add_usage(spent, NS(usage=NS(input_tokens=0, output_tokens=0, cache_creation_input_tokens=1_000_000,
                                     cache_read_input_tokens=1_000_000)), ai.OUTLINE_MODEL)
    assert ai.cost_usd(spent) == pytest.approx(2.0 * 1.25 + 2.0 * 0.1)
    assert ai.total_tokens(spent) == 2_000_000


def test_tokens_already_spent_survive_a_later_api_error():
    spent = {}
    client = FakeClient(_tool_use("get_drug_evidence", {"drug": "x"}))      # the next call raises
    with pytest.raises(IndexError):
        _ask(client, tools={"get_drug_evidence": lambda i: []}, usage=spent)
    assert spent["input_tokens"] == 500 and spent["output_tokens"] == 50


def test_prompt_carries_the_market_and_reader():
    sg = ai.system_prompt(region="SG", audience="Everyday", today="2026-10-30")
    assert "Singapore" in sg and "995" in sg and "no medical training" in sg and "2026-10-30" in sg
    assert "911" in ai.system_prompt(region="US", audience="Clinician", today="2026-10-30")


def test_unknown_audience_uses_the_default():
    assert ai.AUDIENCES[ai.DEFAULT_AUDIENCE] in ai.system_prompt(region="SG", audience="?", today="x")


# ---------------------------------------------------------------- spend limits
@pytest.fixture
def fresh_limits():
    ai._QUOTA.update(day="", count=0)
    ai._ANSWERS.clear()
    yield
    ai._QUOTA.update(day="", count=0)
    ai._ANSWERS.clear()


def test_daily_limit_stops_and_resets_next_day(fresh_limits):
    assert [ai.take_quota("d1", 2) for _ in range(3)] == [True, True, False]
    assert ai.take_quota("d2", 2) is True


def test_answer_cache_expires_and_evicts(fresh_limits, monkeypatch):
    ai.remember_answer("k", {"v": 1})
    assert ai.cached_answer("k") == {"v": 1}
    monkeypatch.setattr(ai.time, "time", lambda: 10 ** 12)
    assert ai.cached_answer("k") is None
    monkeypatch.undo()
    for i in range(ai.ANSWER_CACHE_SIZE + 5):
        ai.remember_answer(i, i)
    assert ai.cached_answer(0) is None and ai.cached_answer(ai.ANSWER_CACHE_SIZE + 4) == ai.ANSWER_CACHE_SIZE + 4


# ------------------------------------------------------------------- in the app
def _ask_app(*questions):
    at = AppTest.from_file(APP, default_timeout=120).run()
    for q in questions:
        at.chat_input[0].set_value(q).run()
    assert not at.exception
    return at


def _all_text(at):
    msg = at.chat_message[-1]
    return " ".join(str(getattr(e, "value", "")) for e in msg)


@pytest.fixture
def api(monkeypatch, fresh_limits):
    """The app with a key set, and the model scripted: no call leaves this machine."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(ai, "make_client", lambda key: object())
    state = NS(calls=[], result=None, spent={}, plan=None)

    def fake_answer(question, **kw):
        state.calls.append((question, kw))
        kw["usage"].update(state.spent)
        if isinstance(state.result, Exception):
            raise state.result
        return state.result
    monkeypatch.setattr(ai, "answer", fake_answer)
    monkeypatch.setattr(ai, "outline", lambda question, **kw: state.plan)
    return state


def _ok(**fields):
    ev = {"S1": {"text": "It works by blocking receptors.", "src": "FDA label", "url": "https://e.org", "section": ""}}
    answer = ai.verify(_raw("**Blocks receptors.**", ["S1"], **fields), _ev("It works by blocking receptors."))
    return {"status": "ok", "answer": answer, "evidence": ev, "usage": {}, "tools": [], "raw": None}


def test_no_key_means_no_ai_and_no_mention_of_it(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = _ask_app()
    assert "ai_audience" not in [r.key for r in at.sidebar.radio]


def test_the_card_shows_the_answer_with_its_source_and_tag(api):
    api.result = _ok(safety=[{"level": "red", "text": "Stop if a rash occurs.", "sources": ["S1"]}],
                     followups=["Side effects of x?"])
    at = _ask_app(NO_LOOKUP_Q)
    body = " ".join(e.value for e in at.chat_message[-1].markdown)
    assert "Blocks receptors" in body and "ai-red" in body and "FDA label" in body
    assert "Side effects of x?" in [b.label for b in at.chat_message[-1].button]
    assert any("AI summary of the linked sources" in c.value for c in at.chat_message[-1].caption)
    assert "ai_audience" in [r.key for r in at.sidebar.radio]


def test_an_answer_without_a_source_says_so(api):
    answer = ai.verify(_raw("**Agonists activate receptors.**"), _ev())
    api.result = {"status": "ok", "answer": answer, "evidence": {}, "usage": {}, "tools": [], "raw": None}
    at = _ask_app(NO_LOOKUP_Q)
    assert any("AI answer, not from a source" in c.value for c in at.chat_message[-1].caption)


def test_off_topic_has_no_source_caption(api):
    answer = ai.verify(_raw("This app only covers medicines.", kind="off_topic"), _ev())
    api.result = {"status": "ok", "answer": answer, "evidence": {}, "usage": {}, "tools": [], "raw": None}
    at = _ask_app(NO_LOOKUP_Q)
    assert not any("AI" in c.value for c in at.chat_message[-1].caption)


@pytest.mark.parametrize("status", ["unverified", "failed", "refusal"])
def test_an_answer_that_cant_be_shown_falls_back_to_the_sources(api, status):
    api.result = {"status": status, "answer": None, "evidence": {}, "usage": {}, "tools": [], "raw": None}
    at = _ask_app(NO_LOOKUP_Q)
    assert len(api.calls) == 1
    assert any("source text directly" in c.value for c in at.chat_message[-1].caption)


def test_an_api_failure_falls_back_to_the_sources(api):
    api.result = ConnectionError("no route")
    at = _ask_app(NO_LOOKUP_Q)
    assert any("isn't available" in c.value for c in at.chat_message[-1].caption)


@pytest.mark.parametrize("q", ["i want to kill myself", "how do i make meth at home", "hello", "help"])
def test_safety_wording_and_small_talk_never_reach_the_model(api, q):
    _ask_app(q)
    assert api.calls == []


def test_a_repeated_question_costs_nothing(api):
    api.result = _ok()
    _ask_app(NO_LOOKUP_Q)
    at = AppTest.from_file(APP, default_timeout=120).run()      # another visitor, same question
    at.chat_input[0].set_value(NO_LOOKUP_Q).run()
    assert len(api.calls) == 1
    assert "Blocks receptors" in " ".join(e.value for e in at.chat_message[-1].markdown)


def _captions(at):
    return " ".join(c.value for c in at.chat_message[-1].caption)


def _card(at):
    return " ".join(e.value for e in at.chat_message[-1].markdown)


def test_the_outline_reaches_the_answer(api):
    api.result, api.plan = _ok(), dict(_PLAN)
    _ask_app(NO_LOOKUP_Q)
    assert api.calls[0][1]["plan"] == _PLAN and api.calls[0][1]["tools_until"] > 0


def test_a_line_with_no_source_is_marked_and_the_mark_explained(api):
    api.result = _ok(sections=_points("Melatonin for sleep.", sources=()))
    body = _card(_ask_app(NO_LOOKUP_Q))
    assert "†" in body and "General knowledge, not from a linked source" in body


def test_with_a_source_on_every_line_there_is_no_mark(api):
    api.result = _ok(sections=_points("It works by blocking receptors."))
    body = _card(_ask_app(NO_LOOKUP_Q))
    assert "†" not in body and "General knowledge" not in body


@pytest.mark.parametrize("level, before", [("red", True), ("amber", False)])
def test_a_danger_comes_before_the_points_and_a_caution_after(api, level, before):
    api.result = _ok(safety=[{"level": level, "text": "Take care with this.", "sources": ["S1"]}],
                     sections=_points("Tiredness is common.", sources=()))
    body = _card(_ask_app(NO_LOOKUP_Q))
    assert (body.index("Take care with this") < body.index("Tiredness is common")) is before


def test_a_chip_says_which_label(api):
    api.result = _ok()
    api.result["evidence"]["S1"]["section"] = "Risperidone · Uses"
    assert "FDA: Risperidone" in _card(_ask_app(NO_LOOKUP_Q))


def test_the_source_text_is_tucked_away(api):
    api.result = _ok()
    assert "Source text (1)" in [e.label for e in _ask_app(NO_LOOKUP_Q).expander]


def test_a_table_cell_can_say_which_drug_its_about_once_a_phone_stacks_the_rows(api):
    api.result = _ok(shape="compare", table_columns=["Feature", "Sertraline", "Fluoxetine"],
                     table_rows=[{"cells": ["Action", "Blocks receptors", "It works by blocking receptors"], "sources": ["S1"]},
                                 {"cells": ["Stopping", "Taper", "Gentler"], "sources": []}])
    body = _card(_ask_app(NO_LOOKUP_Q))
    assert "<td>Action</td>" in body and "<span class='ai-col'>Sertraline: </span>Blocks receptors" in body
    assert "<span class='ai-col'>Fluoxetine: </span>It works by blocking receptors <a" in body
    assert "<span class='ai-col'>Fluoxetine: </span>Gentler<sup" in body


def test_an_answer_stays_where_it_was_when_the_next_question_comes(api):
    api.result = _ok(followups=["Side effects of x?"], sections=_points("It works by blocking receptors.", sources=["S1"]))
    at = _ask_app(NO_LOOKUP_Q)
    before = [c.type for c in at.chat_message[1].children.values()]
    at.chat_input[0].set_value("what is the half life of drugx").run()
    after = [c.type for c in at.chat_message[1].children.values()]
    assert len(after) == len(before) and after[-1] == before[-1] and "expander" in after
    assert not [c for c in at.chat_message[1].children.values() if c.type == "button"]     # its follow-ups went, though


def test_a_list_of_medicines_says_where_they_stand_in_singapore(api):
    api.result = _ok(shape="drugs_for", sections=_points("**Risperidone** for irritability.", "**Atomoxetine** for attention.", sources=()))
    body = _card(_ask_app(NO_LOOKUP_Q))
    assert "In Singapore" in body and "open data" in body and "Atomoxetine" in body and "HSA register" in body


def _market(shape, *points, columns=(), region="SG"):
    answer = {"shape": shape, "headline": "", "sections": [{"title": "", "points": [{"text": p} for p in points]}],
              "table_columns": list(columns)}
    return v._ai_market_line(answer, {"region": region})


def test_the_singapore_line_groups_medicines_by_status():
    assert _market("drugs_for", "**Risperidone**", "**Atomoxetine**") == [("Prescription-only (POM)", "Risperidone"),
                                                                          ("Not in HSA's open data", "Atomoxetine")]
    assert _market("compare", columns=["Ibuprofen", "Risperidone"]) == [("GSL, P or POM by product", "Ibuprofen"),
                                                                        ("Prescription-only (POM)", "Risperidone")]


def test_the_singapore_line_doesnt_list_every_drug_to_say_the_same_thing():
    assert _market("drugs_for", "**Risperidone**", "**Aripiprazole** and **olanzapine**") == [("All", "Prescription-only (POM)")]
    assert _market("compare", columns=["Sertraline", "Fluoxetine"]) == [("Both", "Prescription-only (POM)")]
    assert _market("drugs_for", "**Risperidone**", "**Aripiprazole**", "**Olanzapine**", "**Atomoxetine**") == \
        [("Not in HSA's open data", "Atomoxetine"), ("The rest", "Prescription-only (POM)")]


def test_a_drug_hsas_open_data_doesnt_have_isnt_said_to_be_unregistered():
    # Cymbalta has a Singapore licence on the National Drug Formulary, but no row in the open data
    assert v._ai_not_listed("Duloxetine").startswith("Duloxetine isn't in HSA's open listing") and "isn't known" in v._ai_not_listed("Duloxetine")
    assert _market("drugs_for", "**Duloxetine**", "**Risperidone**") == [("Not in HSA's open data", "Duloxetine"), ("Prescription-only (POM)", "Risperidone")]


def test_a_breath_test_is_not_a_medicine_to_look_up():
    assert _market("drugs_for", "**Omeprazole**", "Confirm eradication with a urea breath test") == [("P or POM by product", "Omeprazole")]


def test_the_singapore_line_is_only_for_that_market_and_only_names_drugs_compared():
    assert _market("drugs_for", "**Risperidone**", region="US") == []
    assert _market("compare", "**Simvastatin** with **clarithromycin**") == []
    assert _market("drugs_for", "Rest and fluids") == []


def test_a_class_of_drug_doesnt_count_as_the_substance_it_acts_on():
    assert [e["display"] for e in v._ai_named_drugs("**Spironolactone** (aldosterone antagonist): add-on")] == ["Spironolactone"]
    assert _market("drugs_for", "**Spironolactone** (aldosterone antagonist)", "Dopamine agonists", "Serotonin reuptake inhibitors") == \
        [("Prescription-only (POM)", "Spironolactone")]
    names = [e["display"] for e in v._ai_named_drugs("**Calcium channel blockers** such as amlodipine lower aldosterone, raising potassium")]
    assert names == ["Amlodipine"]


def test_a_long_list_of_statuses_comes_down_to_what_can_be_bought():
    assert _market("drugs_for", "**Paracetamol**, **ibuprofen**, **aspirin**", "**Flunarizine**", "**Sumatriptan**, **propranolol**", "**Rizatriptan**") == [
        ("Some without prescription", "Paracetamol, Ibuprofen, Aspirin, Flunarizine"), ("Prescription-only (POM)", "Sumatriptan, Propranolol"),
        ("Not in HSA's open data", "Rizatriptan")]


def test_a_brand_is_given_its_own_status_not_its_ingredients():
    assert _market("compare", columns=["Panadol", "Nurofen"])[0][0] != "GSL or POM by product"
    assert _market("compare", columns=["Paracetamol", "Risperidone"])[0] == ("GSL or POM by product", "Paracetamol")


def test_the_same_status_for_all_of_them_reads_as_a_sentence(api):
    api.result = _ok(shape="drugs_for", sections=_points("**Risperidone** for irritability.", "**Aripiprazole** for irritability.", sources=()))
    assert "In Singapore: both <b>Prescription-only (POM)</b>" in _card(_ask_app(NO_LOOKUP_Q))


def test_where_the_class_varies_the_route_is_given():
    by_route = sgr.availability_by_route(sgr.lookup("salbutamol"))
    assert "oral: Pharmacy-only (P)" in by_route and "respiratory (inhalation): Prescription-only (POM)" in by_route
    assert sgr.availability_by_route(sgr.lookup("risperidone")) == ""


# ----------------------------------------------------- looking up what the outline names
def _no_load(monkeypatch, slow=()):
    """_ai_load and _ai_drug_tool replaced: records what was asked for, holds up `slow` until released."""
    state = NS(loads=[], release=threading.Event())

    def load(en, opts, topics, want_market, label_only=False):
        state.loads.append((en["query"], label_only))
        if en["query"] in slow:
            state.release.wait(5)
        return {"ctx": {}, "query": en["query"]}
    monkeypatch.setattr(v, "_ai_load", load)
    monkeypatch.setattr(v, "_ai_drug_tool", lambda inp, opts: [{"text": f"About {inp['drug']}", "src": "FDA label"}])
    monkeypatch.setattr(v, "_ai_encyclopedia_tool", lambda inp, opts: [{"text": f"Article on {inp['subject']}", "src": "Wikipedia"}])
    return state


def test_each_label_is_fetched_once_labels_only(monkeypatch):
    state = _no_load(monkeypatch)
    loads = v._AiLabelLoads({"region": "SG"})
    for name in ("Risperidone", "risperidone", "Aripiprazole"):
        loads.want(name, ["irritability"])
    out = loads.read(time.monotonic() + 5)
    assert sorted(state.loads) == [("aripiprazole", True), ("risperidone", True)] and len(out) == 2


def test_there_is_a_limit_to_the_labels_for_one_question(monkeypatch):
    state = _no_load(monkeypatch)
    loads = v._AiLabelLoads({"region": "SG"})
    for name in ("Metformin", "Gliclazide", "Sitagliptin", "Empagliflozin", "Semaglutide", "Pioglitazone", "Risperidone",
                 "Aripiprazole", "Sertraline", "Fluoxetine"):
        loads.want(name)
    loads.read(time.monotonic() + 5)
    assert len(state.loads) == v._AI_LIST_LABELS


def test_a_label_still_coming_at_the_deadline_is_left_out(monkeypatch):
    state = _no_load(monkeypatch, slow=("risperidone",))
    loads = v._AiLabelLoads({"region": "SG"})
    loads.want("Risperidone")
    loads.want("Aripiprazole")
    out = loads.read(time.monotonic() + 0.5)
    state.release.set()
    assert [s["text"] for s in out] == ["About Aripiprazole"]


def test_a_drug_with_no_label_is_not_searched_for_twice(monkeypatch):
    state = _no_load(monkeypatch)
    monkeypatch.setattr(v, "_ai_load", lambda en, opts, topics, want_market, label_only=False: state.loads.append(en["query"]) or {"error": "no label"})
    loads = v._AiLabelLoads({"region": "SG"})
    loads.want("Gliclazide")
    assert loads.read(time.monotonic() + 5) == [] and state.loads == ["gliclazide"]


@pytest.mark.parametrize("carried_over, looked_up", [(True, ["Sertraline"]), (False, [])])
def test_a_drug_carried_over_from_the_last_question_has_its_evidence_read_at_once(monkeypatch, carried_over, looked_up):
    read = []
    monkeypatch.setattr(v, "_ai_read_drugs", lambda names, *a, **k: read.extend(names) or [])
    parsed = {"mode": "answer", "intents": ["brands"], "raw": "can i buy it without a prescription", "used_context": carried_over,
              "entities": [{"kind": "free", "query": "Sertraline", "display": "Sertraline"}]}
    v._ai_prefetch(parsed, {"region": "SG"}, NS(), time.monotonic() + 5)
    assert read == looked_up


def test_a_named_drug_still_coming_at_the_deadline_gets_its_article_instead(monkeypatch):
    state = _no_load(monkeypatch, slow=("risperidone",))
    out = v._ai_read_drugs(["Risperidone", "Aripiprazole"], ["uses"], [], 3, {"region": "SG"}, time.monotonic() + 0.5)
    state.release.set()
    assert [s["text"] for s in out] == ["About Aripiprazole", "Article on Risperidone"]


def test_an_article_that_is_slow_too_doesnt_hold_up_the_answer(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(v, "_ai_encyclopedia_tool", lambda inp, opts: release.wait(5) and [])
    monkeypatch.setattr(v, "_AI_LATE_ARTICLE_S", 0.1)
    t = time.monotonic()
    assert v._ai_articles_instead(["Risperidone"], [], {"region": "SG"}) == [] and time.monotonic() - t < 2
    release.set()


def _outline_names(monkeypatch, plan, parsed=None):
    monkeypatch.setattr(ai, "outline", lambda question, **kw: plan)
    wanted = []
    v._ai_outline_then_labels("q", None, [], parsed or {"entities": []}, NS(want=lambda name, words=(): wanted.append(name)),
                              {"region": "SG", "_ai_usage": {}})
    return wanted


def test_each_line_gets_a_label_before_any_line_gets_two(monkeypatch):
    plan = {"shape": "drugs_for", "lines": ["SGLT2 inhibitors (empagliflozin, dapagliflozin)",
                                            "Sulfonylureas (gliclazide, glimepiride)"], "warning": ""}
    assert _outline_names(monkeypatch, plan) == ["Empagliflozin", "Gliclazide", "Dapagliflozin", "Glimepiride"]


def test_the_outlines_drugs_are_looked_up_only_when_they_help(monkeypatch):
    lines = ["Pravastatin has few interactions"]
    # a list, or a question about a class
    assert _outline_names(monkeypatch, {"shape": "drugs_for", "lines": lines, "warning": ""}) == ["Pravastatin"]
    assert _outline_names(monkeypatch, {"shape": "compare", "lines": lines, "warning": ""}) == ["Pravastatin"]
    # a question naming its drug has that drug's evidence, and an explanation's example needs none
    named = {"entities": [{"kind": "library", "query": "simvastatin", "display": "Simvastatin"}]}
    assert _outline_names(monkeypatch, {"shape": "compare", "lines": lines, "warning": ""}, named) == []
    assert _outline_names(monkeypatch, {"shape": "explain", "lines": lines, "warning": ""}) == []


def test_no_answer_from_fda_is_asked_once_more_before_it_is_believed(monkeypatch):
    asked = []
    monkeypatch.setattr(cs, "_openfda_fetch", lambda search, limit=1: asked.append(search) or [])
    assert cs.openfda_label(generic_name="sertraline", main_route="ORAL") == {}
    assert len(asked) == 3


def test_no_label_for_a_known_medicine_isnt_remembered(monkeypatch):
    asked = []
    monkeypatch.setattr(v.cs, "openfda_label", lambda **kw: asked.append(kw) or {})
    for _ in range(2):
        assert v.cached_openfda_label("risperidone", "NOT-A-ROUTE") == {}
    assert len(asked) == 2
    asked.clear()
    for _ in range(2):
        v.cached_openfda_label("benzene", "NOT-A-ROUTE")
    assert len(asked) == 1


def test_a_helper_module_whose_file_changed_is_reloaded(tmp_path, monkeypatch):
    path = tmp_path / "helper_for_reload.py"
    path.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import helper_for_reload as helper
    monkeypatch.setitem(__import__("sys").modules, "helper_for_reload", helper)
    assert v._up_to_date(helper).VALUE == 1
    helper.VALUE = 9            # stays while the file is unchanged
    assert v._up_to_date(helper).VALUE == 9
    path.write_text("VALUE = 2\n")
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert v._up_to_date(helper).VALUE == 2


def test_a_server_that_loaded_an_old_helper_before_file_times_were_noted_reloads_it(tmp_path, monkeypatch):
    path = tmp_path / "old_helper_for_reload.py"
    path.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import old_helper_for_reload as helper
    monkeypatch.setitem(__import__("sys").modules, "old_helper_for_reload", helper)
    path.write_text("VALUE = 2\nNEWEST = True\n")
    assert v._up_to_date(helper, "NEWEST").VALUE == 2


def test_an_answer_shows_its_tokens_and_cost_in_sgd(api):
    api.result, api.spent = _ok(), {"input_tokens": 3000, "output_tokens": 200, "usd": 0.0004}
    assert "3,200 tokens · ≈ S\\$0.0005" in _captions(_ask_app(NO_LOOKUP_Q))


def test_off_topic_shows_what_it_spent(api):
    answer = ai.verify(_raw("This app only covers medicines.", kind="off_topic"), _ev())
    api.result = {"status": "ok", "answer": answer, "evidence": {}, "usage": {}, "tools": [], "raw": None}
    api.spent = {"input_tokens": 900, "output_tokens": 40}
    assert "940 tokens" in _captions(_ask_app(NO_LOOKUP_Q))


def test_a_failed_attempt_still_shows_what_it_spent(api):
    api.result = {"status": "unverified", "answer": None, "evidence": {}, "usage": {}, "tools": [], "raw": None}
    api.spent = {"input_tokens": 2000, "output_tokens": 100}
    assert "2,100 tokens" in _captions(_ask_app(NO_LOOKUP_Q))


def test_an_api_error_still_shows_what_it_spent(api):
    api.result, api.spent = ConnectionError("no route"), {"input_tokens": 500, "output_tokens": 50}
    assert "550 tokens" in _captions(_ask_app(NO_LOOKUP_Q))


def test_a_saved_answer_shows_no_tokens(api):
    api.result, api.spent = _ok(), {"input_tokens": 3000, "output_tokens": 200}
    _ask_app(NO_LOOKUP_Q)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value(NO_LOOKUP_Q).run()
    assert "0 tokens · S\\$0 (saved answer)" in _captions(at)


def test_a_fixed_reply_shows_no_tokens(api):
    assert "0 tokens · S\\$0" in _captions(_ask_app("how do i make meth at home"))
    assert api.calls == []


def test_no_key_shows_no_token_count(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "tokens" not in _captions(_ask_app("how do i make meth at home"))


def test_daily_limit_falls_back_without_calling(api, monkeypatch):
    monkeypatch.setenv("AI_DAILY_LIMIT", "0")
    api.result = _ok()
    at = _ask_app(NO_LOOKUP_Q)
    assert api.calls == []
    assert any("limit has been reached" in c.value for c in at.chat_message[-1].caption)


def test_session_limit_falls_back_without_calling(api, monkeypatch):
    monkeypatch.setenv("AI_SESSION_LIMIT", "1")
    api.result = _ok()
    at = _ask_app(NO_LOOKUP_Q, "what's the capital of spain")
    assert len(api.calls) == 1
    assert any("limit has been reached" in c.value for c in at.chat_message[-1].caption)


def test_a_follow_up_sends_the_earlier_exchange(api):
    api.result = _ok()
    _ask_app(NO_LOOKUP_Q, "what's the capital of spain")
    assert api.calls[1][1]["history"] == [(NO_LOOKUP_Q, "**Blocks receptors.**")]


def test_the_key_is_never_in_what_the_page_shows(api):
    api.result = ConnectionError("no route")
    at = _ask_app(NO_LOOKUP_Q)
    assert "test-key-not-real" not in " ".join(str(getattr(e, "value", "")) for e in at.main)
