"""Research features, with no network and no API calls: the calculator, detailed answers, answers
about the evidence, and how papers, registered trials and potency figures become snippets.
The live checks of PubMed (Europe PMC), ClinicalTrials.gov and ChEMBL are marked `network`."""
import math

import pytest

import ai_answer as ai
import chem_sources as cs
import chem_v4 as v


def _raw(headline="**No.**", sources=(), **extra):
    return dict({"kind": "answer", "shape": "", "headline": headline, "headline_sources": list(sources), "safety": [],
                 "sections": [], "chain": [], "chain_sources": [], "table_columns": [], "table_rows": [],
                 "followups": []}, **extra)


def _points(*texts, sources=("S1",), line=0):
    return [{"title": "", "points": [{"text": t, "sources": list(sources), "line": line} for t in texts]}]


# -------------------------------------------------------------- the calculator
@pytest.mark.parametrize("formula, mass", [("NaCl", 58.44), ("H2O", 18.015), ("Ca(OH)2", 74.092),
                                           ("CuSO4.5H2O", 249.68), ("C6H12O6", 180.156), ("Al2(SO4)3", 342.13),
                                           ("K4[Fe(CN)6]", 368.35)])
def test_molar_mass(formula, mass):
    assert ai.molar_mass(formula) == pytest.approx(mass, abs=0.02)


@pytest.mark.parametrize("formula", ["", "Xx2", "NaCl)", "(NaCl", "salt", "na cl"])
def test_molar_mass_refuses_what_is_not_a_formula(formula):
    with pytest.raises(ValueError):
        ai.molar_mass(formula)


@pytest.mark.parametrize("expression, value", [
    ("0.5 * 0.25 * M('NaCl')", 7.305), ("-log10(0.01)", 2.0), ("18 * 15", 270.0), ("(98.6 - 32) * 5 / 9", 37.0),
    ("2 ^ 10", 1024.0), ("0.1 × 0.25 ÷ 0.5", 0.05), ("sqrt(16) + abs(-2)", 6.0), ("ln(exp(3))", 3.0),
    ("1 / NA * 6.02214076e23", 1.0), ("M(NaCl)", 58.44), ("round(7.305, 1)", 7.3), ("max(2, 3) - min(2, 3)", 1.0)])
def test_calculate(expression, value):
    assert ai.calculate(expression)[0] == pytest.approx(value, abs=0.01)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo hi')", "open('x')", "(1).__class__", "x + 1", "1 / 0", "9 ** 9 ** 9",
    "[1, 2][0]", "lambda: 1", "'a' * 3", "", "1 if 1 else 2", "M('NaCl').real", "a" * 400])
def test_calculate_refuses_everything_but_arithmetic(expression):
    with pytest.raises(ValueError):
        ai.calculate(expression)


def test_a_later_step_uses_an_earlier_result():
    out = ai.calculation_snippets([{"name": "n", "label": "moles", "expression": "0.5*0.25", "unit": "mol"},
                                   {"name": "m", "label": "mass of NaCl", "expression": "n*M('NaCl')", "unit": "g"}],
                                  {"250", "0.5"})
    assert len(out) == 2 and "7.305 g" in out[1]["text"] and "58.44 g/mol" in out[1]["text"]
    assert out[1]["src"] == "Calculator"


def test_a_step_that_cannot_be_worked_out_is_left_out():
    out = ai.calculation_snippets([{"name": "a", "label": "x", "expression": "nonsense(", "unit": ""},
                                   {"name": "b", "label": "y", "expression": "2*3", "unit": ""}])
    assert [s["text"].split(":")[0] for s in out] == ["y"]


def test_halves_round_up_as_people_round_them():
    # 7.305 is 7.30499… in binary floating point, and '%.3g' would give 7.3
    assert ai._sig(7.305, 3) == "7.31" and ai._sig(0.125, 2) == "0.13" and ai._sig(270.0, 6) == "270"
    assert ai._sig(6.02214076e23, 3) == "6.02 × 10^23"


def test_a_calculated_figure_passes_the_number_check_and_another_does_not():
    ev = ai.Evidence("how many grams of NaCl do I need for 250 mL of 0.5 M solution")
    ev.format(ai.calculation_snippets([{"name": "n", "label": "moles", "expression": "0.5*0.25", "unit": "mol"},
                                       {"name": "m", "label": "mass", "expression": "n*M('NaCl')", "unit": "g"}],
                                      ev.allowed))
    for good in ("**About 7.3 g of NaCl** (0.5 mol/L × 0.25 L × 58.44 g/mol).", "You need 7.31 g.", "About 7 g.",
                 "7,305 mg of NaCl."):
        assert ai.claim_ok(good, ["S2"], ev)[0], good
    for bad in ("You need 7.5 g.", "You need 73 g.", "About 8 g."):
        assert not ai.claim_ok(bad, ["S2"], ev)[0], bad


def test_a_number_the_model_supplied_is_pointed_out():
    out = ai.calculation_snippets([{"name": "p", "label": "pH", "expression": "4.76 + log10(0.2/0.1)", "unit": ""}],
                                  {"0.2", "0.1"})
    assert "4.76" in out[0]["text"] and "supplied by the AI model" in out[0]["text"]
    # the reader's own numbers, in other units, are not flagged: 250 mL as 0.25 L
    out = ai.calculation_snippets([{"name": "v", "label": "litres", "expression": "0.25 * 0.5", "unit": ""}], {"250", "0.5"})
    assert "supplied by the AI" not in out[0]["text"]


def test_the_hidden_variants_are_not_shown_to_the_model():
    ev = ai.Evidence("")
    shown = ev.format([{"text": "mass: 7.305 g.", "src": "Calculator", "also": "7.3 ; 7.31"}])
    assert "7.31" not in shown and "7.31" in ev.stated(["S1"])


# -------------------------------------------------------------- detailed answers
def test_a_detailed_answer_has_about_twice_the_room():
    for shape, (points, _warnings) in ai.SHAPES.items():
        assert ai.caps(shape, True)[0] > points
        assert ai.outline_room(shape, True) > ai.outline_room(shape)
    assert ai.caps("no such shape") == ai._ANY_SHAPE


@pytest.mark.parametrize("question", ["tell me everything about ketamine", "explain warfarin in detail",
                                      "give me a comprehensive overview of statins", "metformin, in more detail",
                                      "a detailed answer on SSRIs please", "deep dive into PROTACs"])
def test_a_question_can_ask_for_depth(question):
    assert ai.DETAIL_RE.search(question)


@pytest.mark.parametrize("question", ["side effects of metformin", "what is the half life of diazepam",
                                      "how does detailing work"])
def test_most_questions_do_not(question):
    assert not ai.DETAIL_RE.search(question)


def test_verify_keeps_more_points_when_detailed():
    ev = ai.Evidence("")
    ev.add("alpha bravo charlie delta echo foxtrot golf hotel india juliet", "Wikipedia")
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
    raw = _raw(shape="explain", sections=_points(*[f"{w} point" for w in words]))
    assert sum(len(s["points"]) for s in ai.verify(raw, ev)["sections"]) == ai.SHAPES["explain"][0]
    assert sum(len(s["points"]) for s in ai.verify(raw, ev, detail=True)["sections"]) == ai.DETAILED_SHAPES["explain"][0]


def test_the_prompts_say_so_only_when_detailed():
    assert "DETAILED ANSWER" in ai.system_prompt(region="SG", audience="Student", today="2026-11-15", detail=True)
    assert "DETAILED ANSWER" not in ai.system_prompt(region="SG", audience="Student", today="2026-11-15")


# -------------------------------------------------------------- answers about the evidence
def test_a_clause_cut_from_the_end_takes_its_semicolon_with_it():
    ev = ai.Evidence("")
    ev.add("The oral LD50 in rats is 192 mg/kg.", "Wikipedia")
    out = ai.verify(_raw("**Rat oral LD50 is 192 mg/kg**; the sources give no human figure.", ["S1"], shape="fact"), ev)
    assert out["headline"] == "**Rat oral LD50 is 192 mg/kg**"


def test_an_evidence_answer_is_built_from_the_papers_not_forced_back_to_the_outline():
    ev = ai.Evidence("")
    ev.add("Kosiborod et al., N Engl J Med 2023 (randomised trial). Semaglutide improved the symptom score by 7.8 points.",
           "PubMed")
    plan = {"shape": "evidence", "lines": ["Improves symptoms in HFpEF.", "STEP-HFpEF improved symptoms.",
                                           "SELECT cut cardiovascular events.", "HFrEF untested."], "warning": ""}
    raw = _raw("**Improves symptoms in HFpEF.**", shape="evidence",
               sections=_points("**Kosiborod 2023**: symptom score improved by 7.8 points with semaglutide."))
    out = ai.verify(raw, ev, plan)
    # the lines the answer left out are not put back as unsourced points
    assert [p["text"] for s in out["sections"] for p in s["points"]] == \
        ["**Kosiborod 2023**: symptom score improved by 7.8 points with semaglutide."]


def test_a_figure_from_no_abstract_is_dropped_from_an_evidence_answer():
    ev = ai.Evidence("")
    ev.add("Kosiborod et al., N Engl J Med 2023 (randomised trial). Semaglutide improved the symptom score.", "PubMed")
    raw = _raw("**Improves symptoms.**", shape="evidence",
               sections=_points("**Kosiborod 2023**: score improved by 12.4 points.", "It improved the symptom score."))
    out = ai.verify(raw, ev)
    assert [p["text"] for s in out["sections"] for p in s["points"]] == ["It improved the symptom score."]


def test_the_outline_schema_and_tools_cover_the_research_sources():
    assert {"literature", "trials", "calculations"} <= set(ai.OUTLINE_SCHEMA["properties"])
    assert {"search_literature", "search_trials", "calculate"} <= {t["name"] for t in ai.TOOLS}
    assert "potency" in ai.TOPICS and {"evidence", "calculate"} <= set(ai.SHAPES)
    assert set(ai.SHAPES) == set(ai.DETAILED_SHAPES)


# -------------------------------------------------------------- papers, trials, potency → snippets
PAPER = {"pmid": "37622681", "title": "Semaglutide in Patients with Heart Failure with Preserved Ejection Fraction and Obesity.",
         "authorString": "Kosiborod MN, Abildstrøm SZ, Borlaug BA.", "pubYear": "2023", "citedByCount": 1434,
         "journalInfo": {"journal": {"isoabbreviation": "N Engl J Med"}},
         "pubTypeList": {"pubType": ["Randomized Controlled Trial", "Journal Article"]},
         "abstractText": "BACKGROUND: " + "Background sentence. " * 80 + "RESULTS: The score improved by 7.8 points. "
                         "CONCLUSIONS: Semaglutide led to larger reductions in symptoms."}


def test_a_paper_record_is_read():
    p = cs._paper(PAPER)
    assert (p["first_author"], p["et_al"], p["year"], p["cited"], p["kind"], p["journal"]) == \
        ("Kosiborod", True, "2023", 1434, "randomised trial", "N Engl J Med")
    assert p["url"] == "https://pubmed.ncbi.nlm.nih.gov/37622681/"


def test_markup_in_a_title_is_removed():
    assert cs._plain("A selective BCL-X&lt;sub&gt;L&lt;/sub&gt; PROTAC") == "A selective BCL-XL PROTAC"


def test_a_structured_abstract_keeps_its_headings_apart_from_its_text():
    assert cs._plain("<h4>Background</h4>The drug works.<h4>Results</h4>It did.") == "Background: The drug works. Results: It did."


def test_a_saved_conversation_keeps_every_line_and_its_source():
    answer = v._b("ai_answer", headline="**7.31 g of NaCl**", sources=[("Calculator", "")],
                  safety=[{"level": "red", "text": "Corrosive.", "sources": []}], chain=["a", "b"], chain_sources=[],
                  table_columns=["Feature", "X"], table_rows=[{"cells": ["Half-life", "2 h"], "sources": [("FDA: X", "https://e.org")]}],
                  sections=[{"title": "Working", "points": [{"text": "0.125 mol.", "sources": [("PubMed: Kosiborod 2023", "https://p.org")]},
                                                            {"text": "From memory.", "sources": []}]}], market=[])
    msgs = [{"role": "user", "text": "how many grams of NaCl"},
            {"role": "assistant", "blocks": [answer, v._b("buttons", items=[("More", "more")]),
                                             v._b("ai_sources", items=[("Calculated by the app", "0.5 × 0.25 = 0.125 mol", "Calculator", "")]),
                                             v._b("ai_reading", papers=[dict(cs._paper(PAPER), why="most cited")], trials=TRIALS["trials"], trials_total=114),
                                             v._b("caption", v="⏱ 3.0 s · AI summary · ≈ S\\$0.01"), {"t": "something new"}]}]
    md = v.chat_markdown(msgs)
    for part in ("## how many grams of NaCl", "**7.31 g of NaCl** [Calculator]", "> **Danger:** Corrosive. †", "a → b †",
                 "| Half-life | 2 h | [[FDA: X](https://e.org)] |", "**Working**", "- 0.125 mol. [[PubMed: Kosiborod 2023](https://p.org)]",
                 "- From memory. †", "**Source text**", "**Reading list**", "NCT06308653", "S$0.01"):
        assert part in md, part
    assert v.chat_markdown([]).startswith("# Drug & Chemical Explorer")


@pytest.mark.parametrize("terms, kept", [
    (["psilocybin", "depression", "randomized trial"], ["psilocybin", "depression"]),
    (["semaglutide", "heart failure"], ["semaglutide", "heart failure"]),
    (["metformin", "Metformin", "", "  "], ["metformin"]),
    (['lecanemab"', "Alzheimer's disease"], ["lecanemab", "Alzheimer's disease"])])
def test_search_terms_keep_the_subject_and_drop_the_kind_of_paper(terms, kept):
    assert cs._lit_terms(terms) == kept


def test_a_paper_snippet_names_the_study_and_keeps_results_and_conclusion():
    s = v._ai_paper_snippets({"papers": [cs._paper(PAPER)]})[0]
    assert s["text"].startswith("Kosiborod et al., N Engl J Med 2023 (randomised trial). Semaglutide in Patients")
    assert "7.8 points" in s["text"] and "larger reductions" in s["text"] and len(s["text"]) <= ai.MAX_SNIPPET_CHARS
    assert s["src"] == "PubMed" and "1,434 citations" in s["section"]
    # how often it's cited is for the reader to weigh the paper by, never a finding to quote
    assert "1,434" not in s["text"] and "1434" not in s["text"]


def test_a_short_abstract_is_kept_whole():
    assert v._ai_abstract_core("One sentence. Another one.") == "One sentence. Another one."


TRIALS = {"total": 114, "read": 40, "by_status": {"recruiting": 12}, "ok": True, "trials": [
    {"nct": "NCT06308653", "title": "Psilocybin for Major Depressive Disorder (MDD)", "status": "active not recruiting",
     "phase": "phase 3", "enrollment": 238, "sponsor": "Usona Institute", "start": "2024-03", "completion": "2026-01",
     "outcome": "Change in MADRS", "has_results": False, "type": "interventional",
     "url": "https://clinicaltrials.gov/study/NCT06308653"}]}


def test_trial_snippets_give_the_count_then_each_trial():
    out = v._ai_trial_snippets(TRIALS, "depression", "psilocybin")
    assert "114 registered studies for psilocybin in depression" in out[0]["text"]
    assert "By status" not in out[0]["text"]          # only 40 of the 114 were read: no tally of a part
    assert out[1]["text"].startswith("NCT06308653: Psilocybin for Major Depressive Disorder (MDD). Phase 3 interventional")
    assert "238 participants" in out[1]["text"] and "Results posted on the registry: no." in out[1]["text"]
    assert out[1]["section"].startswith("NCT06308653") and out[1]["src"] == "ClinicalTrials.gov"


def test_an_nct_number_needs_no_source_but_a_trial_size_does():
    ev = ai.Evidence("")
    for s in v._ai_trial_snippets(TRIALS, "depression", "psilocybin"):
        ev.add(s["text"], s["src"], s["url"], s["section"])
    assert ai.claim_ok("**NCT06308653**: phase 3, 238 participants, active.", ["S2"], ev)[0]
    assert not ai.claim_ok("**NCT06308653**: phase 3, 500 participants.", ["S2"], ev)[0]


def test_no_trials_means_no_snippets():
    assert v._ai_trial_snippets({"total": 0, "trials": []}, "x", "y") == []


def test_potency_snippets_say_how_many_measurements_a_figure_rests_on(monkeypatch):
    monkeypatch.setattr(v, "cached_potency", lambda name: {"ok": True, "url": "https://example.org", "targets": [
        {"target": "Bcr/Abl fusion protein", "type": "IC50", "text": "220 nM", "n": 49, "low_text": "1.1 nM", "high_text": "10 µM"},
        {"target": "Kinase X", "type": "Kd", "text": "40 nM", "n": 1, "low_text": "40 nM", "high_text": "40 nM"}]})
    out = v._ai_potency_snippets("Imatinib", {})
    assert out[0]["text"] == ("Imatinib at Bcr/Abl fusion protein (human): IC50 about 220 nM, the median of 49 published "
                              "binding measurements, which range from 1.1 nM to 10 µM.")
    assert "one published binding measurement" in out[1]["text"] and out[0]["src"] == "ChEMBL"
    ev = ai.Evidence("")
    ev.add(out[0]["text"], "ChEMBL")
    assert ai.claim_ok("**BCR-ABL**: IC50 about 220 nM, the median of 49 measurements.", ["S1"], ev)[0]
    assert not ai.claim_ok("**BCR-ABL**: IC50 25 nM.", ["S1"], ev)[0]


@pytest.mark.parametrize("nm, text", [(407.4, "410 nM"), (38.9, "39 nM"), (23442.3, "23 µM"), (3.4, "3.4 nM"),
                                      (0.05, "50 pM"), (2_500_000, "2.5 mM")])
def test_concentrations_are_given_in_the_unit_one_would_say(nm, text):
    assert cs._nm_text(nm) == text


def test_median():
    assert cs._median([3.0, 1.0, 2.0]) == 2.0 and cs._median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_the_reading_list_names_papers_and_trials_once_each():
    paper = dict(cs._paper(PAPER), why="most cited")
    opts = {"_ai_research": [("papers", {"papers": [paper]}), ("papers", {"papers": [paper]}), ("trials", TRIALS)]}
    reading = v._ai_reading_list(opts)
    assert len(reading["papers"]) == 1 and len(reading["trials"]) == 1 and reading["trials_total"] == 114
    md = v._ai_reading_md(reading)
    assert "[Semaglutide in Patients with Heart Failure" in md and "pubmed.ncbi.nlm.nih.gov/37622681" in md
    assert "NCT06308653" in md and "1 of 114 on ClinicalTrials.gov" in md
    assert v._ai_reading_list({}) == {}


def test_a_slow_chembl_does_not_switch_off_the_literature_search(monkeypatch):
    """EBI serves both from one host; their failures are counted apart."""
    monkeypatch.setattr(cs, "_host_state", {})
    cs._record("www.ebi.ac.uk", False)
    cs._record("www.ebi.ac.uk", False)
    assert cs.host_degraded(cs.CHEMBL_BASE) and not cs.host_degraded("europepmc")


# -------------------------------------------------------------- live
@pytest.mark.network
def test_literature_search_finds_the_landmark_trial():
    res = cs.literature_search(("semaglutide", "heart failure"))
    assert res["ok"] and any(p["first_author"] == "Kosiborod" and p["why"] == "most cited" for p in res["papers"])
    assert any(p["why"] == "recent" for p in res["papers"]) and all(p["abstract"] and p["url"] for p in res["papers"])


@pytest.mark.network
def test_clinical_trials_lists_registered_studies():
    res = cs.clinical_trials("heart failure", "semaglutide")
    assert res["ok"] and res["total"] >= 10 and any(t["nct"] == "NCT04788511" for t in res["trials"])


@pytest.mark.network
def test_chembl_potency_puts_the_studied_targets_first():
    res = cs.chembl_potency("imatinib")
    if res.get("ok") is False:
        pytest.skip("ChEMBL didn't answer")
    names = " ".join(t["target"] for t in res["targets"][:4])
    assert "ABL1" in names and "Kit" in names and all(t["n"] >= 2 and math.isfinite(t["median_nm"]) for t in res["targets"])
    assert cs.chembl_potency("zzzznotadrug") == {}
