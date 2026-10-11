"""
The question bank (question_bank.json) and everything that uses it:

* every example phrasing is understood (mode, topic, question form);
* phrasings never seen while building the bank (tests/heldout_questions.py);
* safety routing (crisis / refuse / pets) — including things that must NOT trip it;
* the 'Short answer' verdicts, checked offline against small fake source texts.

Offline: no network needed.
"""
import re

import pytest

import chat_router as cr
import chem_v4 as v
import question_bank as qb
from heldout_questions import BATCH_1, BATCH_2, BATCH_3

KNOWN_TOPICS = set(cr.INTENT_TITLES) | {"population"}
KNOWN_MODES = {"answer", "compare", "interaction", "relation", "class_list", "condition", "help",
               "thanks", "crisis", "refuse", "pets", "unknown", "clarify", "multi"}


# ------------------------------------------------------------------ the bank
def test_bank_is_well_formed():
    types = qb.types()
    assert len(types) >= 70
    ids = [t["id"] for t in types]
    assert len(ids) == len(set(ids)), "duplicate question-type ids"
    for t in types:
        exp = t.get("expect") or {}
        assert exp.get("mode") in KNOWN_MODES, t["id"]
        assert set(exp.get("topics") or []) <= KNOWN_TOPICS, t["id"]
        assert t.get("examples") and t.get("form") and t.get("answer"), t["id"]
        assert set(t["form"]) <= set(qb.load()["forms"]), t["id"]
        slots = qb.slots_for(t)
        for ex in t["examples"]:
            for name in re.findall(r"\{(\w+)\}", ex):
                assert name in slots, f"{t['id']}: unknown slot {{{name}}}"


def _check(qtype, question):
    exp = qtype["expect"]
    p = cr.parse_query(question)
    problems = []
    if p["mode"] != exp["mode"]:
        problems.append(f"mode={p['mode']}")
    if exp.get("topics") and p["mode"] == "answer" and (p["intents"] or ["-"])[0] not in exp["topics"]:
        problems.append(f"topic={p['intents']}")
    if exp.get("population") and exp["population"] not in p.get("population", []):
        problems.append(f"population={p.get('population')}")
    if exp.get("same_as") and not p.get("same_as"):
        problems.append("same_as missing")
    if not qb.form_ok(p.get("form", ""), qtype["form"]):
        problems.append(f"form={p.get('form')}")
    return problems


def _bank_failures():
    fails = []
    for qt in qb.types():
        slots = qb.slots_for(qt)
        for ex in qt["examples"]:
            for i in (0, 1):
                q = qb.instantiate(ex, slots, i)
                probs = _check(qt, q)
                if probs:
                    fails.append(f"{qt['id']}: {q!r} {probs}")
    return fails


def test_every_bank_example_is_understood():
    fails = _bank_failures()
    assert not fails, "\n".join(fails[:30])


def test_rules_alone_understand_the_bank(monkeypatch):
    # the example matcher is a safety net, not a crutch: the keyword rules on
    # their own should already handle (nearly) every bank phrasing
    monkeypatch.setattr(cr, "USE_TEMPLATE_MATCHER", False)
    fails = _bank_failures()
    total = sum(len(t["examples"]) * 2 for t in qb.types())
    assert len(fails) <= total * 0.02, "\n".join(fails[:30])


# ------------------------------------------------------- unseen phrasings
def _heldout_failures(batch):
    fails = []
    for q, mode, topics, forms, extra in batch:
        p = cr.parse_query(q)
        probs = []
        if p["mode"] != mode:
            probs.append(f"mode={p['mode']}")
        if topics and p["mode"] == "answer" and (p["intents"] or ["-"])[0] not in topics:
            probs.append(f"topic={p['intents']}")
        if forms and not qb.form_ok(p.get("form", ""), list(forms)):
            probs.append(f"form={p.get('form')}")
        if extra.get("population") and extra["population"] not in p.get("population", []):
            probs.append(f"population={p.get('population')}")
        if extra.get("allergen") and extra["allergen"] not in (p.get("allergen") or "").lower():
            probs.append(f"allergen={p.get('allergen')!r}")
        if extra.get("same_as") and not p.get("same_as"):
            probs.append("same_as missing")
        if probs:
            fails.append(f"{q!r}: {probs}")
    return fails


@pytest.mark.parametrize("batch", [BATCH_1, BATCH_2, BATCH_3], ids=["batch1", "batch2", "batch3"])
def test_heldout_questions(batch):
    fails = _heldout_failures(batch)
    assert len(fails) <= len(batch) * 0.03, "\n".join(fails)


def test_combination_brand_is_not_the_same_as_its_ingredient():
    # REGRESSION 2026-10-12: Augmentin (amoxicillin + clavulanate) was merged
    # with amoxicillin, so 'amoxicillin vs augmentin' lost the comparison
    p = cr.parse_query("amoxicillin vs augmentin")
    assert p["mode"] == "compare" and not p["same_as"]


@pytest.mark.parametrize("q,mode", [
    ("what's the antidote for paracetamol overdose", "answer"),   # its own overdose
    ("why do they give naloxone in an overdose", "relation"),     # the link to overdose
    ("how does sertraline work in the brain", "answer"),          # a body part isn't a target
])
def test_relation_targets(q, mode):
    assert cr.parse_query(q)["mode"] == mode


@pytest.mark.parametrize("q,topic", [
    ("ibuprofen who should not", "warnings"),
    ("metformin process in body", "pk"),
    ("ibuprofen kidney", "warnings"),
    ("ibuprofen after drinking", "interactions"),
])
def test_vocabulary_matcher_rescues_unfamiliar_wording(monkeypatch, q, topic):
    # terse phrasings no keyword rule covers — answered via the bank's vocabulary
    p = cr.parse_query(q)
    assert p["intents"][0] == topic and p["template"], (q, p["intents"], p["template"])
    monkeypatch.setattr(cr, "USE_TEMPLATE_MATCHER", False)
    assert cr.parse_query(q)["intents"][0] != topic        # the rules alone miss it


def test_slang_is_expanded():
    assert cr.expand_shorthand("does it make u sleepy b4 bed") == "does it make you sleepy before bed"
    assert cr.expand_shorthand("100 u of insulin") == "100 u of insulin"     # units, not 'you'
    assert "IM" in cr.expand_shorthand("given IM")                          # intramuscular


def test_every_showcase_example_is_understood():
    # the examples shown to users must themselves work — never teach a dud
    for cat, qs in qb.load()["showcase"].items():
        for q in qs:
            p = cr.parse_query(q)
            assert p["mode"] not in ("unknown", "clarify", "multi"), (cat, q, p["mode"])
            if p["mode"] == "answer":
                assert p["intents"][0] != "search", (cat, q, p["intents"])


def test_suggestions_fill_in_the_drug():
    qs = qb.suggestions_for("metformin", 3, skip_topics=("uses",))
    assert len(qs) == 3 and all("metformin" in q for q in qs)
    assert not any("used for" in q for q in qs)
    assert len(set(qb.showcase(6, seed=1))) == 6


# ------------------------------------------------------------- question form
@pytest.mark.parametrize("q,form", [
    ("does drugx cause weight gain", "yes_no"), ("is drugx safe in pregnancy", "yes_no"),
    ("how long does drugx last", "how_long"), ("half life of drugx", "how_long"),
    ("how much drugx can i take", "how_much"), ("what's the max dose of drugx", "how_much"),
    ("how often can i take drugx", "how_often"), ("how many times a day drugx", "how_often"),
    ("when should i take drugx", "when"), ("should i take drugx in the morning or at night", "when"),
    ("why does drugx cause cough", "why"), ("which is better drugx or drugx", "which"),
    ("drugx vs drugx", "which"), ("how common is nausea on drugx", "how_common"),
    ("how does drugx work", "how"), ("list of statins", "list"), ("what is drugx", "what"),
    ("drugx", "statement"), ("what if i miss a dose of drugx", "what"),
])
def test_question_form(q, form):
    assert qb.question_form(q) == form


# ----------------------------------------------------------- safety routing
@pytest.mark.parametrize("q,mode", [
    ("i want to kill myself", "crisis"), ("how many paracetamol would kill me", "crisis"),
    ("thinking about ending my life", "crisis"), ("how much amitriptyline is fatal for me", "crisis"),
    ("how to make meth", "refuse"), ("how can i extract codeine from tablets", "refuse"),
    ("my dog ate my ibuprofen", "pets"), ("can cats have paracetamol", "pets"),
])
def test_safety_routing(q, mode):
    assert cr.parse_query(q)["mode"] == mode


@pytest.mark.parametrize("q", [
    "does penicillin cause bacteria to die",     # not self-harm
    "is paracetamol lethal in overdose",          # a toxicology question → overdose answer
    "how is morphine made",                       # educational synthesis
    "i have a dog, can i take ibuprofen",         # about the person, not the pet
    "what is the lethal dose of caffeine",
])
def test_safety_routing_false_positives(q):
    assert cr.parse_query(q)["mode"] not in ("crisis", "refuse", "pets"), q


def test_pet_question_flags_urgency():
    assert cr.parse_query("my cat licked ibuprofen gel")["urgent"]
    assert not cr.parse_query("is ibuprofen safe for dogs")["urgent"]


# ------------------------------------------------------ short-answer logic
def _ctx(docs=(), **kw):
    """A minimal loaded-compound context with an injected source corpus."""
    base = {"label": {}, "grouped": {}, "wiki_secs": {}, "wiki": {}, "has_parent": False,
            "region": "SG", "market": {}, "drug_hit": None, "clin_chembl": {}, "smiles": "",
            "compound_name": "Testdrug"}
    base.update(kw)
    base["_corpus"] = [dict({"url": "", "section": ""}, **d) for d in docs]
    return base


def _short(blocks):
    assert blocks and blocks[0]["t"] == "answer", blocks
    return " ".join(t for t, _s, _u in blocks[0]["items"])


def _p(raw, intents, form, **kw):
    return dict({"raw": raw, "intents": intents, "form": form, "focus": []}, **kw)


class TestEffectVerdict:
    def test_listed_on_label(self):
        ctx = _ctx([{"text": "Weight gain was reported in 4% of patients.", "src": "FDA label",
                     "topic": "side_effects"}])
        out = _short(v._effect_verdict(ctx, "Testdrug", _p("does testdrug cause weight gain", [], "yes_no")))
        assert out.startswith("**Yes — Testdrug's US FDA label lists it")

    def test_counteracting_is_not_causing(self):
        # REGRESSION 2026-10-12: 'counteracting the weight gain caused by
        # antipsychotics' was read as metformin CAUSING weight gain
        ctx = _ctx([{"text": "It is effective in counteracting the weight gain caused by olanzapine.",
                     "src": "Wikipedia", "topic": "uses"}])
        out = _short(v._effect_verdict(ctx, "Testdrug", _p("does testdrug cause weight gain", [], "yes_no")))
        assert out.startswith("**No — the sources describe Testdrug as working against this")

    def test_side_effect_list_beats_overdose_case(self):
        ctx = _ctx([{"text": "After an overdose, drowsiness was observed.", "src": "FDA label",
                     "topic": "overdose"},
                    {"text": "Somnolence occurred in 14% of patients.", "src": "FDA label",
                     "topic": "side_effects"}])
        out = _short(v._effect_verdict(ctx, "Testdrug", _p("does testdrug make you sleepy", [], "yes_no")))
        assert "14%" in out

    def test_sources_say_no(self):
        ctx = _ctx([{"text": "Testdrug does not cause weight gain.", "src": "Wikipedia",
                     "topic": "side_effects"}])
        out = _short(v._effect_verdict(ctx, "Testdrug", _p("does testdrug cause weight gain", [], "yes_no")))
        assert out.startswith("**Not usually")

    def test_intended_effect(self):
        ctx = _ctx([{"text": "Testdrug is used to lower blood sugar in type 2 diabetes.",
                     "src": "FDA label", "topic": "uses"}])
        out = _short(v._effect_verdict(ctx, "Testdrug",
                                       _p("does testdrug lower blood sugar", [], "yes_no")))
        assert "intended effect" in out

    def test_mechanism_effect_is_not_called_intended(self):
        ctx = _ctx([{"text": "Its glucocorticoid effects include increased blood glucose.",
                     "src": "FDA label", "topic": "mechanism"}])
        out = _short(v._effect_verdict(ctx, "Testdrug",
                                       _p("does testdrug raise blood sugar", [], "yes_no")))
        assert "known effect of how Testdrug works" in out

    def test_symptom_word_is_not_a_prefix(self):
        # REGRESSION 2026-10-12: 'sleepy' matched 'sleeplessness'
        ctx = _ctx([{"text": "Stop use if sleeplessness persists.", "src": "FDA label", "topic": "warnings"}])
        out = _short(v._effect_verdict(ctx, "Testdrug", _p("does testdrug make you sleepy", [], "yes_no")))
        assert out.startswith("**Not mentioned")

    def test_two_word_effect_must_be_a_phrase(self):
        # REGRESSION 2026-10-12: 'increased sweating … weight' read as weight gain
        ctx = _ctx([{"text": "Increased sweating, tremor and changes in weight were reported.",
                     "src": "FDA label", "topic": "side_effects"}])
        out = _short(v._effect_verdict(ctx, "Testdrug", _p("does testdrug cause weight gain", [], "yes_no")))
        assert out.startswith("**Not mentioned")

    def test_not_mentioned_is_not_a_no(self):
        out = _short(v._effect_verdict(_ctx(), "Testdrug", _p("does testdrug cause hair loss", [], "yes_no")))
        assert out.startswith("**Not mentioned") and "doesn't rule it out" in out

    def test_synonyms(self):
        ctx = _ctx([{"text": "Somnolence occurred in 14% of patients.", "src": "FDA label",
                     "topic": "side_effects"}])
        out = _short(v._effect_verdict(ctx, "Testdrug", _p("does testdrug make you sleepy", [], "yes_no")))
        assert out.startswith("**Yes")


def test_interactant_verdict_warns_or_says_unknown():
    ctx = _ctx([{"text": "Avoid alcohol during treatment and for 3 days afterwards.",
                 "src": "FDA label", "topic": "interactions"}])
    out = _short(v._interactant_verdict(ctx, "Testdrug", _p("", [], "yes_no", interactant="alcohol")))
    assert out.startswith("**Caution")
    out = _short(v._interactant_verdict(_ctx(), "Testdrug", _p("", [], "yes_no", interactant="alcohol")))
    assert "isn't proof it's safe" in out


def test_controlled_substance_verdict():
    ctx = _ctx(label={"controlled_substance": "Testdrug contains codeine, a Schedule II controlled substance."})
    assert "Schedule II" in _short(v._controlled_verdict(ctx, "Testdrug", _p("", [], "yes_no")))
    ctx = _ctx(label={"controlled_substance": "Testdrug is not a controlled substance."})
    assert _short(v._controlled_verdict(ctx, "Testdrug", _p("", [], "yes_no"))).startswith("**No")


def test_availability_verdict_singapore():
    sg = {"classes": ["POM"], "class_counts": {"POM": 4}, "combination_only": False,
          "combo_class_counts": {}, "partners": []}
    ctx = _ctx(market={"sg": sg})
    out = _short(v._availability_verdict(ctx, "Testdrug",
                                         _p("do i need a prescription for testdrug", [], "yes_no")))
    assert "Prescription-only in Singapore" in out
    out = _short(v._availability_verdict(_ctx(market={}), "Testdrug",
                                         _p("is testdrug available", [], "yes_no")))
    assert "No HSA-registered" in out


def test_half_life_rule_of_thumb():
    ctx = _ctx(label={"pharmacokinetics": "The elimination half-life of Testdrug is 20 to 50 hours."})
    out = _short(v._duration_lead(ctx, "Testdrug",
                                  _p("how long does testdrug stay in your system", ["pk"], "how_long")))
    assert "Half-life" in out and "100–250 hours" in out


@pytest.mark.parametrize("smiles,raw,expect", [
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "is ibuprofen chiral", "Yes — 1 stereocentre"),
    ("CC(=O)Oc1ccccc1C(=O)O", "is aspirin chiral", "No — no stereocentres"),
    ("CC(=O)Oc1ccccc1C(=O)O", "is aspirin acidic", "A (weak) acid"),
    ("CN1CCC[C@H]1c1cccnc1", "is nicotine a base", "A (weak) base"),
])
def test_chemistry_verdicts_are_computed(smiles, raw, expect):
    out = _short(v._chem_verdict(_ctx(smiles=smiles), "X", _p(raw, ["properties"], "yes_no")))
    assert expect in out


def test_ingredient_verdict():
    ctx = _ctx(label={"inactive_ingredients": "lactose monohydrate, magnesium stearate, gelatin"})
    assert "lists lactose" in _short(v._ingredients_verdict(ctx, "X", _p("does it contain lactose", [], "yes_no")))
    assert _short(v._ingredients_verdict(ctx, "X", _p("is it gluten free", [], "yes_no"))).startswith("**Not listed")
    assert "gelatin" in _short(v._ingredients_verdict(ctx, "X", _p("is it halal", [], "yes_no")))


def test_medication_guide_is_split_into_topics():
    mg = ("Medication Guide Testdrug What is the most important information I should know about "
          "Testdrug? It can cause bleeding. How should I take Testdrug? Take it once daily. If you "
          "miss a dose, take it as soon as you remember. What should I avoid while taking Testdrug? "
          "Avoid alcohol. What are the possible side effects of Testdrug? Nausea and headache. "
          "How should I store Testdrug? Store at room temperature.")
    topics = {head: tp for head, _b, tp in v.medguide_sections(mg)}
    assert topics["How should I take Testdrug?"] == "dosing"
    assert topics["What should I avoid while taking Testdrug?"] == "interactions"
    assert topics["What are the possible side effects of Testdrug?"] == "side_effects"
    assert topics["How should I store Testdrug?"] == "storage"


def test_allergy_verdict_flags_same_class():
    drug = {"name": "Amoxicillin", "classes": []}
    ctx = _ctx([{"text": "Contraindicated in patients with a history of hypersensitivity to "
                         "penicillins.", "src": "FDA label", "topic": "warnings"}], drug_hit=drug)
    out = _short(v._allergy_verdict(ctx, "Amoxicillin", _p("", ["warnings"], "yes_no", allergen="Penicillin")))
    assert "mentions Penicillin allergy" in out or "is itself in" in out
