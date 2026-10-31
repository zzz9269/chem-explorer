"""
Offline checks for 'drugs for <condition>' when the condition has no ATC group
(2026-10-30: 'drugs for autism' → "I couldn't identify autism"). The parser used
to send it to PubChem as a chemical name; the label search then has to tell a real
indication from a label that merely mentions the condition.
"""
import pytest

import chat_router as cr
import chem_sources as cs


@pytest.mark.parametrize("q", [
    "drugs for autism", "drugs for ASD", "medication for autism", "what treats autism",
    "autism treatment", "autism medication", "what medications are used for autism",
    "how to treat autism", "how is autism treated", "treatment of autism in children",
    "medicine for my son with autism",
])
def test_autism_questions_ask_for_its_medicines(q):
    p = cr.parse_query(q)
    assert p["mode"] == "condition"
    assert p["condition"]["label"] == "autism spectrum disorder"
    assert "autistic" in p["condition"]["search"]   # the labels' word, not the asker's


def test_any_named_condition_is_searched_not_looked_up_as_a_chemical():
    p = cr.parse_query("drugs for fibromyalgia")
    assert p["mode"] == "condition" and p["condition"]["search"] == ["fibromyalgia"]
    assert not p["entities"]


# REGRESSION: matched 'covid' in the viral-infections group → a list of HIV antivirals
def test_long_covid_is_not_acute_covid():
    p = cr.parse_query("drugs for long covid")
    assert p["mode"] == "condition" and p["condition"]["search"] == ["long covid"]


def test_second_condition_is_kept_for_a_follow_up():
    assert cr.parse_query("drugs for autism and adhd")["condition"]["also"] == ["adhd"]


# REGRESSION: in CONDITIONS, but 'adhd' stayed a leftover word so it went to PubChem
@pytest.mark.parametrize("q", ["drugs for adhd", "best drug for adhd", "adhd medication"])
def test_adhd_uses_its_atc_group(q):
    p = cr.parse_query(q)
    assert p["mode"] == "condition" and p["condition"]["label"] == "ADHD" and p["condition"]["codes"]


@pytest.mark.parametrize("q, mode", [
    ("drugs for hypertension", "condition"),            # still the ATC list
    ("drugs for kids", "clarify"),
    ("what is it for", "clarify"),
    ("what drugs are in the statin class", "class_list"),
    ("drugs to avoid in G6PD deficiency", "concept"),
])
def test_other_shapes_keep_their_meaning(q, mode):
    p = cr.parse_query(q)
    assert p["mode"] == mode
    assert not (p.get("condition") or {}).get("search")


@pytest.mark.parametrize("q", ["side effects of drugs for autism", "how do drugs for autism work",
                               "drugs for sale", "is amoxicillin good for autism", "2-methylpentane"])
def test_not_a_request_for_a_list(q):
    assert cr.parse_query(q)["mode"] != "condition"


def test_quote_finds_a_real_indication():
    text = ("1 INDICATIONS AND USAGE Risperidone is indicated for schizophrenia ( 1.1 ). "
            "Treatment of irritability associated with autistic disorder ( 1.3 ). Other text here.")
    assert cs.label_use_quote(text, "autistic") == "Treatment of irritability associated with autistic disorder."


# REGRESSION: shown as '…lithium or valproate, for the treatment of acute manic or mixed episodes
# associated with Bipolar I Disorder Treatment of irritability associated with autistic disorder
# 1.1 Schizophrenia Risperidone tablets are indicated for the treatment of schizophrenia.'
def test_quote_prefers_the_indications_own_sentence_to_the_opening_list():
    text = ("1 INDICATIONS AND USAGE Risperidone tablets are an atypical antipsychotic indicated for: Treatment of "
            "schizophrenia ( 1.1 ) As monotherapy or adjunctive therapy with lithium or valproate, for the treatment "
            "of acute manic or mixed episodes associated with Bipolar I Disorder ( 1.2 ) Treatment of irritability "
            "associated with autistic disorder ( 1.3 ) 1.1 Schizophrenia Risperidone tablets are indicated for the "
            "treatment of schizophrenia. Efficacy was established in 4 short-term trials in adults. 1.3 Irritability "
            "Associated with Autistic Disorder Risperidone tablets are indicated for the treatment of irritability "
            "associated with autistic disorder, including symptoms of aggression towards others, deliberate "
            "self-injuriousness, temper tantrums, and quickly changing moods. Efficacy was established in 3 trials.")
    quote = cs.label_use_quote(text, "autistic disorder")
    assert "aggression towards others" in quote and quote.endswith("quickly changing moods.")
    assert "lithium" not in quote and "Schizophrenia" not in quote and not quote.startswith("1.3")


def test_a_label_with_only_the_opening_list_still_gives_a_quote():
    text = ("INDICATIONS AND USAGE Drug is indicated for: Treatment of pain Treatment of irritability associated "
            "with autistic disorder 1.1 Pain Drug is indicated for pain.")
    assert "autistic disorder" in cs.label_use_quote(text, "autistic disorder")


# REGRESSION: clomiphene was listed for endometriosis from these sentences
@pytest.mark.parametrize("text, term", [
    ("Clomiphene is indicated for treatment of ovulatory dysfunction. Endometriosis and Endometrial Carcinoma. "
     "The incidence of endometriosis and endometrial carcinoma increases with age.", "endometriosis"),
    ("Indicated for the treatment of pain. Not indicated for the treatment of lupus.", "lupus"),
    ("Treatment of pain. Endometriosis must be excluded before treatment.", "endometriosis"),
    ("The tablets are white and marked on one side with the word lupus.", "lupus"),
    # phentermine: obesity is its use, and diabetes only one of the risk factors that qualify a patient
    ("Short-term adjunct in a regimen of weight reduction in patients with an initial body mass index greater "
     "than 27 in the presence of other risk factors (e.g., controlled hypertension, diabetes, hyperlipidemia).",
     "diabetes"),
])
def test_quote_rejects_a_mention_that_isnt_a_use(text, term):
    assert cs.label_use_quote(text, term) == ""


def test_heading_then_its_indication_is_a_use():
    text = "INDICATIONS AND USAGE Endometriosis. Danazol capsules are indicated for the treatment of endometriosis."
    assert "indicated for the treatment" in cs.label_use_quote(text, "endometriosis")


@pytest.mark.parametrize("generic, names", [
    ("METHYLPREDNISOLONE SODIUM SUCCINATE", ("Methylprednisolone",)),
    ("ARIPIPRAZOLE ORALLY DISINTEGRATING", ("Aripiprazole",)),
    ("BETAMETHASONE SODIUM PHOSPHATE AND BETAMETHASONE ACETATE", ("Betamethasone",)),
    ("DEXTROAMPHETAMINE SACCHARATE, AMPHETAMINE ASPARTATE, DEXTROAMPHETAMINE SULFATE AND AMPHETAMINE SULFATE",
     ("Amphetamine", "Dextroamphetamine")),
    ("DIMETHYL FUMARATE", ("Dimethyl fumarate",)),
    ("CV, PREGABALIN", ("Pregabalin",)),
])
def test_label_generic_names_group_by_ingredient(generic, names):
    assert cs._base_ingredients(generic) == names
