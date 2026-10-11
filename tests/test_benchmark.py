"""
Drug-identity benchmark (live sources) — the cases where the app has matched,
or could match, the WRONG drug, product or label. Run after any change to
name resolution, label selection, salt/tautomer handling or brand lookup:

    python -m pytest -m network            # everything live (~3–6 min)
    python -m pytest -m "network and not slow"   # label + resolver only (~1 min)

Live databases change, so a failure is either a regression in our rules or a
real-world change worth a look — the assertion message says which case.
Add a case here whenever a wrong-match bug is reported (reports/data_reports.jsonl).
"""
import re

import pytest

import chem_sources as cs
import chem_v4 as v

pytestmark = pytest.mark.network


def _gen(label):
    return (label.get("generic_names") or [""])[0].upper()


# ----------------------------------------------------------- FDA label choice
# (search name, expectation) — expectation is a regex the label's generic name
# must match, or None for "no label at all".
LABEL_CASES = [
    # glucosamine is a US supplement, not a drug — every hit is someone else's
    # product (Chromagen's 'levomefolate glucosamine', OTC joint blends)
    ("glucosamine", None),
    ("metformin", r"^METFORMIN HYDROCHLORIDE$"),          # not the sitagliptin combo
    ("ibuprofen", r"^IBUPROFEN$"),
    ("atorvastatin calcium", r"^ATORVASTATIN CALCIUM$"),
    ("warfarin", r"^WARFARIN SODIUM$"),
    ("acetaminophen", r"^ACETAMINOPHEN$"),
    ("doxycycline hyclate", r"^DOXYCYCLINE HYCLATE$"),
    ("fluticasone furoate", r"FUROATE"),                   # ester ≠ propionate (Flonase)
    # diphenoxylate is only sold with atropine — the Rx combo IS the right label
    ("diphenoxylate", r"DIPHENOXYLATE HYDROCHLORIDE AND ATROPINE SULFATE"),
]


@pytest.mark.parametrize("name,expect", LABEL_CASES, ids=[c[0] for c in LABEL_CASES])
def test_fda_label_choice(name, expect):
    label = cs.openfda_label(generic_name=name)
    if expect is None:
        assert not label, f"{name}: expected no label, got {label.get('brand_names')} " \
                          f"({_gen(label)})"
        return
    assert label, f"{name}: no label found"
    assert re.search(expect, _gen(label)), f"{name}: wrong label {_gen(label)!r}"
    assert label.get("effective_date"), f"{name}: label revision date missing"


@pytest.mark.parametrize("name,route", [
    # REGRESSION 2026-10-11: acetaminophen → IV injection label
    ("acetaminophen", "ORAL"), ("ibuprofen", "ORAL"), ("metformin", "ORAL"),
    ("doxycycline hyclate", "ORAL"),
    # a blanket 'prefer oral' rule got these wrong (oral vancomycin is C. diff
    # only; timolol is mostly an eye drop) — the drug's main form wins
    ("vancomycin", "INTRAVENOUS"), ("timolol", "OPHTHALMIC"),
    ("albuterol", "RESPIRATORY (INHALATION)"),
])
def test_label_is_for_the_drugs_main_form(name, route):
    label = cs.openfda_label(generic_name=name)
    assert route in [r.upper() for r in label.get("routes") or []], \
        f"{name}: label route {label.get('routes')} ({(label.get('brand_names') or [''])[0]})"


def test_singapore_form_steers_label_choice():
    # US labels for diclofenac are mostly topical gels; Singapore sells mostly
    # tablets, so in SG mode the oral label must be chosen
    sg_route = v.market_route("SG", "diclofenac")
    label = cs.openfda_label(generic_name="diclofenac", main_route=sg_route)
    assert sg_route == "ORAL" and "ORAL" in [r.upper() for r in label.get("routes") or []]


def test_known_boxed_warning_is_captured():
    label = cs.openfda_label(generic_name="warfarin")
    assert label.get("has_boxed_section") and re.search(r"bleed", label.get("boxed_warning", ""), re.I)
    assert "bleeding" in v.boxed_status(label).lower()


def test_absent_boxed_warning_is_scoped_to_this_label():
    # oral acetaminophen has none, but the IV product does — never 'the drug has none'
    label = cs.openfda_label(generic_name="acetaminophen")
    assert not label.get("has_boxed_section")
    assert v.boxed_status(label).startswith("None on this label (oral)")


def test_benzene_label_is_not_treated_as_medicine():
    # homeopathic 'Benzinum' is an OTC label — must never make benzene a drug
    label = cs.openfda_label(generic_name="benzene")
    assert not v.openfda_is_medicine(label, None, {})


# ------------------------------------------------------------- name resolver
RESOLVE_CASES = [
    # (input, region, acceptable first-candidate CIDs or None,
    #  brand-register ingredients regex or None)
    ("glucosamine", "SG", {439213}, None),
    ("aspirin", "SG", {2244}, None),
    # PubChem knows these brands as synonyms → straight to the ingredient
    ("Lipitor", "US", {60822, 60823}, None),        # atorvastatin calcium / free acid
    ("Panadol", "SG", {1983}, None),                 # paracetamol
    # brands PubChem doesn't list → register lookup (RxNorm / HSA)
    ("Pepto-Bismol", "US", None, r"^bismuth subsalicylate$"),
    ("Dhamotil", "SG", None, r"atropine.*\+.*diphenoxylate"),
]


@pytest.mark.parametrize("text,region,cids,brand", RESOLVE_CASES,
                         ids=[f"{c[0]}-{c[1]}" for c in RESOLVE_CASES])
def test_resolver(text, region, cids, brand):
    res = v.resolve_input(text, region)
    assert res["candidates"], f"{text}: nothing resolved ({res['error']})"
    if cids is not None:
        assert res["candidates"][0]["cid"] in cids, \
            f"{text}: resolved to CID {res['candidates'][0]['cid']}, expected one of {cids}"
    if brand is not None:
        ings = " + ".join((res["brand"] or {}).get("ingredients") or []).lower()
        assert re.search(brand, ings), f"{text}: brand → {ings or 'no brand mapping'}"
        # every ingredient of a combination brand is offered, not just the first
        assert len(res["candidates"]) == len(res["brand"]["ingredients"])


def test_smiles_input_is_taken_literally():
    res = v.resolve_input("CCO", "US")
    assert res["source"] == "smiles" and res["candidates"][0]["smiles"] == "CCO"


@pytest.mark.slow
def test_misspelling_is_not_silently_matched():
    # 'glucossamine' must fail loudly, never resolve to some other product
    res = v.resolve_input("glucossamine", "SG")
    assert not res["candidates"] and res["error"]


# -------------------------------------------------------------- full pipeline
def _load(text, region="SG"):
    res = v.resolve_input(text, region)
    assert res["candidates"], f"{text}: nothing resolved"
    return v.load_compound(text, res["candidates"][0], res["source"], region=region)


@pytest.mark.slow
def test_glucosamine_page_has_no_borrowed_label():
    ctx = _load("glucosamine")
    assert ctx["compound_name"] == "D-Glucosamine"
    assert not ctx["label"], f"borrowed label: {ctx['label'].get('brand_names')}"
    assert ctx["is_drug"]                     # still a known (library) medicine


@pytest.mark.slow
def test_salt_keeps_its_identity_and_gets_a_parent():
    ctx = _load("diphenoxylate hydrochloride")
    assert ctx["has_parent"], "salt should get a parent (free base) bundle"
    assert ctx["parent"]["cid"] == 13505
    assert "ATROPINE" in _gen(ctx["label"]), "only label is the Lomotil combination"


@pytest.mark.slow
def test_salt_pk_comes_from_parent_and_says_so():
    # REGRESSION 2026-10-11: the salt's own thin PK heading (one Wikipedia
    # line) hid the parent's measured ADME data; parent lines linked to the
    # SALT's PubChem page
    ctx = _load("diphenoxylate hydrochloride")
    items, scope = v._pk_items(ctx)
    parent = [(s, src, u) for s, src, u in items if v.PARENT_TAG in src]
    assert scope == "parent" and len(parent) >= 3
    assert all(u.endswith("/compound/13505") for _s, src, u in parent if src.startswith("PubChem"))
    row = v.compare_profile(ctx)
    assert row.get("_parent_pk") and "parent" in row["Half-life"]


@pytest.mark.slow
def test_doxycycline_tautomer_is_recovered():
    ctx = _load("doxycycline")
    assert ctx["clin_chembl"].get("chembl_id") == "CHEMBL1433", \
        f"got {ctx['clin_chembl'].get('chembl_id')!r}"


@pytest.mark.slow
def test_plain_chemical_is_not_a_drug():
    ctx = _load("benzene", "US")
    assert not ctx["is_drug"] and not ctx["label"]


@pytest.mark.slow
def test_metformin_gets_its_own_label():
    ctx = _load("metformin", "US")
    assert ctx["is_drug"]
    assert _gen(ctx["label"]) == "METFORMIN HYDROCHLORIDE", _gen(ctx["label"])
